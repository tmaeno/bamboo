"""PanDA documentation graph navigator.

Builds a hierarchical graph from the PanDA WMS ReadTheDocs HTML pages,
summarises every node with an LLM, stores embeddings in a dedicated Qdrant
collection (``panda_docs``), and provides a dual-strategy search:

- **Flat semantic search** — fast cosine similarity over all nodes at all
  levels; good at paraphrase/synonym matches.
- **LLM-guided top-down traversal** — LLM reads page summaries, picks
  relevant pages, then drills into section summaries with parent context;
  catches non-obvious relevance that embedding distance misses.

Both strategies run in parallel; results are merged and deduped by URL.

Staleness is detected by comparing the GitHub tree SHA stored in
``bamboo/data/panda_docs_meta.json`` against the upstream repo.  The index
is rebuilt automatically when the SHA changes or the file is absent.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage

from bamboo.config import get_settings
from bamboo.llm import get_embeddings, get_extraction_llm, get_summary_llm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PANDA_DOCS_ORG = "tmaeno"
_PANDA_DOCS_TREE_URL = (
    f"https://api.github.com/repos/{_PANDA_DOCS_ORG}/panda-docs/git/trees/main?recursive=1"
)
_PANDA_DOCS_HTML_BASE = "https://panda-wms.readthedocs.io/en/latest"

_GH_HEADERS = {
    "User-Agent": "bamboo-panda-doc-navigator",
    "X-GitHub-Api-Version": "2022-11-28",
}

_EXCLUDED_RST_PATHS: frozenset[str] = frozenset({
    "docs/source/advanced/task_params.rst",
    "docs/source/advanced/gdpconfig.rst",
})

_COLLECTION = "panda_docs"
_META_FILE = Path(__file__).parent.parent / "data" / "panda_docs_meta.json"

# ---------------------------------------------------------------------------
# LLM prompt templates
# ---------------------------------------------------------------------------

_SUMMARIZE_PROMPT = (
    "Summarize the following PanDA WMS documentation section in 2-3 sentences.\n"
    "Preserve verbatim: all parameter names, option flags (e.g. --nFilesPerJob), "
    "error codes, class names, and any text that appeared in code formatting.\n"
    "Keep it concise and factual.\n\n"
    "Title: {title}\n\n"
    "Content:\n{content}"
)

_TRAVERSAL_PAGE_PROMPT = (
    "Search query: {query}\n\n"
    "Review these PanDA WMS documentation page summaries. "
    "Return a JSON array of IDs for pages that are likely to contain relevant information. "
    "Return an empty array [] if none are relevant.\n\n"
    "Pages:\n{pages_text}\n\n"
    "Return ONLY a JSON array of IDs, e.g.: [\"id1\", \"id2\"]"
)

_TRAVERSAL_SECTION_PROMPT = (
    "Search query: {query}\n\n"
    "You are exploring page: \"{page_title}\"\n"
    "Page context: {page_summary}\n\n"
    "Review these section summaries. "
    "Return a JSON array of IDs for sections likely to contain relevant information. "
    "Return an empty array [] if none are relevant.\n\n"
    "Sections:\n{sections_text}\n\n"
    "Return ONLY a JSON array of IDs, e.g.: [\"id1\", \"id2\"]"
)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class DocNode:
    id: str
    url: str
    level: int          # 0=page, 1=h2-section, 2=h3-subsection, …
    title: str
    content: str
    summary: str = ""
    parent_id: str | None = None
    children: list[str] = field(default_factory=list)


@dataclass
class DocResult:
    title: str
    url: str
    content: str
    parent_summary: str  # LLM summary of the direct parent node
    breadcrumb: list[str]  # ancestor titles from root to parent
    source: str          # "semantic" | "llm_traversal" | "both"
    score: float = 0.0


# ---------------------------------------------------------------------------
# PandaDocNavigator
# ---------------------------------------------------------------------------


class PandaDocNavigator:
    """Build and search a hierarchical graph of PanDA WMS documentation.

    Usage::

        nav = PandaDocNavigator()
        results = await nav.search("memory limit for jobs", top_k=5)

    :meth:`ensure_initialized` is called automatically by :meth:`search`, so
    explicit initialisation is only needed when you want to control timing.
    """

    def __init__(self) -> None:
        self._graph: dict[str, DocNode] = {}
        self._page_ids: list[str] = []
        self._initialized = False
        self._settings = get_settings()
        self._embeddings = get_embeddings()
        self._summary_llm = get_summary_llm()
        self._extraction_llm = get_extraction_llm()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def ensure_initialized(self) -> None:
        """Ensure the doc index is built and loaded into memory."""
        if self._initialized:
            return
        needs_rebuild = await self._check_staleness()
        if needs_rebuild:
            await self._build_index()
        else:
            await self._load_graph_from_qdrant()
        self._initialized = True

    async def search(self, query: str, top_k: int = 5) -> list[DocResult]:
        """Search the doc graph. Runs flat semantic + LLM traversal in parallel."""
        await self.ensure_initialized()

        semantic_task = self._semantic_search(query, top_k=top_k)
        traversal_task = self._llm_traversal(query)

        semantic_results, traversal_results = await asyncio.gather(
            semantic_task, traversal_task, return_exceptions=True
        )

        if isinstance(semantic_results, BaseException):
            logger.warning("PandaDocNavigator: semantic search failed: %s", semantic_results)
            semantic_results = []
        if isinstance(traversal_results, BaseException):
            logger.warning("PandaDocNavigator: LLM traversal failed: %s", traversal_results)
            traversal_results = []

        # Merge and dedup by URL; track which strategy found each result.
        merged: dict[str, DocResult] = {}
        for r in semantic_results:
            merged[r.url] = r
        for r in traversal_results:
            if r.url in merged:
                merged[r.url].source = "both"
            else:
                merged[r.url] = r

        # "both" first, then by descending semantic score.
        results = sorted(
            merged.values(),
            key=lambda r: (0 if r.source == "both" else 1, -r.score),
        )
        return results[:top_k]

    # ------------------------------------------------------------------
    # Staleness detection
    # ------------------------------------------------------------------

    async def _check_staleness(self) -> bool:
        current_sha = await self._fetch_tree_sha()
        if current_sha is None:
            logger.warning("PandaDocNavigator: GitHub unreachable — using existing index")
            return not _META_FILE.exists()
        meta = self._read_meta()
        if meta is None or meta.get("sha") != current_sha:
            logger.info(
                "PandaDocNavigator: SHA changed (%s → %s) — rebuilding index",
                meta.get("sha") if meta else "none",
                current_sha,
            )
            return True
        return False

    async def _fetch_tree_sha(self) -> str | None:
        try:
            import httpx  # noqa: PLC0415
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(_PANDA_DOCS_TREE_URL, headers=_GH_HEADERS)
                resp.raise_for_status()
                return resp.json().get("sha")
        except Exception as exc:
            logger.warning("PandaDocNavigator: failed to fetch tree SHA: %s", exc)
            return None

    def _read_meta(self) -> dict | None:
        try:
            return json.loads(_META_FILE.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def _write_meta(self, sha: str) -> None:
        from datetime import datetime, timezone  # noqa: PLC0415
        _META_FILE.parent.mkdir(parents=True, exist_ok=True)
        _META_FILE.write_text(json.dumps({
            "sha": sha,
            "built_at": datetime.now(timezone.utc).isoformat(),
        }))

    # ------------------------------------------------------------------
    # Graph loading from Qdrant
    # ------------------------------------------------------------------

    async def _load_graph_from_qdrant(self) -> None:
        """Rebuild in-memory graph by scrolling all Qdrant points."""
        client = self._make_qdrant_client()
        try:
            all_points = []
            offset = None
            while True:
                points, next_offset = await client.scroll(
                    collection_name=_COLLECTION,
                    offset=offset,
                    limit=100,
                    with_payload=True,
                    with_vectors=False,
                )
                all_points.extend(points)
                if next_offset is None:
                    break
                offset = next_offset
        finally:
            await client.close()

        self._graph = {}
        for point in all_points:
            p = point.payload
            node = DocNode(
                id=p["node_id"],
                url=p["url"],
                level=p["level"],
                title=p["title"],
                content=p["content"],
                summary=p["summary"],
                parent_id=p.get("parent_id"),
                children=p.get("children", []),
            )
            self._graph[node.id] = node

        self._page_ids = [nid for nid, n in self._graph.items() if n.level == 0]
        logger.info(
            "PandaDocNavigator: loaded %d nodes (%d pages) from Qdrant",
            len(self._graph), len(self._page_ids),
        )

    # ------------------------------------------------------------------
    # Index build pipeline
    # ------------------------------------------------------------------

    async def _build_index(self) -> None:
        logger.info("PandaDocNavigator: building doc index — this may take a few minutes")

        rst_paths, tree_sha = await self._fetch_rst_paths()
        if not rst_paths:
            logger.warning("PandaDocNavigator: no RST paths found — aborting build")
            return

        html_pages = await self._fetch_html_pages(rst_paths)

        from bs4 import BeautifulSoup  # noqa: PLC0415 — lazy import
        all_nodes: list[DocNode] = []
        for rst_path, html in html_pages.items():
            all_nodes.extend(self._parse_page_to_nodes(rst_path, html, BeautifulSoup))

        logger.info("PandaDocNavigator: parsed %d nodes across %d pages", len(all_nodes), len(html_pages))

        await self._summarize_nodes(all_nodes)
        await self._embed_and_upsert(all_nodes)

        self._graph = {n.id: n for n in all_nodes}
        self._page_ids = [n.id for n in all_nodes if n.level == 0]

        if tree_sha:
            self._write_meta(tree_sha)

        logger.info(
            "PandaDocNavigator: index ready — %d nodes, %d pages",
            len(all_nodes), len(self._page_ids),
        )

    async def _fetch_rst_paths(self) -> tuple[list[str], str | None]:
        try:
            import httpx  # noqa: PLC0415
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(_PANDA_DOCS_TREE_URL, headers=_GH_HEADERS)
                resp.raise_for_status()
                data = resp.json()
            sha = data.get("sha")
            rst_paths = [
                item["path"]
                for item in data.get("tree", [])
                if item.get("type") == "blob"
                and item.get("path", "").startswith("docs/source/")
                and item["path"].endswith(".rst")
                and item["path"] not in _EXCLUDED_RST_PATHS
            ]
            logger.info("PandaDocNavigator: discovered %d RST paths", len(rst_paths))
            return rst_paths, sha
        except Exception as exc:
            logger.warning("PandaDocNavigator: failed to fetch RST paths: %s", exc)
            return [], None

    async def _fetch_html_pages(self, rst_paths: list[str]) -> dict[str, str]:
        import httpx  # noqa: PLC0415

        async def _fetch_one(client: Any, path: str) -> tuple[str, str]:
            try:
                r = await client.get(_rst_path_to_html_url(path))
                r.raise_for_status()
                return path, r.text
            except Exception as exc:
                logger.debug("PandaDocNavigator: skipped %s: %s", path, exc)
                return path, ""

        async with httpx.AsyncClient(timeout=20) as client:
            fetched = await asyncio.gather(*[_fetch_one(client, p) for p in rst_paths])

        result = {path: html for path, html in fetched if html}
        logger.info("PandaDocNavigator: fetched %d/%d HTML pages", len(result), len(rst_paths))
        return result

    def _parse_page_to_nodes(self, rst_path: str, html: str, BeautifulSoup: Any) -> list[DocNode]:
        """Recursively parse an HTML page into DocNode tree using section nesting.

        ReadTheDocs renders each heading level as a nested ``<section>`` element,
        so the DOM hierarchy directly reflects the heading hierarchy.
        Orphan sub-sections (inconsistent nesting) become children of the nearest
        ancestor or the page root.
        """
        soup = BeautifulSoup(html, "html.parser")
        main = (
            soup.find("div", role="main")
            or soup.find("article")
            or soup.find("div", class_="document")
            or soup.body
        )
        if not main:
            return []

        nodes: list[DocNode] = []
        page_url = _rst_path_to_html_url(rst_path)

        def _node_id(url: str) -> str:
            return str(uuid.uuid5(uuid.NAMESPACE_URL, url))

        def _direct_text(element: Any) -> str:
            parts = []
            for child in element.children:
                if getattr(child, "name", None) == "section":
                    continue
                if hasattr(child, "get_text"):
                    parts.append(child.get_text(" ", strip=True))
                elif isinstance(child, str):
                    parts.append(child.strip())
            return re.sub(r"\s+", " ", " ".join(parts)).strip()

        def _parse_section(elem: Any, level: int, parent_id: str | None) -> DocNode:
            heading = elem.find(["h1", "h2", "h3", "h4"], recursive=False)
            title = heading.get_text(" ", strip=True) if heading else ""
            anchor = elem.get("id", "")
            url = f"{page_url}#{anchor}" if anchor else page_url
            node = DocNode(
                id=_node_id(url),
                url=url,
                level=level,
                title=title,
                content=_direct_text(elem),
                parent_id=parent_id,
            )
            nodes.append(node)
            for child_sec in elem.find_all("section", recursive=False):
                child = _parse_section(child_sec, level + 1, node.id)
                node.children.append(child.id)
            return node

        top_sections = main.find_all("section", recursive=False)
        if top_sections:
            for sec in top_sections:
                _parse_section(sec, level=0, parent_id=None)
        else:
            body_text = re.sub(r"\s+", " ", main.get_text(" ", strip=True)).strip()
            file_title = rst_path.split("/")[-1].replace(".rst", "").replace("_", " ").title()
            nodes.append(DocNode(
                id=_node_id(page_url),
                url=page_url,
                level=0,
                title=file_title,
                content=body_text,
            ))

        return nodes

    async def _summarize_nodes(self, nodes: list[DocNode]) -> None:
        semaphore = asyncio.Semaphore(10)

        async def _one(node: DocNode) -> None:
            if not node.content.strip():
                node.summary = node.title
                return
            prompt = _SUMMARIZE_PROMPT.format(
                title=node.title,
                content=node.content[:3000],
            )
            async with semaphore:
                try:
                    resp = await self._summary_llm.ainvoke([HumanMessage(content=prompt)])
                    node.summary = resp.content.strip()
                except Exception as exc:
                    logger.warning(
                        "PandaDocNavigator: summarize failed for %r: %s", node.title, exc
                    )
                    node.summary = node.content[:300]

        await asyncio.gather(*[_one(n) for n in nodes])
        logger.info("PandaDocNavigator: summarized %d nodes", len(nodes))

    async def _embed_and_upsert(self, nodes: list[DocNode]) -> None:
        from qdrant_client.models import Distance, PointStruct, VectorParams  # noqa: PLC0415

        client = self._make_qdrant_client()
        try:
            collections = await client.get_collections()
            existing = {c.name for c in collections.collections}
            if _COLLECTION in existing:
                await client.delete_collection(_COLLECTION)
            await client.create_collection(
                collection_name=_COLLECTION,
                vectors_config=VectorParams(
                    size=self._settings.embedding_dimension,
                    distance=Distance.COSINE,
                ),
            )

            texts = [f"{n.title} {n.summary}" for n in nodes]
            embeddings = await asyncio.to_thread(self._embeddings.embed_documents, texts)

            points = [
                PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, node.url)),
                    vector=emb,
                    payload={
                        "node_id": node.id,
                        "url": node.url,
                        "level": node.level,
                        "title": node.title,
                        "content": node.content,
                        "summary": node.summary,
                        "parent_id": node.parent_id,
                        "children": node.children,
                    },
                )
                for node, emb in zip(nodes, embeddings)
            ]

            batch = 100
            for i in range(0, len(points), batch):
                await client.upsert(collection_name=_COLLECTION, points=points[i:i + batch])

            logger.info("PandaDocNavigator: upserted %d vectors to Qdrant", len(points))
        finally:
            await client.close()

    # ------------------------------------------------------------------
    # Search strategies
    # ------------------------------------------------------------------

    async def _semantic_search(self, query: str, top_k: int) -> list[DocResult]:
        query_emb = await asyncio.to_thread(self._embeddings.embed_query, query)
        client = self._make_qdrant_client()
        try:
            response = await client.query_points(
                collection_name=_COLLECTION,
                query=query_emb,
                limit=top_k,
                score_threshold=0.5,
                with_payload=True,
            )
        finally:
            await client.close()

        results = []
        for point in response.points:
            node = self._graph.get(point.payload["node_id"])
            if node:
                results.append(self._make_result(node, source="semantic", score=point.score))
        return results

    async def _llm_traversal(self, query: str) -> list[DocResult]:
        """Top-down LLM traversal: choose pages, then sections within each page."""
        if not self._page_ids:
            return []

        page_items = [(pid, self._graph[pid]) for pid in self._page_ids if pid in self._graph]
        pages_text = "\n".join(
            f'- ID: "{pid}", Title: {node.title!r}, Summary: {node.summary}'
            for pid, node in page_items
        )
        chosen_page_ids = await self._llm_select(
            _TRAVERSAL_PAGE_PROMPT.format(query=query, pages_text=pages_text),
            candidates=[pid for pid, _ in page_items],
        )
        if not chosen_page_ids:
            return []

        async def _explore_page(page_id: str) -> list[DocResult]:
            page = self._graph.get(page_id)
            if page is None:
                return []
            if not page.children:
                return [self._make_result(page, source="llm_traversal")]

            child_items = [(cid, self._graph[cid]) for cid in page.children if cid in self._graph]
            sections_text = "\n".join(
                f'- ID: "{cid}", Title: {node.title!r}, Summary: {node.summary}'
                for cid, node in child_items
            )
            chosen_ids = await self._llm_select(
                _TRAVERSAL_SECTION_PROMPT.format(
                    query=query,
                    page_title=page.title,
                    page_summary=page.summary,
                    sections_text=sections_text,
                ),
                candidates=[cid for cid, _ in child_items],
            )
            return [
                self._make_result(self._graph[cid], source="llm_traversal")
                for cid in chosen_ids
                if cid in self._graph
            ]

        page_results = await asyncio.gather(
            *[_explore_page(pid) for pid in chosen_page_ids],
            return_exceptions=True,
        )
        results: list[DocResult] = []
        for pr in page_results:
            if isinstance(pr, list):
                results.extend(pr)
        return results

    async def _llm_select(self, prompt: str, candidates: list[str]) -> list[str]:
        """Prompt the LLM to select relevant IDs from *candidates*. Returns a filtered subset."""
        try:
            response = await self._extraction_llm.ainvoke([HumanMessage(content=prompt)])
            text = response.content.strip()
            match = re.search(r"\[.*?\]", text, re.DOTALL)
            if not match:
                return []
            selected: list[str] = json.loads(match.group())
            candidate_set = set(candidates)
            return [s for s in selected if s in candidate_set]
        except Exception as exc:
            logger.warning("PandaDocNavigator: LLM select failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Result construction
    # ------------------------------------------------------------------

    def _make_result(self, node: DocNode, source: str, score: float = 0.0) -> DocResult:
        """Build a DocResult, walking the parent chain for breadcrumbs."""
        breadcrumb: list[str] = []
        parent_summary = ""
        parent_id = node.parent_id
        while parent_id:
            parent = self._graph.get(parent_id)
            if parent is None:
                break
            breadcrumb.insert(0, parent.title)
            if not parent_summary:
                parent_summary = parent.summary
            parent_id = parent.parent_id
        return DocResult(
            title=node.title,
            url=node.url,
            content=node.content,
            parent_summary=parent_summary,
            breadcrumb=breadcrumb,
            source=source,
            score=score,
        )

    # ------------------------------------------------------------------
    # Qdrant client helper
    # ------------------------------------------------------------------

    def _make_qdrant_client(self) -> Any:
        from qdrant_client import AsyncQdrantClient  # noqa: PLC0415
        kwargs: dict[str, Any] = {
            "url": self._settings.qdrant_url,
            "check_compatibility": False,
        }
        if self._settings.qdrant_api_key:
            kwargs["api_key"] = self._settings.qdrant_api_key
        return AsyncQdrantClient(**kwargs)


# ---------------------------------------------------------------------------
# Module-level helper (mirrors panda_mcp_client._rst_path_to_html_url)
# ---------------------------------------------------------------------------


def _rst_path_to_html_url(rst_path: str) -> str:
    rel = rst_path.removeprefix("docs/source/").removesuffix(".rst") + ".html"
    return f"{_PANDA_DOCS_HTML_BASE}/{rel}"
