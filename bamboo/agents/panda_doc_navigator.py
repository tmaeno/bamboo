"""PanDA documentation graph navigator.

Builds a hierarchical graph from the PanDA WMS ReadTheDocs HTML pages,
summarises every node with an LLM, stores embeddings in a dedicated Qdrant
collection (``panda_docs``), and provides a dual-strategy search:

- **Flat semantic search** — fast cosine similarity over all nodes at all
  levels; good at paraphrase/synonym matches.
- **LLM-guided top-down traversal** — LLM reads page summaries, picks
  relevant pages, then drills into section summaries with parent context;
  catches non-obvious relevance that embedding distance misses.

All three strategies run in parallel; results are merged via Reciprocal Rank Fusion.

Staleness is detected by comparing the GitHub tree SHA stored in
``bamboo/data/panda_docs_meta.json`` against the upstream repo.  The index
is rebuilt automatically when the SHA changes or the file is absent.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage

from bamboo.config import get_settings
from bamboo.llm import get_embeddings, get_extraction_llm, get_reranker, get_summary_llm
from bamboo.utils.narrator import counting, say, thinking

logger = logging.getLogger(__name__)


def _embed_documents_silently(emb, texts: list[str]) -> list[list[float]]:
    _sink = io.StringIO()
    with contextlib.redirect_stdout(_sink), contextlib.redirect_stderr(_sink):
        return emb.embed_documents(texts)


def _embed_query_silently(emb, query: str) -> list[float]:
    _sink = io.StringIO()
    with contextlib.redirect_stdout(_sink), contextlib.redirect_stderr(_sink):
        return emb.embed_query(query)


def _predict_silently(model, pairs: list[tuple[str, str]]) -> list[float]:
    _sink = io.StringIO()
    with contextlib.redirect_stdout(_sink), contextlib.redirect_stderr(_sink):
        return model.predict(pairs).tolist()


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



def invalidate_doc_cache() -> bool:
    """Delete the doc-index metadata, forcing a full rebuild on next use.

    Returns ``True`` if the cache file existed and was deleted, ``False`` if it
    was already absent.
    """
    if _META_FILE.exists():
        _META_FILE.unlink()
        return True
    return False


# ---------------------------------------------------------------------------
# LLM prompt templates
# ---------------------------------------------------------------------------

_SUMMARIZE_PROMPT = (
    "Summarize the following PanDA WMS documentation section in 2-3 sentences.\n"
    "Preserve verbatim: all parameter names, option flags (e.g. --nFilesPerJob), "
    "error codes, class names, and any text that appeared in code formatting.\n"
    "Keep it concise and factual.\n\n"
    "Also classify the section:\n"
    '- "concept": explains PanDA terminology, system concepts, data structures, '
    "task/job states, or how a mechanism works\n"
    '- "other": operational guide, FAQ, API reference, parameter table, or how-to\n\n'
    "Title: {title}\n\n"
    "Content:\n{content}\n\n"
    'Return ONLY a JSON object: {{"summary": "...", "doc_type": "concept|other"}}'
)

_TRAVERSAL_PAGE_PROMPT = (
    "Search query: {query}\n\n"
    "Review these PanDA WMS documentation page summaries. "
    "Return a JSON array of IDs for pages whose content is relevant to the query — "
    "including pages that explain the concept, describe the status/error, "
    "list the parameters involved, or provide context needed to understand it. "
    "Exclude pages that are only about unrelated workflows or unrelated system components. "
    "Return an empty array [] if none apply.\n\n"
    "Pages:\n{pages_text}\n\n"
    "Return ONLY a JSON array of IDs, e.g.: [\"id1\", \"id2\"]"
)

_TRAVERSAL_SECTION_PROMPT = (
    "Search query: {query}\n\n"
    "You are exploring page: \"{page_title}\"\n"
    "Page context: {page_summary}\n\n"
    "Review these section summaries. "
    "Return a JSON array of IDs for sections that answer, explain, or provide relevant "
    "context for the query — including sections that describe the relevant status, "
    "error cause, or parameter. "
    "Exclude sections that are only about unrelated actions or unrelated system components. "
    "Return an empty array [] if none apply.\n\n"
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
    doc_type: str = "other"  # "concept" | "other"


@dataclass
class DocResult:
    title: str
    url: str
    content: str
    parent_summary: str  # LLM summary of the direct parent node
    breadcrumb: list[str]  # ancestor titles from root to parent
    source: str          # "semantic" | "llm_traversal" | "both"
    score: float = 0.0
    doc_type: str = "other"  # "concept" | "other"


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
        self._bm25: Any = None
        self._bm25_ids: list[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def ensure_initialized(self) -> None:
        """Ensure the doc index is built and loaded into memory."""
        if self._initialized:
            return
        say("Checking PanDA doc index …")
        needs_rebuild = await self._check_staleness()
        if needs_rebuild:
            await self._build_index()
        else:
            say("Doc index up to date — loading from Qdrant")
            await self._load_graph_from_qdrant()
        self._build_bm25_index()
        self._initialized = True

    async def search(
        self,
        query: str,
        top_k: int = 5,
        keyword_query: str | None = None,
    ) -> list[DocResult]:
        """Search the doc graph. Runs semantic, LLM traversal, and BM25 in parallel.

        query         — natural-language query for semantic search and LLM traversal.
        keyword_query — exact-term query for BM25 (defaults to query if not provided).
        """
        await self.ensure_initialized()
        _kw = keyword_query or query
        say(f"NL query (semantic+LLM): {query!r}")
        say(f"Keyword query (BM25):    {_kw!r}")

        semantic_task = self._semantic_search(query, top_k=top_k)
        traversal_task = self._llm_traversal(query)
        bm25_task = asyncio.to_thread(self._bm25_search, _kw, top_k)

        semantic_results, traversal_results, bm25_results = await asyncio.gather(
            semantic_task, traversal_task, bm25_task, return_exceptions=True
        )

        if isinstance(semantic_results, BaseException):
            logger.warning("PandaDocNavigator: semantic search failed: %s", semantic_results)
            semantic_results = []
        if isinstance(traversal_results, BaseException):
            logger.warning("PandaDocNavigator: LLM traversal failed: %s", traversal_results)
            traversal_results = []
        if isinstance(bm25_results, BaseException):
            logger.warning("PandaDocNavigator: BM25 search failed: %s", bm25_results)
            bm25_results = []

        say(
            f"Raw hits — semantic:{len(semantic_results)} "
            f"llm:{len(traversal_results)} "
            f"bm25:{len(bm25_results)}"
        )

        # Deduplicate by URL; label each result with every strategy that found it.
        merged: dict[str, DocResult] = {}
        _strategies: dict[str, list[str]] = {}
        for strategy_name, result_list in [
            ("semantic", semantic_results),
            ("llm", traversal_results),
            ("bm25", bm25_results),
        ]:
            for r in result_list:
                _strategies.setdefault(r.url, []).append(strategy_name)
                if r.url not in merged:
                    merged[r.url] = r
        for url, r in merged.items():
            r.source = ",".join(_strategies[url])
        results = list(merged.values())

        reranker = get_reranker()
        try:
            pairs = [(query, f"{r.title}\n{r.content[:500]}") for r in results]
            scores = await asyncio.to_thread(_predict_silently, reranker, pairs)
            for r, s in zip(results, scores):
                r.score = float(s)
            results = sorted(results, key=lambda r: -r.score)
            filtered = [r for r in results if r.score > 0.0]
            rejected = [r for r in results if r.score <= 0.0]
            n_before = len(results)
            if filtered:
                results = filtered
            say(
                f"Reranked: {len(filtered)}/{n_before} passed threshold "
                f"(top score {results[0].score:.2f})"
            )
            for r in rejected:
                label = f"{r.breadcrumb[-1]} › {r.title}" if r.breadcrumb else r.title
                say(f"  ✗ \\[{r.source}] {label}  ({r.score:.2f})")
        except Exception as exc:
            logger.warning("PandaDocNavigator: reranking failed: %s", exc)

        # Reserved slots: top (top_k - 2) diagnostic + top 2 concept (by cross-encoder rank)
        n_diag = max(1, top_k - 2)
        diag = results[:n_diag]
        seen = {r.url for r in diag}
        concept = [r for r in results if r.doc_type == "concept" and r.url not in seen][:2]
        final = diag + concept

        say(f"PanDA doc search: {len(final)} result(s) ({len(diag)} diagnostic + {len(concept)} concept)")
        for r in diag:
            label = f"{r.breadcrumb[-1]} › {r.title}" if r.breadcrumb else r.title
            say(f"  \\[{r.source}] {label}  ({r.score:.2f})")
        for r in concept:
            label = f"{r.breadcrumb[-1]} › {r.title}" if r.breadcrumb else r.title
            say(f"  \\[{r.source}] {label}  ({r.score:.2f}) [concept]")
        return final

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
        with thinking("Loading PanDA doc index from Qdrant"):
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
                doc_type=p.get("doc_type", "other"),
            )
            self._graph[node.id] = node

        self._page_ids = [nid for nid, n in self._graph.items() if n.level == 0]
        logger.info(
            "PandaDocNavigator: loaded %d nodes (%d pages) from Qdrant",
            len(self._graph), len(self._page_ids),
        )
        say(f"Loaded {len(self._graph)} nodes ({len(self._page_ids)} pages) from Qdrant")

    # ------------------------------------------------------------------
    # Index build pipeline
    # ------------------------------------------------------------------

    async def _build_index(self) -> None:
        say("Building PanDA doc index — this may take a few minutes")
        logger.info("PandaDocNavigator: building doc index — this may take a few minutes")

        with thinking("Fetching PanDA doc file list"):
            rst_paths, tree_sha = await self._fetch_rst_paths()
        if not rst_paths:
            logger.warning("PandaDocNavigator: no RST paths found — aborting build")
            return
        say(f"Discovered {len(rst_paths)} RST paths")

        with thinking(f"Downloading {len(rst_paths)} HTML pages"):
            html_pages = await self._fetch_html_pages(rst_paths)
        say(f"Fetched {len(html_pages)}/{len(rst_paths)} pages")

        from bs4 import BeautifulSoup  # noqa: PLC0415 — lazy import
        all_nodes: list[DocNode] = []
        for rst_path, html in html_pages.items():
            all_nodes.extend(self._parse_page_to_nodes(rst_path, html, BeautifulSoup))

        logger.info("PandaDocNavigator: parsed %d nodes across %d pages", len(all_nodes), len(html_pages))
        say(f"Parsed {len(all_nodes)} nodes across {len(html_pages)} pages")

        with counting(f"Summarising {len(all_nodes)} nodes", total=len(all_nodes)) as advance:
            await self._summarize_nodes(all_nodes, advance_fn=advance)
        say(f"Summarised {len(all_nodes)} nodes")

        with thinking("Embedding and indexing nodes"):
            await self._embed_and_upsert(all_nodes)

        self._graph = {n.id: n for n in all_nodes}
        self._page_ids = [n.id for n in all_nodes if n.level == 0]

        if tree_sha:
            self._write_meta(tree_sha)

        logger.info(
            "PandaDocNavigator: index ready — %d nodes, %d pages",
            len(all_nodes), len(self._page_ids),
        )
        say(f"Doc index ready — {len(all_nodes)} nodes, {len(self._page_ids)} pages")

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

    @staticmethod
    def _iter_sections(elem: Any):
        """Yield direct child <section> or <div class='section'> elements.

        Handles both HTML5 Sphinx output (<section>) and the classic Sphinx
        theme (<div class="section">) so subsections are parsed correctly
        regardless of the ReadTheDocs theme in use.
        """
        for child in elem.children:
            if not hasattr(child, "name"):
                continue
            if child.name == "section":
                yield child
            elif child.name == "div" and "section" in (child.get("class") or []):
                yield child

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
            for child_sec in self._iter_sections(elem):
                child = _parse_section(child_sec, level + 1, node.id)
                node.children.append(child.id)
            return node

        top_sections = list(self._iter_sections(main))
        if not top_sections:
            # ReadTheDocs wraps content in an anonymous <div> with no class.
            # Look one level deeper into any plain div children.
            for child in main.children:
                if hasattr(child, "name") and child.name == "div":
                    top_sections = list(self._iter_sections(child))
                    if top_sections:
                        break
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

    async def _summarize_nodes(self, nodes: list[DocNode], advance_fn=None) -> None:
        semaphore = asyncio.Semaphore(10)

        async def _one(node: DocNode) -> None:
            try:
                if not node.content.strip():
                    node.summary = node.title
                    node.doc_type = "other"
                    return
                prompt = _SUMMARIZE_PROMPT.format(
                    title=node.title,
                    content=node.content[:3000],
                )
                async with semaphore:
                    try:
                        resp = await self._summary_llm.ainvoke([HumanMessage(content=prompt)])
                        raw = resp.content.strip()
                        if raw.startswith("```"):
                            raw = "\n".join(
                                line for line in raw.splitlines() if not line.startswith("```")
                            ).strip()
                        try:
                            parsed = json.loads(raw)
                            node.summary = parsed.get("summary", "") or node.content[:300]
                            node.doc_type = parsed.get("doc_type", "other")
                            if node.doc_type not in ("concept", "other"):
                                node.doc_type = "other"
                        except (json.JSONDecodeError, AttributeError):
                            node.summary = raw or node.content[:300]
                            node.doc_type = "other"
                    except Exception as exc:
                        logger.warning(
                            "PandaDocNavigator: summarize failed for %r: %s", node.title, exc
                        )
                        node.summary = node.content[:300]
                        node.doc_type = "other"
            finally:
                if advance_fn:
                    advance_fn()

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
            embeddings = await asyncio.to_thread(_embed_documents_silently, self._embeddings, texts)

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
                        "doc_type": node.doc_type,
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

    def _build_bm25_index(self) -> None:
        """Build an in-memory BM25 index over node content for exact-term matching."""
        from rank_bm25 import BM25Okapi  # noqa: PLC0415
        self._bm25_ids = list(self._graph.keys())
        corpus = [self._graph[nid].content.split() for nid in self._bm25_ids]
        self._bm25 = BM25Okapi(corpus)
        logger.info("PandaDocNavigator: BM25 index built over %d nodes", len(self._bm25_ids))

    def _bm25_search(self, query: str, top_k: int) -> list[DocResult]:
        """Keyword search using BM25 — reliable for exact technical identifiers.

        Scores are normalised to [0, 1] by dividing by the top result's score so
        they are comparable to cosine-similarity scores from semantic search.
        """
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(query.split())
        ranked = sorted(zip(scores, self._bm25_ids), reverse=True)[:top_k]
        max_score = ranked[0][0] if ranked and ranked[0][0] > 0 else 1.0
        return [
            self._make_result(self._graph[nid], source="bm25", score=float(s) / max_score)
            for s, nid in ranked
            if s > 0
        ]

    async def _semantic_search(self, query: str, top_k: int) -> list[DocResult]:
        query_emb = await asyncio.to_thread(_embed_query_silently, self._embeddings, query)
        client = self._make_qdrant_client()
        try:
            response = await client.query_points(
                collection_name=_COLLECTION,
                query=query_emb,
                limit=top_k,
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
        """Top-down LLM traversal: choose pages, then recursively drill into sections."""
        if not self._page_ids:
            return []

        page_items = [(pid, self._graph[pid]) for pid in self._page_ids if pid in self._graph]

        def _child_titles(node: DocNode) -> str:
            children = [self._graph[c] for c in node.children[:6] if c in self._graph]
            if not children:
                return ""
            def _first_sentence(text: str) -> str:
                end = text.find(". ")
                return text[: end + 1] if end != -1 else text[:150]
            parts = [f"    - {c.title!r}: {_first_sentence(c.summary)}" for c in children]
            return "  Sections:\n" + "\n".join(parts)

        pages_text = "\n".join(
            f'- ID: "{pid}", Title: {node.title!r}, Summary: {node.summary}{_child_titles(node)}'
            for pid, node in page_items
        )
        chosen_page_ids = await self._llm_select(
            _TRAVERSAL_PAGE_PROMPT.format(query=query, pages_text=pages_text),
            candidates=[pid for pid, _ in page_items],
        )
        say(f"LLM selected {len(chosen_page_ids)} relevant doc page(s)")
        if not chosen_page_ids:
            return []

        async def _explore_node(node: DocNode, depth: int) -> list[DocResult]:
            if not node.children or depth >= 2:
                return [self._make_result(node, source="llm_traversal")]
            child_items = [(cid, self._graph[cid]) for cid in node.children if cid in self._graph]
            sections_text = "\n".join(
                f'- ID: "{cid}", Title: {child.title!r}, Summary: {child.summary}'
                for cid, child in child_items
            )
            chosen_ids = await self._llm_select(
                _TRAVERSAL_SECTION_PROMPT.format(
                    query=query,
                    page_title=node.title,
                    page_summary=node.summary,
                    sections_text=sections_text,
                ),
                candidates=[cid for cid, _ in child_items],
            )
            sub_results: list[DocResult] = []
            for cid in chosen_ids:
                child = self._graph.get(cid)
                if child:
                    sub_results.extend(await _explore_node(child, depth + 1))
            return sub_results

        async def _explore_page(page_id: str) -> list[DocResult]:
            page = self._graph.get(page_id)
            return await _explore_node(page, depth=0) if page else []

        page_results = await asyncio.gather(
            *[_explore_page(pid) for pid in chosen_page_ids],
            return_exceptions=True,
        )
        results: list[DocResult] = []
        for pr in page_results:
            if isinstance(pr, list):
                results.extend(pr)
        say(f"LLM traversal: {len(results)} section(s) found")
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
            doc_type=node.doc_type,
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
