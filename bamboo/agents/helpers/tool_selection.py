"""Tool-list rendering (Phase 0) and budget-gated retrieval selection (Phase 1)
for orchestration prompts.

``render_tools`` is the single formatter both code-gen paths use
(``investigation_session._format_available_tools`` and
``context_enricher._build_tools_description``). It renders each tool either with
its full JSON Schema (for the tools named in ``full_schema_for``) or as a compact
one-liner, and — when a ``token_budget`` is supplied — fills greedily against the
*actual* assembled text and stops before the budget, returning the names it had to
omit. With ``token_budget=None`` it renders everything, reproducing the historical
per-path output verbatim (the Phase-0 safe-refactor default).

The retrieval ``ToolSelector`` that decides ``full_schema_for`` from the current
symptom is added in Phase 1; see the plan and docs/AGENTS.md.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Descriptor normalisation
# ---------------------------------------------------------------------------


def _as_descriptor(obj: Any) -> dict[str, Any]:
    """Normalise an ``McpTool`` (or any attribute-bearing tool) or a unified
    descriptor dict to a plain dict with the keys the renderer needs.

    The investigate path already produces these dicts
    (``_unified_tool_descriptors``); the explorer path passes ``McpTool``
    objects — both are accepted so the renderer is source-agnostic.
    """
    if isinstance(obj, dict):
        return obj
    return {
        "name": obj.name,
        "description": getattr(obj, "description", "") or "",
        "parameters_schema": getattr(obj, "parameters_schema", {}) or {},
        "external_access": getattr(obj, "external_access", True),
        "read_only": getattr(obj, "read_only", True),
    }


def _param_names(d: dict[str, Any]) -> str:
    schema = d.get("parameters_schema") or {}
    return ", ".join((schema.get("properties") or {}).keys())


# ---------------------------------------------------------------------------
# Per-style line formatters (kept byte-for-byte compatible with the original
# call-site output so Phase 0 is a pure refactor)
# ---------------------------------------------------------------------------


def _investigate_full(d: dict[str, Any]) -> str:
    access = "external" if d.get("external_access") else "internal"
    # read_only defaults True (a tool only modifies state when explicitly tagged).
    rw = "read-only" if d.get("read_only", True) else "MODIFIES-STATE"
    schema = json.dumps(d.get("parameters_schema") or {}, indent=None)
    return (
        f"- {d['name']} [{access}, {rw}]: {(d.get('description') or '').strip()}\n"
        f"    args schema: {schema}"
    )


def _investigate_compact(d: dict[str, Any]) -> str:
    access = "external" if d.get("external_access") else "internal"
    rw = "read-only" if d.get("read_only", True) else "MODIFIES-STATE"
    return (
        f"- {d['name']} [{access}, {rw}]: {(d.get('description') or '').strip()}\n"
        f"    args: ({_param_names(d)})"
    )


def _explorer_full(d: dict[str, Any]) -> str:
    schema = json.dumps(d.get("parameters_schema") or {}, indent=None)
    return (
        f"- {d['name']}({_param_names(d)})\n  {d.get('description') or ''}\n"
        f"  args schema: {schema}"
    )


def _explorer_compact(d: dict[str, Any]) -> str:
    return f"- {d['name']}({_param_names(d)})\n  {d.get('description') or ''}"


_STYLES: dict[str, dict[str, Any]] = {
    "investigate": {
        "full": _investigate_full,
        "compact": _investigate_compact,
        "sep": "\n",
        "empty": "(none)",
    },
    "explorer": {
        "full": _explorer_full,
        "compact": _explorer_compact,
        "sep": "\n\n",
        "empty": "",
    },
}


def _fallback_count_tokens(text: str) -> int:
    """Conservative *over*-counting token estimate when no tokenizer is given.

    Tool schemas are punctuation-heavy JSON that tokenises denser than prose, so
    we deliberately over-count (~chars/3) to bias toward showing fewer tools
    rather than overflowing a small local context.
    """
    return (len(text) + 2) // 3


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------


def render_tools(
    descriptors: Iterable[Any],
    *,
    full_schema_for: Optional[set[str]] = None,
    style: str = "investigate",
    token_budget: Optional[int] = None,
    count_tokens: Optional[Callable[[str], int]] = None,
    base_tokens: int = 0,
) -> tuple[str, list[str]]:
    """Render tool descriptors for an orchestration prompt.

    Returns ``(rendered_text, omitted_names)``.

    Args:
        descriptors: ``McpTool`` objects or unified descriptor dicts.
        full_schema_for: tool names to render with their full JSON schema; every
            other tool is rendered compact. ``None`` => *all* tools get the full
            schema (the behaviour-preserving Phase-0 default). Pass ``set()`` to
            render everything compact.
        style: ``"investigate"`` or ``"explorer"`` — selects the per-path line
            format and separator.
        token_budget: when given, greedily add tool blocks (in the order passed —
            callers pass the highest-priority tools first) and stop before the
            assembled text plus ``base_tokens`` would exceed the budget. Omitted
            tool names are returned. When ``None``, every tool is rendered and
            ``omitted_names`` is empty.
        count_tokens: token counter for the budget path; defaults to a
            conservative char-based over-estimate.
        base_tokens: tokens already consumed by the rest of the prompt (system +
            task context + prior turns + utterance), so the budget reflects the
            *real* assembled prompt.
    """
    descs = [_as_descriptor(o) for o in descriptors]
    fmt = _STYLES[style]
    if not descs:
        return fmt["empty"], []

    def render_line(d: dict[str, Any]) -> str:
        full = full_schema_for is None or d["name"] in full_schema_for
        return fmt["full"](d) if full else fmt["compact"](d)

    if token_budget is None:
        return fmt["sep"].join(render_line(d) for d in descs), []

    count = count_tokens or _fallback_count_tokens
    chosen: list[str] = []
    omitted: list[str] = []
    for d in descs:
        line = render_line(d)
        candidate = fmt["sep"].join(chosen + [line])
        # Always keep at least the first (highest-priority) tool; a later, smaller
        # block may still fit after a big one didn't, so we don't break early.
        if chosen and base_tokens + count(candidate) > token_budget:
            omitted.append(d["name"])
        else:
            chosen.append(line)
    text = fmt["sep"].join(chosen) if chosen else fmt["empty"]
    return text, omitted


# ---------------------------------------------------------------------------
# Phase 1 — budget-gated retrieval selection
# ---------------------------------------------------------------------------


class RetrievalUnavailable(RuntimeError):
    """Raised when the catalogue is over budget but the vector store can't be
    reached — the caller aborts the turn rather than degrade (see the plan's
    scoped-failure invariant)."""


@dataclass
class Selection:
    """Outcome of :meth:`ToolSelector.select`.

    ``ordered`` is the candidate tool names in render order (full-schema picks
    first, then the compact tail); ``full_schema_names`` is the subset to render
    with their JSON schema. ``dropped`` is the selector's *estimate* of what won't
    fit — :func:`render_tools` (called by the caller with the real token budget)
    is the authoritative gate.
    """

    ordered: list[str]
    full_schema_names: set[str]
    dropped: list[str] = field(default_factory=list)


def config_namespace(settings: Any) -> str:
    """Stable id for the active (MCP catalogue + provider) so a shared vector store
    isn't thrashed by bamboo instances configured differently."""
    raw = f"{getattr(settings, 'mcp_servers_config', '')}|{getattr(settings, 'llm_provider', '')}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def _content_hash(d: dict[str, Any]) -> str:
    payload = json.dumps(
        [d.get("name"), d.get("description"), d.get("parameters_schema") or {}],
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _index_text(d: dict[str, Any]) -> str:
    """Text embedded for a tool (source #2) — name + description + param names,
    never the schema (structural noise that hurts recall)."""
    params = _param_names(d)
    suffix = f" (args: {params})" if params else ""
    return f"{d['name']}: {(d.get('description') or '').strip()}{suffix}"


def _merge_sources(s1: list[str], s2: list[str], reserved_explore: int) -> list[str]:
    """Priority-order the two candidate lists. Source #1 (validated past) leads,
    but the top ``reserved_explore`` source-#2 (description) picks are interleaved
    near the front so newly-added tools always get a shot at the full-schema slots.
    Deduped (a tool in both keeps its earliest position)."""
    s1 = list(dict.fromkeys(s1))
    s2 = list(dict.fromkeys(s2))
    reserved = s2[: max(0, reserved_explore)]
    rest2 = s2[max(0, reserved_explore):]
    out: list[str] = []
    seen: set[str] = set()

    def add(name: str) -> None:
        if name not in seen:
            seen.add(name)
            out.append(name)

    i1 = 0
    for r in reserved:  # 1:1 interleave reserved #2 with #1 at the front
        if i1 < len(s1):
            add(s1[i1])
            i1 += 1
        add(r)
    for n in s1[i1:]:
        add(n)
    for n in rest2:
        add(n)
    return out


class ToolSelector:
    """Budget-gated, two-source tool selection for orchestration prompts.

    Source #1 = tools used by similar *human-approved* past investigations
    (the ``ProcedureTriggers`` vector section); source #2 = tools whose
    descriptions match the symptom (the ``ToolCatalogue`` section). Indexing and
    retrieval reuse the existing :class:`VectorDatabaseClient` + embeddings; all
    vectors are scoped by ``config_namespace`` so a shared store isn't thrashed.
    """

    def __init__(
        self,
        vector_db: Any,
        embeddings: Any,
        *,
        candidate_k: int = 40,
        reserved_explore: int = 4,
        max_full_schemas: int = 25,
        catalogue_section: str = "ToolCatalogue",
        triggers_section: str = "ProcedureTriggers",
    ):
        self._vector_db = vector_db
        # ``embeddings`` may be an Embeddings object or a zero-arg factory (e.g.
        # ``get_embeddings``) so callers like ``build_deps`` don't eagerly load a
        # local model that a non-tool-selecting run never needs.
        self._embeddings = embeddings
        self._candidate_k = candidate_k
        self._reserved_explore = reserved_explore
        # Relevance cap: at most this many tools get full schemas, even if more fit
        # the token budget (large catalogues hurt selection accuracy / cost).
        self._max_full_schemas = max_full_schemas
        self._catalogue_section = catalogue_section
        self._triggers_section = triggers_section

    def _emb(self) -> Any:
        e = self._embeddings
        if callable(e) and not hasattr(e, "embed_query"):
            e = e()
            self._embeddings = e  # cache the resolved object
        return e

    @staticmethod
    def _vid(*parts: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, "::".join(parts)))

    async def ensure_index(self, tools: Iterable[Any], *, config_namespace: str) -> None:
        """Upsert the ``ToolCatalogue`` vectors for *tools*, embedding only the
        ones that are new or whose content hash changed (idempotent, cheap in
        steady state). Stale vectors for removed tools are harmless — ``select``
        intersects results with the live tool list — so they are not pruned here.
        """
        pending: list[tuple[str, str, str, str]] = []  # (vid, text, name, hash)
        for o in tools:
            d = _as_descriptor(o)
            h = _content_hash(d)
            vid = self._vid("toolcat", config_namespace, d["name"])
            try:
                existing = await self._vector_db.get_document(vid)
            except Exception:  # noqa: BLE001 — treat as missing, re-index
                existing = None
            if existing and (existing.get("metadata") or {}).get("tool_hash") == h:
                continue
            pending.append((vid, _index_text(d), d["name"], h))
        if not pending:
            return
        vectors = self._emb().embed_documents([t for (_v, t, _n, _h) in pending])
        for (vid, text, name, h), emb in zip(pending, vectors, strict=True):
            await self._vector_db.upsert_section_vector(
                vid,
                emb,
                text,
                self._catalogue_section,
                {"config_namespace": config_namespace, "tool_name": name, "tool_hash": h},
            )
        logger.info("tool catalogue: indexed %d new/changed tools", len(pending))

    async def index_procedure_run(
        self,
        *,
        tool_names: list[str],
        prompt_text: str,
        config_namespace: str,
        weight: float = 1.0,
    ) -> None:
        """Add (or update) one ``ProcedureTriggers`` point — the validated
        ``(approved prompt → tool signature)`` example that feeds source #1. One
        point per approved run; identical prompts for the same signature collapse
        to the same id (exact-dedup). ``weight`` upranks finalized-with-Cause runs.
        """
        tool_names = sorted({t for t in tool_names if t})
        if not tool_names or not (prompt_text or "").strip():
            return
        signature = ",".join(tool_names)
        vid = self._vid("proctrig", config_namespace, signature, prompt_text.strip())
        emb = self._emb().embed_query(prompt_text.strip())
        await self._vector_db.upsert_section_vector(
            vid,
            emb,
            prompt_text.strip(),
            self._triggers_section,
            {"config_namespace": config_namespace, "tools": tool_names, "weight": weight},
        )

    async def select(
        self,
        query: str,
        tools: Iterable[Any],
        *,
        budget: int,
        config_namespace: str,
        base_tokens: int = 0,
        count_tokens: Optional[Callable[[str], int]] = None,
        style: str = "investigate",
        exclude_names: frozenset[str] = frozenset(),
    ) -> Selection:
        """Pick which tools get full schemas vs. compact, bounded by *budget*.

        Called only when the catalogue is over budget. Raises
        :class:`RetrievalUnavailable` if the (essential) description search fails —
        the caller then aborts the turn rather than degrade.
        """
        descs = {
            d["name"]: d
            for d in (_as_descriptor(o) for o in tools)
            if d["name"] not in exclude_names
        }
        count = count_tokens or _fallback_count_tokens
        q_emb = self._emb().embed_query(query)

        # Source #2 (essential coverage): failure => fail-hard.
        try:
            rows2 = await self._vector_db.search_similar(
                q_emb,
                limit=self._candidate_k,
                score_threshold=0.0,
                filter_conditions={"section": self._catalogue_section, "config_namespace": config_namespace},
            )
        except Exception as exc:  # noqa: BLE001
            raise RetrievalUnavailable(f"tool-catalogue retrieval failed: {exc}") from exc
        s2: list[str] = []
        for r in rows2:
            name = (r.get("metadata") or {}).get("tool_name")
            if name in descs:
                s2.append(name)

        # Source #1 (validated past, best-effort): empty/failure is non-fatal.
        s1: list[str] = []
        try:
            rows1 = await self._vector_db.search_similar(
                q_emb,
                limit=self._candidate_k,
                score_threshold=0.0,
                filter_conditions={"section": self._triggers_section, "config_namespace": config_namespace},
            )
            scored = []
            for r in rows1:
                meta = r.get("metadata") or {}
                weight = float(meta.get("weight", 1.0) or 1.0)
                scored.append((float(r.get("score", 0.0)) * weight, meta.get("tools") or []))
            scored.sort(key=lambda x: -x[0])
            for _score, tool_list in scored:
                for name in tool_list:
                    if name in descs:
                        s1.append(name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("procedure-triggers retrieval failed (descriptions only): %s", exc)

        priority = _merge_sources(s1, s2, self._reserved_explore)

        # Budget-aware full/compact split (an estimate; render_tools enforces the
        # real ceiling on the assembled prompt).
        remaining = max(0, budget - base_tokens)
        sep_cost = count(_STYLES[style]["sep"])
        used = 0
        full: list[str] = []
        for name in priority:
            # Relevance cap: stop promoting to full once we've shown enough, even if
            # more would fit the token budget (the rest fall through to compact).
            if self._max_full_schemas is not None and len(full) >= self._max_full_schemas:
                break
            cost = count(_STYLES[style]["full"](descs[name])) + (sep_cost if full else 0)
            if used + cost <= remaining:
                full.append(name)
                used += cost
            else:
                break  # priority order: stop promoting to full once one doesn't fit
        full_set = set(full)
        compact: list[str] = []
        dropped: list[str] = []
        for name in priority:
            if name in full_set:
                continue
            cost = count(_STYLES[style]["compact"](descs[name])) + (sep_cost if (full or compact) else 0)
            if used + cost <= remaining:
                compact.append(name)
                used += cost
            else:
                dropped.append(name)
        return Selection(ordered=full + compact, full_schema_names=full_set, dropped=dropped)
