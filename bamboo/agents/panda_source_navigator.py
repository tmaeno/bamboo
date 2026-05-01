"""Iterative LLM agent for navigating panda-server source code."""

from __future__ import annotations

import ast
import importlib.util
import json
import math
import re
from pathlib import Path
from typing import Any

_method_index_cache: dict[str, list[dict[str, Any]]] | None = None


def _get_pkg_root() -> Path | None:
    spec = importlib.util.find_spec("pandaserver")
    return Path(spec.origin).parent if spec and spec.origin else None


def _tokenize(text: str) -> list[str]:
    """Split on whitespace and camelCase boundaries. 'ramCount' → ['ram', 'count']."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _build_index(pkg_root: Path) -> dict[str, list[dict[str, Any]]]:
    """AST-parse all .py files → {rel_path: [{name, qualname, docstring_first_line, line, module}]}."""
    index: dict[str, list[dict[str, Any]]] = {}
    for py_file in sorted(pkg_root.rglob("*.py")):
        rel = str(py_file.relative_to(pkg_root))
        try:
            tree = ast.parse(py_file.read_text(errors="replace"))
        except SyntaxError:
            continue
        entries: list[dict[str, Any]] = []
        # Class methods — track class context explicitly
        for cls_node in ast.walk(tree):
            if isinstance(cls_node, ast.ClassDef):
                for item in cls_node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        doc = ast.get_docstring(item) or ""
                        entries.append({
                            "name": item.name,
                            "qualname": f"{cls_node.name}.{item.name}",
                            "docstring_first_line": doc.split("\n")[0].strip(),
                            "line": item.lineno,
                            "module": rel,
                        })
        # Top-level functions
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                doc = ast.get_docstring(node) or ""
                entries.append({
                    "name": node.name,
                    "qualname": node.name,
                    "docstring_first_line": doc.split("\n")[0].strip(),
                    "line": node.lineno,
                    "module": rel,
                })
        if entries:
            index[rel] = entries
    return index


def _get_index(pkg_root: Path) -> dict[str, list[dict[str, Any]]]:
    global _method_index_cache
    if _method_index_cache is None:
        _method_index_cache = _build_index(pkg_root)
    return _method_index_cache


def _keyword_filter(
    index: dict[str, list[dict[str, Any]]],
    query: str,
    max_results: int = 30,
) -> list[dict[str, Any]]:
    """Return methods whose name+docstring keyword-match the query.

    Uses camelCase-aware tokenization and a majority-match threshold (≥60% of
    query tokens must appear) so that natural-language words in the query that
    don't appear in source identifiers don't kill the match.
    """
    tokens = _tokenize(query)
    if not tokens:
        return []
    threshold = max(1, math.ceil(len(tokens) * 0.6))
    results: list[dict[str, Any]] = []
    for entries in index.values():
        for e in entries:
            haystack = set(_tokenize(e["qualname"] + " " + e["docstring_first_line"]))
            if sum(1 for t in tokens if t in haystack) >= threshold:
                results.append(e)
    return results[:max_results]


def _read_method(pkg_root: Path, module: str, qualname: str) -> dict[str, Any] | None:
    """Return full source of a named method: {module, qualname, source, line_start, line_end}."""
    py_file = pkg_root / module
    try:
        source_text = py_file.read_text(errors="replace")
        tree = ast.parse(source_text)
    except Exception:
        return None
    lines = source_text.splitlines()
    target = qualname.split(".")[-1]
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == target:
            body = "\n".join(lines[node.lineno - 1 : node.end_lineno])
            return {
                "module": module,
                "qualname": qualname,
                "line_start": node.lineno,
                "line_end": node.end_lineno,
                "source": body,
            }
    return None


def _parse_nav_decision(text: str) -> dict[str, Any]:
    """Parse LLM JSON navigation decision. Fail-open to 'done' on any error."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {"action": "done"}
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return {"action": "done"}


class PandaSourceNavigator:
    """Iteratively navigates panda-server AST to answer a code question.

    Each round the LLM sees the accumulated source it has read so far and
    decides whether to read more methods or declare the question answered.
    This mirrors how Claude Code uses grep + read in a loop, applied to an
    AST-indexed view of the pandaserver package.
    """

    MAX_ROUNDS = 3

    def __init__(self) -> None:
        self._llm = None

    @property
    def llm(self):
        if self._llm is None:
            from bamboo.llm import get_extraction_llm
            self._llm = get_extraction_llm()
        return self._llm

    async def navigate(self, question: str) -> str:
        """Return a synthesized answer to a code question about panda-server."""
        from langchain_core.messages import HumanMessage, SystemMessage
        from bamboo.llm.prompts import PANDA_SOURCE_NAV_PROMPT, PANDA_SOURCE_SYNTHESIS_PROMPT

        pkg_root = _get_pkg_root()
        if pkg_root is None:
            return "pandaserver package not installed — cannot navigate source."

        index = _get_index(pkg_root)
        candidates = _keyword_filter(index, question)
        if not candidates:
            return "No methods found matching the query in pandaserver."

        accumulated: list[dict[str, Any]] = []

        for _ in range(self.MAX_ROUNDS):
            already_read = [f"{s['module']}::{s['qualname']}" for s in accumulated]
            prompt = PANDA_SOURCE_NAV_PROMPT.format(
                question=question,
                candidates=json.dumps(candidates, indent=2),
                already_read=json.dumps(already_read),
            )
            response = await self.llm.ainvoke([
                SystemMessage(content="You are navigating panda-server Python source code."),
                HumanMessage(content=prompt),
            ])
            decision = _parse_nav_decision(response.content)

            if decision.get("action") == "done":
                break

            for ref in decision.get("read", []):
                src = _read_method(pkg_root, ref["module"], ref["qualname"])
                if src:
                    accumulated.append(src)

            follow_symbols = decision.get("follow_up_symbols") or []
            if follow_symbols:
                candidates = _keyword_filter(index, " ".join(follow_symbols))
            else:
                break

        if not accumulated:
            return "Found candidate methods but could not read their source."

        prompt = PANDA_SOURCE_SYNTHESIS_PROMPT.format(
            question=question,
            sources=json.dumps(accumulated, indent=2),
        )
        response = await self.llm.ainvoke([
            SystemMessage(content="You are analyzing panda-server Python source code."),
            HumanMessage(content=prompt),
        ])
        return response.content
