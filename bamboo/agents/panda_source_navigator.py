"""Iterative LLM agent for navigating panda-server / pandajedi source code."""

from __future__ import annotations

import ast
import asyncio
import json
import re
from pathlib import Path
from typing import Any

_PANDA_PACKAGES = ["pandaserver", "pandajedi"]

_pkg_roots_cache: dict[str, Path] = {}


def _get_pkg_roots() -> dict[str, Path]:
    global _pkg_roots_cache
    if not _pkg_roots_cache:
        from importlib.metadata import Distribution, PackageNotFoundError, packages_distributions
        # packages_distributions() maps import name → [dist name, ...] from on-disk
        # metadata — independent of sys.modules, so pandaclient's monkey-patch
        # of sys.modules["pandaserver"] cannot mislead it.
        pkg_to_dist = packages_distributions()
        for pkg in _PANDA_PACKAGES:
            for dist_name in pkg_to_dist.get(pkg, []):
                try:
                    dist = Distribution.from_name(dist_name)
                    pkg_dir = Path(dist.locate_file("")).resolve() / pkg
                    if pkg_dir.is_dir():
                        _pkg_roots_cache[pkg] = pkg_dir
                        break
                except PackageNotFoundError:
                    pass
    return _pkg_roots_cache


def _grep_candidates(
    pkg_roots: dict[str, Path],
    terms: list[str],
    max_results: int = 30,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Return (methods, stats) where methods whose body contains one or more of *terms*.

    Each .py file is pre-filtered with a fast ``in`` check before AST parsing.
    Results are sorted descending by the number of matching terms.
    Module paths are package-prefixed: ``pandaserver/taskbuffer/task_utils.py``.
    stats keys: files_scanned, files_hit (contain term in text), methods_found.
    """
    if not terms:
        return [], {"files_scanned": 0, "files_hit": 0, "methods_found": 0}
    results: list[dict[str, Any]] = []
    files_scanned = 0
    files_hit = 0
    for pkg_name, pkg_root in pkg_roots.items():
        for py_file in sorted(pkg_root.rglob("*.py")):
            files_scanned += 1
            try:
                source_text = py_file.read_text(errors="replace")
            except Exception:
                continue
            if not any(t in source_text for t in terms):
                continue
            files_hit += 1
            try:
                tree = ast.parse(source_text)
            except SyntaxError:
                continue
            lines = source_text.splitlines()
            rel = f"{pkg_name}/{py_file.relative_to(pkg_root)}"
            for cls_node in ast.walk(tree):
                if not isinstance(cls_node, ast.ClassDef):
                    continue
                for item in cls_node.body:
                    if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        continue
                    body = "\n".join(lines[item.lineno - 1 : item.end_lineno])
                    score = sum(1 for t in terms if t in body)
                    if score:
                        doc = ast.get_docstring(item) or ""
                        results.append({
                            "name": item.name,
                            "qualname": f"{cls_node.name}.{item.name}",
                            "docstring_first_line": doc.split("\n")[0].strip(),
                            "line": item.lineno,
                            "module": rel,
                            "_score": score,
                        })
            for node in tree.body:
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                body = "\n".join(lines[node.lineno - 1 : node.end_lineno])
                score = sum(1 for t in terms if t in body)
                if score:
                    doc = ast.get_docstring(node) or ""
                    results.append({
                        "name": node.name,
                        "qualname": node.name,
                        "docstring_first_line": doc.split("\n")[0].strip(),
                        "line": node.lineno,
                        "module": rel,
                        "_score": score,
                    })
    results.sort(key=lambda x: x["_score"], reverse=True)
    for r in results:
        r.pop("_score")
    stats = {"files_scanned": files_scanned, "files_hit": files_hit, "methods_found": len(results)}
    return results[:max_results], stats


def _read_method(module: str, qualname: str) -> dict[str, Any] | None:
    """Return full source of a named method: {module, qualname, source, line_start, line_end}.

    ``module`` must be a package-prefixed path, e.g. ``pandaserver/task/taskrefiner.py``.
    """
    pkg_name, rel_path = module.split("/", 1)
    pkg_root = _get_pkg_roots().get(pkg_name)
    if pkg_root is None:
        return None
    py_file = pkg_root / rel_path
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


_MIN_WORDS = 2
_MAX_FRAG_HITS = 10


def _grep_sliding_window(
    pkg_roots: dict[str, Path],
    phrase: str,
    min_words: int = _MIN_WORDS,
) -> tuple[list[dict[str, Any]], dict[str, int], str]:
    """Try all K-word windows of *phrase*, largest K first.

    Single-token phrases (no spaces) are searched directly — they are camelCase /
    snake_case identifiers or short fragments and need no windowing.

    Multi-word phrases use a sliding window: K decreases from len(words) down to
    min_words.  Stops at the first K that yields any candidates.  Never degrades
    below min_words, so noisy single-word fallbacks are avoided.

    Returns (candidates, stats, term_used).
    """
    words = phrase.split()
    if len(words) == 1:
        candidates, stats = _grep_candidates(pkg_roots, [phrase])
        return candidates, stats, phrase
    for k in range(len(words), min_words - 1, -1):
        for start in range(len(words) - k + 1):
            term = " ".join(words[start : start + k])
            candidates, stats = _grep_candidates(pkg_roots, [term])
            if candidates:
                return candidates, stats, term
    return [], {"files_scanned": 0, "files_hit": 0, "methods_found": 0}, phrase


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
    """Iteratively navigates pandaserver/pandajedi AST to answer a code question.

    Candidate methods are found by asking the LLM to extract specific grep terms
    from the question, then searching method bodies for those exact strings.
    The LLM then reads selected methods and may follow up with additional symbols
    across up to MAX_ROUNDS rounds before synthesising a final answer.
    """

    MAX_ROUNDS = 3

    def __init__(self) -> None:
        self._llm = None
        self.last_rounds_used: int = 0
        self.last_term_extraction_succeeded: bool = False
        self.last_grep_terms: list[str] = []
        self.last_candidates_count: int = 0
        self.last_max_overlap: int = 0

    @property
    def llm(self):
        if self._llm is None:
            from bamboo.llm import get_extraction_llm
            self._llm = get_extraction_llm()
        return self._llm

    async def navigate(self, question: str) -> str:
        """Return a synthesized answer to a code question about panda-server/pandajedi."""
        from langchain_core.messages import HumanMessage, SystemMessage
        from bamboo.llm.prompts import (
            PANDA_SOURCE_NAV_PROMPT,
            PANDA_SOURCE_SYNTHESIS_PROMPT,
            SOURCE_GREP_TERMS_PROMPT,
        )
        from bamboo.utils.narrator import say, show_block, thinking

        pkg_roots = _get_pkg_roots()
        if not pkg_roots:
            return "Neither pandaserver nor pandajedi is installed — cannot navigate source."

        for pkg_name, pkg_root in pkg_roots.items():
            say(f"pkg_root: {pkg_name} → {pkg_root}")

        # Step 0: ask LLM which exact strings to grep for in the source
        with thinking("Extracting source grep terms"):
            resp = await self.llm.ainvoke([
                HumanMessage(content=SOURCE_GREP_TERMS_PROMPT.format(question=question))
            ])
        raw = resp.content.strip()
        if raw.startswith("```"):
            raw = "\n".join(l for l in raw.splitlines() if not l.startswith("```")).strip()
        try:
            grep_terms: list[str] = list(dict.fromkeys(
                t for t in json.loads(raw) if isinstance(t, str) and t
            ))
            self.last_term_extraction_succeeded = True
        except Exception:
            grep_terms = [w for w in question.split() if len(w) >= 6]
            self.last_term_extraction_succeeded = False
        self.last_grep_terms = grep_terms
        say(f"Suggested grep terms: {grep_terms}")

        loop = asyncio.get_event_loop()

        # Per-fragment sliding-window search with overlap-count ranking.
        qualname_counts: dict[str, int] = {}
        qualname_to_candidate: dict[str, dict[str, Any]] = {}

        with thinking("Scanning source files"):
            for fragment in grep_terms:
                frag_candidates, _, term_used = await loop.run_in_executor(
                    None, _grep_sliding_window, pkg_roots, fragment
                )
                say(f"  fragment {fragment!r} → term {term_used!r} → {len(frag_candidates)} hit(s)")
                if len(frag_candidates) > _MAX_FRAG_HITS:
                    say("  (skipping — too many hits, fragment is too generic)")
                    continue
                for c in frag_candidates:
                    say(f"    \\[{c['module']}] {c['qualname']}")
                seen_in_fragment: set[str] = set()
                for c in frag_candidates:
                    qn = f"{c['module']}::{c['qualname']}"
                    if qn in seen_in_fragment:
                        continue
                    seen_in_fragment.add(qn)
                    qualname_counts[qn] = qualname_counts.get(qn, 0) + 1
                    qualname_to_candidate.setdefault(qn, c)

        # Keep only highest-overlap candidates; fall back to all if none score > 1.
        all_ranked = sorted(
            qualname_to_candidate.values(),
            key=lambda c: qualname_counts[f"{c['module']}::{c['qualname']}"],
            reverse=True,
        )
        max_count = qualname_counts[f"{all_ranked[0]['module']}::{all_ranked[0]['qualname']}"] if all_ranked else 0
        candidates = (
            [c for c in all_ranked if qualname_counts[f"{c['module']}::{c['qualname']}"] >= 2]
            if max_count >= 2
            else all_ranked
        )[:30]

        self.last_candidates_count = len(candidates)
        self.last_max_overlap = max_count

        if not candidates:
            say("No candidates found")
            return "No methods found matching the query in pandaserver/pandajedi."

        say(f"{len(candidates)} candidate(s) found (max overlap score: {max_count})")
        show_block(
            "candidates",
            "\n".join(
                f"[{qualname_counts[c['module'] + '::' + c['qualname']]}] {c['module']}::{c['qualname']}  |  {c['docstring_first_line'][:80]}"
                for c in candidates
            ),
        )

        accumulated: list[dict[str, Any]] = []
        accumulated_qualnames: set[str] = set()

        rounds_completed = 0
        for round_num in range(self.MAX_ROUNDS):
            # Read every candidate in this round automatically.
            new_this_round = 0
            for candidate in candidates:
                qualname = candidate["qualname"]
                full_qn = f"{candidate['module']}::{qualname}"
                if full_qn in accumulated_qualnames:
                    continue
                say(f"Reading {candidate['module']}::{qualname}")
                src = _read_method(candidate["module"], qualname)
                if src:
                    accumulated.append(src)
                    accumulated_qualnames.add(full_qn)
                    new_this_round += 1
                    show_block(
                        f"{qualname} (lines {src['line_start']}–{src['line_end']})",
                        src["source"],
                        max_lines=80,
                    )
                else:
                    say(f"  (not found)")

            if not accumulated:
                break

            # Ask LLM whether follow-up symbols are needed, based on what was read.
            sources_read = json.dumps(
                [{"qualname": s["qualname"], "source": s["source"]} for s in accumulated],
                indent=2,
            )
            prompt = PANDA_SOURCE_NAV_PROMPT.format(
                question=question,
                sources_read=sources_read,
            )
            say(f"Round {round_num + 1}/{self.MAX_ROUNDS}: asking LLM for follow-up "
                f"(read so far: {list(accumulated_qualnames)})")
            with thinking(f"Round {round_num + 1} — follow-up decision"):
                response = await self.llm.ainvoke([
                    SystemMessage(content="You are navigating panda-server Python source code."),
                    HumanMessage(content=prompt),
                ])
            show_block(f"follow-up decision (round {round_num + 1})", response.content)
            decision = _parse_nav_decision(response.content)

            rounds_completed = round_num + 1
            if decision.get("action") != "follow_up":
                relevant = decision.get("relevant_qualnames")
                if relevant:
                    accumulated = [s for s in accumulated if s["qualname"] in relevant]
                    say(f"LLM filtered to {len(accumulated)} relevant method(s): {[s['qualname'] for s in accumulated]}")
                say("LLM declared done")
                break

            follow_symbols = decision.get("follow_up_symbols") or []
            if not follow_symbols:
                break
            say(f"Follow-up symbols: {follow_symbols}")
            with thinking("Scanning source files"):
                candidates, _ = await loop.run_in_executor(
                    None, _grep_candidates, pkg_roots, follow_symbols
                )
            say(f"{len(candidates)} candidate(s) for follow-up")
            if not candidates:
                break
            show_block(
                "follow-up candidates",
                "\n".join(
                    f"{c['module']}::{c['qualname']}  |  {c['docstring_first_line'][:80]}"
                    for c in candidates
                ),
            )

        self.last_rounds_used = rounds_completed

        if not accumulated:
            return "Found candidate methods but could not read their source."

        say(f"Synthesizing from {len(accumulated)} method(s): "
            f"{[s['qualname'] for s in accumulated]}")
        prompt = PANDA_SOURCE_SYNTHESIS_PROMPT.format(
            question=question,
            sources=json.dumps(accumulated, indent=2),
        )
        with thinking("Synthesizing answer"):
            response = await self.llm.ainvoke([
                SystemMessage(content="You are analyzing panda-server Python source code."),
                HumanMessage(content=prompt),
            ])
        show_block("source navigator result", response.content)
        return response.content
