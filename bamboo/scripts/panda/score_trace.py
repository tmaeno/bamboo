"""Throwaway diagnostic: is the use-time trace complete, sound and honest?

Not production code.  It exists to answer three design questions with numbers,
the way ``census_code_map.py`` and ``score_reading.py`` did.

**The oracle is the source, so the scoring is by hand.**  Nothing else in the
system computes why a branch ran, so there is nothing to compare against
automatically; this prints the trace beside the marked function and collects a
verdict per case.  Three verdicts rather than one, because they fail
differently and only one of them is silent:

``complete``  did the trace list every assignment that can set the names it
              walked?  A missing one is the dangerous kind -- pruning is by
              elimination, so an explanation that never becomes a candidate
              lets a wrong one survive with confidence.
``sound``     is every site and guard it lists actually there?
``honest``    where a structure the path condition cannot see was in the way,
              did the trace say so?  This is the one a reader cannot check for
              themselves, which is why it is scored separately and why the
              pass mark for it is zero failures rather than a proportion.

**The sample is drawn once, stratified, and the holdout is sealed.**  Two
walkthroughs picked by hand would only ever test the blind spots that were
predicted from reading the code, and the corpus has punished reasoning from a
list of known failures before.  The strata come from the map's own arms: an
``if`` chain (71.0%), no ``if`` but inside a try, a loop or a ``with``
(18.1%), and plain straight-line code (4.9%).

Honest about what this cannot be: the author of the walk is also its scorer,
so sealing the holdout buys "not tuned against these cases" and not blindness.
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Optional

import click

from bamboo.codemap import reading as reading_mod
from bamboo.codemap import trace as trace_mod
from bamboo.codemap.panda import provenance
from bamboo.codemap.panda.plugin import PandaCodeMapPlugin

logger = logging.getLogger(__name__)


def _case_id(case: dict) -> str:
    return f"{case['owner']}@{case['line']}"


def _render(case: dict, roots: dict, budget: trace_mod.Budget) -> list[str]:
    out: list[str] = []
    steps, note = trace_mod.walk(
        roots,
        file=case["file"],
        owner=case["owner"],
        lines=[case["line"]],
        classify=provenance.classify,
        budget=budget,
    )
    out.append(f"  stratum {case.get('stratum', '?')}   tier {case.get('tier', '?')}"
               f"   outcome {case.get('outcome', '?')}")
    out.append(f"  structures the map noted: {', '.join(case.get('structs') or []) or 'none'}")
    if note:
        out.append(f"  NOTE  {note}")
    out.append("  -- what the trace says " + "-" * 50)
    for step in steps:
        head = step.name if step.kind == trace_mod.TRACE_UNBOUND else (
            f"{step.name} = {step.value}" if step.name else step.value
        )
        out.append(f"    {step.line or '':>5}  {step.kind:<11} {head}")
        if step.guards:
            out.append(f"           when   {' · '.join(step.guards)}")
        elif step.kind != trace_mod.TRACE_UNBOUND:
            out.append("           when   no test above it")
        for unsaid in step.unseen:
            out.append(f"           also   {unsaid}")
        if step.terminal:
            detail = f" -- {step.detail}" if step.detail else ""
            out.append(f"           stops  {step.terminal}{detail}")
    if not steps:
        out.append("    (nothing)")
    out.append("  -- the source " + "-" * 59)
    region = reading_mod.region_for(
        roots, file=case["file"], owner=case["owner"], line=case["line"]
    )
    out.append(region.marked([case["line"]]) if region else "    (not found)")
    return out


@click.command("score-trace")
@click.option("--sample", type=click.Path(exists=True, path_type=Path), default=None)
@click.option(
    "--verdicts",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Tally this file instead of printing a worksheet.",
)
@click.option("--out", type=click.Path(path_type=Path), default=None)
@click.option(
    "--scan",
    is_flag=True,
    help=(
        "Ignore the sample and run the whole-population check instead: every "
        "arm the stored map can locate, asking whether the walk closed a "
        "self.<field> while a binding for it was in scope."
    ),
)
@click.option("--map-id", default="panda", show_default=True)
@click.option("--top", default=12, show_default=True, help="Owners listed by the scan.")
@click.option("--depth", type=int, default=trace_mod.DEFAULT_BUDGET.depth)
@click.option("--source-root", type=click.Path(path_type=Path), default=None)
def main(
    sample: Optional[Path],
    verdicts: Optional[Path],
    out: Optional[Path],
    scan: bool,
    map_id: str,
    top: int,
    depth: int,
    source_root: Optional[Path],
) -> None:
    if scan:
        asyncio.run(_scan(map_id, source_root, trace_mod.Budget(depth=depth), top))
        return
    if sample is None:
        raise click.UsageError(
            "give --sample, or --scan for the whole-population check"
        )
    cases = json.loads(sample.read_text())
    seal = hashlib.sha256(sample.read_bytes()).hexdigest()[:16]
    if verdicts is not None:
        _tally(cases, verdicts, seal)
        return
    roots = PandaCodeMapPlugin._resolve_roots(source_root)
    budget = trace_mod.Budget(depth=depth)
    lines = [f"sample {sample.name}  sha256 {seal}  n={len(cases)}  depth={depth}", ""]
    for index, case in enumerate(cases, 1):
        lines.append("=" * 78)
        lines.append(f"case {index}/{len(cases)}  {_case_id(case)}")
        lines.extend(_render(case, roots, budget))
        lines.append("")
        lines.append(f"VERDICT {_case_id(case)}  complete=?  sound=?  honest=?  note=")
        lines.append("")
    text = "\n".join(lines)
    if out is not None:
        out.write_text(text)
        click.echo(f"worksheet written to {out} ({len(cases)} case(s), sha256 {seal})")
    else:
        click.echo(text)


def _tally(cases: list[dict], verdicts: Path, seal: str) -> None:
    """Count the three verdicts and, for each miss, what kind it was.

    The kinds are not enumerated here on purpose: they come out of the notes
    the reading produced, and fixing a vocabulary in advance is how a sample
    stops being able to turn up a shape that was not predicted.
    """
    scored: dict[str, dict[str, str]] = {}
    notes: list[tuple[str, str]] = []
    for line in verdicts.read_text().splitlines():
        if not line.startswith("VERDICT "):
            continue
        body = line[len("VERDICT ") :]
        case_id, _, rest = body.partition("  ")
        fields = {}
        note = ""
        for part in rest.split("  "):
            part = part.strip()
            if part.startswith("note="):
                note = part[len("note=") :].strip()
            elif "=" in part:
                key, _, value = part.partition("=")
                fields[key.strip()] = value.strip()
        scored[case_id.strip()] = fields
        if note:
            notes.append((case_id.strip(), note))
    counts = {k: Counter() for k in ("complete", "sound", "honest")}
    for fields in scored.values():
        for key, counter in counts.items():
            counter[fields.get(key, "?")] += 1
    click.echo(f"sample sha256 {seal}  n={len(cases)}  scored={len(scored)}")
    for key, counter in counts.items():
        total = sum(v for k, v in counter.items() if k != "?")
        yes = counter.get("y", 0)
        click.echo(
            f"  {key:<9} {yes}/{total}"
            + (f"   unscored {counter['?']}" if counter.get("?") else "")
        )
    if notes:
        click.echo("\n  what the misses were:")
        for case_id, note in notes:
            click.echo(f"    {case_id}\n      {note}")

# ---------------------------------------------------------------------------
# The whole-population check (P1-26 A).
#
# Written as a *second reading*, sharing no line with the walk it checks.
# P1-25's completeness check missed ``self.X = ...`` because it re-counted the
# same six binding forms the implementation enumerated, from the same list --
# both were blind in the same place.  So this descends the tree from the top
# where ``_field`` climbs from an expression, and it enumerates *more* forms
# than the walk is going to learn (``with ... as self.x``, ``for self.x in``),
# so an under-implementation of A shows up here as a residue rather than a pass.
#
# And it asks the question P1-25's check could not.  Not "did a step appear for
# this name" -- one did, a wrong one -- but "was a stronger answer in scope
# while the walk gave a weaker one".


def _attribute_targets(node: ast.AST) -> list[str]:
    """``self.<field>`` names *node* binds, in every form that binds one."""

    def named(target: ast.AST) -> list[str]:
        if (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
        ):
            return [target.attr]
        if isinstance(target, (ast.Tuple, ast.List)):
            return [name for element in target.elts for name in named(element)]
        if isinstance(target, ast.Starred):
            return named(target.value)
        return []

    if isinstance(node, ast.Assign):
        return [name for target in node.targets for name in named(target)]
    if isinstance(node, (ast.AugAssign, ast.AnnAssign)):
        return named(node.target)
    if isinstance(node, (ast.For, ast.AsyncFor)):
        return named(node.target)
    if isinstance(node, (ast.With, ast.AsyncWith)):
        return [
            name
            for item in node.items
            if item.optional_vars is not None
            for name in named(item.optional_vars)
        ]
    return []


def _descend(node: ast.AST, stop: tuple = ()) -> list[ast.AST]:
    """Every node under *node*, top-down, without entering *stop* kinds.

    ``ast.walk`` would do, and that is exactly why this does not use it: the
    walk under test walks, and a shared traversal is a shared blind spot.
    """
    found: list[ast.AST] = []
    for child in ast.iter_child_nodes(node):
        found.append(child)
        if isinstance(child, stop):
            continue
        found.extend(_descend(child, stop))
    return found


_NESTED = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)


def _bound_here(func: ast.AST) -> set[str]:
    """Fields assigned in *func* itself, nested functions excluded."""
    return {
        name
        for node in _descend(func, stop=_NESTED)
        for name in _attribute_targets(node)
    }


def _bound_in_class(cls: ast.ClassDef) -> set[str]:
    """Fields assigned anywhere in *cls*.

    Bases are not followed.  A base in another module would need the corpus,
    and the residue that leaves is itself part of the measurement: if A's
    stage 3 reads bases and this does not, the check stops counting cases A
    fixed, which is the direction that cannot flatter it.
    """
    return {
        name
        for method in cls.body
        if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
        for name in _bound_here(method)
    }


_PARSED: dict[str, object] = {}


def _tree_for(roots: dict, rel: str):
    """Parse ``<package>/<path>`` out of *roots*, the way ``region_for`` does."""
    if rel not in _PARSED:
        _PARSED[rel] = None
        package, _, relative = rel.partition("/")
        root = roots.get(package)
        if root is not None and relative:
            try:
                _PARSED[rel] = ast.parse((root / relative).read_text(errors="replace"))
            except (OSError, SyntaxError, ValueError):
                _PARSED[rel] = None
    return _PARSED[rel]


def _scope_of(roots: dict, file: str, owner: str):
    """``(func, class)`` for a step's owner, read from this tree."""
    tree = _tree_for(roots, file)
    if tree is None:
        return None, None
    wanted = owner.rsplit("::", 1)[-1]
    cls_name, _, func_name = wanted.rpartition(".")
    for node in _descend(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for method in node.body:
            if (
                isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
                and method.name == func_name
                and (not cls_name or node.name == cls_name)
            ):
                return method, node
    for node in _descend(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == func_name
        ):
            return node, None
    return None, None


def _weaker_answers(steps: list, roots: dict, tally: Counter) -> list:
    """Steps that closed a ``self.<field>`` while a binding was in scope.

    *tally* carries what this could not decide.  A check that drops a case it
    cannot read looks exactly like a check that found nothing, which is the
    shape P1-25's rule 6 is about: the unreadable ones are counted and printed
    beside the finding rather than left out of it.
    """
    out = []
    scopes: dict = {}
    for step in steps:
        if step.kind != trace_mod.TRACE_UNBOUND or not step.name.startswith("self."):
            continue
        tally["self.<field> steps"] += 1
        if not step.terminal:
            tally["  left open by the walk"] += 1
            continue
        tally["  closed by the walk"] += 1
        field = step.name.split(".", 1)[1]
        key = (step.file, step.owner)
        if key not in scopes:
            scopes[key] = _scope_of(roots, step.file, step.owner)
        func, cls = scopes[key]
        if func is None:
            tally["  scope not found in this tree"] += 1
            continue
        if field in _bound_here(func):
            out.append((step, "same function"))
        elif cls is not None and field in _bound_in_class(cls):
            out.append((step, "enclosing class"))
        elif cls is None:
            tally["  no enclosing class"] += 1
    return out


def _handovers(junction) -> list:
    from bamboo.codemap.models import Handover

    return list(
        {
            (e.entry, e.via, e.reached_by): Handover(
                entry=e.entry,
                via=e.via or "",
                reached_by=e.reached_by,
                fields=dict(e.arg_binding),
            )
            for e in junction.entry_points
            if e.arg_binding
        }.values()
    )


async def _scan(map_id: str, source_root: Optional[Path], budget, top: int) -> None:
    """Walk every arm the map can locate, and count the weaker answers."""
    from bamboo.codemap.lookup import CodeMap
    from bamboo.codemap.models import JunctionNode
    from bamboo.database.graph_database_client import GraphDatabaseClient

    roots = PandaCodeMapPlugin._resolve_roots(source_root)
    graph_db = GraphDatabaseClient()
    await graph_db.connect()
    try:
        code_map = CodeMap(graph_db, map_id=map_id)
        junctions = await code_map._find(JunctionNode)
    finally:
        await graph_db.close()

    arms = walked = steps_total = 0
    hits: list = []
    tally: Counter = Counter()
    for junction in junctions:
        if junction.anchor is None:
            continue
        lines = sorted(
            {b.line for b in junction.branches if b.line} or {junction.anchor.line_start}
        )
        arms += len(lines)
        steps, _note = trace_mod.walk(
            roots,
            file=junction.anchor.file,
            owner=junction.owner,
            lines=lines,
            handovers=_handovers(junction),
            classify=provenance.classify,
            budget=budget,
        )
        if not steps:
            continue
        walked += len(lines)
        steps_total += len(steps)
        found = _weaker_answers(steps, roots, tally)
        if found:
            hits.extend((junction.owner, step, where) for step, where in found)

    click.echo(
        f"junctions {len(junctions)}   arms {arms}   walked {walked}   steps {steps_total}"
    )
    # Split before it is counted.  A ``classify()`` terminal reached while a
    # binding is also in scope is stage 1 doing its job: a guard that reads
    # ``self.cur`` arrives here undecided, and the field is still the
    # database however its ``__init__`` obtained the handle.  Folding the
    # two together would set a target A is designed not to meet.
    handed = [h for h in hits if h[1].terminal == trace_mod.STOP_PARAMETER]
    classified = [h for h in hits if h[1].terminal != trace_mod.STOP_PARAMETER]
    handed_owners = len({s.owner for _o, s, _w in handed})
    click.echo(
        "\nhanded off although a binding was in scope: "
        f"{len(handed)} step(s) over {handed_owners} owner(s)   <- what A has to take to zero"
    )
    for where, count in Counter(w for _o, _s, w in handed).most_common():
        click.echo(f"    {where:<18} {count}")
    click.echo(
        f"\nanswered by classify() with a binding also in scope: {len(classified)}"
        "   <- stage 1 winning, and it is meant to"
    )
    for terminal, count in Counter(s.terminal for _o, s, _w in classified).most_common():
        click.echo(f"    {terminal}: {count}")
    click.echo("\n  what the check saw, so a silent drop cannot read as a clean bill:")
    for label, count in tally.most_common():
        click.echo(f"    {label:<32} {count}")
    hits = handed
    by_owner = Counter(owner for owner, _s, _w in hits)
    click.echo("\n  where they are:")
    for owner, count in by_owner.most_common(top):
        click.echo(f"    {count:>4}  {owner}")
    if len(by_owner) > top:
        click.echo(f"    … {len(by_owner) - top} more owner(s)")


if __name__ == "__main__":
    main()
