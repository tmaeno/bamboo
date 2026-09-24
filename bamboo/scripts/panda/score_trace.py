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

**The sample is drawn here, stratified, and nothing is sealed.**  Two
walkthroughs picked by hand would only ever test the blind spots that were
predicted from reading the code, and the corpus has punished reasoning from a
list of known failures before.  Drawing lives in this file rather than beside
it because the last sample was drawn by a script that is gone: the cases
survived in a JSON file nobody could regenerate, then the JSON went too, and a
sealed holdout that cannot be opened is worse than no holdout.  So the draw is
a flag here, the sample it writes is committed, and none of it is sealed.

**The strata are what the walk has to resolve**, not the shape of the guard
above the arm -- that axis was tried and it cut across the thing being scored.
Each arm goes to the rarest kind of step its walk produced, and the rarity is
measured in the same pass rather than fixed here, because it moves: the walk
returned 2969 steps over 830 arms when this was first sized and returns four
times that now.

Honest about what this cannot be: the author of the walk is also its scorer,
so sealing the holdout buys "not tuned against these cases" and not blindness.
"""

from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import logging
import random
from collections import Counter
from pathlib import Path
from typing import Optional

import click

from bamboo.codemap import models as models_mod
from bamboo.codemap import reading as reading_mod
from bamboo.codemap import trace as trace_mod
from bamboo.codemap.panda import provenance
from bamboo.codemap.panda.plugin import PandaCodeMapPlugin

# The level vocabulary rather than a second copy of it, for the reason
# ``log_level`` is public: one of them has to be kept current.
from bamboo.codemap.panda.recognizers.selection import _LOG_LEVELS

logger = logging.getLogger(__name__)


def _case_id(case: dict) -> str:
    return f"{case['owner']}@{case['line']}"


def _skeleton_rows(skeleton) -> list[str]:
    """What the walk says this function prints, laid out as source.

    Scored beside the steps rather than instead of them, because they fail
    differently: a step is wrong when it names a site the source does not have,
    and a skeleton row is wrong when the pattern it renders is not what that
    line would print.  The nesting is kept because it carries the part a list
    cannot -- two rows under one ``if`` were printed together or not at all.
    """
    if not skeleton:
        return ["    (nothing)"]
    out = []
    for row in skeleton:
        pad = "  " * row.depth
        if row.kind == models_mod.SKELETON_BRANCH:
            out.append(f"           {pad}{row.text}")
        elif row.kind == models_mod.SKELETON_ARM:
            out.append(f"    {row.line:>5} ARM {pad}{row.text}")
        else:
            body = row.pattern or f"-- {row.refused} --"
            out.append(f"    {row.line:>5}  |  {pad}{body}")
            if row.value:
                out.append(f"           {pad}with the value: {row.value}")
    return out


def _render(case: dict, roots: dict, budget: trace_mod.Budget) -> list[str]:
    out: list[str] = []
    # ``observed`` is what makes the walk render the narrower pattern -- the
    # line with the value in its hole beside the line with every hole open.
    # Left at its default the skeleton still comes back, and the half of it
    # this sample is meant to score does not.
    steps, note, skeleton = trace_mod.walk(
        roots,
        file=case["file"],
        owner=case["owner"],
        lines=[case["line"]],
        observed=case.get("outcome") or "",
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
    out.append("  -- what the skeleton says is printed " + "-" * 36)
    out.extend(_skeleton_rows(skeleton))
    out.append("  -- the source " + "-" * 59)
    region = reading_mod.region_for(
        roots, file=case["file"], owner=case["owner"], line=case["line"]
    )
    out.append(region.marked([case["line"]]) if region else "    (not found)")
    return out


async def _draw(
    map_id: str,
    source_root: Optional[Path],
    budget,
    seed: int,
    per_stratum: int,
    out: Optional[Path],
) -> None:
    """Walk every arm, stratify by what the walk had to resolve, and sample.

    The rarity order is computed here rather than written down, so a stratum
    that stops being rare stops being over-sampled.  An arm goes to the rarest
    kind of step its own walk produced: sampling by the commonest instead puts
    almost every case in one stratum, since nearly every walk binds something.
    """
    from bamboo.codemap.lookup import CodeMap
    from bamboo.codemap.models import JunctionNode
    from bamboo.database.graph_database_client import GraphDatabaseClient

    roots = PandaCodeMapPlugin._resolve_roots(source_root)
    graph_db = GraphDatabaseClient()
    await graph_db.connect()
    try:
        junctions = await CodeMap(graph_db, map_id=map_id)._find(JunctionNode)
    finally:
        await graph_db.close()

    arms: list[dict] = []
    kind_counts: Counter = Counter()
    for junction in junctions:
        if junction.anchor is None:
            continue
        by_line = {b.line: b for b in junction.branches if b.line}
        lines = sorted(by_line) or [junction.anchor.line_start]
        for line in lines:
            steps, _note, _skeleton = trace_mod.walk(
                roots,
                file=junction.anchor.file,
                owner=junction.owner,
                lines=[line],
                observed=(by_line.get(line).outcome if by_line.get(line) else ""),
                handovers=_handovers(junction),
                classify=provenance.classify,
                budget=budget,
            )
            kinds = {
                f"unbound:{step.terminal or 'none'}"
                if step.kind == trace_mod.TRACE_UNBOUND
                else step.kind
                for step in steps
            }
            kind_counts.update(kinds)
            branch = by_line.get(line)
            arms.append(
                {
                    "file": junction.anchor.file,
                    "owner": junction.owner,
                    "line": line,
                    "outcome": branch.outcome if branch else "",
                    "tier": branch.tier if branch else 0,
                    "kinds": sorted(kinds),
                }
            )

    rarity = dict(kind_counts)
    for arm in arms:
        arm["stratum"] = (
            min(arm["kinds"], key=lambda k: (rarity[k], k)) if arm["kinds"] else "no steps"
        )
    strata: dict[str, list[dict]] = {}
    for arm in arms:
        strata.setdefault(arm["stratum"], []).append(arm)

    click.echo(f"arms {len(arms)}   strata {len(strata)}   seed {seed}")
    drawn: list[dict] = []
    for name in sorted(strata, key=lambda s: (-len(strata[s]), s)):
        pool = sorted(strata[name], key=lambda a: (a["owner"], a["line"]))
        take = min(per_stratum, len(pool))
        picked = random.Random(f"{seed}:{name}").sample(pool, take)
        drawn.extend(picked)
        click.echo(f"  {len(pool):>4} arm(s)  drew {take}   {name}")
    click.echo(f"\nn={len(drawn)}")
    # Rule 4: say what a sample this size would catch, or the number is decor.
    for rate in (0.10, 0.05):
        caught = 1 - (1 - rate) ** len(drawn)
        click.echo(f"  a defect in {rate:.0%} of arms is caught with probability {caught:.0%}")
    if out is not None:
        out.write_text(json.dumps(drawn, indent=1))
        click.echo(f"\nsample written to {out}")


@click.command("score-trace")
@click.option("--sample", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--draw", is_flag=True, help="Walk every arm, stratify, and write a sample.")
@click.option("--seed", type=int, default=None, help="The draw is this and the population.")
@click.option("--per-stratum", default=5, show_default=True)
@click.option(
    "--verdicts",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Tally this file instead of printing a worksheet.",
)
@click.option("--out", type=click.Path(path_type=Path), default=None)
@click.option(
    "--audit",
    is_flag=True,
    help=(
        "Score the sample against the source mechanically instead of "
        "printing a worksheet: what the walk listed beside what the "
        "function holds, with the difference named."
    ),
)
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
    draw: bool,
    seed: Optional[int],
    per_stratum: int,
    verdicts: Optional[Path],
    out: Optional[Path],
    audit: bool,
    scan: bool,
    map_id: str,
    top: int,
    depth: int,
    source_root: Optional[Path],
) -> None:
    if scan:
        asyncio.run(_scan(map_id, source_root, trace_mod.Budget(depth=depth), top))
        return
    if draw:
        if seed is None:
            raise click.UsageError("--draw needs --seed: a draw nobody can repeat is a guess")
        asyncio.run(
            _draw(
                map_id, source_root, trace_mod.Budget(depth=depth), seed, per_stratum, out
            )
        )
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
    if audit:
        _audit(
            cases,
            PandaCodeMapPlugin._resolve_roots(source_root),
            trace_mod.Budget(depth=depth),
            seal,
        )
        return
    roots = PandaCodeMapPlugin._resolve_roots(source_root)
    budget = trace_mod.Budget(depth=depth)
    lines = [f"sample {sample.name}  sha256 {seal}  n={len(cases)}  depth={depth}", ""]
    for index, case in enumerate(cases, 1):
        lines.append("=" * 78)
        lines.append(f"case {index}/{len(cases)}  {_case_id(case)}")
        lines.extend(_render(case, roots, budget))
        lines.append("")
        for artefact in ("walk", "skeleton"):
            lines.append(
                f"VERDICT {artefact}:{_case_id(case)}  "
                "complete=?  sound=?  honest=?  note="
            )
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
    # Kept apart because they are two artefacts, not two readings of one: the
    # walk can be right about why an arm ran while the skeleton is wrong about
    # what the function prints, and an average over both would hide either.
    artefacts: dict[str, dict[str, Counter]] = {}
    for case_id, fields in scored.items():
        artefact, _, _rest = case_id.partition(":")
        if not _rest:
            artefact = "walk"
        bucket = artefacts.setdefault(
            artefact, {k: Counter() for k in ("complete", "sound", "honest")}
        )
        for key, counter in bucket.items():
            counter[fields.get(key, "?")] += 1
    click.echo(f"sample sha256 {seal}  n={len(cases)}  scored={len(scored)}")
    for artefact in sorted(artefacts):
        click.echo(f"  {artefact}")
        for key, counter in artefacts[artefact].items():
            total = sum(v for k, v in counter.items() if k != "?")
            yes = counter.get("y", 0)
            click.echo(
                f"    {key:<9} {yes}/{total}"
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
        steps, _note, _skeleton = trace_mod.walk(
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


# --------------------------------------------------------------------------
# The audit: the machine half of a scoring round, kept as an artefact.
#
# P1-35 reported five figures and kept nothing that could produce them again
# -- the checks were heredocs and went with the shell.  That is the failure
# the retired holdout had and the failure a tool named in a plan for six
# rounds had, so the third time it is written down.
#
# **Nothing here decides whether a site can reach an arm.**  That predicate is
# what a fix round changes, and an oracle holding its own copy would agree
# with whatever the walk had just been taught.  What the audit states instead
# is the positional fact -- below the arm, inside a loop the arm is also in --
# and names every step it holds for, so the list is read rather than counted.


def _audit_names(target: ast.AST, name: str) -> bool:
    """Whether *target*, an assignment target, binds the plain name *name*."""
    if isinstance(target, ast.Name):
        return target.id == name
    if isinstance(target, (ast.Tuple, ast.List)):
        return any(_audit_names(element, name) for element in target.elts)
    if isinstance(target, ast.Starred):
        return _audit_names(target.value, name)
    return False


def _binding_lines(func: ast.AST, name: str) -> dict[int, str]:
    """``line -> how`` for every binding of *name* in *func*.

    Read with :func:`_descend` for the reason that function already gives.
    ``self.<field>`` is read as an attribute target instead, which is what
    lets one check cover the steps the walk reaches by descending into
    another method of the same class.
    """
    found: dict[int, str] = {}
    if name.startswith("self."):
        field = name.split(".", 1)[1]
        for node in [func, *_descend(func)]:
            if isinstance(node, ast.Assign):
                targets: list = list(node.targets)
            elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
                targets = [node.target]
            else:
                continue
            for target in targets:
                if (
                    isinstance(target, ast.Attribute)
                    and target.attr == field
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                ):
                    found[node.lineno] = "field"
        return found
    for node in [func, *_descend(func)]:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    found[node.lineno] = "="
                elif isinstance(target, (ast.Tuple, ast.List)) and _audit_names(target, name):
                    found[node.lineno] = "tuple"
        elif isinstance(node, ast.AugAssign):
            if isinstance(node.target, ast.Name) and node.target.id == name:
                found[node.lineno] = "augmented"
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            if isinstance(node.target, ast.Name) and node.target.id == name:
                found[node.lineno] = "annotated"
        elif isinstance(node, (ast.For, ast.AsyncFor)) and _audit_names(node.target, name):
            found[node.lineno] = "for"
        elif isinstance(node, ast.withitem) and node.optional_vars is not None:
            if _audit_names(node.optional_vars, name):
                found[node.context_expr.lineno] = "with"
        elif isinstance(node, ast.NamedExpr):
            if isinstance(node.target, ast.Name) and node.target.id == name:
                found[node.lineno] = "walrus"
        elif isinstance(node, ast.ExceptHandler) and node.name == name:
            found[node.lineno] = "except"
    return found


def _scope_at(roots: dict, file: str, owner: str, line: int):
    """The innermost function holding *line*, or the module for a module arm.

    Not :func:`_scope_of`, which finds a function by name.  ``datasetManager``
    holds seven classes with a ``run`` method and the map records the owner as
    ``run``: a lookup by name reads the first of the seven and then calls every
    binding in the real one a fabrication.  Forty of those were reported before
    this was noticed.  An arm has a line, so the line picks the scope.
    """
    tree = _tree_for(roots, file)
    if tree is None:
        return None
    if owner.rsplit("::", 1)[-1] == "<module>":
        return tree
    holding = [
        node
        for node in [tree, *_descend(tree)]
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.lineno <= line <= (node.end_lineno or node.lineno)
    ]
    return min(holding, key=lambda n: (n.end_lineno or n.lineno) - n.lineno) if holding else tree


def _statement_holding(func: ast.AST, line: int):
    """The innermost statement of *func* on *line*."""
    holding = [
        node
        for node in [func, *_descend(func)]
        if isinstance(node, ast.stmt)
        and node.lineno <= line <= (node.end_lineno or node.lineno)
    ]
    if not holding:
        return None
    return min(holding, key=lambda n: (n.end_lineno or n.lineno) - n.lineno)


def _structures_over(func: ast.AST, line: int) -> list[tuple[str, int]]:
    """``(kind, line)`` for every loop, ``try`` body, handler or ``with``
    whose body holds *line*."""
    out: list[tuple[str, int]] = []
    for node in [func, *_descend(func)]:
        start = getattr(node, "lineno", 0)
        if not start or not (start <= line <= (getattr(node, "end_lineno", 0) or 0)):
            continue
        if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            if node.body and node.body[0].lineno <= line <= (node.body[-1].end_lineno or 0):
                out.append(("loop", start))
        elif isinstance(node, ast.Try) and node.handlers:
            if node.body and node.body[0].lineno <= line <= (node.body[-1].end_lineno or 0):
                out.append(("try", start))
        elif isinstance(node, ast.ExceptHandler):
            out.append(("except", start))
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            out.append(("with", start))
    return out


def _logging_calls(func: ast.AST) -> dict[int, str]:
    """``line -> level`` for every logging call in *func*."""
    found: dict[int, str] = {}
    for node in [func, *_descend(func)]:
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _LOG_LEVELS
            and node.args
        ):
            found[node.lineno] = node.func.attr
    return found


def _audit(cases: list[dict], roots: dict, budget: trace_mod.Budget, seal: str) -> None:
    """Score the sample against the source, mechanically, and print the tally.

    Four numbers and a list.  ``complete`` and ``sound`` and ``honest`` are the
    same three questions the worksheet asks by hand; ``skeleton`` asks whether
    every logging call of the arm's function carries a row.  The list is the
    positional one, and it is a list because it is the one a reader has to
    settle.
    """
    names = names_complete = 0
    step_total = step_exists = 0
    honest_total = honest_said = 0
    calls_total = calls_listed = 0
    module_arms = 0
    missing: list[str] = []
    ghost: list[str] = []
    below: list[str] = []
    in_a_shared_loop: list[str] = []
    silent: list[str] = []

    for case in cases:
        steps, _note, skeleton = trace_mod.walk(
            roots,
            file=case["file"],
            owner=case["owner"],
            lines=[case["line"]],
            observed=case.get("outcome") or "",
            classify=provenance.classify,
            budget=budget,
        )
        cid = _case_id(case)

        # sound: every step that names a line, read in its own frame.
        for step in steps:
            if not step.line:
                continue
            step_total += 1
            where = _scope_at(
                roots, step.file or case["file"], step.owner or case["owner"], step.line
            )
            if where is None:
                continue
            if step.kind == trace_mod.TRACE_WRITE:
                # An arm has no name to look up; what it claims is that the
                # line is a statement of this function, and that is checkable.
                if _statement_holding(where, step.line) is not None:
                    step_exists += 1
                else:
                    ghost.append(f"{cid}  arm@{step.line} is no statement of this function")
            elif step.name and step.line in _binding_lines(where, step.name):
                step_exists += 1
            else:
                ghost.append(f"{cid}  {step.name}@{step.line} binds nothing in the source")

        # honest: a step under a structure the path condition cannot see.
        for step in steps:
            if not step.line:
                continue
            where = _scope_at(
                roots, step.file or case["file"], step.owner or case["owner"], step.line
            )
            wants = _structures_over(where, step.line) if where is not None else []
            if not wants:
                continue
            honest_total += 1
            honest_said += all(
                any(said.startswith(f"{kind} at {start}:") for said in step.unseen)
                for kind, start in wants
            )

        func = _scope_at(roots, case["file"], case["owner"], case["line"])
        if func is None:
            continue

        # complete: every binding the arm's own function holds for a name the
        # walk explained there.
        listed: dict[str, set] = {}
        for step in steps:
            if step.kind not in (trace_mod.TRACE_BINDING, trace_mod.TRACE_LOOP):
                continue
            if step.name and not step.name.startswith("self.") and step.owner == case["owner"]:
                listed.setdefault(step.name, set()).add(step.line)
        for name, lines in sorted(listed.items()):
            names += 1
            held = _binding_lines(func, name)
            names_complete += not (set(held) - lines)
            missing.extend(
                f"{cid}  {name}@{line} ({held[line]}) is in the source and not listed"
                for line in sorted(set(held) - lines)
            )

        # The positional relation, stated rather than judged.  Only in the
        # arm's own frame: a field explained from another method has no
        # ordering against the arm at all, and reading one as though it had is
        # how two of these were miscounted by hand.
        arm_loops = {at for kind, at in _structures_over(func, case["line"]) if kind == "loop"}
        for step in steps:
            if step.kind not in (trace_mod.TRACE_BINDING, trace_mod.TRACE_LOOP):
                continue
            if not step.name or step.name.startswith("self.") or not step.line:
                continue
            if step.owner != case["owner"] or step.line <= case["line"]:
                continue
            shared = {
                at for kind, at in _structures_over(func, step.line) if kind == "loop"
            } & arm_loops
            (in_a_shared_loop if shared else below).append(
                f"{cid}  {step.name}@{step.line} is below the arm"
                + (f", in loop(s) {sorted(shared)} with it" if shared else "")
            )

        # skeleton: the logging calls of the arm's function carry a row.
        if case["owner"].rsplit("::", 1)[-1] == "<module>":
            # A module arm has no function to bound the check by, and the
            # skeleton does not claim the whole file.  Counted, not folded in.
            module_arms += 1
            continue
        rows = {row.line for row in skeleton if row.kind == models_mod.SKELETON_PRINT}
        for line, level in sorted(_logging_calls(func).items()):
            calls_total += 1
            if line in rows:
                calls_listed += 1
            else:
                silent.append(f"{cid}  the {level} call@{line} prints and has no row")

    click.echo(f"sample sha256 {seal}  n={len(cases)}\n")
    click.echo(f"  complete   {names_complete}/{names} name(s), every binding of it listed")
    click.echo(f"  sound      {step_exists}/{step_total} step(s) name a site the source has")
    click.echo(f"  honest     {honest_total and honest_said}/{honest_total} "
               "step(s) under a structure say so")
    click.echo(f"  skeleton   {calls_listed}/{calls_total} logging call(s) carry a row"
               f"   ({module_arms} module arm(s) not checked)")
    click.echo(
        f"\nbelow the arm and in no loop it is in: {len(below)} step(s)"
        "   -- a position, not a verdict"
    )
    for line in below:
        click.echo(f"    {line}")
    click.echo(f"\nbelow it but inside a loop it is in too: {len(in_a_shared_loop)} step(s)")
    for line in in_a_shared_loop:
        click.echo(f"    {line}")
    for label, rows_ in (("missing", missing), ("fabricated", ghost), ("silent", silent)):
        click.echo(f"\n{label}: {len(rows_)}")
        for line in rows_:
            click.echo(f"    {line}")


if __name__ == "__main__":
    main()
