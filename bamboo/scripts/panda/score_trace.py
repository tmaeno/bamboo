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
@click.option("--sample", type=click.Path(exists=True, path_type=Path), required=True)
@click.option(
    "--verdicts",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Tally this file instead of printing a worksheet.",
)
@click.option("--out", type=click.Path(path_type=Path), default=None)
@click.option("--depth", type=int, default=trace_mod.DEFAULT_BUDGET.depth)
@click.option("--source-root", type=click.Path(path_type=Path), default=None)
def main(
    sample: Path,
    verdicts: Optional[Path],
    out: Optional[Path],
    depth: int,
    source_root: Optional[Path],
) -> None:
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


if __name__ == "__main__":
    main()
