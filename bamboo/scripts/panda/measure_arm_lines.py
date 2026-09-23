"""Throwaway diagnostic: how well does a line name the arm that printed it?

Not production code.  It exists so that one number stops being a document.
The plan this round came from carried a breakdown of arm-sentence ambiguity --
1849 sentences, 396 naming one arm -- that no code computed and that nothing
could reproduce; it is retired here, and what replaces it is whatever this
prints today.

Two populations, and keeping them apart is the point:

**The map's arm sentences.**  One per ``junction x branch x emit`` the build
recorded as reporting a decision.  This is what ``line_shape`` pools into a
single question per subject, by majority, and it knows nothing about the value
actually observed.

**The skeleton's lines.**  One per ``(value term, printed line, reachable
arm)`` the use-time walk renders.  A different and smaller population by
construction, and reported beside the map's sentences *for the same
derivations*, because the interesting comparison is like for like and the
global map-side figure is a different denominator.

Both are scored by :func:`strategy.discrimination`, which is shared with the
report rather than restated, so a difference between the two cadences cannot
turn out to be a difference between two scorers.

It also answers the question that found the last round's defect: **do these
patterns match the lines production actually wrote?**  Given ``--evidence`` it
puts every rendered pattern against the log lines already captured in that
file, and prints what matched.  A pattern checked only against the rule that
generated it is not checked; ``set\\ .*=None`` passed every test in the suite
and matched a line about a different field entirely.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections import Counter
from pathlib import Path
from typing import Optional

import click

from bamboo.codemap import models as models_mod
from bamboo.codemap import strategy as strategy_mod
from bamboo.codemap.lookup import CodeMap
from bamboo.codemap.models import REPORTS_DECISION, JunctionNode
from bamboo.codemap.panda.plugin import PandaCodeMapPlugin


def _captured(evidence: Path) -> list[str]:
    """Every production log line the evidence file already holds.

    Harvested by key rather than through the evidence model because what is
    wanted is the raw lines whatever shape they were stored in, and this
    script is a diagnostic that should not fail when that shape moves.
    """
    lines: list[str] = []

    def walk(node: object) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in ("lines", "sample", "samples") and isinstance(value, list):
                    lines.extend(one for one in value if isinstance(one, str))
                else:
                    walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(json.loads(evidence.read_text()))
    return lines


def _against_production(patterns: dict[str, tuple[str, str]], lines: list[str]) -> None:
    """Put every rendered pattern against the lines production actually wrote.

    The check that found the defect this round opens with, and the only one
    that could have.  A pattern is generated from a rule, and testing it
    against that rule proves the rule was applied -- not that the result asks
    what it looks like it asks.  ``set\\ .*=None`` was rendered correctly by a
    correct rule and matched ``set task_status=pending oldTask=False ...``,
    because ``.*`` had nothing to stop it.

    What is printed is every match with its source, because whether a match is
    the line the pattern meant is a judgement about this corpus that no
    assertion here can make.  A reader checks them.
    """
    click.echo(f"\nagainst {len(lines)} captured production line(s)")
    hits: Counter = Counter()
    first: dict[str, str] = {}
    broken = 0
    compiled = []
    for pattern, (term, text) in patterns.items():
        try:
            compiled.append((re.compile(pattern), pattern, term, text))
        except re.error as exc:
            broken += 1
            click.echo(f"  unusable pattern {pattern!r}: {exc}")
    for line in lines:
        for rx, pattern, _term, _text in compiled:
            found = rx.search(line)
            if found:
                hits[pattern] += 1
                # The matching region, not the head of the line.  A line here
                # runs to 300 characters and the match is routinely past 240,
                # so printing the first 130 showed a reader a line that had
                # nothing to do with the pattern above it -- which is the one
                # thing this output exists to let them check.
                first.setdefault(
                    pattern,
                    line[max(0, found.start() - 30) : found.end() + 15],
                )
    click.echo(f"  {len(compiled)} distinct pattern(s), {broken} that would not compile")
    click.echo(f"  {len(hits)} of them matched a real line")
    for pattern, count in hits.most_common():
        term, text = patterns[pattern]
        click.echo(f"\n  {count:>5}x  {term}")
        click.echo(f"         pattern  {pattern}")
        click.echo(f"         source   {text[:110]}")
        click.echo(f"         matched  ...{first[pattern]}...")


def _line(counted: Counter, total: int) -> str:
    parts = [f"{counted[kind]:>5}  {kind}" for kind in strategy_mod.DISCRIMINATION]
    return f"  {total} sentence(s)\n" + "\n".join(f"  {part}" for part in parts)


async def _run(
    map_id: str,
    source_root: Optional[Path],
    out: Optional[Path],
    evidence: Optional[Path] = None,
) -> None:
    from bamboo.database.graph_database_client import GraphDatabaseClient

    roots = PandaCodeMapPlugin._resolve_roots(source_root)
    graph_db = GraphDatabaseClient()
    await graph_db.connect()
    try:
        code_map = CodeMap(graph_db, map_id=map_id)

        # The whole map, independent of any symptom: what the build has to work
        # with before a question is asked of it.
        everywhere = [
            strategy_mod.Sentence(emit.template, junction.owner, branch.line, junction.log_files)
            for junction in await code_map._find(JunctionNode)
            for branch in junction.branches
            for emit in branch.emits
            if emit.reports == REPORTS_DECISION
        ]
        click.echo("the map's arm sentences, over the whole map")
        click.echo(_line(strategy_mod.discrimination(everywhere), len(everywhere)))

        terms = [t for t in await code_map.vocabulary() if t.kind == "value"]
        from_map: Counter = Counter()
        from_trace: Counter = Counter()
        totals = {"terms": len(terms), "map": 0, "trace": 0, "arms": 0, "anchored": 0}
        shape: Counter = Counter()
        patterns: dict[str, tuple[str, str]] = {}
        rows = []
        for term in terms:
            strategy = await strategy_mod.derive(code_map, term.symptom)
            strategy_mod.attach_traces(strategy, roots)
            walked = strategy_mod.skeleton_sentences(strategy.readings)
            from_trace += strategy_mod.discrimination(walked)
            from_map[len({e.log_pattern for e in strategy.readings if e.log_pattern})] += 1
            totals["map"] += len({e.log_pattern for e in strategy.readings if e.log_pattern})
            totals["trace"] += len(walked)
            totals["arms"] += sum(len(entry.lines) for entry in strategy.readings)
            totals["anchored"] += len(
                {row.line for e in strategy.readings for row in e.skeleton if row.value}
            )
            for entry in strategy.readings:
                for row in entry.skeleton:
                    if row.kind != models_mod.SKELETON_PRINT:
                        continue
                    shape["rows"] += 1
                    if row.pattern:
                        patterns.setdefault(row.pattern, (term.key, row.text))
                    else:
                        shape["refused"] += 1
                    if row.value:
                        shape["with a value in the hole"] += 1
                        patterns.setdefault(row.value, (term.key, row.text))
                    shape[f"reachable from {min(len(row.arms), 3)} arm(s)"] += 1
            rows.append(
                {
                    "term": f"{term.symptom.subject}={term.symptom.observed}",
                    "map": len({e.log_pattern for e in strategy.readings if e.log_pattern}),
                    "trace": len(walked),
                    "skeleton": [
                        {"owner": e.owner, **row.model_dump()}
                        for e in strategy.readings
                        for row in e.skeleton
                    ],
                }
            )
    finally:
        await graph_db.close()

    click.echo(f"\nover the {totals['terms']} value term(s) of the vocabulary")
    click.echo(f"  {totals['arms']} arm(s) read, {totals['anchored']} of them anchored")
    click.echo(
        f"\n  the map settled {totals['map']} shared sentence(s) across them all, "
        f"by majority: {dict(sorted(from_map.items()))} term(s) by count of sentences"
    )
    click.echo("\nthe skeleton's lines, in those derivations")
    click.echo(_line(from_trace, totals["trace"]))
    click.echo("\nthe shape of those rows")
    for name, count in sorted(shape.items()):
        click.echo(f"  {count:>5}  {name}")
    if evidence is not None:
        _against_production(patterns, _captured(evidence))
    if out is not None:
        out.write_text(json.dumps({"totals": totals, "rows": rows}, indent=1, default=str))
        click.echo(f"\nwritten to {out}")


@click.command("measure-arm-lines")
@click.option("--map-id", default="panda", show_default=True)
@click.option("--source-root", type=click.Path(path_type=Path), default=None)
@click.option("--out", type=click.Path(path_type=Path), default=None)
@click.option(
    "--evidence",
    type=click.Path(path_type=Path, exists=True),
    default=None,
    help="Put the rendered patterns against the log lines this file already holds.",
)
def main(
    map_id: str, source_root: Optional[Path], out: Optional[Path], evidence: Optional[Path]
) -> None:
    """Score the map's arm sentences and the skeleton's lines."""
    asyncio.run(_run(map_id, source_root, out, evidence))


if __name__ == "__main__":
    main()
