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

**The trace's predicted lines.**  One per ``(value term, arm)`` the use-time
walk could anchor, which is a different and smaller population by
construction: the value is half the question, so the same arm asked about two
values is two lines.  Reported beside the map's sentences *for the same
derivations*, because the interesting comparison is like for like and the
global map-side figure is a different denominator.

Both are scored by :func:`strategy.discrimination`, which is shared with the
report rather than restated, so a difference between the two cadences cannot
turn out to be a difference between two scorers.
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from pathlib import Path
from typing import Optional

import click

from bamboo.codemap import strategy as strategy_mod
from bamboo.codemap.lookup import CodeMap
from bamboo.codemap.models import REPORTS_DECISION, JunctionNode
from bamboo.codemap.panda.plugin import PandaCodeMapPlugin


def _line(counted: Counter, total: int) -> str:
    parts = [f"{counted[kind]:>5}  {kind}" for kind in strategy_mod.DISCRIMINATION]
    return f"  {total} sentence(s)\n" + "\n".join(f"  {part}" for part in parts)


async def _run(map_id: str, source_root: Optional[Path], out: Optional[Path]) -> None:
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
        rows = []
        for term in terms:
            strategy = await strategy_mod.derive(code_map, term.symptom)
            strategy_mod.attach_traces(strategy, roots)
            walked = strategy_mod.predicted_sentences(strategy.readings)
            from_trace += strategy_mod.discrimination(walked)
            from_map[len({e.log_pattern for e in strategy.readings if e.log_pattern})] += 1
            totals["map"] += len({e.log_pattern for e in strategy.readings if e.log_pattern})
            totals["trace"] += len(walked)
            totals["arms"] += sum(len(entry.lines) for entry in strategy.readings)
            totals["anchored"] += len({line.line for e in strategy.readings for line in e.predicted})
            rows.append(
                {
                    "term": f"{term.symptom.subject}={term.symptom.observed}",
                    "map": len({e.log_pattern for e in strategy.readings if e.log_pattern}),
                    "trace": len(walked),
                    "predicted": [
                        {"owner": e.owner, **line.model_dump()}
                        for e in strategy.readings
                        for line in e.predicted
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
    click.echo("\nthe trace's predicted lines, in those derivations")
    click.echo(_line(from_trace, totals["trace"]))
    if out is not None:
        out.write_text(json.dumps({"totals": totals, "rows": rows}, indent=1, default=str))
        click.echo(f"\nwritten to {out}")


@click.command("measure-arm-lines")
@click.option("--map-id", default="panda", show_default=True)
@click.option("--source-root", type=click.Path(path_type=Path), default=None)
@click.option("--out", type=click.Path(path_type=Path), default=None)
def main(map_id: str, source_root: Optional[Path], out: Optional[Path]) -> None:
    """Score the map's arm sentences and the trace's predicted lines."""
    asyncio.run(_run(map_id, source_root, out))


if __name__ == "__main__":
    main()
