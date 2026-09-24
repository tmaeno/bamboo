"""Throwaway diagnostic: draw symptoms at random and read what the map says.

Not production code.  It exists because the plan named it for six rounds and
it was never there -- every draw until now was done by hand, which is the same
failure that retired the sealed holdout: a procedure a document names and no
artefact performs.  What it does is small and the point is that it is written
down.

**The draw is recorded before the report is read.**  That ordering is the whole
test: a sample looked at first and recorded afterwards can always be called
unrepresentative once its answers are in.  So ``--report`` requires ``--out``,
and the file is written before the first report is rendered.

**The population is the map's vocabulary** -- the set of questions the map
claims it can answer -- rather than the values production was seen to hold.
Measured when that was decided: the two symptoms that had found the most were
both absent from the production side, because one is a transient the sample
never caught and the other belongs to a table the evidence only holds failures
for.  Drawing from what the map advertises is what makes this a test of the
advertisement.

``--population`` draws from a plain list of keys instead, one per line, for
categories that are not vocabulary entries -- the junctions with no resolvable
caller, the tables holding no spec.  Those are read by hand; the tool only
draws and records them.
"""

from __future__ import annotations

import asyncio
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Optional

import click

from bamboo.codemap.lookup import CodeMap


def _argv(kind: str, key: str, subject: str, observed: str) -> list[str]:
    """How ``derive-strategy`` is addressed for a term of this kind.

    A value term names its subject and the value; everything else resolves
    through the description, which is the interface a reader would use.
    """
    if kind == "value" and subject and observed:
        return ["--subject", subject, "--observed", observed]
    return ["--describe", key]


async def _vocabulary(map_id: str) -> list[dict]:
    from bamboo.database.graph_database_client import GraphDatabaseClient

    db = GraphDatabaseClient()
    await db.connect()
    try:
        terms = await CodeMap(db, map_id=map_id).vocabulary()
    finally:
        await db.close()
    # Sorted by key so the draw depends on the map's content and not on the
    # order the store happened to return.
    return [
        {
            "kind": term.kind,
            "key": term.key,
            "subject": term.symptom.subject,
            "observed": term.symptom.observed or "",
        }
        for term in sorted(terms, key=lambda t: t.key)
    ]


@click.command("review-symptoms")
@click.option("--seed", type=int, required=True, help="The draw is this and the population.")
@click.option("-k", "--count", default=3, show_default=True, help="How many to draw.")
@click.option("--map-id", default="panda", show_default=True)
@click.option(
    "--kind",
    default=None,
    help="Restrict the population to one kind (value, cut, step, chain).",
)
@click.option(
    "--population",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Draw from this list of keys, one per line, instead of the vocabulary.",
)
@click.option(
    "--out",
    type=click.Path(path_type=Path),
    default=None,
    help="Write the draw here.  Required by --report, and written before it.",
)
@click.option(
    "--report",
    is_flag=True,
    help="After recording the draw, run derive-strategy for each and print it.",
)
@click.option("--evidence", type=click.Path(path_type=Path), default=None)
@click.option("--source-root", type=click.Path(path_type=Path), default=None)
def main(
    seed: int,
    count: int,
    map_id: str,
    kind: Optional[str],
    population: Optional[Path],
    out: Optional[Path],
    report: bool,
    evidence: Optional[Path],
    source_root: Optional[Path],
) -> None:
    """Draw *count* symptoms with *seed*, record them, then read them."""
    if report and out is None:
        raise click.UsageError(
            "--report needs --out: the draw is recorded before it is read, or a "
            "sample can be called unrepresentative once its answers are in."
        )

    if population is not None:
        rows = [
            {"kind": "-", "key": line.strip(), "subject": "", "observed": ""}
            for line in population.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
        where = str(population)
    else:
        rows = asyncio.run(_vocabulary(map_id))
        if kind:
            rows = [row for row in rows if row["kind"] == kind]
        where = f"{map_id} vocabulary" + (f" ({kind})" if kind else "")

    if count > len(rows):
        raise click.UsageError(f"asked for {count} of a population of {len(rows)}")
    drawn = random.Random(seed).sample(rows, count)

    click.echo(f"population {len(rows)} from {where}   seed {seed}   n={count}")
    for i, row in enumerate(drawn, 1):
        click.echo(f"  {i}  {row['kind']:<6} {row['key']}")

    if out is not None:
        out.write_text(
            json.dumps(
                {"seed": seed, "population": len(rows), "where": where, "drawn": drawn},
                indent=1,
            )
        )
        click.echo(f"\ndraw recorded in {out}")

    if not report:
        return
    if population is not None:
        click.echo("\n(a population file is read by hand -- nothing to derive)")
        return

    for i, row in enumerate(drawn, 1):
        click.echo(f"\n{'=' * 24} {i}/{count}  {row['key']} {'=' * 24}")
        argv = [sys.executable, "-m", "bamboo.cli", "derive-strategy"]
        argv += _argv(row["kind"], row["key"], row["subject"], row["observed"])
        if evidence is not None:
            argv += ["--evidence", str(evidence)]
        if source_root is not None:
            argv += ["--source-root", str(source_root)]
        subprocess.run(argv, check=False)


if __name__ == "__main__":
    main()
