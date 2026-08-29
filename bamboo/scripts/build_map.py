"""``bamboo build-map`` — extract a Code Map from a target system's source.

An offline build step, deliberately separate from ``analyze``:

* the costs differ by orders of magnitude -- a build parses a whole source tree
  once per release, an analyze run must answer in seconds;
* a Code Map's lifetime is tied to a *code version*, not to an incident, which
  is what makes the version stamp meaningful;
* ``analyze`` runs unattended and read-only, and a build writes.

Consequently ``analyze`` never rebuilds inline: a missing or stale map is
reported as a degraded run rather than silently repaired.  The cost of that
separation is that nobody rebuilding leaves the map quietly out of date, which
is why ``check-map`` is not optional -- the two commands only make sense
together.

Usage::

    # Build from the installed distribution
    bamboo build-map

    # Build a specific release from a checkout
    bamboo build-map --source-root /path/to/panda

    # Inspect without touching the database
    bamboo build-map --dry-run
"""

from __future__ import annotations

import asyncio
import logging
from collections import Counter
from pathlib import Path
from typing import Optional

import click

from bamboo.codemap import gates
from bamboo.codemap.factory import available_map_ids, get_code_map_plugin
from bamboo.codemap.models import MapFragment
from bamboo.codemap.store import store_fragment

logger = logging.getLogger(__name__)


def _report(fragment: MapFragment, results: list[gates.GateResult], top: int) -> bool:
    """Print the build summary and return True when every gate passed."""
    click.echo(f"\nmap_id       : {fragment.map_id}")
    click.echo(f"derived_from : {fragment.derived_from}")

    totals = gates.slice_totals(fragment)
    if totals:
        click.echo("\nslice coverage:")
        for slice_name, counts in sorted(totals.items()):
            candidates = counts["candidates"]
            explained = counts["explained"]
            pct = f"{explained * 100 // candidates}%" if candidates else "-"
            click.echo(
                f"  {slice_name:<14} {explained:>5}/{candidates:<5} ({pct:>4})  "
                f"in {counts['files']} file(s)"
            )

    rows = gates.coverage_matrix(fragment)
    weak = [r for r in rows if r[4] < 1.0]
    if weak:
        # Files where candidates were found but not explained use a different
        # idiom; naming them is the point, since an unreported gap reads as
        # full coverage.
        click.echo(f"\nlowest-coverage files ({len(weak)} below 100%):")
        for slice_name, file, candidates, explained, ratio in weak[:top]:
            click.echo(
                f"  {ratio:6.0%}  {slice_name:<12} {explained:>4}/{candidates:<4} {file}"
            )
        if len(weak) > top:
            click.echo(f"  … {len(weak) - top} more")

    if fragment.junctions:
        bases = Counter(j.attribution for j in fragment.junctions)
        total = sum(bases.values())
        click.echo("\njunction attribution:")
        for basis in ("certain", "heuristic", "unresolved"):
            count = bases.get(basis, 0)
            click.echo(f"  {basis:<12} {count:>5}  ({count * 100 // total if total else 0}%)")
        mix = gates.attribution_mix(fragment)
        if mix:
            # Named per attribute because the weak bases are not spread evenly:
            # they concentrate on the attributes several classes declare, which
            # are also the ones the reasoning starts from most often.
            click.echo("  attributes with guessed or unresolved writes:")
            for attribute, heuristic, unresolved in mix[:top]:
                click.echo(f"    {attribute:<20} heuristic={heuristic:<4} unresolved={unresolved}")
            if len(mix) > top:
                click.echo(f"    … {len(mix) - top} more")

        drift = gates.outcomes_outside_declared_subsets(fragment)
        if drift:
            # Not a gate: the declared lists are purpose-built subsets, so an
            # outcome outside them is normal.  Shown because a value no
            # declaration mentions is still worth a glance.
            click.echo(f"  outcomes no declared list mentions ({len(drift)}):")
            for subject, outcome, where in drift[:top]:
                click.echo(f"    {subject} = {outcome!r}  ({where})")
            if len(drift) > top:
                click.echo(f"    … {len(drift) - top} more")

    if fragment.boundaries:
        systems = Counter(b.system for b in fragment.boundaries)
        click.echo(
            "\nboundaries: "
            + ", ".join(f"{system}={count}" for system, count in systems.most_common())
        )
        thin = gates.unobservable_boundaries(fragment)
        if thin:
            # What crossed a boundary and was never logged cannot be recovered
            # afterwards, so the limit is worth knowing before an incident
            # turns on it rather than during one.
            click.echo(f"  logging under half of what they receive ({len(thin)}):")
            for interface, observed, carried in thin[:top]:
                click.echo(f"    {observed:>3}/{carried:<3}  {interface}")
            if len(thin) > top:
                click.echo(f"    … {len(thin) - top} more")

    click.echo("\ngates:")
    all_passed = True
    for result in results:
        click.echo(f"  {result.summary()}")
        if not result.passed:
            all_passed = False
            for failure in result.failures[:top]:
                click.echo(f"      - {failure}")
            if len(result.failures) > top:
                click.echo(f"      … {len(result.failures) - top} more")
        if result.note:
            click.echo(f"      note: {result.note}")
    return all_passed


async def _store(fragment: MapFragment) -> dict[str, int]:
    from bamboo.database.graph_database_client import GraphDatabaseClient

    graph_db = GraphDatabaseClient()
    await graph_db.connect()
    try:
        return await store_fragment(fragment, graph_db)
    finally:
        await graph_db.close()


@click.command("build-map")
@click.option(
    "--map-id",
    default="panda",
    show_default=True,
    help="Which Code Map to build.",
)
@click.option(
    "--source-root",
    default=None,
    metavar="PATH",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help=(
        "Analyse packages under PATH instead of the installed distribution. "
        "Use this to map a specific release rather than whatever happens to be "
        "installed next to bamboo."
    ),
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Extract and run gates, but write nothing to the database.",
)
@click.option(
    "--strict",
    is_flag=True,
    help=(
        "Exit non-zero if any gate fails.  Off by default because a gate "
        "failure means two independent expressions of the same fact disagree "
        "-- often the upstream code, not the extraction -- and that is a "
        "signal to look, not a reason to refuse the build."
    ),
)
@click.option("--top", default=10, show_default=True, help="Rows per listing.")
@click.option("-v", "--verbose", is_flag=True, help="DEBUG logging.")
def main(
    map_id: str,
    source_root: Optional[Path],
    dry_run: bool,
    strict: bool,
    top: int,
    verbose: bool,
) -> None:
    """Extract a Code Map and store it under its own labels in the graph DB."""
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)

    try:
        plugin = get_code_map_plugin(map_id)
    except ValueError as exc:
        raise click.ClickException(
            f"{exc}  Available maps: {', '.join(available_map_ids())}"
        ) from exc

    try:
        plugin.prepare(source_root)
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc

    fragment = plugin.run()
    results = gates.run_all(fragment)
    all_passed = _report(fragment, results, top)

    if dry_run:
        click.echo("\n--dry-run: nothing written.")
    else:
        written = asyncio.run(_store(fragment))
        click.echo(
            "\nstored: "
            + ", ".join(f"{k}={v}" for k, v in written.items() if v)
        )

    if not all_passed and strict:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
