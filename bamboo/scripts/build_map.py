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
        for basis in ("certain", "container", "structural", "unresolved"):
            count = bases.get(basis, 0)
            click.echo(f"  {basis:<12} {count:>5}  ({count * 100 // total if total else 0}%)")
        thin = gates.unresolved_attributes(fragment)
        if thin:
            # Each of these is a candidate for one line of type annotation
            # upstream, which is the intended remedy -- cheaper and more honest
            # than another inference rule here.
            click.echo("  attributes with unresolved writes (annotation candidates):")
            for attribute, writes in thin[:top]:
                click.echo(f"    {attribute:<20} {writes} write(s)")
            if len(thin) > top:
                click.echo(f"    … {len(thin) - top} more")

        sinks = gates.unreachable_values(fragment)
        if sinks:
            # The plan's "in-edges but no out-edge" invariant, as a report:
            # a terminal status is supposed to be a sink and nothing in the
            # source says which ones those are.
            click.echo(f"  written but nothing selects rows on ({len(sinks)} subject(s)):")
            for subject, values in sinks[:top]:
                click.echo(f"    {subject:<30} {values}")
            if len(sinks) > top:
                click.echo(f"    … {len(sinks) - top} more")

        outside = gates.carried_from_outside(fragment)
        if outside:
            # A terminal, in the same sense as a boundary: the walk stops and
            # the answer is complete rather than missing.
            click.echo(f"  carried from a field the map does not explain ({len(outside)}):")
            for subject, source in outside[:top]:
                click.echo(f"    {subject} <- {source}")

        gated = sorted(
            {gate for subject in fragment.subjects for gate in subject.selection_gates}
        )
        if gated:
            # The same shape as the line above, on the read side: a query's
            # reach depends on a table nothing here keeps current, so a row
            # can be missed for a reason no branch condition mentions.
            click.echo(f"  queries bounded by a table nothing here writes ({len(gated)}):")
            for table in gated[:top]:
                users = sorted(
                    s.name for s in fragment.subjects if table in s.selection_gates
                )
                click.echo(f"    {table:<34} gates {len(users)} subject(s)")

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

    if fragment.filter_stages:
        chains = Counter(s.owner for s in fragment.filter_stages)
        click.echo(f"\nfilter stages: {len(fragment.filter_stages)} in {len(chains)} chain(s)")
        for owner, count in chains.most_common(top):
            click.echo(f"  {count:>3}  {owner}")
        blind = gates.unexplainable_rejections(fragment)
        if blind:
            # What a rejection did not log cannot be checked afterwards, so
            # the map should not promise "read the log" for these.
            click.echo(
                f"  rejections logging none of what they tested ({len(blind)}):"
            )
            for tag, inputs, where in blind[:top]:
                click.echo(f"    {tag:<22} tests {inputs}  ({where})")
            if len(blind) > top:
                click.echo(f"    … {len(blind) - top} more")

    if fragment.diagnostics:
        fields = Counter(d.field for d in fragment.diagnostics)
        distinct = {d.template for d in fragment.diagnostics}
        click.echo(
            f"\ndiagnostic templates: {len(fragment.diagnostics)} "
            f"({len(distinct)} distinct) in {len(fields)} field(s)"
        )
        for field, count in fields.most_common(top):
            click.echo(f"  {count:>3}  {field}")
        if len(fields) > top:
            click.echo(f"    … {len(fields) - top} more field(s)")
        # Said out loud because it is the one part of the map promotion does
        # not filter: these are here precisely because their fields are not
        # subjects, and a reader who assumed otherwise would expect a junction.
        click.echo(
            "  outside promotion: a free-text field is not a subject, and this "
            "answers\n  \"who wrote this line\" rather than \"why is it this value\""
        )

    if fragment.enumeration_writes:
        bound: dict[str, set[str]] = {}
        for write in fragment.enumeration_writes:
            bound.setdefault(write.field, set()).add(write.namespace)
        constants = {w.constant for w in fragment.enumeration_writes}
        click.echo(
            f"\nenumeration bindings: {len(fragment.enumeration_writes)} write(s) "
            f"of {len(constants)} constant(s) into {len(bound)} field(s)"
        )
        for field, spaces in sorted(bound.items()):
            click.echo(f"  {field}  <-  {', '.join(sorted(spaces))}")
        # The same note the templates carry, for the same reason, plus what
        # this one buys: values are reused across enumerations, so a code seen
        # in a record cannot be decoded without knowing which field held it.
        click.echo(
            "  outside promotion: an index, not a claim that the field is a "
            "subject.\n  It is what makes a value observed in a record decodable, "
            "since the\n  index is keyed on (namespace, value) and values repeat "
            "across namespaces"
        )

    if fragment.boundaries:
        systems = Counter(b.system for b in fragment.boundaries)
        click.echo(
            "\nboundaries: "
            + ", ".join(f"{system}={count}" for system, count in systems.most_common())
        )
        shared = [b for b in fragment.boundaries if b.transport == "shared_table"]
        if shared:
            # Listed rather than counted: a shared table is investigated by
            # querying it, so the columns and the verbs are the investigation.
            # "DELETE, INSERT" on a command table is a finding on its own.
            click.echo(f"  through a shared table ({len(shared)}):")
            for channel in shared:
                click.echo(
                    f"    {channel.interface}  [{', '.join(channel.operations)}]"
                    f"  in {len(channel.carried_values)} / out {len(channel.handed_over)}"
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
        if result.note and not result.passed:
            # A note explains why a failure matters; on a passing gate it is
            # prose nobody needs, and it made the least interesting gates as
            # loud as the ones that found something.
            click.echo(f"      note: {result.note}")
    return all_passed


def _report_triggers(fragment: MapFragment, plugin: object, top: int) -> None:
    """Print how junctions are reached, and what follows from it.

    Plugin-specific, like the table findings: what counts as an entry point is
    a fact about how the target system is deployed, not about the map model.
    """
    from bamboo.codemap.panda.recognizers import trigger

    reached, total = getattr(plugin, "trigger_reach", (0, 0))
    if not total:
        return
    kinds = Counter(e.trigger for j in fragment.junctions for e in j.entry_points)
    click.echo(
        f"\nentry points: {reached}/{total} junction(s) reached  ("
        + ", ".join(f"{k}={v}" for k, v in kinds.most_common())
        + ")"
    )

    # Where to look, which is a different question from what starts it: the
    # proxy mixins declare no logger and the knights that call them do.
    own = [j for j in fragment.junctions if j.log_files]
    borrowed = [j for j in fragment.junctions if not j.log_files and j.caller_log_files]
    silent = [j for j in fragment.junctions if not j.log_files and not j.caller_log_files]
    click.echo(
        f"log files: {len(own)} junction(s) from their own logger, "
        f"{len(borrowed)} from a caller's, {len(silent)} from neither"
    )
    if silent:
        # Not a defect: a base class logging through a caller's MsgWrapper has
        # no file to name.  Listed because an unnamed file is an observation
        # the map cannot offer, and that has to be visible rather than implied.
        by_module = Counter(j.owner.split("::")[0] for j in silent)
        click.echo("  no log file named by the source:")
        for module, count in by_module.most_common(top):
            click.echo(f"    {module:<58} {count} junction(s)")
        if len(by_module) > top:
            click.echo(f"    … {len(by_module) - top} more module(s)")

    fragile = trigger.fragile_subjects(fragment.junctions)
    if fragile:
        # The question the map exists to answer for a stalled task: nothing
        # re-evaluates these, so waiting will not help.
        click.echo(f"  no self-repairing trigger reaches ({len(fragile)}):")
        for subject, triggers in fragile[:top]:
            click.echo(f"    {subject:<34} only {', '.join(triggers)}")

    differing = trigger.differing_arguments(fragment.junctions)
    if differing:
        # Structure, not trivia: an argument one entry omits is a guard that
        # cannot fire on that path, so the candidate causes differ by entry.
        click.echo(f"  entries handing over different arguments ({len(differing)}):")
        for subject, owner, supplied in differing[:top]:
            click.echo(f"    {subject} at {owner.split('::')[-1]}")
            for entry, args in supplied.items():
                click.echo(f"      {entry.split('/')[-1]:<38} {args}")


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

    # Plugin-specific findings.  Kept out of the fragment because they describe
    # the target system rather than the map: a table holding no spec is not a
    # node, and a contradiction about one is a fact about PanDA's schema.
    conflicts = getattr(plugin, "table_conflicts", {})
    if conflicts:
        all_passed = False
        click.echo(f"\n[FAIL] table-class-agrees: {len(conflicts)} table(s) disagree")
        for table, classes in sorted(conflicts.items()):
            click.echo(f"      - {table} reads as {sorted(classes)}")
        click.echo("      note: a table holds one kind of row; two answers means one reading is wrong.")
    dropped = getattr(plugin, "unpromoted", (0, 0))
    if any(dropped):
        # Not a coverage gap: an attribute no predicate gates, no vocabulary
        # declares and no closed literal set fills is a field nobody
        # investigates.  Reporting the count keeps the filter visible.
        click.echo(
            f"\nnot promoted: {dropped[0]} subject(s), {dropped[1]} junction(s) "
            "— no criterion fired"
        )
    unexplained = getattr(plugin, "unexplained_steps", [])
    if unexplained:
        # The funnel will report candidates disappearing here and the map has
        # nothing to say about where they went -- the one way this slice can be
        # quietly wrong, so it is named rather than left in a ratio.
        all_passed = False
        click.echo(
            f"\n[FAIL] selection-steps-explained: {len(unexplained)} step(s) count a cut "
            "with no readable reason"
        )
        for step in unexplained[:top]:
            click.echo(f"      - {step}")

    _report_triggers(fragment, plugin, top)

    uncovered = getattr(plugin, "uncovered_tables", set())
    if uncovered:
        # Not a gap in extraction: these are written by the code and hold no
        # spec, so nothing about them can become a subject.  Named because
        # "the map is silent here" is worth knowing before an incident needs it.
        click.echo(f"\ntables written but holding no spec ({len(uncovered)}):")
        click.echo("  " + ", ".join(sorted(uncovered)[:top]))
        if len(uncovered) > top:
            click.echo(f"  … {len(uncovered) - top} more")

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
