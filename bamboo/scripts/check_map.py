"""``bamboo check-map`` — check a Code Map against the running system.

``build-map`` verifies the map against the source it came from.  That catches
extraction bugs, but it cannot see the two things only a deployment knows:
whether the diagnostics the map offers as observables actually survive
production's log level, and whether the transitions its branch tables predict
are the ones that happen.

Those need production, which is why this is a separate command on a separate
cadence.  The map does not change between runs; its trustworthiness does.  A
build nobody checks goes quietly out of date, so the two commands only make
sense together.

Evidence is fetched over PanDA's async grep API and written to a file, then
the gates read the file.  Separating the two means the gates re-run offline,
tests use a fixture instead of the network, and the checking half stays
developable without the API allowlist.

Usage::

    # Fetch fresh evidence and check against it
    bamboo check-map --fetch

    # Re-check against evidence already on disk
    bamboo check-map

    # Check a specific release, and say where its logs are
    bamboo check-map --fetch --source-root /path/to/panda \\
        --log-file jedi:panda-jedi.log --log-file server:panda-server.log
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

import click

from bamboo.codemap import evidence, gates
from bamboo.codemap.factory import available_map_ids, get_code_map_plugin
from bamboo.codemap.models import MapFragment

logger = logging.getLogger(__name__)

DEFAULT_EVIDENCE = Path(".bamboo") / "codemap-evidence.json"


def _targets(fragment: MapFragment, declared: dict[str, str]) -> dict[str, str]:
    """``{log filename: the service that writes it}`` for the files the map uses.

    Derived from the map rather than configured, because the map is what
    knows: each node carries the file its module logs to, so the set of
    questions to ask production is a property of what was extracted.  The
    service comes from the module that *declares* the logger -- the proxy
    mixins name two files, and each of those is declared in one package, so
    the pair resolves without guessing.
    """
    declaring = {filename: rel_path for rel_path, filename in declared.items()}
    targets: dict[str, str] = {}
    for node in list(fragment.filter_stages) + list(fragment.junctions):
        for filename in node.log_files:
            owner = declaring.get(filename)
            if owner:
                targets[filename] = evidence.service_for_module(owner)
    return targets


def _parse_overrides(specs: tuple[str, ...]) -> dict[str, str]:
    """Turn ``service:filename`` options into extra targets."""
    extra: dict[str, str] = {}
    for spec in specs:
        service, _, filename = spec.partition(":")
        if not service or not filename:
            raise click.BadParameter(f"expected service:filename, got {spec!r}")
        extra[filename] = service
    return extra


def _report_levels(ev: evidence.Evidence, top: int) -> None:
    """Print what each log file actually contains.

    Per file, not per service: PanDA configures a level per logger, so one
    threshold for a whole machine group would be a number established for
    something else.
    """
    click.echo("\nlog files:")
    rows = sorted(ev.log_filenames())
    for filename in rows[:top]:
        status = ev.file_status(filename)
        results = ev.matching(evidence.ANY_LINE_PATTERN, log_filename=filename)
        if status != "present":
            click.echo(f"  {filename:<34} {status}")
            continue
        counts = evidence.level_histogram(ev, log_filename=filename)
        threshold = evidence.effective_level(ev, log_filename=filename) or "unknown"
        detail = ", ".join(f"{lv}={counts[lv]}" for lv in evidence.LEVELS if counts[lv])
        # A truncated sample cannot say what production does *not* emit, so the
        # mark is on the line that reports the threshold rather than buried.
        truncated = "" if ev.conclusive(evidence.ANY_LINE_PATTERN, log_filename=filename) else " [partial]"
        click.echo(
            f"  {filename:<34} {threshold:<8}{truncated:<10} {sum(counts.values())} line(s) "
            f"from {len(results)} machine(s)"
        )
        if detail:
            click.echo(f"    {detail}")
    if len(rows) > top:
        click.echo(f"  … {len(rows) - top} more")


def _report_dropped(fragment: MapFragment, ev: evidence.Evidence, top: int) -> None:
    """List the observables production does not emit.

    The point of the command: an observable below the threshold has to come
    out of the strategy, because a step that fetches a line which does not
    exist is worse than having no step at all.  Stages whose file is absent
    are left out here -- ``code-paths-are-live`` reports those, and a path
    that never ran is a different finding from one that runs quietly.
    """
    dropped = []
    for stage in fragment.filter_stages:
        if not stage.log_level or not stage.log_files:
            continue
        live = [f for f in stage.log_files if ev.file_status(f) == "present"]
        if not live:
            continue
        if all(
            evidence.below_threshold(
                stage.log_level, evidence.effective_level(ev, log_filename=f)
            )
            for f in live
        ):
            where = stage.anchor.as_ref() if stage.anchor else stage.owner
            dropped.append((stage.criteria_tag or stage.funnel_label, stage.log_level, where))
    if not dropped:
        return
    click.echo(f"\nobservables production does not emit ({len(dropped)}):")
    for tag, level, where in dropped[:top]:
        click.echo(f"  {tag:<24} {level:<6} {where}")
    if len(dropped) > top:
        click.echo(f"  … {len(dropped) - top} more")


@click.command("check-map")
@click.option("--map-id", default="panda", show_default=True, help="Which Code Map to check.")
@click.option(
    "--source-root",
    default=None,
    metavar="PATH",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Check the map built from PATH instead of the installed distribution.",
)
@click.option(
    "--evidence",
    "evidence_path",
    default=DEFAULT_EVIDENCE,
    show_default=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Where production evidence is read from, and written to by --fetch.",
)
@click.option(
    "--fetch",
    is_flag=True,
    help=(
        "Query production over the async grep API and overwrite the evidence "
        "file.  Requires the caller's DN in the server's allowAsyncRequest list."
    ),
)
@click.option(
    "--log-file",
    "log_file_specs",
    multiple=True,
    metavar="SERVICE:FILENAME",
    help=(
        "Sample an extra log file the map does not name, e.g. "
        "jedi:panda-JediTaskBuffer.log.  Repeatable.  The files the map does "
        "name are queried anyway; this is for looking beyond them."
    ),
)
@click.option(
    "--timeout",
    default=evidence.POLL_TIMEOUT_SECONDS,
    show_default=True,
    help="Seconds to wait for each grep to come back.",
)
@click.option(
    "--strict",
    is_flag=True,
    help=(
        "Exit non-zero if any gate fails.  Off by default: a production gate "
        "failing means the map and the deployment have drifted, which is a "
        "signal to look rather than a reason to fail a pipeline."
    ),
)
@click.option("--top", default=10, show_default=True, help="Rows per listing.")
@click.option("-v", "--verbose", is_flag=True, help="DEBUG logging.")
def main(
    map_id: str,
    source_root: Optional[Path],
    evidence_path: Path,
    fetch: bool,
    log_file_specs: tuple[str, ...],
    timeout: float,
    strict: bool,
    top: int,
    verbose: bool,
) -> None:
    """Check a Code Map against production and report where it has drifted."""
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

    # The map is a pure function of the source, so rebuilding it here is both
    # cheap and the only way to be sure the evidence is being compared against
    # the version named in the report.
    fragment = plugin.run()

    if fetch:
        targets = _targets(fragment, getattr(plugin, "declared_log_files", {}))
        targets.update(_parse_overrides(log_file_specs))
        if not targets:
            raise click.ClickException(
                "The map names no log files, so there is nothing to ask production."
            )
        # Two kinds of question per file: what levels it carries (a histogram,
        # over a small recent window) and what its rejection and funnel lines
        # say (the lines themselves, over a wide one).  Only the stages'
        # own files are worth the second kind.
        stage_files = {
            name: service
            for name, service in targets.items()
            if any(name in stage.log_files for stage in fragment.filter_stages)
        }
        queries = evidence.sample_queries(targets) + evidence.reading_queries(stage_files)
        click.echo(
            f"querying {len(queries)} question(s) over {len(targets)} log file(s) "
            f"across {len(set(targets.values()))} service(s)"
        )
        ev = asyncio.run(evidence.collect(queries, timeout=timeout))
        ev.save(evidence_path)
        click.echo(f"evidence written to {evidence_path}")
    else:
        if not evidence_path.exists():
            raise click.ClickException(
                f"No evidence at {evidence_path}.  Run with --fetch to collect it."
            )
        ev = evidence.Evidence.load(evidence_path)

    click.echo(f"\nmap_id       : {fragment.map_id}")
    click.echo(f"derived_from : {fragment.derived_from}")
    click.echo(f"evidence     : {evidence_path} ({ev.fetched_at})")

    broken = ev.failures()
    if broken:
        # Named before the gates so that "not authorized" or a wrong filename
        # reads as what it is, rather than as production disagreeing with the
        # map.  A query that did not run is not a finding about PanDA.
        click.echo(f"\nqueries that did not run ({len(broken)}):")
        for result in broken[:top]:
            click.echo(f"  {result.query.service}/{result.machine}: {result.error}")
        if len(broken) > top:
            click.echo(f"  … {len(broken) - top} more")

    _report_levels(ev, top)
    _report_dropped(fragment, ev, top)

    confirmed, promised, unconfirmed = gates.templates_confirmed(fragment, ev)
    if promised:
        # A report and not a gate: asserting a template has *gone* needs every
        # matching line, which a busy broker log does not yield under a cap.
        # An unconfirmed template is unknown, not missing.
        click.echo(f"\ndiagnostics confirmed in production: {confirmed}/{promised}")
        for label in unconfirmed[:top]:
            click.echo(f"  unconfirmed  {label}")
        if len(unconfirmed) > top:
            click.echo(f"  … {len(unconfirmed) - top} more")

    click.echo("\ngates:")
    all_passed = True
    for result in gates.run_production(fragment, ev):
        click.echo(f"  {result.summary()}")
        if not result.passed:
            all_passed = False
        for failure in result.failures[:top]:
            click.echo(f"      - {failure}")
        if len(result.failures) > top:
            click.echo(f"      … {len(result.failures) - top} more")
        for item in result.inconclusive[:top]:
            click.echo(f"      ? {item}")
        if len(result.inconclusive) > top:
            click.echo(f"      … {len(result.inconclusive) - top} more inconclusive")
        if result.note:
            click.echo(f"      note: {result.note}")

    if not all_passed and strict:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
