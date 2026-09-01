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

The report is ordered by what the reader has to decide, not by how the checks
run.  Three distinctions do the work, and each of them was learned by getting
the output wrong first:

* **not every failure asks for a change.**  A mapped log file that no machine
  has means the map is right and this deployment does not run that code; a tag
  production emits that no stage explains is an extraction miss.  Printing both
  as ``FAIL`` made them look like one thing.
* **a bounded sample is not the log.**  Every gate here already knows whether
  what it read was whole, and reporting the verdict without that turned "we did
  not look" into "it is not there" -- the one mistake this whole layer exists to
  avoid.
* **a count needs its unit.**  Query answers, log files, filter stages, tags and
  step pairs are not comparable quantities, and printing all five as "checked"
  invited exactly the comparison that means nothing.

Usage::

    # Fetch fresh evidence and check against it
    bamboo check-map --fetch

    # Re-check against evidence already on disk
    bamboo check-map

    # Every folded row
    bamboo check-map --full

    # Check a specific release, and say where its logs are
    bamboo check-map --fetch --source-root /path/to/panda \\
        --log-file jedi:panda-jedi.log --log-file server:panda-server.log
"""

from __future__ import annotations

import asyncio
import logging
import textwrap
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import click

from bamboo.codemap import evidence, gates
from bamboo.codemap.factory import available_map_ids, get_code_map_plugin
from bamboo.codemap.models import MapFragment

logger = logging.getLogger(__name__)

DEFAULT_EVIDENCE = Path(".bamboo") / "codemap-evidence.json"

# Width the prose wraps to.  Notes are whole sentences -- they carry the reason a
# finding matters -- so they wrap rather than run off the edge.
_WIDTH = 96

_KIND_LABEL = {
    gates.CHECK_BROKEN: "broken check",
    gates.MAP_DEFECT: "map defect",
    gates.DEPLOYMENT_FACT: "deployment difference",
}


def _count(number: int, noun: str) -> str:
    """``1 map defect`` / ``2 map defects`` -- the verdict line is a sentence."""
    return f"{number} {noun}" if number == 1 else f"{number} {noun}s"


def _clip(text: str, width: int = 62) -> str:
    """One representative row, shortened to keep a summary line scannable.

    The rows it shortens lead with their reason for that purpose: cutting the
    tail of "X is on no machine, so Y never ran" still says what happened,
    where cutting the tail of "Y at <long anchor>: X is on no machine" does not.
    """
    return text if len(text) <= width else text[: width - 1] + "…"


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


def _age(fetched_at: str) -> str:
    """How old the evidence is, in words.

    Printed because the map does not change between runs and its
    trustworthiness does: a verdict read off week-old evidence is a statement
    about last week's deployment.
    """
    try:
        when = datetime.fromisoformat(fetched_at)
    except ValueError:
        return "age unknown"
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    seconds = (datetime.now(timezone.utc) - when).total_seconds()
    for size, unit in ((86400, "d"), (3600, "h"), (60, "m")):
        if seconds >= size:
            return f"{seconds / size:.0f}{unit} ago"
    return "just now"


def _size(count: int) -> str:
    """Bytes as the size a reader recognises."""
    for unit, size in (("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)):
        if count >= size:
            return f"{count // size}{unit}"
    return f"{count}B"


def _note(text: str, indent: str) -> None:
    """Print a gate's note, wrapped."""
    click.echo(
        textwrap.fill(
            text,
            width=_WIDTH,
            initial_indent=f"{indent}note: ",
            subsequent_indent=f"{indent}      ",
        )
    )


def _report_header(
    fragment: MapFragment, ev: evidence.Evidence, path: Path, plugin: object, top: int
) -> None:
    """What was checked, against what, and how old it is."""
    click.echo(f"map        {fragment.map_id} @ {fragment.derived_from}")
    parsed = getattr(plugin, "module_count", 0)
    if parsed:
        excluded = getattr(plugin, "excluded_module_count", 0)
        click.echo(
            f"source     {parsed} module(s) analysed · {excluded} test module(s) excluded "
            "(test code is not the system's behaviour)"
        )
    click.echo(f"evidence   {path} · {ev.fetched_at} ({_age(ev.fetched_at)})")
    machines = {
        service: len({r.machine for r in ev.results if r.query.service == service})
        for service in sorted(ev.services())
    }
    click.echo(
        f"           {len({r.query.key() for r in ev.results})} query(ies) · "
        f"{len(ev.results)} answer(s) · "
        + ", ".join(f"{service} {count}" for service, count in machines.items())
        + " machine(s)"
    )
    status = Counter(ev.file_status(f) for f in ev.log_filenames())
    click.echo(
        f"files      {len(ev.log_filenames())} asked · {status['present']} present · "
        f"{status['absent']} absent"
        + (f" · {status['unknown']} unknown" if status["unknown"] else "")
    )
    broken = ev.failures()
    if broken:
        # Named up here, before any verdict: "not authorized" or a wrong
        # filename is not production disagreeing with the map, and a query that
        # did not run is not a finding about PanDA.
        click.echo(f"errors     {len(broken)} query(ies) did not run")
        for result in broken[:top]:
            click.echo(f"           {result.query.service}/{result.machine}: {result.error}")
        if len(broken) > top:
            click.echo(f"           … {len(broken) - top} more")


def _report_sample(ev: evidence.Evidence) -> None:
    """How much of what was asked came back whole.

    The section that was missing, and its absence was the worst of the report's
    problems: the gates establish this to decide what they may conclude, and
    without it on screen a gate that could not conclude anything looks exactly
    like one that checked and found nothing.
    """
    present = sorted(f for f in ev.log_filenames() if ev.file_status(f) == "present")
    rows: list[str] = []
    for name, pattern, needs_lines in (
        ("levels", evidence.ANY_LINE_PATTERN, False),
        ("tags", evidence.TAG_PATTERN, True),
        ("funnel", evidence.FUNNEL_PATTERN, True),
    ):
        whole, asked = evidence.sample_state(ev, pattern, present, needs_lines=needs_lines)
        if not asked:
            continue
        word = gates.COMPLETE if whole == asked else gates.PARTIAL.upper()
        rows.append(f"{name:<7} {word:<9} {whole}/{asked} file(s) answered whole")
    if ev.records:
        # A record query has no truncation to report -- the API answers with
        # the rows or it errors -- so what bounds this sample is how many tasks
        # were asked about, and that has to be said as plainly as a bound is.
        jobs = sum(len(r.jobs) for r in ev.records)
        failed = sum(1 for r in ev.records if r.error)
        word = gates.COMPLETE if len(ev.records) >= ev.tasks_available else gates.PARTIAL.upper()
        rows.append(
            f"{'records':<7} {word:<9} {len(ev.records)}/{ev.tasks_available} task(s) "
            f"asked, {jobs} job row(s)" + (f", {failed} error(s)" if failed else "")
        )
    if not rows:
        return
    for index, row in enumerate(rows):
        click.echo(("sample     " if index == 0 else "           ") + row)
    capped, total, bounds = evidence.bounds_hit(ev)
    if capped:
        shape = ", ".join(
            f"{_size(window)} window / {matches} matches per machine"
            for window, matches in sorted(bounds)
        )
        click.echo(f"           {capped} of {total} answer(s) hit a bound ({shape})")
    click.echo("           absence in a bounded sample proves nothing -- and for a line a")
    click.echo("           branch has to fire to write, neither does a complete one.")


def _report_verdict(results: list[gates.GateResult]) -> None:
    """The one line that says whether anything needs doing."""
    failing = [r for r in results if not r.passed]
    counts = Counter(r.kind for r in failing)
    click.echo(
        "\nverdict    "
        + " · ".join(
            _count(counts.get(kind, 0), _KIND_LABEL[kind]) for kind in gates.KIND_ORDER
        )
    )
    worst = next(
        (r for kind in gates.KIND_ORDER for r in failing if r.kind == kind), None
    )
    if worst is None:
        click.echo(
            "           nothing to change: the map and the deployment agree "
            "wherever they can be compared."
        )
    else:
        click.echo(f"           {worst.finding or worst.gate}.")


def _report_findings(results: list[gates.GateResult], top: int, full: bool) -> None:
    """Every failure, worst kind first, headed by what it means.

    The gate's slug is kept as the pointer but demoted: it is an identifier, and
    ``code-paths-are-live`` does not tell a reader what went wrong.
    """
    failing = [
        r for kind in gates.KIND_ORDER for r in results if not r.passed and r.kind == kind
    ]
    if not failing:
        return
    click.echo("\nfindings")
    for result in failing:
        click.echo(f"\n  {result.finding or result.gate}  [{result.gate}]")
        shown = result.failures if full else result.failures[:top]
        for failure in shown:
            click.echo(f"    {failure}")
        if len(result.failures) > len(shown):
            click.echo(f"    … {len(result.failures) - len(shown)} more (--full)")
        if result.note:
            _note(result.note, "    ")


def _report_production(
    ev: evidence.Evidence,
    results: list[gates.GateResult],
    confirmed: list[str],
    top: int,
    full: bool,
) -> None:
    """What production said, as facts rather than as row dumps.

    The level table used to be the largest block in the report and its whole
    payload was one sentence.  It is that sentence now, plus the files that
    differ from it -- which is also what keeps a finding out of a ``… 12 more``.
    """
    by_gate = {r.gate: r for r in results}
    click.echo("\nproduction")

    present = sorted(f for f in ev.log_filenames() if ev.file_status(f) == "present")
    levels = Counter(
        evidence.effective_level(ev, log_filename=f) or "unknown" for f in present
    )
    if levels:
        click.echo(
            "  log level    "
            + ", ".join(f"{count} at {level}" for level, count in levels.most_common())
        )
        common = levels.most_common(1)[0][0]
        odd = [
            f
            for f in present
            if (evidence.effective_level(ev, log_filename=f) or "unknown") != common
        ]
        if odd and not full:
            click.echo(f"               not at {common}: " + ", ".join(odd[:top]))
        if full:
            for filename in present:
                counts = evidence.level_histogram(ev, log_filename=filename)
                detail = ", ".join(
                    f"{level}={counts[level]}" for level in evidence.LEVELS if counts[level]
                )
                level = evidence.effective_level(ev, log_filename=filename) or "?"
                click.echo(f"               {filename:<34} {level:<8} {detail}")

    observables = by_gate.get("observables-are-emitted")
    if observables:
        click.echo(
            f"  observables  {observables.checked} stage(s) · "
            f"{len(observables.failures)} dropped by level · "
            f"{len(observables.inconclusive)} not concluded"
        )
    funnel = by_gate.get("funnel-order-matches")
    if funnel:
        state = (
            "all in the map's order"
            if not funnel.failures
            else f"{len(funnel.failures)} out of the map's order"
        )
        click.echo(f"  funnel       {funnel.checked} step pair(s) · {state}")
    if confirmed:
        # The only production check on version skew there is, and one-sided.
        # The unconfirmed half is not counted here on purpose: a line nobody saw
        # may simply not have fired, so the ratio invited a false reading.
        click.echo(
            f"  agreement    {len(confirmed)} diagnostic line(s) of the map found "
            "verbatim in production"
        )
        click.echo("               one-sided: a line not seen may simply not have fired")
        if full:
            for label in sorted(confirmed):
                click.echo(f"               {label}")

    histories = evidence.observed_task_status(ev)
    if histories:
        moving = {task: rows for task, rows in histories.items() if len(rows) > 1}
        writers = Counter(
            filename for rows in histories.values() for _stamp, _status, filename in rows
        )
        click.echo(
            f"  transitions  {len(histories)} task(s) · {len(moving)} seen to change "
            f"status · {len(evidence.observed_pairs(histories))} distinct pair(s)"
        )
        # Which component's log holds this is what the map is for, so it is
        # reported as the answer rather than assumed as the question: every file
        # was asked and these are the ones that carry it.
        click.echo(
            "               logged by "
            + ", ".join(
                f"{name.removeprefix('panda-').removesuffix('.log')} {count}"
                for name, count in writers.most_common(None if full else top)
            )
        )
        pairs = evidence.observed_pairs(histories)
        for (before, after), count in pairs.most_common(None if full else 5):
            click.echo(f"               {before} -> {after}  {count}x")
        if not full and len(pairs) > 5:
            click.echo(f"               … {len(pairs) - 5} more pair(s)")
        click.echo(
            "               pairs are a report: a step the sample missed leaves its"
        )
        click.echo(
            "               neighbours adjacent, and nothing in the log marks the gap"
        )


def _report_unconcluded(results: list[gates.GateResult], full: bool) -> None:
    """What was looked at and not decided, one line per gate.

    Kept visible rather than folded away entirely: "we could not tell" is a
    result, and the seven near-identical rows it used to print were not.
    """
    rows = [(r.gate, r.inconclusive) for r in results if r.inconclusive]
    if not rows:
        return
    click.echo(f"\nnot concluded ({sum(len(items) for _, items in rows)})")
    for gate, items in rows:
        if full:
            click.echo(f"  {gate}")
            for item in items:
                click.echo(f"    {item}")
            continue
        click.echo(f"  {gate:<26} {len(items):>3}  {_clip(items[0])}")
        if len(items) > 1:
            click.echo(f"  {'':<26} {'':>3}  … {len(items) - 1} more (--full)")


def _report_gates(results: list[gates.GateResult]) -> None:
    """The audit trail: what ran, over how much, in whose units."""
    click.echo("\ngates")
    for result in results:
        mark = "  [partial]" if result.sample == gates.PARTIAL else ""
        click.echo(
            f"  {result.verdict:<8} {result.gate:<24} {result.checked:>4} "
            f"{result.unit:<14} {result.question or ''}{mark}"
        )


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
    "--tasks",
    default=evidence.DEFAULT_TASK_SAMPLE,
    show_default=True,
    help=(
        "With --fetch, how many of the tasks seen in the logs to pull job "
        "records for.  One request each, so this is the API cost."
    ),
)
@click.option(
    "--strict",
    is_flag=True,
    help=(
        "Exit non-zero if a gate fails in a way that asks for a change -- a map "
        "defect or a check that could not run.  A deployment difference does "
        "not count: the map and the source agree there, and the deployment "
        "simply does not exercise that code, which is no reason to fail a "
        "pipeline."
    ),
)
@click.option(
    "--full",
    is_flag=True,
    help=(
        "Print every folded row: each log file's level histogram, every "
        "confirmed diagnostic, and all failure and not-concluded rows."
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
    tasks: int,
    strict: bool,
    full: bool,
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
        # Three kinds of question.  What levels a file carries (a histogram,
        # over a small recent window); what its rejection and funnel lines say
        # (the lines themselves, over a wide one), which only the stages' own
        # files are worth; and which task statuses were set (a sequence, so
        # nothing may be trimmed), which every file is asked because where a
        # transition is logged is the thing being established.
        stage_files = {
            name: service
            for name, service in targets.items()
            if any(name in stage.log_files for stage in fragment.filter_stages)
        }
        queries = (
            evidence.sample_queries(targets)
            + evidence.reading_queries(stage_files)
            + evidence.transition_queries(targets)
        )
        click.echo(
            f"querying {len(queries)} question(s) over {len(targets)} log file(s) "
            f"across {len(set(targets.values()))} service(s)"
        )
        ev = asyncio.run(evidence.collect(queries, timeout=timeout))

        # A second round, and it has to be second: the tasks to ask about come
        # out of the transition lines the first round returned.  That is not a
        # convenience -- every endpoint returning a *population* of tasks scopes
        # it to one userName, so the logs are the only unscoped source of ids
        # there is, and the per-id endpoints impose no such check.
        task_ids = sorted(evidence.observed_task_status(ev))
        if task_ids:
            click.echo(
                f"fetching job records for {min(len(task_ids), tasks)} "
                f"of {len(task_ids)} task(s) seen in the logs"
            )
            ev.records, ev.tasks_available = asyncio.run(
                evidence.collect_job_records(task_ids, sample=tasks)
            )
        ev.save(evidence_path)
        click.echo(f"evidence written to {evidence_path}")
    else:
        if not evidence_path.exists():
            raise click.ClickException(
                f"No evidence at {evidence_path}.  Run with --fetch to collect it."
            )
        ev = evidence.Evidence.load(evidence_path)

    results = gates.run_production(fragment, ev)
    confirmed = gates.templates_confirmed(fragment, ev)

    click.echo()
    _report_header(fragment, ev, evidence_path, plugin, top)
    _report_sample(ev)
    _report_verdict(results)
    _report_findings(results, top, full)
    _report_production(ev, results, confirmed, top, full)
    _report_unconcluded(results, full)
    _report_gates(results)

    # Only the failures that ask for a change.  A deployment difference is
    # information the map should carry, not a broken build.
    if strict and any(result.actionable for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
