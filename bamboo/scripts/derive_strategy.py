"""``bamboo derive-strategy`` — ask the Code Map about one observed value.

The third cadence.  ``build-map`` extracts, ``check-map`` says how far the
extraction can be trusted, and this one only reads: it takes a subject and the
value a record actually holds, and returns the junctions that could have
settled it, the log to read for each, and whether anything is going to move the
row on.

Two phases, and the split is the same one ``check-map`` makes for the same
reason -- first contact turns up the errors in the model, so the plan is worth
reading before a query goes out::

    # What the map says.  No production access at all.
    bamboo derive-strategy --subject JediTaskSpec.status --observed pending --task 52266181

    # Put its questions to production, and evaluate the answers
    bamboo derive-strategy --subject JediTaskSpec.status --observed pending --task 52266181 \\
        --fetch --evidence /tmp/strategy.json

    # Evaluate again offline against what was fetched
    bamboo derive-strategy --subject JediTaskSpec.status --observed pending --task 52266181 \\
        --evidence /tmp/strategy.json

The report leads with what the reader has to decide.  For a row that is not
moving, "which junction put it here" is only half the question and often the
less useful half -- so the follow-up, *will anything pick this up*, is printed
before the candidate list rather than after it.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

import click

from bamboo.codemap import evidence as evidence_mod
from bamboo.codemap import strategy as strategy_mod
from bamboo.codemap.lookup import CodeMap
from bamboo.codemap.models import (
    ELIMINATED,
    SEEN,
    UNASKABLE,
    UNSETTLED,
    Strategy,
    Symptom,
)

logger = logging.getLogger(__name__)

DEFAULT_EVIDENCE = Path(".bamboo") / "strategy-evidence.json"

# Kept apart from ``check-map``'s file on purpose.  The two ask different
# questions of different files, and one overwriting the other would silently
# leave a gate reading a sample collected for something else.

_WIDTH = 96

_VERDICT_ORDER = (SEEN, UNSETTLED, UNASKABLE, ELIMINATED)


def _short(owner: str) -> str:
    """``pandaserver/taskbuffer/db_proxy_mods/x.py::m`` -> ``x.py::m``.

    The directory is dropped for the listing only.  It carries no information
    a reader of this report uses -- the package does not decide which service
    runs the code, which is the one thing they might reach for it for -- and
    keeping it pushes the log file, which they do use, off the line.
    """
    where, sep, method = owner.partition("::")
    return where.rsplit("/", 1)[-1] + sep + method


def _report_header(strategy: Strategy, evidence_path: Optional[Path], ev) -> None:
    symptom = strategy.symptom
    scope = f" · task {symptom.task_id}" if symptom.task_id else ""
    click.echo(f"\nsymptom    {symptom.subject} = {symptom.observed!r}{scope}")
    click.echo(f"map        {strategy.map_id} @ {strategy.derived_from}")
    if ev is None:
        click.echo("evidence   none asked for -- this is what the map says on its own")
        return
    answered = sum(1 for o in strategy.observations if o.verdict != "not_asked")
    click.echo(
        f"evidence   {evidence_path} · {ev.fetched_at} · "
        f"{answered}/{len(strategy.observations)} question(s) answered"
    )
    if strategy.observations and not answered:
        # The likeliest way to get a confidently empty report: an evidence file
        # collected for another symptom carries no answer to this one's pattern,
        # and every candidate then comes back unsettled for the wrong reason.
        click.echo("           this file holds no answer to these questions -- fetch again")


def _report_verdict(strategy: Strategy, evaluated: bool) -> None:
    """The one line that says where the question stands."""
    counts = {v: [c for c in strategy.candidates if c.verdict == v] for v in _VERDICT_ORDER}
    click.echo(f"\nverdict    {len(strategy.candidates)} junction(s) can write this value")
    if not evaluated:
        click.echo("           nothing asked of production yet -- run again with --fetch")
        return
    seen = counts[SEEN]
    if seen:
        for candidate in seen:
            click.echo(f"           {_short(candidate.owner)} wrote it -- {candidate.because}")
    remaining = len(strategy.candidates) - len(counts[ELIMINATED])
    click.echo(
        f"           {len(counts[ELIMINATED])} ruled out, {remaining} left "
        f"({len(counts[UNASKABLE])} of them with no log to ask)"
    )
    if not seen:
        click.echo(
            "           no writer was seen: an absence is not an answer unless the "
            "file carries the line"
        )


def _report_follow_up(strategy: Strategy) -> None:
    """Whether the row is going anywhere, printed before who put it there.

    The order is the finding.  A task sat in ``finishing`` while the query that
    rescues it ran every cycle, because the table that query joins had stopped
    being updated -- nothing about the writer would have led anywhere.
    """
    follow = strategy.follow_up
    if follow is None:
        return
    click.echo("\nwill anything move it on")
    selects = "yes" if follow.selected else "no"
    repair = "yes" if follow.self_repairing else "no"
    click.echo(f"  a query selects on this value   {selects}")
    click.echo(
        f"  a re-evaluating trigger reaches  {repair}"
        + (f"  ({', '.join(follow.triggers)})" if follow.triggers else "")
    )
    if follow.selection_gates:
        click.echo(f"  what bounds that query's reach   {', '.join(follow.selection_gates)}")
        click.echo("                                   nothing in the map writes these")
    for line in click.wrap_text(
        follow.question, width=_WIDTH, initial_indent="  → ", subsequent_indent="    "
    ).splitlines():
        click.echo(line)


def _report_candidates(strategy: Strategy, top: int, full: bool, evaluated: bool) -> None:
    candidates = strategy_mod.survivors(strategy) if evaluated else strategy.candidates
    stated = sum(1 for c in strategy.candidates if c.tier == 1)
    groups = {tuple(c.log_files) for c in strategy.candidates}
    click.echo(
        f"\ncandidates ({len(strategy.candidates)}) "
        f"· {stated} state the value outright, {len(strategy.candidates) - stated} settle it "
        "at run time"
    )
    click.echo(
        f"           {len(groups)} distinguishable by log file"
        + (
            f", {sum(1 for c in strategy.candidates if not c.log_files)} named by none"
            if any(not c.log_files for c in strategy.candidates)
            else ""
        )
    )
    shown = candidates if full else candidates[:top]
    for candidate in shown:
        mark = f"{candidate.verdict:<10}" if evaluated else f"tier {candidate.tier}   "
        click.echo(f"  {mark} {_short(candidate.owner)}")
        detail = ", ".join(candidate.log_files) or "no log file names it"
        click.echo(f"  {'':<10} {detail}")
        if evaluated and candidate.because:
            click.echo(f"  {'':<10} {candidate.because}")
        if full:
            click.echo(f"  {'':<10} {candidate.owner}")
            if candidate.triggers:
                click.echo(f"  {'':<10} triggers: {', '.join(candidate.triggers)}")
            for condition in candidate.conditions:
                click.echo(f"  {'':<10} when: {condition}")
    if len(candidates) > len(shown):
        click.echo(f"  … {len(candidates) - len(shown)} more (--full)")


def _report_observations(strategy: Strategy, top: int, full: bool, evaluated: bool) -> None:
    """What was asked of production, and what came back.

    Controls are listed with the probes rather than folded away: the reason a
    file's silence did or did not count is the part of this a reader has to be
    able to check.
    """
    if not strategy.observations:
        return
    probes = [o for o in strategy.observations if o.role == strategy_mod.PROBE]
    if not evaluated:
        queries = strategy_mod.queries(strategy)
        click.echo(
            f"\nquestions to put to production: {len(queries)} over {len(probes)} log file(s), "
            f"{len(strategy.observations) - len(probes)} of them controls"
        )
        click.echo(f"  probe    {probes[0].pattern}")
        click.echo(f"  control  {evidence_mod.TRANSITION_PATTERN}  (does this file carry the line)")
        return
    click.echo("\nquestions put to production")
    controls = {o.log_file: o for o in strategy.observations if o.role == strategy_mod.CONTROL}
    if controls:
        # The number that says how much of the elimination is real.  Measured
        # against the existing production sample, five of these twelve files
        # carry ``set task_status=`` at all; the silence of the other seven is
        # about the line, not about the junction.
        carrying = sum(1 for c in controls.values() if c.verdict == "seen")
        click.echo(
            f"  {carrying}/{len(controls)} file(s) carry this line at all -- "
            "no other file's silence can rule anything out"
        )
    shown = probes if full else probes[:top]
    for probe in shown:
        control = controls.get(probe.log_file)
        carries = "" if control is None else f" · carries the line: {control.verdict}"
        click.echo(f"  {probe.verdict:<13} {probe.log_file}{carries}")
        for line in probe.sample:
            click.echo(f"                {line.strip()[:_WIDTH - 16]}")
        if full:
            click.echo(f"                settles: {', '.join(_short(o) for o in probe.settles)}")
    if len(probes) > len(shown):
        click.echo(f"  … {len(probes) - len(shown)} more (--full)")


def _report_findings(strategy: Strategy, top: int, full: bool) -> None:
    for title, rows in (("findings", strategy.findings), ("gaps", strategy.gaps)):
        if not rows:
            continue
        click.echo(f"\n{title}")
        shown = rows if full else rows[:top]
        for row in shown:
            for line in click.wrap_text(
                row, width=_WIDTH, initial_indent="  ", subsequent_indent="    "
            ).splitlines():
                click.echo(line)
        if len(rows) > len(shown):
            click.echo(f"  … {len(rows) - len(shown)} more (--full)")


async def _derive(map_id: str, version: Optional[str], symptom: Symptom) -> Strategy:
    from bamboo.database.graph_database_client import GraphDatabaseClient

    graph_db = GraphDatabaseClient()
    await graph_db.connect()
    try:
        code_map = CodeMap(graph_db, map_id=map_id, version=version)
        return await strategy_mod.derive(code_map, symptom)
    finally:
        await graph_db.close()


@click.command("derive-strategy")
@click.option("--map-id", default="panda", show_default=True, help="Which Code Map to read.")
@click.option(
    "--version",
    default=None,
    help=(
        "Pin to one ``derived_from``.  Several versions coexist on purpose: an "
        "incident from months ago has to be read against the code that ran then."
    ),
)
@click.option("--subject", required=True, help="Qualified subject, e.g. JediTaskSpec.status.")
@click.option("--observed", required=True, help="The value the record actually holds.")
@click.option(
    "--task",
    "task_id",
    default=None,
    help=(
        "The entity this is about.  Without it a match still confirms a writer "
        "is live, but nothing can be ruled out -- elimination needs a pattern "
        "scoped to one row."
    ),
)
@click.option(
    "--evidence",
    "evidence_path",
    default=DEFAULT_EVIDENCE,
    show_default=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Where production answers are read from, and written to by --fetch.",
)
@click.option(
    "--fetch",
    is_flag=True,
    help=(
        "Put the derived questions to production over the async grep API and "
        "overwrite the evidence file.  Requires the caller's DN in the "
        "server's allowAsyncRequest list."
    ),
)
@click.option(
    "--timeout",
    default=evidence_mod.POLL_TIMEOUT_SECONDS,
    show_default=True,
    help="Seconds to wait for each grep to come back.",
)
@click.option("--full", is_flag=True, help="Print every folded row, with conditions and anchors.")
@click.option("--top", default=10, show_default=True, help="Rows per listing.")
@click.option("-v", "--verbose", is_flag=True, help="DEBUG logging.")
def main(
    map_id: str,
    version: Optional[str],
    subject: str,
    observed: str,
    task_id: Optional[str],
    evidence_path: Path,
    fetch: bool,
    timeout: float,
    full: bool,
    top: int,
    verbose: bool,
) -> None:
    """Ask the Code Map which code could have produced an observed value."""
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)

    symptom = Symptom(subject=subject, observed=observed, task_id=task_id)
    try:
        strategy = asyncio.run(_derive(map_id, version, symptom))
    except LookupError as exc:
        raise click.ClickException(str(exc)) from exc

    ev = None
    if fetch:
        queries = strategy_mod.queries(strategy)
        if not queries:
            raise click.ClickException(
                "The map offers no observable for this subject, so there is "
                "nothing to ask production.  See the gaps below."
            )
        click.echo(
            f"asking {len(queries)} question(s) over "
            f"{len({q.log_filename for q in queries})} log file(s) "
            f"across {len({q.service for q in queries})} service(s)"
        )
        ev = asyncio.run(evidence_mod.collect(queries, timeout=timeout))
        ev.save(evidence_path)
        click.echo(f"evidence written to {evidence_path}")
    elif evidence_path.exists():
        ev = evidence_mod.Evidence.load(evidence_path)

    if ev is not None:
        strategy = strategy_mod.evaluate(strategy, ev)

    _report_header(strategy, evidence_path if ev else None, ev)
    _report_verdict(strategy, ev is not None)
    _report_follow_up(strategy)
    _report_candidates(strategy, top, full, ev is not None)
    _report_observations(strategy, top, full, ev is not None)
    _report_findings(strategy, top, full)


if __name__ == "__main__":
    main()
