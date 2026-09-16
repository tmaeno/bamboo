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
from bamboo.codemap import reading as reading_mod
from bamboo.codemap import strategy as strategy_mod
from bamboo.codemap.lookup import CodeMap
from bamboo.codemap.models import (
    ELIMINATED,
    LEAD_MAP,
    SEEN,
    UNASKABLE,
    UNSETTLED,
    Strategy,
    Symptom,
)
from bamboo.codemap.panda.plugin import PandaCodeMapPlugin

logger = logging.getLogger(__name__)

DEFAULT_EVIDENCE = Path(".bamboo") / "strategy-evidence.json"

# Kept apart from ``check-map``'s file on purpose.  The two ask different
# questions of different files, and one overwriting the other would silently
# leave a gate reading a sample collected for something else.

_WIDTH = 96

_VERDICT_ORDER = (SEEN, UNSETTLED, UNASKABLE, ELIMINATED)


def _report_header(strategy: Strategy, evidence_path: Optional[Path], ev) -> None:
    symptom = strategy.symptom
    scope = f" · task {symptom.task_id}" if symptom.task_id else ""
    asked = (
        f"the distribution around {symptom.focus}"
        if strategy.localization is not None
        else f"{symptom.subject} = {symptom.observed!r}"
    )
    click.echo(f"\nsymptom    {asked}{scope}")
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
    seen = counts[SEEN]
    for candidate in seen:
        click.echo(f"           {strategy_mod.short_owner(candidate.owner)} wrote it -- {candidate.because}")
        for branch in candidate.named:
            # The arm, on the verdict line, because for a junction reached from
            # six of them the arm *is* the answer and the reason is in it.
            click.echo(f"           {'':<10} {_arm(branch)}{_at(branch)}")
    if not evaluated:
        click.echo("           nothing asked of production yet -- run again with --fetch")
        return
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
    if follow.selected_by:
        # The reader, not the writers: this is the query that has to pick the
        # row up, so it is the one an investigation goes and reads.
        for index, owner in enumerate(follow.selected_by):
            label = "  which query selects it        " if index == 0 else " " * 33
            click.echo(f"{label} {strategy_mod.short_owner(owner)}")
        click.echo(
            "                                   "
            + (
                ", ".join(follow.reader_log_files)
                if follow.reader_log_files
                else "the map holds no log for it -- it settles nothing, so it is not a junction"
            )
        )
    if follow.selection_gates:
        click.echo(f"  what bounds that query's reach   {', '.join(follow.selection_gates)}")
        click.echo("                                   nothing in the map writes these")
    for line in click.wrap_text(
        follow.question, width=_WIDTH, initial_indent="  → ", subsequent_indent="    "
    ).splitlines():
        click.echo(line)


def _at(branch) -> str:
    """``:1423`` -- where to go and read, when the map knows."""
    return f"  :{branch.line}" if branch.line else ""


def _arm(branch) -> str:
    """What to call this arm: its tags, or failing those the words it leaves."""
    return " ".join(branch.tags) or (branch.messages[0] if branch.messages else branch.outcome)


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
        click.echo(f"  {mark} {strategy_mod.short_owner(candidate.owner)}")
        detail = ", ".join(candidate.log_files) or "no log file names it"
        click.echo(f"  {'':<10} {detail}")
        if evaluated and candidate.because:
            click.echo(f"  {'':<10} {candidate.because}")
        if candidate.row_precondition:
            # Always shown, not folded into --full: it changes how the line
            # above is to be read, and a reader who stops at the default
            # listing is exactly the one who would otherwise over-read it.
            click.echo(
                f"  {'':<10} only lands on a row where "
                f"{' and '.join(candidate.row_precondition)}"
            )
        for branch in candidate.named:
            # Always shown: when one arm is named, which arm it is *is* the
            # answer.  Six of these reach ``exhausted`` from one function and
            # differ only in the reason they give.
            click.echo(f"  {'':<10} → {_arm(branch)}{_at(branch)}")
            for condition in branch.conditions:
                click.echo(f"  {'':<10}   when: {condition}")
        if full:
            click.echo(f"  {'':<10} {candidate.owner}")
            if candidate.triggers:
                click.echo(f"  {'':<10} triggers: {', '.join(candidate.triggers)}")
            for branch in candidate.branches:
                if branch.matched:
                    continue  # already shown above, with its reason
                click.echo(f"  {'':<10} {_arm(branch)}{_at(branch)}")
                for condition in branch.conditions:
                    click.echo(f"  {'':<10}   when: {condition}")
    if len(candidates) > len(shown):
        click.echo(f"  … {len(candidates) - len(shown)} more (--full)")


def _report_observations(strategy: Strategy, top: int, full: bool, evaluated: bool) -> None:
    """What was asked of production, and what came back.

    Controls are listed with the probes rather than folded away: the reason a
    file's silence did or did not count is the part of this a reader has to be
    able to check.

    The row-count question is listed separately because it is about a different
    thing.  The probe asks whether the code decided the value; this asks whether
    the row took it, and a write can announce the first and do neither -- which
    is the only failure the map cannot settle from the decision alone.
    """
    if not strategy.observations:
        return
    probes = [o for o in strategy.observations if o.role == strategy_mod.PROBE]
    rows = [o for o in strategy.observations if o.role == strategy_mod.ROWS]
    if not evaluated:
        queries = strategy_mod.queries(strategy)
        controls = [o for o in strategy.observations if o.role == strategy_mod.CONTROL]
        click.echo(
            f"\nquestions to put to production: {len(queries)} over {len(probes)} log file(s), "
            f"{len(controls)} of them controls"
        )
        if probes:
            click.echo(f"  probe    {probes[0].pattern}   (did the code decide it)")
            click.echo(f"  control  {controls[0].pattern if controls else '-'}"
                       "   (does this file carry the line)")
        if rows:
            click.echo(f"  rows     {rows[0].pattern}   (did the row take it)")
            click.echo(f"           in {', '.join(sorted({o.log_file for o in rows}))}")
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
            # Timestamp and tail, not the first eighty characters.  The logger
            # name, the run key and a link sit in between; the half that says
            # *why* is at the end -- "no candidates. brokerage failed for 1
            # input datasets" is the finding, and the head is boilerplate.
            body = line.strip()
            click.echo(f"                {body if full else body[:19] + ' … ' + body[-74:]}")
        if full:
            click.echo(f"                settles: {', '.join(strategy_mod.short_owner(o) for o in probe.settles)}")
    if len(probes) > len(shown):
        click.echo(f"  … {len(probes) - len(shown)} more (--full)")


def _report_leads(strategy: Strategy, top: int, full: bool) -> None:
    """Where this answer goes next, and where it stops.

    Printed as something to run, not as prose.  A lead that continues is a whole
    question, so it is shown as the command that asks it -- that the next hop
    can be taken by hand is the contract the loop runs on, and a report that
    only described it would leave nobody able to check that.
    """
    if not strategy.leads:
        return
    # Continuing leads first: one is a question that can be asked now and the
    # other is a place the walk ran out, and a reader deciding what to do next
    # wants them in that order.
    ordered = sorted(strategy.leads, key=lambda lead: (lead.symptom is None, lead.field))
    # Folded for reading only.  Several candidates open the same field and the
    # strategy keeps every one of them, because which of them survives the
    # evidence is not decided until later -- see ``_deduped``.
    ordered = [
        lead
        for index, lead in enumerate(ordered)
        if lead.field not in {earlier.field for earlier in ordered[:index]}
    ]
    click.echo("\nwhere this goes next")
    shown = ordered if full else ordered[:top]
    for lead in shown:
        # The supplier, on every line.  A lead the extraction recorded and one
        # assembled from a call are not the same claim, and a reader who cannot
        # tell them apart reads the map's coverage as better than it is.
        via = "" if lead.source == LEAD_MAP else f"  [{lead.source}]"
        if lead.symptom is not None:
            ask = f"--subject {lead.symptom.subject} --observed {lead.symptom.observed}"
            if lead.symptom.task_id:
                ask += f" --task {lead.symptom.task_id}"
            click.echo(f"  ask   {ask}{via}")
        else:
            click.echo(f"  stops {lead.field}  -- {lead.stop}{via}")
        for line in click.wrap_text(
            lead.why, width=_WIDTH, initial_indent=" " * 8, subsequent_indent=" " * 8
        ).splitlines():
            click.echo(line)
    if len(ordered) > len(shown):
        click.echo(f"  … {len(ordered) - len(shown)} more (--full)")


def _report_reading(
    strategy: Strategy, roots: Optional[dict], top: int, full: bool
) -> None:
    """The code to read, and the production line to read it against.

    One entry per function, not per arm: the surviving arms routinely sit
    together and reading the same text twice invites two answers about one
    piece of code.  Whether the pair can be made is stated either way -- an arm
    the map records no line for is shown as silent rather than left out, since
    a query invented to fill the gap would come back empty and an empty query
    is what this design reads as evidence.

    Without a source tree only the coordinates are printed.  The map names the
    text; fetching it is a separate step, for the same reason deriving and
    asking production are: the plan can be checked before anything is read.
    """
    if not strategy.readings:
        return
    click.echo("\ncode to read")
    shown = strategy.readings if full else strategy.readings[:top]
    for entry in shown:
        outcomes = ", ".join(entry.outcomes[:3]) or "-"
        click.echo(f"  {strategy_mod.short_owner(entry.owner):<52} {outcomes}")
        marks = ",".join(str(line) for line in entry.lines[:6]) or "-"
        region = (
            reading_mod.region_for(
                roots,
                file=entry.file,
                owner=entry.owner,
                line=entry.lines[0],
                expected_sha=entry.blob_sha,
            )
            if roots and entry.lines
            else None
        )
        where = f"{entry.file}"
        if region is not None:
            where += (
                f"  {region.line_start}-{region.line_end} "
                f"({region.line_end - region.line_start + 1} lines)"
            )
        click.echo(f"        {where}   arms at {marks}")
        if region is not None and region.off_version:
            # Said before anything is read from it.  The map's lines address the
            # snapshot it was built from, and a different one puts the mark on
            # plausible code that is not the code -- which reads as an answer.
            click.echo(
                "        the tree given is not the one this map was built from: "
                "the lines above are this tree's, not the map's"
            )
        if entry.silent:
            click.echo(
                "        silent -- the map records no line production prints for this arm"
            )
        else:
            click.echo(
                f"        match against  {entry.log_pattern}"
                f"  in {', '.join(entry.log_files[:2]) or 'no file'}"
            )
        if full and region is not None:
            click.echo(region.marked(entry.lines))
    if len(strategy.readings) > len(shown):
        click.echo(f"  … {len(strategy.readings) - len(shown)} more (--full)")


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


def _report_resolution(description: str, matches: list) -> None:
    """What the description was taken to mean, and what else it could have.

    Shown rather than assumed.  The alternatives are the part a reader has to
    be able to check: a description naming two things is a fact about the
    description, and silently picking one of them is the retrieval failure the
    map exists to remove, put back one layer up.
    """
    click.echo(f"\ndescribed  {description}")
    top, rest = matches[0], matches[1:3]
    click.echo(
        f"resolved   {top.term.kind:<6} {top.term.key}   "
        f"({', '.join(top.words)})  {top.score:.2f}"
    )
    for other in rest:
        click.echo(f"  also     {other.term.kind:<6} {other.term.key}  {other.score:.2f}")


def _report_localization(strategy: Strategy, top: int, full: bool) -> None:
    """Which step of the chain took the list, and how much of it.

    The cuts first and the funnel second, because they answer from two
    directions and the first needs no assumptions: a rejection line is complete
    on its own, while the counts are aggregated over passes that cannot be told
    apart.
    """
    local = strategy.localization
    if local is None:
        return
    click.echo(f"\nchain      {strategy_mod.short_owner(local.chain)}")
    click.echo(f"           {', '.join(local.log_files[:2]) or 'no log file names it'}")
    for owner in local.chains_sharing_the_file:
        click.echo(f"           with {strategy_mod.short_owner(owner)} in the same file")
    if local.entered is not None:
        click.echo(
            f"           {local.entered} candidate(s) in, {local.left} out, "
            f"over {local.passes} pass(es) · sample {local.sample}"
        )
    took = [cut for cut in local.cuts if cut.sites]
    click.echo(f"\nwhat took the list ({len(took)} of {len(local.cuts)} step(s) named a candidate)")
    shown = took if full else took[:top]
    for cut in shown:
        click.echo(
            f"  {len(cut.sites):5d}  {cut.tag or cut.funnel_label:<22} "
            f"{cut.funnel_label:<24} #{cut.order}{_at(cut)}"
        )
        for reason in cut.reasons[: None if full else 1]:
            click.echo(f"  {'':<7} said: {reason}")
        for condition in cut.conditions[: None if full else 1]:
            click.echo(f"  {'':<7} when: {condition[:78]}")
        if full:
            click.echo(f"  {'':<7} sites: {', '.join(cut.sites[:12])}")
    if len(took) > len(shown):
        click.echo(f"  … {len(took) - len(shown)} more (--full)")
    if not took:
        click.echo("  nothing was seen -- the questions came back empty, which settles nothing")

    if not local.funnel:
        return
    click.echo("\nthe funnel, in the order the map runs it")
    click.echo("           most … fewest left after each step, pooled over passes")
    click.echo("           no difference is taken: the log marks no boundary between passes, so")
    click.echo("           two steps' maxima can come from two different ones")
    steps = local.funnel if full else local.funnel[:top]
    for step in steps:
        where = f"#{step.order}" if step.order is not None else "  ?"
        spread = f"{step.most:5d} … {step.fewest:<5d}"
        click.echo(f"  {where:>5}  {spread}  {step.label}")
    if len(local.funnel) > len(steps):
        click.echo(f"  … {len(local.funnel) - len(steps)} more (--full)")


async def _derive(
    map_id: str,
    version: Optional[str],
    symptom: Optional[Symptom],
    description: Optional[str],
    task_id: Optional[str],
    observed_diag: Optional[str],
) -> tuple[Strategy, list]:
    from bamboo.database.graph_database_client import GraphDatabaseClient

    graph_db = GraphDatabaseClient()
    await graph_db.connect()
    try:
        code_map = CodeMap(graph_db, map_id=map_id, version=version)
        matches: list = []
        if symptom is None:
            # Resolution needs the map, so it happens here rather than in the
            # command: the vocabulary is the map's, and building a second copy
            # of it outside would be a second thing to keep current.
            matches = strategy_mod.resolve(description or "", await code_map.vocabulary())
            if not matches:
                raise LookupError(
                    "nothing in this map's vocabulary matches that description.  "
                    "Name a subject and a value instead, or say it in the map's own "
                    "words -- a status, a rejection reason, a brokerage step."
                )
            symptom = matches[0].term.symptom.model_copy(
                update={
                    "task_id": task_id or strategy_mod.entity_in(description or ""),
                    "observed_diag": observed_diag,
                }
            )
        return await strategy_mod.derive(code_map, symptom), matches
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
@click.option(
    "--describe",
    "description",
    default=None,
    help=(
        "What is wrong, in words.  Resolved against the map's own vocabulary -- "
        "every question it can be asked, 420 of them for PanDA -- so the "
        "derivation is chosen by what the words meant and not by a flag per "
        "symptom.  What it resolved to is printed, with the alternatives."
    ),
)
@click.option("--subject", default=None, help="Qualified subject, e.g. JediTaskSpec.status.")
@click.option("--observed", default=None, help="The value the record actually holds.")
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
    "--observed-diag",
    "observed_diag",
    default=None,
    help=(
        "The message the record carries, when it is already to hand.  PanDA "
        "writes the reason a branch took into the same text it logs, so this "
        "names the arm and not only the junction.  With --fetch it is read from "
        "the task row instead."
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
@click.option(
    "--source-root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help=(
        "Read the code the map points at from this tree.  Without it only the "
        "coordinates are printed; the map names the text either way."
    ),
)
@click.option("--full", is_flag=True, help="Print every folded row, with conditions and anchors.")
@click.option("--top", default=10, show_default=True, help="Rows per listing.")
@click.option("-v", "--verbose", is_flag=True, help="DEBUG logging.")
def main(
    map_id: str,
    version: Optional[str],
    description: Optional[str],
    subject: Optional[str],
    observed: Optional[str],
    task_id: Optional[str],
    observed_diag: Optional[str],
    evidence_path: Path,
    fetch: bool,
    timeout: float,
    source_root: Optional[Path],
    full: bool,
    top: int,
    verbose: bool,
) -> None:
    """Ask the Code Map what it has to say about a symptom."""
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)

    if bool(description) == bool(subject or observed):
        raise click.UsageError(
            "Give either --describe or a --subject and --observed pair.  The first "
            "resolves against the map's vocabulary; the second names the entry itself."
        )
    symptom = None
    if subject or observed:
        if not (subject and observed):
            raise click.UsageError("--subject and --observed are only meaningful together.")
        symptom = Symptom(
            subject=subject, observed=observed, task_id=task_id, observed_diag=observed_diag
        )
    try:
        strategy, matches = asyncio.run(
            _derive(map_id, version, symptom, description, task_id, observed_diag)
        )
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
        if strategy.symptom.task_id is not None and strategy.localization is None:
            # One row, one call, no window.  Asked alongside the greps rather
            # than instead of them: the record holds the last message written
            # to the field, so it confirms an arm and never rules one out, and
            # the log line is what survives a later junction overwriting it.
            ev.tasks = asyncio.run(
                evidence_mod.collect_task_records([strategy.symptom.task_id])
            )
        ev.save(evidence_path)
        click.echo(f"evidence written to {evidence_path}")
    elif evidence_path.exists():
        ev = evidence_mod.Evidence.load(evidence_path)

    if ev is not None:
        strategy = strategy_mod.evaluate(strategy, ev)

    # A record that names an arm settles the question without a grep, so the
    # listing is ordered by verdict even when nothing was asked of production.
    named = any(c.named for c in strategy.candidates)
    _report_header(strategy, evidence_path if ev else None, ev)
    if matches:
        _report_resolution(description or "", matches)
    if strategy.localization is not None:
        _report_localization(strategy, top, full)
    else:
        _report_verdict(strategy, ev is not None)
        _report_follow_up(strategy)
        _report_candidates(strategy, top, full, ev is not None or named)
    _report_observations(strategy, top, full, ev is not None)
    _report_leads(strategy, top, full)
    _report_reading(
        strategy,
        PandaCodeMapPlugin._resolve_roots(source_root) if source_root else None,
        top,
        full,
    )
    _report_findings(strategy, top, full)


if __name__ == "__main__":
    main()
