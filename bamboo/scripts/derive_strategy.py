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
from typing import Callable, Optional

import click

from bamboo.codemap import evidence as evidence_mod
from bamboo.codemap import models as models_mod
from bamboo.codemap import reading as reading_mod
from bamboo.codemap import strategy as strategy_mod
from bamboo.codemap.lookup import CodeMap
from bamboo.codemap.models import (
    ACTS,
    ARRIVES_BY_DISPATCH,
    ARRIVES_THROUGH_DOOR,
    ELIMINATED,
    LEAD_MAP,
    SEEN,
    UNASKABLE,
    UNSETTLED,
    WALK_ASKED,
    WALK_BUDGET,
    WALK_TERMINAL,
    Hop,
    Investigation,
    Lead,
    Strategy,
    Symptom,
)
from bamboo.codemap.panda.plugin import PandaCodeMapPlugin

logger = logging.getLogger(__name__)

DEFAULT_EVIDENCE = Path(".bamboo") / "strategy-evidence.json"

#: Hops a walk takes before it stops of its own accord.  Small because the
#: measured walks are: the passthrough chains in this corpus are at most four
#: fields long, so anything deeper is a loop or a mistake -- and because each
#: hop reaches production once, the count is what stands in for the person who
#: would otherwise say "that's enough".
DEFAULT_HOPS = 3

# Kept apart from ``check-map``'s file on purpose.  The two ask different
# questions of different files, and one overwriting the other would silently
# leave a gate reading a sample collected for something else.

_WIDTH = 96

#: Openers listed under one folded destination before the rest are counted.
#: Enough to show that a descent was reached by more than one helper -- which
#: is what says whether a task is reassigning or waiting on its jobs -- without
#: turning four destinations into thirty lines.
_REASONS_SHOWN = 3

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


#: How many creating functions to name before counting the rest.  A job row is
#: made in twelve places, which is itself the answer to "where does one come
#: from" -- so the count is printed rather than the list quietly cut.
_TOP_CREATORS = 4


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
    click.echo(f"  anything acts on this value     {selects}")
    click.echo(
        f"  a re-evaluating trigger reaches  {repair}"
        + (f"  ({', '.join(follow.triggers)})" if follow.triggers else "")
    )
    # A query where there is one: it is the statement that could have missed the
    # row, which is the question.  An update acts on the row by the same value
    # but is the picking up rather than a chance to have failed at it, so it is
    # named as itself and only when nothing queries.
    actors = follow.selected_by or follow.updated_by
    heading = (
        "  which query selects it        "
        if follow.selected_by
        else "  which update acts on it       "
    )
    if actors:
        for index, owner in enumerate(actors):
            label = heading if index == 0 else " " * 33
            click.echo(f"{label} {strategy_mod.short_owner(owner)}")
        click.echo(
            "                                   "
            + (
                ", ".join(follow.reader_log_files)
                if follow.reader_log_files
                else "the map holds no log for it -- it settles nothing, so it is not a junction"
            )
        )
        if follow.reader_acts != ACTS:
            # Beside the name rather than only in the sentence below, because
            # the heading above says "which query selects it" and a reader
            # takes that as the thing that will pick the row up.  This is what
            # the line above is worth.
            click.echo(f"                                   {follow.reader_acts}")
    if follow.selection_gates:
        click.echo(f"  what bounds that query's reach   {', '.join(follow.selection_gates)}")
        click.echo("                                   nothing in the map writes these")
    # Where a row of this kind starts existing.  A branch table answers why a
    # row holds the value it holds and cannot answer why there is a row at all,
    # which is the question behind "did the command arrive".
    if follow.created_by:
        shown = follow.created_by[:_TOP_CREATORS]
        for index, owner in enumerate(shown):
            label = "  where a row of this kind is made" if index == 0 else " " * 33
            click.echo(f"{label} {strategy_mod.short_owner(owner)}")
        if len(follow.created_by) > len(shown):
            # Said as a count rather than truncated silently: twelve functions
            # create a job row, and a list cut to three reads like all of them.
            click.echo(
                " " * 33 + f" … {len(follow.created_by) - len(shown)} more place(s) make them"
            )
    elif follow.selected:
        click.echo("  where a row of this kind is made  nothing in this map makes them")
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
    traced = {entry.owner for entry in strategy.readings if entry.trace}
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
        if candidate.tier != 1:
            # Why a candidate whose value nothing states is on the list at all.
            # Shown here for the reason ``row_precondition`` just below is: it
            # changes how the line above is to be read, and the reader who
            # stops at this listing is the one who would otherwise take four
            # rows as four equal answers.  The expression was already in the
            # report -- a hundred and eighty lines further down, in ``code to
            # read``, where it is found by whoever had already gone there.
            #
            # Most of what it says is that the value was copied from another
            # field, which is the fact the reader needs and the one thing the
            # candidate set cannot act on: ruling a copy out would need the
            # source field's value set to be closed, and two of this corpus's
            # hundred and eight subjects have one.  So it is handed over as
            # evidence instead of being spent as a claim.
            sources = sorted({b.outcome for b in candidate.branches if b.tier != 1})
            if sources:
                click.echo(f"  {'':<10} value from {', '.join(sources)}")
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
        for handover in candidate.handovers:
            if handover.reached_by != ARRIVES_BY_DISPATCH:
                continue
            if candidate.owner in traced:
                # The fields are the *input* to the walk, not an answer for a
                # reader.  Once the walk has crossed, its ``handover`` step
                # says the same thing about the one name that mattered, with
                # the expression actually supplied -- and the 4,620 fields
                # printed across the vocabulary were drawn on 232 times, so
                # 95% of this block never told anyone anything.  Measured at
                # 491 lines and 103,127 characters, 41% of them repeated
                # verbatim inside a single report.
                continue
            # Kept when nothing walked: there is then no step to carry it, and
            # this is the only line that says who filled ``self.taskList``.
            # No call runs between the two, so a reader without a source tree
            # cannot recover it from the arm's own module either.
            where = f"{handover.entry.rsplit('/', 1)[-1]}::{handover.via}"
            click.echo(
                f"  {'':<10} built by {where} with "
                + ", ".join(f"{name}={expr}" for name, expr in sorted(handover.fields.items()))
            )
        if full:
            click.echo(f"  {'':<10} {candidate.owner}")
            if candidate.triggers:
                click.echo(f"  {'':<10} triggers: {', '.join(candidate.triggers)}")
            for handover in candidate.handovers:
                if handover.reached_by == ARRIVES_BY_DISPATCH:
                    continue
                where = f"{handover.entry.rsplit('/', 1)[-1]}::{handover.via}"
                if handover.reached_by == ARRIVES_THROUGH_DOOR:
                    # The facade rewrites the arguments, so this entry's are
                    # not read.  "with nothing" would be a claim; the door is
                    # the whole of what this line can say.
                    click.echo(f"  {'':<10} reached through {where}")
                    continue
                click.echo(
                    f"  {'':<10} called from {where} with "
                    + ", ".join(sorted(handover.fields))
                )
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
    # Folded by destination for reading only.  The strategy keeps every lead,
    # because which of their openers survives the evidence is not decided until
    # later -- see ``_deduped``.
    #
    # **The reasons are not folded with them.**  Since a descent is named by the
    # entity, several helpers reach one kind of row and which helper did it is
    # the whole of what distinguishes them: ``runImpl`` asks ``reassignShare``
    # in one arm and ``getPandaIDsWithTask_JEDI`` in another, and keeping the
    # first says the task is reassigning when it is waiting on its jobs.
    grouped: dict[str, list[Lead]] = {}
    for lead in ordered:
        grouped.setdefault(lead.field, []).append(lead)
    click.echo("\nwhere this goes next")
    fields = list(grouped) if full else list(grouped)[:top]
    for field in fields:
        leads = grouped[field]
        lead = leads[0]
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
            click.echo(f"  stops {field}  -- {lead.stop}{via}")
        reasons = list(dict.fromkeys(one.why for one in leads if one.why))
        said = reasons if full else reasons[:_REASONS_SHOWN]
        for reason in said:
            for line in click.wrap_text(
                reason, width=_WIDTH, initial_indent=" " * 8, subsequent_indent=" " * 8
            ).splitlines():
                click.echo(line)
        if len(reasons) > len(said):
            click.echo(f"        … {len(reasons) - len(said)} more opener(s) (--full)")
    if len(grouped) > len(fields):
        click.echo(f"  … {len(grouped) - len(fields)} more (--full)")


def _report_walk(entry, top: int, full: bool) -> None:
    """Why the arms of one reading ran with the value they did.

    Printed under the reading rather than as its own section: the walk starts
    at those arms, and a list of steps away from the function they belong to
    is a list of facts about nothing in particular.

    **Guards and what the guard cannot say are kept on separate lines.**  Only
    ``ast.If`` contributes to a path condition, and 86% of the map's arms have
    a ``try``, a handler, a loop or an early exit on the way to them, so a
    step with nothing under ``when`` is one the walk found no test for -- not
    one that is unconditional.  Folding the two together is how a reader comes
    to read silence as certainty.
    """
    if entry.trace_note:
        click.echo(f"        {entry.trace_note}")
    if not entry.trace:
        return
    click.echo(
        "        why it ran -- necessary conditions only; 'also' is what the "
        "guard cannot say"
    )
    shown = entry.trace if full else entry.trace[:top]
    for step in shown:
        if step.kind == models_mod.TRACE_UNBOUND:
            head = step.name
        elif step.name:
            head = f"{step.name} = {step.value}"
        else:
            head = step.value
        click.echo(f"          {step.line or '':>5}  {step.kind:<11} {head}")
        if step.guards:
            click.echo(f"                 when   {' · '.join(step.guards)}")
        elif step.kind != models_mod.TRACE_UNBOUND:
            click.echo("                 when   no test above it")
        for unsaid in step.unseen if full else step.unseen[:2]:
            click.echo(f"                 also   {unsaid}")
        if step.terminal:
            detail = f" -- {step.detail}" if step.detail else ""
            click.echo(f"                 stops  {step.terminal}{detail}")
        elif step.kind == models_mod.TRACE_HANDOVER and step.detail:
            # Computed all along and shown only for terminal steps, so the one
            # line that says who handed the value over was being dropped while
            # the candidate block above repeated the whole argument list.
            click.echo(f"                 from   {step.detail}")
    if len(entry.trace) > len(shown):
        click.echo(f"          … {len(entry.trace) - len(shown)} more step(s) (--full)")


def _report_skeleton(entry, top: int, full: bool) -> None:
    """What this function prints, in source order, with the arms in place.

    Printed beside the map's ``match against`` rather than in place of it, and
    labelled so the two cannot be confused: the map's sentence is what the
    questions already asked were built from, and swapping the two would make
    every answer already collected read as unasked.

    Laid out as source rather than as a list because the nesting is the part a
    list cannot carry.  Two rows under one ``if`` were printed together or not
    at all; two rows either side of an ``else`` cannot both have been.  A
    reader with a grepped region in the other hand reads that off the
    indentation, and no per-row verdict is printed for them -- which line
    proves what is theirs to conclude, and this side of the derivation has not
    seen the log.
    """
    if not entry.skeleton:
        return
    printed = [row for row in entry.skeleton if row.kind != models_mod.SKELETON_BRANCH]
    click.echo(
        f"        what it prints, in source order ({len(printed)} line(s) and arm(s), "
        "not what was asked)"
    )
    shown = entry.skeleton if full else _skeleton_head(entry.skeleton, top)
    for row in shown:
        pad = "  " * row.depth
        if row.kind == models_mod.SKELETON_BRANCH:
            click.echo(f"                 {pad}{row.text}")
            continue
        if row.kind == models_mod.SKELETON_ARM:
            click.echo(f"          {row.line:>5} ARM {pad}{row.text}")
            continue
        body = row.pattern or f"-- {row.refused} --"
        click.echo(f"          {row.line:>5}  |  {pad}{body}")
        if row.value:
            click.echo(f"                 {pad}with the value: {row.value}")
            click.echo(f"                 {pad}  {row.hole} -- {row.because}")
    if len(shown) < len(entry.skeleton):
        click.echo(f"          … {len(entry.skeleton) - len(shown)} more row(s) (--full)")


def _skeleton_head(rows, top: int) -> list:
    """The first *top* printed rows, with the branch headers that lead to them.

    Cutting the list at *top* would drop the headers an early row sits under
    and leave the rows that survive hanging at an indentation nothing
    explains.  Counting only the rows that carry a line keeps the cut where a
    reader expects it while the structure above each one stays.
    """
    kept: list = []
    seen = 0
    for row in rows:
        if row.kind != models_mod.SKELETON_BRANCH:
            seen += 1
            if seen > top:
                break
        kept.append(row)
    while kept and kept[-1].kind == models_mod.SKELETON_BRANCH:
        kept.pop()
    return kept


def _report_discrimination(strategy: Strategy) -> None:
    """How far those lines narrow the answer, beside what the map asks now.

    The gate the parent plan put on replacing the map's shared sentence with
    these, and it is a gate rather than a switch: what is printed here is the
    count, and the questions that go out are still the map's.

    The two are not counted against a common denominator, because they do not
    have one.  The map settles **one** sentence per subject, by majority over
    every writer of it, and the skeleton produces one row per line the
    function prints, counted once per arm that row could have been printed
    alongside -- so the honest statement is how many of the tree's lines would
    settle an arm, beside how many sentences the map had to pool them into.
    """
    walked = strategy_mod.skeleton_sentences(strategy.readings)
    if not walked:
        return
    counted = strategy_mod.discrimination(walked)
    asked = {entry.log_pattern for entry in strategy.readings if entry.log_pattern}
    click.echo(f"\nhow far a line narrows it ({len(walked)} line(s) from the tree)")
    for kind in strategy_mod.DISCRIMINATION:
        click.echo(f"  {counted[kind]:>5}  {kind}")
    click.echo(
        f"  beside {len(asked)} shared sentence(s) the map settled by majority over "
        f"{len(strategy.readings)} function(s), which is what was actually asked"
    )


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
        for fanout in entry.dispatch:
            # A candidate set and the line that settles it, printed together.
            # The plan's three legs for a run-time dispatch -- candidates at
            # build, the run-time attribute at use, the log for proof -- only
            # hold when the third travels with the first.
            click.echo(
                f"        runs one of {len(fanout.candidates)}: "
                f"{', '.join(fanout.candidates)}"
                + (f"  (default {fanout.default})" if fanout.default else "")
            )
            click.echo(
                f"          which one ran is in the log: ask for "
                f"{fanout.announced_by!r} in "
                f"{', '.join(entry.log_files[:2]) or 'no file the map names'}"
                if fanout.announced_by
                else "          nothing prints which one ran -- the set is all the map can say"
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
        _report_skeleton(entry, top, full)
        _report_walk(entry, top, full)
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
    why = "  the name itself" if top.exact else ""
    click.echo(
        f"resolved   {top.term.kind:<6} {top.term.key}   "
        f"({', '.join(top.words)})  {top.score:.2f}{why}"
    )
    for other in rest:
        click.echo(f"  also     {other.term.kind:<6} {other.term.key}  {other.score:.2f}")
    # Said out loud rather than left to the sort.  Where the top two score the
    # same, which one is printed above comes from how the keys happen to be
    # spelled, and a reader who is not told that reads a decision where there
    # was none.  Not said when the top is exact: there the order came from the
    # description and it is an answer.
    if not top.exact and top.tied_with:
        shown = {other.term.key for other in rest}
        with_them = ", ".join(key for key in top.tied_with if key in shown)
        if with_them:
            click.echo(
                f"  tied     {top.term.key} and {with_them} score the same -- "
                "the order between them is the sort's, not an answer"
            )


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
    if local.emitted_by:
        # Before any evidence, and that is the point: production writes the tag
        # per rejected site, so the description already names code.  Without
        # this the report stopped at the chain and left the stage to be found
        # among a hundred and nine, as deep as forty-seventh in the listing.
        click.echo(f"\nwhat writes {local.describes} ({len(local.emitted_by)} stage(s), from the map)")
        for cut in local.emitted_by:
            click.echo(
                f"  {strategy_mod.short_owner(cut.owner)}  #{cut.order}{_at(cut)}"
                f"  {cut.funnel_label}"
            )
            click.echo(f"  {'':<7} in: {', '.join(cut.log_files) or 'no log file names it'}")
            for condition in cut.conditions[: None if full else 2]:
                click.echo(f"  {'':<7} when: {condition[:78]}")

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


async def walk(
    code_map: CodeMap,
    symptom: Symptom,
    budget: int = DEFAULT_HOPS,
    settle: Optional[Callable[[Strategy], Strategy]] = None,
) -> Investigation:
    """Follow the map from *symptom* until it stops, one derivation per hop.

    **The loop is here and not in the derivation, and that is the design.**  A
    hop needs evidence before the next one is worth taking, and collecting
    evidence is a separate cadence -- a ``derive`` that walked would have to
    reach production from inside itself, which is the split the two phases
    exist to keep.  So the map enumerates, *settle* selects, and this only
    decides whether to go round again.  That seam is also where a reasoner can
    sit without being able to invent anything: it may pick among the leads the
    map opened, and it may not add one.

    *settle* is what applies the evidence -- fetching it, loading it, or
    neither.  Handed in rather than built here so this is testable against a
    fixture and so an offline walk and a fetching one are the same code path.
    """
    investigation = Investigation(budget=budget)
    asked = symptom
    opened, source = "", LEAD_MAP
    while True:
        strategy = await strategy_mod.derive(code_map, asked)
        if settle is not None:
            strategy = settle(strategy)
        investigation.visited.append(strategy_mod.visit_key(asked))
        investigation.hops.append(
            Hop(
                number=len(investigation.hops),
                symptom=asked,
                opened=opened,
                source=source,
                strategy=strategy,
            )
        )
        lead, repeats = strategy_mod.next_question(strategy, set(investigation.visited))
        # Recorded even when a hop is available, because a walk can both find
        # somewhere to go and have come back past somewhere it has been.
        investigation.cycles += [r for r in repeats if r not in investigation.cycles]
        if lead is None or lead.symptom is None:
            investigation.stopped = WALK_ASKED if repeats else WALK_TERMINAL
            return investigation
        if len(investigation.hops) >= budget:
            # Checked after choosing, so the reason is "the budget ran out"
            # rather than "there was nothing left" -- those are opposite things
            # to do next, and a walk that conflated them would tell a reader to
            # stop looking when it had simply been told to stop.
            investigation.stopped = WALK_BUDGET
            return investigation
        asked, opened, source = lead.symptom, lead.field, lead.source


def _report_trace(investigation: Investigation) -> None:
    """The path, the terminal it reached, and what opened each step.

    Printed before the hops themselves because it is the product.  An
    investigation is a path rather than a lookup, so "where it went and why it
    stopped" is the answer and each hop's detail is the working.
    """
    click.echo(
        f"\ntrace      {len(investigation.hops)} hop(s) of {investigation.budget} · "
        f"stopped because {investigation.stopped}"
    )
    for hop in investigation.hops:
        symptom = hop.symptom
        asked = (
            f"the distribution around {symptom.focus}"
            if symptom.kind == "distribution"
            else f"{symptom.subject} = {symptom.observed!r}"
        )
        opened = f"   ← {hop.opened} [{hop.source}]" if hop.opened else ""
        click.echo(f"  hop {hop.number}  {asked}{opened}")
    for repeat in investigation.cycles:
        click.echo(
            f"  loop     {repeat} came round again -- the fields copy from each "
            f"other, so the way out is a terminal and not another hop"
        )


async def _investigate(
    map_id: str,
    version: Optional[str],
    symptom: Optional[Symptom],
    description: Optional[str],
    task_id: Optional[str],
    observed_diag: Optional[str],
    budget: int,
    settle: Optional[Callable[[Strategy], Strategy]],
) -> tuple[Investigation, list]:
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
        # The description only ever chooses the first question.  Every hop after
        # it is chosen by what the map opened and the evidence left standing,
        # which is why resolution happens once and outside the walk.
        return await walk(code_map, symptom, budget=budget, settle=settle), matches
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
        "Read the code the map points at from this tree.  Defaults to the "
        "installed distribution, which is what the stored map is usually "
        "built from; a tree that is not the map's is refused rather than "
        "traced, since a trace is shaped like an answer."
    ),
)
@click.option(
    "--max-hops",
    "max_hops",
    default=1,
    show_default=True,
    help=(
        "How far to follow the map.  A symptom is a path rather than a lookup: "
        "the value a task sits in came from a field that came from another, and "
        "each step is chosen by what the evidence left standing, not by the "
        "description.  One by default because every hop puts its own questions "
        "to production."
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
    max_hops: int,
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
    # One evidence file for the whole walk.  A later hop asks about the same
    # task in the same logs, so its answers belong beside the first hop's rather
    # than in a file of their own -- and keeping one file is what lets the whole
    # investigation be replayed offline afterwards.
    loaded = (
        evidence_mod.Evidence.load(evidence_path)
        if not fetch and evidence_path.exists()
        else None
    )
    fetched: list = []

    def settle(strategy: Strategy) -> Strategy:
        if not fetch:
            return strategy_mod.evaluate(strategy, loaded) if loaded else strategy
        asked = strategy_mod.queries(strategy)
        if not asked:
            click.echo(
                "           the map offers no observable here, so production is "
                "not asked -- see the gaps below"
            )
            return strategy
        click.echo(
            f"asking {len(asked)} question(s) over "
            f"{len({q.log_filename for q in asked})} log file(s) "
            f"across {len({q.service for q in asked})} service(s)"
        )
        answers = asyncio.run(evidence_mod.collect(asked, timeout=timeout))
        if strategy.symptom.task_id is not None and strategy.localization is None:
            # One row, one call, no window.  Asked alongside the greps rather
            # than instead of them: the record holds the last message written
            # to the field, so it confirms an arm and never rules one out, and
            # the log line is what survives a later junction overwriting it.
            answers.tasks = asyncio.run(
                evidence_mod.collect_task_records([strategy.symptom.task_id])
            )
        fetched.append(answers)
        return strategy_mod.evaluate(strategy, answers)

    try:
        investigation, matches = asyncio.run(
            _investigate(
                map_id,
                version,
                symptom,
                description,
                task_id,
                observed_diag,
                max_hops,
                settle,
            )
        )
    except LookupError as exc:
        raise click.ClickException(str(exc)) from exc

    ev = loaded
    if fetched:
        ev = fetched[0].model_copy(
            update={
                "results": [r for answer in fetched for r in answer.results],
                "tasks": [t for answer in fetched for t in answer.tasks],
            }
        )
        ev.save(evidence_path)
        click.echo(f"evidence written to {evidence_path}")

    # The installed distribution by default: the walk computes from the source
    # rather than merely pointing at it, so having no tree turns the reading
    # back into coordinates, and that is a worse default than the release the
    # map was most likely built from.
    roots = PandaCodeMapPlugin._resolve_roots(source_root)
    for hop in investigation.hops:
        strategy_mod.attach_traces(hop.strategy, roots)

    if len(investigation.hops) > 1 or max_hops > 1:
        _report_trace(investigation)
    for hop in investigation.hops:
        strategy = hop.strategy
        if len(investigation.hops) > 1:
            click.echo(f"\n{'─' * 24} hop {hop.number} {'─' * 24}")
        # A record that names an arm settles the question without a grep, so the
        # listing is ordered by verdict even when nothing was asked of production.
        named = any(c.named for c in strategy.candidates)
        evaluated = ev is not None
        _report_header(strategy, evidence_path if evaluated else None, ev)
        if matches and hop.number == 0:
            _report_resolution(description or "", matches)
        if strategy.localization is not None:
            _report_localization(strategy, top, full)
        else:
            _report_verdict(strategy, evaluated)
            _report_follow_up(strategy)
            _report_candidates(strategy, top, full, evaluated or named)
        _report_observations(strategy, top, full, evaluated)
        _report_leads(strategy, top, full)
        _report_reading(strategy, roots, top, full)
        _report_discrimination(strategy)
        _report_findings(strategy, top, full)


if __name__ == "__main__":
    main()
