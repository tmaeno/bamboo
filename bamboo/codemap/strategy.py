"""Turn an observed value into a plan for finding out why it is that value.

The *use* half of the three cadences.  ``build-map`` extracts, ``check-map``
says how far the extraction can be trusted, and this reads what they left --
nothing here parses source or rebuilds anything.

The question it answers is the one the map was shaped around: *which code
settled this, and whose log says so?*  Asking a source navigator instead means
turning "the task is stuck in pending" into grep terms and ranking thirty
candidates, which fails as no-candidates, too-many-candidates or irrelevant --
all retrieval failures.  Here the symptom names a subject and a value, the
lookup is exact, and the answer does not depend on having phrased the question
well.

**Pruning is by observation, not by condition.**  The plan this implements
expected path conditions to do the narrowing -- substitute the observed values
and see which branch could have fired.  Measured against the real map that is
mostly idle: of the eighteen junctions that can write ``pending``, fourteen
carry no condition at all on the branch that does it, and of the twenty-six
distinct conditions the other four carry, three are settled by an observed
value.  So the conditions are reported, and the narrowing comes from asking
production which log actually holds the line.

**Which is why an absence needs a licence.**  Three of the four ways a grep
comes back empty are not answers, and the fourth needs one more thing: that the
file carries this kind of line at all.  ``panda-DBProxy.log`` never contains
``set task_status=`` however often a proxy junction fires -- the knight that
called it writes that line -- so a naive eliminator would rule out every proxy
candidate at once and keep the wrong one with confidence.  Two guards deal with
it, and they are independent:

* the file asked is where a line *about this junction* can appear -- the
  caller's, or the junction's own only where its module declares a logger;
* every probe is paired with a control asking whether that file carries the
  line shape at all, and an absence eliminates only where the control says yes.

The second is the more important, because it is sound without the map being
right about the first.

**The line shape comes from the map.**  It did not for a long time: junctions
carried no emits at all, so the pattern was a constant covering the one subject
the transition gate needed, and every other subject came back as a capability
gap.  ``Branch.emits`` now holds the sentence each writer leaves, and the probe
is built from the head the most writers of the observed value share -- writers
agreeing on a head are writers one question reaches.

Two things still bound what can be asked, and both are reported as gaps rather
than papered over.  A writer that logs nothing about the value cannot be
observed at all.  And the prefix that scopes a query to a single row names a
task, so a subject whose rows are jobs cannot be scoped by it -- asking anyway
would produce a pattern that never matches, and a silence is what the
eliminator reads as evidence.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from bamboo.codemap import evidence
from bamboo.codemap.evidence import GrepQuery
from bamboo.codemap.lookup import CodeMap
from bamboo.codemap.models import (
    ANSWER_ABSENT,
    ANSWER_INCONCLUSIVE,
    ANSWER_NO_FILE,
    ANSWER_NOT_ASKED,
    ANSWER_SEEN,
    ELIMINATED,
    REPORTS_DECISION,
    REPORTS_ROWS_CHANGED,
    SEEN,
    SELF_REPAIRING_TRIGGERS,
    UNASKABLE,
    UNSETTLED,
    Candidate,
    CandidateBranch,
    FollowUp,
    JunctionNode,
    Observation,
    Strategy,
    Symptom,
)

logger = logging.getLogger(__name__)

#: The two roles a query plays.  A probe asks whether *this* row was moved here;
#: a control asks whether the file carries that kind of line at all.  Without
#: the second, a silent file is indistinguishable from one that never speaks
#: this sentence, and the map's largest group of junctions is reached through
#: files of exactly that kind.
PROBE = "probe"
CONTROL = "control"

#: A third question, about the same junction and a different thing.  The probe
#: asks whether the code decided the value; this asks whether the row took it.
#: They are separate because a write can announce a decision and change nothing
#: -- the count comes back, most callers discard it, and the flush's own log is
#: the one place it is written down.  Kept out of the eliminator: its silence
#: says nothing about which writer fired.
ROWS = "rows"

#: Log line shapes, per subject.  Not read from the map -- see the module
#: docstring.  ``{task}`` and ``{value}`` are filled in; a subject absent from
#: here can be enumerated but not observed.
_LINE_SHAPE: dict[str, str] = {
    evidence.TRANSITION_SUBJECT: r"set task_status={value}",
}

#: Subjects the task prefix can scope.  A job is tagged ``PandaID``, a dataset
#: by its own id, so building a ``jediTaskID=`` pattern for one of those asks a
#: question that cannot match -- which is worse than saying nothing, because a
#: silence is what the eliminator reads.  Reported as a gap instead.
_TASK_SCOPED = ("JediTaskSpec.", "JEDI_Tasks.")

# ``<jediTaskID=52266181 datasetID=685030095>`` is what the log wrapper puts in
# front of the message, so the id and the transition are on one line and one
# pattern can require both.  The trailing class matters: without it
# ``jediTaskID=5226618`` matches ``jediTaskID=52266181`` and the answer is about
# a different task.
_TASK_PREFIX = r"jediTaskID={task}[ >].*"

# Bounds for the probe.  Wide window, because a task that went ``pending``
# hours ago is the case being asked about, and a cap that will not be reached --
# a pattern naming one task id is selective enough that the answer comes back
# whole, which is the only thing that licenses reading its emptiness.
PROBE_TAIL_BYTES = evidence.DEFAULT_TAIL_BYTES
PROBE_MAX_MATCHES = evidence.DEFAULT_MAX_MATCHES
PROBE_KEEP_LINES = 50

# Bounds for the control.  Narrow, because the control is only ever read
# positively -- "this file does carry the line" -- and that direction is sound
# under truncation.  Keeping no lines follows: the answer is a count.
CONTROL_TAIL_BYTES = evidence.TRANSITION_TAIL_BYTES
CONTROL_MAX_MATCHES = evidence.TRANSITION_MAX_MATCHES

# How many matched lines a finding shows.  Enough to read a timestamp and a
# component off the answer; the rest is in the evidence file.
_SAMPLE_LINES = 3


def line_shape(subject: str, producers: list[JunctionNode]) -> Optional[str]:
    """The log line a writer of *subject* leaves, or None when none is known.

    Read from the map first.  The constant below covers one subject and was
    what this had while junctions carried no emits at all; keeping it as the
    fallback costs nothing and means a map built before the emits existed still
    answers for the subject it could always answer for.

    The literal frame is turned into a pattern by anchoring on the text either
    side of the hole the value goes in.  Only lines that *carry the value* can
    do that, which is why the emits say which they are: a line reporting how
    many rows changed is evidence about the same junction and answers a
    different question, so building a value pattern out of it would ask
    production for something that never appears in it.
    """
    heads: dict[str, set[str]] = {}
    for junction in producers:
        for branch in junction.branches:
            if branch.tags:
                # An arm that names its own decision speaks its own sentence,
                # and it is asked for separately in the file that sentence lands
                # in.  Pooled here it would win the shared head whenever the
                # other writers log nothing, and then be put to every
                # candidate's file as though they all said it.
                continue
            for emit in branch.emits:
                if emit.reports != REPORTS_DECISION:
                    continue
                head = emit.template.partition("{}")[0]
                if head.strip():
                    heads.setdefault(head, set()).add(junction.owner)
    if heads:
        # The head is the part a pattern anchors on, so writers agreeing on a
        # head are writers a single question reaches.  Most-shared first, and
        # longest to break a tie: taking the shortest instead picked ``set to``
        # out of one junction over ``set task_status=`` out of four, which is
        # both less selective and about a different sentence.
        best = max(heads, key=lambda head: (len(heads[head]), len(head)))
        return re.escape(best) + "{value}"
    return _LINE_SHAPE.get(subject)


def rows_shape(producers: list[JunctionNode]) -> tuple[Optional[str], list[str]]:
    """The line saying how many rows a write changed, and where it lands.

    Chosen the same way the decision shape is -- the head the most writers
    share -- and returned with its own files, which are not the files the
    decision line lands in.  That is the point of it: the knight announces the
    value in its own log and the proxy reports the row count in the proxy's,
    and only the second can distinguish a write that landed from one that lost
    a compare-and-set.
    """
    heads: dict[str, set[str]] = {}
    files: dict[str, set[str]] = {}
    for junction in producers:
        for branch in junction.branches:
            for emit in branch.emits:
                if emit.reports != REPORTS_ROWS_CHANGED:
                    continue
                head = emit.template.partition("{}")[0]
                if not head.strip():
                    continue
                heads.setdefault(head, set()).add(junction.owner)
                files.setdefault(head, set()).update(emit.log_files)
    if not heads:
        return None, []
    best = max(heads, key=lambda head: (len(heads[head]), len(head)))
    return re.escape(best), sorted(files[best])


def _pattern(
    subject: str, value: str, task_id: Optional[str], producers: list[JunctionNode]
) -> Optional[str]:
    """The regular expression to put to production, or None.

    Values are escaped even though every status in the corpus is alphanumeric:
    the symptom comes from a record, and a pattern assembled from data is one
    place a stray metacharacter turns a precise question into a vague one.
    """
    shape = line_shape(subject, producers)
    if shape is None:
        return None
    pattern = shape.format(value=re.escape(value))
    if task_id is not None:
        if not subject.startswith(_TASK_SCOPED):
            return None
        pattern = _TASK_PREFIX.format(task=re.escape(str(task_id))) + pattern
    return pattern


def control_shape(subject: str, producers: list[JunctionNode]) -> Optional[str]:
    """The probe's sentence with nothing filled in -- does this file say it at all."""
    shape = line_shape(subject, producers)
    return None if shape is None else shape.replace("{value}", "")


def _reaching(junction: JunctionNode, observed: str) -> list:
    """The branches that can have produced *observed*.

    The ones that state it, or -- when none does -- the ones whose value is
    only settled at run time.  A tier-2 branch is never ruled out by the value,
    because "this one could have" is the honest answer for it.
    """
    stated = [b for b in junction.branches if b.outcome == observed]
    return stated or [b for b in junction.branches if b.tier == 2]


def _candidate(junction: JunctionNode, observed: str) -> Candidate:
    stated = any(b.outcome == observed for b in junction.branches)
    return Candidate(
        owner=junction.owner,
        tier=1 if stated else 2,
        log_files=junction.observable_log_files(),
        branches=[
            CandidateBranch(
                outcome=branch.outcome,
                tier=branch.tier,
                line=branch.line,
                tags=list(branch.tags),
                messages=list(branch.messages),
                conditions=list(branch.path_condition),
                row_precondition=list(branch.row_precondition),
            )
            for branch in _reaching(junction, observed)
        ],
        triggers=sorted({entry.trigger for entry in junction.entry_points}),
        entries=sorted({entry.entry for entry in junction.entry_points}),
    )


def _names_tags(text: str, tags: list[str]) -> bool:
    """Whether *text* carries every one of *tags* as a whole token.

    Matched here rather than re-derived from the text, so this layer needs to
    know nothing about how a tag is spelled: the map extracted the tokens and
    this asks whether the message contains them.  Whole tokens because
    ``reason=low`` must not answer for ``reason=low_efficiency``.
    """
    return bool(tags) and all(
        re.search(rf"(?<![\w=]){re.escape(tag)}\b", text) for tag in tags
    )


def _speaks(template: str, text: str) -> bool:
    """Whether *text* contains something *template* could have rendered.

    Contained rather than equal: ``errorDialog`` is appended to, and a frame
    that has to account for the whole field would match none of the records
    that carry two messages.  The holes are the only wildcards, and a frame
    with no literal text never reaches here -- the extraction drops those,
    because a pattern of nothing but wildcards names every arm at once.
    """
    if not template.strip():
        return False
    parts = [re.escape(part) for part in re.split(r"\{[^{}]*\}", template) if part.strip()]
    return bool(parts) and re.search(".*".join(parts), text, re.DOTALL) is not None


def name_the_arm(strategy: Strategy, diag: str) -> Strategy:
    """Mark the arms the record's message names, and settle what that proves.

    One direction only, and the asymmetry is not the usual one about sample
    size.  A match is proof: the message and the branch were written in the
    same block.  A record that names no arm proves nothing at all, because the
    field holds the *last* message written to it and any later junction may
    have overwritten it -- so a junction is never ruled out by this, only
    confirmed.  Within the junction it does rule out: six arms write
    ``exhausted``, and which one is the whole question.

    **Tags first, then frames, and the frames have to be unique.**  A tag is
    exact -- the tokens are the contract, and two arms cannot share them -- so
    every arm carrying them is named.  A frame is the author's wording: it
    drifts between releases and two arms can share one, so it names an arm only
    where it is the only frame in the whole set that matches.  That makes the
    failure "says nothing" rather than "says the wrong thing", and it is needed
    because production mostly writes the untagged half: of thirty tasks found
    in ``exhausted`` with a message on the record, none carried a tag.
    """
    settled = strategy.model_copy(deep=True)
    tagged = False
    for candidate in settled.candidates:
        for branch in candidate.branches:
            # Only ever set: a tagged probe may already have matched this arm
            # from the log, and the two are the same positive evidence reaching
            # it by different routes.
            if _names_tags(diag, branch.tags):
                branch.matched = True
                tagged = True
    if not tagged:
        spoken = [
            (candidate, branch)
            for candidate in settled.candidates
            for branch in candidate.branches
            if any(_speaks(message, diag) for message in branch.messages)
        ]
        if len(spoken) == 1:
            spoken[0][1].matched = True
    for candidate in settled.candidates:
        if candidate.named:
            candidate.verdict = SEEN
            candidate.because = "the record's own message names this branch"
    return settled


def _findings(candidates: list[Candidate], junctions: list[JunctionNode]) -> list[str]:
    """What asking this question turned up about the map itself.

    A candidate no log names is not a gap in production's record; it is a place
    the map cannot point an investigation at, and the two reasons for it are
    different work.  One of these was how ``makeTaskPending_JEDI`` came to
    light: a junction that writes ``pending`` outright whose only call site is
    commented out, which is dead code rather than a missing log.
    """
    reach = {j.owner: j for j in junctions}
    findings = []
    for candidate in sorted(candidates, key=lambda c: c.owner):
        if candidate.log_files:
            continue
        junction = reach[candidate.owner]
        why = (
            "nothing the map recognises reaches it and its module declares no logger"
            if not junction.entry_points and not junction.caller_log_files
            else "its module declares no logger and no caller was resolved"
        )
        findings.append(f"{candidate.owner}: {why} -- it can be neither confirmed nor ruled out")
    return findings


def short_owner(owner: str) -> str:
    """``pandaserver/taskbuffer/db_proxy_mods/x.py::m`` -> ``x.py::m``.

    The directory is dropped for display only.  It carries no information a
    reader uses -- the package does not decide which service runs the code,
    which is the one thing they might reach for it for -- and keeping it pushes
    the log file, which they do use, off the line.
    """
    where, sep, method = owner.partition("::")
    return where.rsplit("/", 1)[-1] + sep + method


def _readers_phrase(selected_by: list[str], reader_files: list[str]) -> str:
    """Name the query that has to pick the row up, and where it says so."""
    named = ", ".join(short_owner(owner) for owner in selected_by[:2])
    if len(selected_by) > 2:
        named += f" and {len(selected_by) - 2} more"
    where = f" ({', '.join(reader_files)})" if reader_files else ""
    return f"{named}{where}"


def _follow_up(
    observed: str,
    selected_values: list[str],
    selected_by: list[str],
    selection_gates: list[str],
    writers: list[JunctionNode],
    carried_from: list[str],
) -> FollowUp:
    """Whether anything will move the value on, and what to ask if not.

    Three states, and telling them apart is what the observed value is for.
    Getting this wrong once is why ``selection_gates`` exists at all: a task
    sat in ``finishing`` and the map said nothing selected that value, which was
    an artefact of reading the query's ``WHERE`` and not its ``FROM`` list.  A
    query did select it, every cycle, and the row was outside what that query
    could see because the table it joins had stopped being updated.

    The reader's own triggers are used where the map names one that is also a
    junction.  Pooling them over the subject's *writers* is what this did while
    the reader was unknown, and it answers a different question -- how work
    reaches anything that touches the subject, rather than how it reaches the
    query that has to pick this row up.  Kept as the fallback, because an empty
    trigger set reads as "nothing reaches it", which is a stronger claim than
    "the map cannot say".
    """
    selected = observed in selected_values
    by_owner = {j.owner: j for j in writers}
    readers = [by_owner[o] for o in selected_by if o in by_owner]
    reader_files = sorted({f for r in readers for f in r.observable_log_files()})
    # The reader's own triggers where it has any.  Most readers are proxy
    # methods the trigger slice reaches through a knight rather than directly,
    # so their entry points are empty -- and reading that as the answer says
    # "nothing reaches this subject", which is a stronger claim than the map
    # can make and, for ``pending``, the opposite of true.
    triggers = sorted({entry.trigger for j in readers for entry in j.entry_points}) or sorted(
        {entry.trigger for j in writers for entry in j.entry_points}
    )
    repairing = bool(set(triggers) & SELF_REPAIRING_TRIGGERS)
    asks = (
        f"{observed!r} is selected by {_readers_phrase(selected_by, reader_files)}"
        if selected_by
        else f"a query selects on {observed!r}"
    )
    if selected and repairing:
        bounded = (
            "bounded by " + ", ".join(selection_gates)
            if selection_gates
            else "bounded by nothing this map can name"
        )
        question = (
            f"{asks} and a re-evaluating trigger reaches this "
            f"subject, so ask why it did not pick the row up -- its reach is {bounded}, "
            "and nothing in the map writes those tables"
        )
    elif selected:
        question = (
            f"{asks}, but only {', '.join(triggers) or 'nothing the map recognises'} "
            "reaches it, and none of those re-evaluate -- so ask whether the "
            "command or message arrived, not which condition blocked it"
        )
    else:
        where = ", ".join(carried_from) if carried_from else "the writers listed above"
        question = (
            f"no query in the map selects on {observed!r}, so waiting will not move the "
            f"row -- ask who wrote the step before it: {where}"
        )
    return FollowUp(
        selected=selected,
        selected_by=list(selected_by),
        reader_log_files=reader_files,
        selection_gates=list(selection_gates),
        triggers=triggers,
        self_repairing=repairing,
        carried_from=carried_from,
        question=question,
    )


def _tag_pattern(template: str, tags: list[str]) -> Optional[str]:
    """The tokens of *tags* in the order *template* writes them.

    The tag rather than the sentence around it, for the reason the brokerage
    slice made the tag the identity of a filter stage: the wording is the
    author's and the tag is the contract.  ``action=set_exhausted since
    reason=many_shorter_jobs`` puts a word between the two, so the order is read
    off the template instead of assumed.
    """
    placed = [(template.find(tag), tag) for tag in tags]
    if any(at < 0 for at, _tag in placed):
        return None
    return ".*".join(re.escape(tag) for _at, tag in sorted(placed))


def _tag_observations(
    producers: list[JunctionNode], task_id: Optional[str], scoped: bool
) -> list[Observation]:
    """A probe per branch that names its own decision, in the file it writes to.

    A second family, because the sentence differs.  ``line_shape`` picks the one
    head the most writers share, which is right for a value every writer
    announces the same way and useless for a branch whose line is about the
    reason -- six arms write ``exhausted`` and the shared head belongs to the
    other writers entirely.

    Its file comes from the emit rather than from the junction: the arm logs in
    the proxy's own file, while the line the shared head matches is written by
    the caller.  That is the whole point of an emit carrying its own files.
    """
    observations: list[Observation] = []
    seen: set[tuple[str, str]] = set()
    for junction in producers:
        for branch in junction.branches:
            if not branch.tags:
                continue
            for emit in branch.emits:
                if emit.reports != REPORTS_DECISION:
                    continue
                body = _tag_pattern(emit.template, branch.tags)
                if body is None:
                    continue
                pattern = (
                    _TASK_PREFIX.format(task=re.escape(str(task_id))) + body
                    if task_id is not None and scoped
                    else body
                )
                for filename in emit.log_files:
                    if (filename, pattern) in seen:
                        continue
                    seen.add((filename, pattern))
                    observations.append(
                        Observation(
                            log_file=filename,
                            pattern=pattern,
                            role=PROBE,
                            services=list(evidence.SERVICES),
                            settles=[junction.owner],
                        )
                    )
                    if pattern != body:
                        observations.append(
                            Observation(
                                log_file=filename,
                                pattern=body,
                                role=CONTROL,
                                services=list(evidence.SERVICES),
                                settles=[],
                                control_for=pattern,
                            )
                        )
    return observations


def _observations(
    candidates: list[Candidate],
    pattern: str,
    control: str,
    with_control: bool,
    rows: Optional[str] = None,
    rows_files: Optional[list[str]] = None,
) -> list[Observation]:
    """One probe per log file, and its control where a control is meaningful.

    Grouped by file rather than by candidate: candidates share files -- the two
    watchdog junctions reach three of them between them -- and one query per
    candidate would ask the same question of the same file several times.

    The control is the probe's own sentence with the entity and the value
    taken out -- it has to be, or it answers about a line the probe was never
    about.  While the shape was a constant this was the constant too, which was
    right for the one subject it covered and silently wrong for any other.

    The control is skipped when the symptom names no entity, because then the
    probe *is* the control and the two would be one query asked twice.  That
    also states the consequence honestly: with nothing to scope the pattern to,
    a match confirms and a silence settles nothing.
    """
    settles: dict[str, list[str]] = {}
    for candidate in candidates:
        for filename in candidate.log_files:
            settles.setdefault(filename, []).append(candidate.owner)
    observations = []
    for filename, owners in sorted(settles.items()):
        observations.append(
            Observation(
                log_file=filename,
                pattern=pattern,
                role=PROBE,
                services=list(evidence.SERVICES),
                settles=sorted(owners),
            )
        )
        if with_control:
            observations.append(
                Observation(
                    log_file=filename,
                    pattern=control,
                    role=CONTROL,
                    services=list(evidence.SERVICES),
                    settles=[],
                    control_for=pattern,
                )
            )
    for filename in rows_files or []:
        if not rows:
            break
        observations.append(
            Observation(
                log_file=filename,
                pattern=rows,
                role=ROWS,
                services=list(evidence.SERVICES),
                # Settles nothing on its own: it says whether the row moved,
                # not which of the candidates moved it.
                settles=[],
            )
        )
    return observations


async def derive(code_map: CodeMap, symptom: Symptom) -> Strategy:
    """Read the map and return what it has to say about *symptom*.

    Offline in the sense that matters: it reads the stored map and touches
    production not at all.  Everything that needs the deployment is expressed as
    an :class:`Observation` to be run later, so the plan can be inspected --
    and its model corrected -- before a single query goes out.
    """
    subject = await code_map.subject(symptom.subject)
    if subject is None:
        known = ", ".join(s.name for s in await code_map.subjects())
        raise LookupError(f"{symptom.subject} is not a subject of this map.  Known: {known}")

    writers = await code_map.writers_of(symptom.subject)
    producers = await code_map.producers_of(symptom.subject, symptom.observed)
    carried = sorted(await code_map.carried_from(symptom.subject))

    candidates = [_candidate(j, symptom.observed) for j in producers]
    pattern = _pattern(symptom.subject, symptom.observed, symptom.task_id, producers)
    rows_head, rows_files = rows_shape(producers)
    # Scoped to the row, like the probe: an unscoped row count answers about
    # every write the proxy made, which is no answer about this one.
    rows_pattern = (
        _TASK_PREFIX.format(task=re.escape(str(symptom.task_id))) + rows_head
        if rows_head and symptom.task_id is not None
        and symptom.subject.startswith(_TASK_SCOPED)
        else None
    )
    if rows_pattern is None:
        rows_files = []

    gaps: list[str] = []
    if pattern is None:
        gaps.append(
            f"no question can be put to production about {symptom.subject}: "
            + (
                "the map records no diagnostic line on any of its writers"
                if line_shape(symptom.subject, producers) is None
                else "the entity given is a task and this subject's rows are not, "
                "so the log prefix that scopes a query to one row does not apply"
            )
        )
    if symptom.task_id is None:
        gaps.append(
            "no entity was named, so a match confirms a writer is live but a silence "
            "rules nothing out -- elimination needs a pattern scoped to one row"
        )

    asked = (
        _observations(
            candidates,
            pattern,
            control_shape(symptom.subject, producers) or "",
            with_control=symptom.task_id is not None,
            rows=rows_pattern,
            rows_files=rows_files,
        )
        if pattern
        else []
    )
    asked += _tag_observations(
        producers, symptom.task_id, symptom.subject.startswith(_TASK_SCOPED)
    )

    strategy = Strategy(
        symptom=symptom,
        map_id=code_map.map_id,
        derived_from=subject.derived_from,
        candidates=candidates,
        observations=asked,
        follow_up=_follow_up(
            symptom.observed,
            subject.selected_values,
            subject.selected_by.get(symptom.observed, []),
            subject.selection_gates,
            writers,
            carried,
        ),
        findings=_findings(candidates, producers),
        gaps=gaps,
    )
    # Applied here when the caller already has the message, so the plan a reader
    # inspects before any query goes out is the narrowed one.  When it arrives
    # with the evidence instead, ``evaluate`` applies the same function.
    return name_the_arm(strategy, symptom.observed_diag) if symptom.observed_diag else strategy


def queries(strategy: Strategy) -> list[GrepQuery]:
    """The grep queries *strategy* wants put to production.

    Both machine groups for every file.  The package a junction lives in does
    not decide which group runs it -- JEDI opens its own TaskBuffer, so
    ``pandaserver`` code called by a knight runs in the JEDI process and logs
    there -- and that is the case most junctions are in.  A service without the
    file answers "No such file or directory", which is one extra query and no
    guess at all.
    """
    return [
        GrepQuery(
            pattern=observation.pattern,
            log_filename=observation.log_file,
            service=service,
            tail_bytes=PROBE_TAIL_BYTES if observation.role == PROBE else CONTROL_TAIL_BYTES,
            max_matches=PROBE_MAX_MATCHES if observation.role == PROBE else CONTROL_MAX_MATCHES,
            keep_lines=PROBE_KEEP_LINES if observation.role == PROBE else 0,
        )
        for observation in strategy.observations
        for service in observation.services
    ]


def _answer(ev: evidence.Evidence, observation: Observation) -> Observation:
    """Fill in what production said about one observation.

    A machine that does not have the file is dropped rather than counted as an
    inconclusive answer.  Both services are asked precisely because the map
    cannot say which runs the code, so one of them reporting the file missing is
    the expected shape of a right answer, not a degraded one.  Only when *every*
    machine says so does the absence become a statement -- and then it is the
    strongest one available, since the file is created on the logger's first
    emit.
    """
    results = ev.matching(observation.pattern, log_filename=observation.log_file)
    settled = observation.model_copy(deep=True)
    if not results:
        settled.verdict = ANSWER_NOT_ASKED
        return settled
    answering = [r for r in results if not evidence.missing_file(r)]
    if not answering:
        settled.verdict = ANSWER_NO_FILE
        return settled
    settled.matched = sum(r.matched for r in answering)
    if settled.matched:
        settled.verdict = ANSWER_SEEN
        settled.sample = [line for r in answering for line in r.lines][:_SAMPLE_LINES]
    elif all(r.conclusive for r in answering):
        settled.verdict = ANSWER_ABSENT
    else:
        settled.verdict = ANSWER_INCONCLUSIVE
    return settled


def _recorded_message(ev: evidence.Evidence, symptom: Symptom) -> Optional[str]:
    """The message the record carries for this symptom's entity, if any."""
    if symptom.task_id is None:
        return None
    for record in ev.tasks:
        if str(record.task_id) == str(symptom.task_id):
            message = record.fields.get("errordialog")
            return str(message) if message else None
    return None


def _settle(
    candidate: Candidate,
    probes: list[Observation],
    controls: dict[tuple[str, str], Observation],
) -> Candidate:
    """Decide what production said about one candidate.

    Three rules, all of them the asymmetry this layer is built on:

    * a line seen proves the writer wrote it, whatever the sample size;
    * a line not seen rules the writer out only where the answer was whole
      *and* the file is shown to carry that line at all;
    * a candidate no log names is not ruled out by anything -- being unable to
      look is not evidence.
    """
    settled = candidate.model_copy(deep=True)
    mine = [probe for probe in probes if candidate.owner in probe.settles]
    if not candidate.log_files and not mine:
        settled.verdict = UNASKABLE
        settled.because = "no log file names it"
        return settled

    asked = [
        (probe.log_file, probe, controls.get((probe.log_file, probe.pattern)))
        for probe in mine
    ]
    # Every seen probe marks what it names before any of them decides the
    # verdict: a tagged probe names one arm, and returning on the first would
    # leave the arm a later probe confirmed unmarked.
    for _filename, probe, _control in asked:
        if probe is None or probe.verdict != ANSWER_SEEN:
            continue
        for branch in settled.branches:
            if branch.tags and _names_tags(probe.pattern, branch.tags):
                branch.matched = True
    for filename, probe, _control in asked:
        if probe is not None and probe.verdict == ANSWER_SEEN:
            settled.verdict = SEEN
            settled.because = f"logged in {filename}"
            if candidate.row_precondition:
                # Confirmed as the decider, not as the writer of the row: this
                # one's statement tests a column it writes, so the line above
                # is what the code said, and a competing write could have made
                # it change nothing.
                settled.because += ", though its write is conditional on the row"
            return settled

    reasons = []
    for filename, probe, control in asked:
        if probe is None:
            reasons.append(f"{filename} was not asked")
        elif probe.verdict == ANSWER_NO_FILE:
            continue  # the logger has never emitted anywhere: nothing ran here
        elif probe.verdict != ANSWER_ABSENT:
            reasons.append(f"{filename} answered {probe.verdict}")
        elif control is not None and control.verdict != ANSWER_SEEN:
            # The guard that matters.  This file's silence is not about the
            # candidate: it does not carry this kind of line at all.
            reasons.append(f"{filename} carries no such line, so its silence says nothing")
        elif control is None:
            reasons.append(f"{filename} has no control, so its silence says nothing")

    if reasons:
        settled.verdict = UNSETTLED
        settled.because = "; ".join(reasons[:2])
    else:
        settled.verdict = ELIMINATED
        settled.because = "absent from every log that would carry the line"
    return settled


def evaluate(strategy: Strategy, ev: evidence.Evidence) -> Strategy:
    """Return *strategy* with production's answers filled in.

    A new object rather than a mutation, so the derivation stays comparable
    against a later run: the map is the same and the deployment is not, which
    is the whole reason build and check are separate cadences.
    """
    settled = strategy.model_copy(deep=True)
    settled.observations = [_answer(ev, o) for o in strategy.observations]
    probes = [o for o in settled.observations if o.role == PROBE]
    # Keyed by file *and* probe, because one file now carries two sentences --
    # the value line every writer shares and the tagged line one arm names
    # itself with -- and a control answers for exactly one of them.
    controls = {
        (o.log_file, o.control_for): o
        for o in settled.observations
        if o.role == CONTROL and o.control_for
    }
    settled.candidates = [_settle(c, probes, controls) for c in settled.candidates]
    diag = _recorded_message(ev, strategy.symptom)
    return name_the_arm(settled, diag) if diag else settled


def survivors(strategy: Strategy) -> list[Candidate]:
    """Candidates the evidence has not ruled out, most settled first."""
    order = {SEEN: 0, UNSETTLED: 1, UNASKABLE: 2, ELIMINATED: 3}
    return sorted(
        (c for c in strategy.candidates if c.verdict != ELIMINATED),
        key=lambda c: (order.get(c.verdict, 9), c.owner),
    )
