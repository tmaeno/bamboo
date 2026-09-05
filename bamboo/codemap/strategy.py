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

**The line shape does not come from the map.**  ``Branch.emits`` is empty for
every junction -- the assembled-string index that survives is about text landing
in a *field*, not in a log -- so the pattern used here is a constant in
:mod:`bamboo.codemap.evidence`, put there for the transition gate.  That covers
one subject.  For any other, this can enumerate and explain but not observe,
and it says so as a capability gap rather than returning a weaker answer: the
gap names the next thing to record.
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
    SEEN,
    SELF_REPAIRING_TRIGGERS,
    UNASKABLE,
    UNSETTLED,
    Candidate,
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

#: Log line shapes, per subject.  Not read from the map -- see the module
#: docstring.  ``{task}`` and ``{value}`` are filled in; a subject absent from
#: here can be enumerated but not observed.
_LINE_SHAPE: dict[str, str] = {
    evidence.TRANSITION_SUBJECT: r"set task_status={value}",
}

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


def line_shape(subject: str) -> Optional[str]:
    """The log line a writer of *subject* leaves, or None when none is known."""
    return _LINE_SHAPE.get(subject)


def _pattern(subject: str, value: str, task_id: Optional[str]) -> Optional[str]:
    """The regular expression to put to production, or None.

    Values are escaped even though every status in the corpus is alphanumeric:
    the symptom comes from a record, and a pattern assembled from data is one
    place a stray metacharacter turns a precise question into a vague one.
    """
    shape = line_shape(subject)
    if shape is None:
        return None
    pattern = shape.format(value=re.escape(value))
    if task_id is not None:
        pattern = _TASK_PREFIX.format(task=re.escape(str(task_id))) + pattern
    return pattern


def probe_files(junction: JunctionNode) -> list[str]:
    """Files where a line about *junction* firing can appear.

    The caller's, plus the junction's own where its module declares a logger.
    An inherited file is deliberately left out: it is a true statement about
    where the code's own output lands -- the SQL comment trace really is in
    ``panda-DBProxy.log`` -- and a false one about where a line saying the
    junction fired appears, because the code writing that line is the caller.
    Production settles it: of thirty-three files asked, ``set task_status=`` is
    in exactly five, all of them callers, and in neither proxy file.
    """
    files = set(junction.caller_log_files)
    if junction.owns_logger:
        files.update(junction.log_files)
    return sorted(files)


def _conditions(junction: JunctionNode, observed: str) -> list[str]:
    """Path conditions of the branches that can reach *observed*, deduplicated.

    Reported rather than evaluated.  Substituting observed values into them is
    what the plan called for, and it stays available -- but it is not what
    narrows the set here, because most of these branches have no condition to
    substitute into.
    """
    stated = [b for b in junction.branches if b.outcome == observed]
    reaching = stated or [b for b in junction.branches if b.tier == 2]
    seen: list[str] = []
    for branch in reaching:
        for condition in branch.path_condition:
            if condition not in seen:
                seen.append(condition)
    return seen


def _candidate(junction: JunctionNode, observed: str) -> Candidate:
    stated = any(b.outcome == observed for b in junction.branches)
    return Candidate(
        owner=junction.owner,
        tier=1 if stated else 2,
        log_files=probe_files(junction),
        conditions=_conditions(junction, observed),
        triggers=sorted({entry.trigger for entry in junction.entry_points}),
        entries=sorted({entry.entry for entry in junction.entry_points}),
    )


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


def _follow_up(
    observed: str,
    selected_values: list[str],
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
    """
    selected = observed in selected_values
    triggers = sorted({entry.trigger for j in writers for entry in j.entry_points})
    repairing = bool(set(triggers) & SELF_REPAIRING_TRIGGERS)
    if selected and repairing:
        bounded = (
            "bounded by " + ", ".join(selection_gates)
            if selection_gates
            else "bounded by nothing this map can name"
        )
        question = (
            f"a query selects on {observed!r} and a re-evaluating trigger reaches this "
            f"subject, so ask why it did not pick the row up -- its reach is {bounded}, "
            "and nothing in the map writes those tables"
        )
    elif selected:
        question = (
            f"a query selects on {observed!r}, but only {', '.join(triggers) or 'nothing the map recognises'} "
            "reaches this subject, and none of those re-evaluate -- so ask whether the "
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
        selection_gates=list(selection_gates),
        triggers=triggers,
        self_repairing=repairing,
        carried_from=carried_from,
        question=question,
    )


def _observations(candidates: list[Candidate], pattern: str, with_control: bool) -> list[Observation]:
    """One probe per log file, and its control where a control is meaningful.

    Grouped by file rather than by candidate: candidates share files -- the two
    watchdog junctions reach three of them between them -- and one query per
    candidate would ask the same question of the same file several times.

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
                    pattern=evidence.TRANSITION_PATTERN,
                    role=CONTROL,
                    services=list(evidence.SERVICES),
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
    pattern = _pattern(symptom.subject, symptom.observed, symptom.task_id)

    gaps: list[str] = []
    if pattern is None:
        gaps.append(
            f"no log line shape is known for {symptom.subject}, so its writers can be "
            "enumerated but not observed -- the map records no diagnostic line on any "
            "junction, and the one shape that exists is a constant in codemap.evidence"
        )
    if symptom.task_id is None:
        gaps.append(
            "no entity was named, so a match confirms a writer is live but a silence "
            "rules nothing out -- elimination needs a pattern scoped to one row"
        )

    return Strategy(
        symptom=symptom,
        map_id=code_map.map_id,
        derived_from=subject.derived_from,
        candidates=candidates,
        observations=(
            _observations(candidates, pattern, with_control=symptom.task_id is not None)
            if pattern
            else []
        ),
        follow_up=_follow_up(
            symptom.observed,
            subject.selected_values,
            subject.selection_gates,
            writers,
            carried,
        ),
        findings=_findings(candidates, producers),
        gaps=gaps,
    )


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


def _settle(
    candidate: Candidate,
    probes: dict[str, Observation],
    controls: dict[str, Observation],
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
    if not candidate.log_files:
        settled.verdict = UNASKABLE
        settled.because = "no log file names it"
        return settled

    asked = [(f, probes.get(f), controls.get(f)) for f in candidate.log_files]
    for filename, probe, _control in asked:
        if probe is not None and probe.verdict == ANSWER_SEEN:
            settled.verdict = SEEN
            settled.because = f"logged in {filename}"
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
    probes = {o.log_file: o for o in settled.observations if o.role == PROBE}
    controls = {o.log_file: o for o in settled.observations if o.role == CONTROL}
    settled.candidates = [_settle(c, probes, controls) for c in settled.candidates]
    return settled


def survivors(strategy: Strategy) -> list[Candidate]:
    """Candidates the evidence has not ruled out, most settled first."""
    order = {SEEN: 0, UNSETTLED: 1, UNASKABLE: 2, ELIMINATED: 3}
    return sorted(
        (c for c in strategy.candidates if c.verdict != ELIMINATED),
        key=lambda c: (order.get(c.verdict, 9), c.owner),
    )
