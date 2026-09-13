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
import math
import re
from collections import Counter
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
    SYMPTOM_DISTRIBUTION,
    UNASKABLE,
    UNSETTLED,
    Candidate,
    CandidateBranch,
    FollowUp,
    FunnelStep,
    JunctionNode,
    Localization,
    MapTerm,
    Match,
    Observation,
    StageCut,
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

# What the log wrapper puts in front of the message, so the id and the line are
# together and one pattern can require both.  Shared with the module that reads
# the answers back: a second spelling of the same convention is a second thing
# to keep current, and there the reader's containment check depends on the two
# being literally the same string.
_TASK_PREFIX = evidence.TASK_PREFIX

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

# A brokerage probe keeps everything it matched.  Its answer is not "did this
# fire" but "which candidates went and why", so the lines are the answer rather
# than a sample of it -- the same reason the transition query keeps its cap's
# worth.  Equal to the cap, so "the answer came back whole" and "nothing was
# dropped writing it down" are one question.
BROKERAGE_KEEP_LINES = evidence.DEFAULT_MAX_MATCHES

# How many matched lines a finding shows.  Enough to read a timestamp and a
# component off the answer; the rest is in the evidence file.
_SAMPLE_LINES = 3


# The entity a description carries.  PanDA ids are long, so this does not have
# to guess: nothing else in a sentence about a task is eight digits.  Kept apart
# from the vocabulary match because an id is not a symptom -- it says which row,
# not which question.
_ENTITY = re.compile(r"\b\d{7,}\b")

# Words of a description.  The same split the vocabulary uses, so that
# ``LowMemory``, ``low memory`` and ``lowmemory`` reduce alike.  Non-ASCII text
# yields nothing here, which is a real limit and is reported as one rather than
# papered over with a hand-built lexicon -- see ``resolve``.
_DESCRIPTION_WORD = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z]+|[a-z]+|\d+")


def entity_in(description: str) -> Optional[str]:
    """The id a description names, if it names exactly one."""
    found = _ENTITY.findall(description)
    return found[0] if len(set(found)) == 1 else None


def resolve(description: str, terms: list[MapTerm], limit: int = 5) -> list[Match]:
    """Rank the vocabulary entries *description* could have meant.

    This is the step that makes a free-text question answerable without a
    search.  The map's vocabulary is closed and enumerable -- 420 entries for
    PanDA -- so the problem is not retrieval but selection out of a known set,
    and selection out of a known set can be measured.

    **Weighted by rarity, computed from the vocabulary itself.**  Counting
    matched words makes ``check`` worth as much as ``lowmemory``, and ``check``
    ends 40 of the 49 funnel steps.  The weight is ``log(N / documents
    containing the word)`` over the entries in hand, so nothing is tuned and the
    weights move on their own when the map grows.  A score is the share of an
    entry's own weight the description accounted for, which is why a two-word
    entry fully said beats a seven-word entry half said.

    **Deterministic on purpose, and measured rather than assumed.**  An LLM
    belongs here only where this is shown not to reach, and where that is can
    only be said by running it: the honest known limit is that the word split
    is ASCII, so a description written in Japanese contributes only the
    identifiers embedded in it.  Returning a ranked list rather than a pick is
    what keeps that limit visible -- a caller that cannot tell one entry from
    the next is told so.
    """
    said = {word.lower() for word in _DESCRIPTION_WORD.findall(description)}
    if not said or not terms:
        return []
    documents = Counter(word for term in terms for word in set(term.words))
    total = len(terms)
    weight = {word: math.log(total / count) + 1.0 for word, count in documents.items()}
    # Normalised by whichever side said more, so neither direction wins on its
    # own: an entry that is fully accounted for but says far less than the
    # description was not what the description was about, and dividing by the
    # entry alone would put a two-word entry above every longer one.
    spoken = sum(weight[word] for word in said if word in weight)
    matches = []
    for term in terms:
        hit = [word for word in term.words if word in said]
        if not hit:
            continue
        carried = sum(weight.get(word, 1.0) for word in term.words)
        against = max(carried, spoken)
        if against <= 0:
            continue
        matches.append(
            Match(term=term, score=sum(weight[word] for word in hit) / against, words=hit)
        )
    matches.sort(key=lambda m: (-m.score, m.term.key))
    return matches[:limit]


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


# ---------------------------------------------------------------------------
# The second symptom class: which step of a chain threw the candidates away
# ---------------------------------------------------------------------------


def _leading(focus: str, chains: dict[str, list]) -> str:
    """Which chain the answer opens with before any evidence has been read.

    Only the opening.  Every chain stays in play, because a description naming
    one cut is a guess about which cut mattered and the file the lines landed in
    is a fact -- narrowing here instead cost exactly what it was meant to
    prevent: a focus resolved to a task broker's ``-job`` dropped the job
    broker's forty-eight stages, and its rejections were then attributed to the
    task broker's ``status check``, condition and all.
    """
    if focus in chains:
        return focus
    named = sorted(
        owner
        for owner, stages in chains.items()
        if any(focus in (stage.criteria_tag, stage.funnel_label) for stage in stages)
    )
    return named[0] if named else sorted(chains)[0]


def _cut(stage) -> StageCut:
    return StageCut(
        tag=stage.criteria_tag,
        funnel_label=stage.funnel_label,
        owner=stage.owner,
        order=stage.order,
        line=stage.anchor.line_start if stage.anchor else None,
        conditions=list(stage.conditions),
        inputs=list(stage.inputs),
        log_files=list(stage.log_files),
    )


def _brokerage_observations(
    files: list[str], owners: list[str], task_id: Optional[str]
) -> list[Observation]:
    """One scoped question per file, and the control that licenses its silence.

    The survey form of these questions reaches its cap on every machine --
    45,000 matches against a bound of 5,000 -- so its emptiness is never
    readable.  Scoped to one task the same question comes back whole.  The
    control is the unscoped pattern, which answers whether the file carries
    rejections at all: ``panda-GenJobBroker.log`` does not exist in this
    deployment, and without the control its silence would read as a chain that
    dropped nothing.
    """
    observations: list[Observation] = []
    for filename in files:
        for pattern in (evidence.TAG_PATTERN, evidence.FUNNEL_PATTERN):
            scoped = evidence.task_scoped(pattern, task_id) if task_id is not None else pattern
            observations.append(
                Observation(
                    log_file=filename,
                    pattern=scoped,
                    role=PROBE,
                    services=[evidence.JEDI],
                    settles=list(owners),
                    keep_lines=BROKERAGE_KEEP_LINES,
                )
            )
            if scoped != pattern:
                observations.append(
                    Observation(
                        log_file=filename,
                        pattern=pattern,
                        role=CONTROL,
                        services=[evidence.JEDI],
                        settles=[],
                        control_for=scoped,
                    )
                )
    return observations


async def localize(code_map: CodeMap, symptom: Symptom) -> Strategy:
    """Read the map for a distribution symptom.

    Brokerage does not settle a value, so there is no branch table and no
    candidate set to prune: every stage runs on every pass and each takes some
    of the list.  What the map contributes here is the chain in order, the
    condition each step tests, where to read it, and which file to ask -- and
    what production contributes is how much of the list each step actually
    took, which is the half that says *which* step is the answer.

    Nothing is filled in from the map about the cuts themselves.  That is the
    honest shape: "this stage removed 161 sites" is not something a map can
    know, and the two-phase split is what keeps the map's part inspectable
    before a query goes out.
    """
    chains = await code_map.chains()
    if not chains:
        raise LookupError("this map holds no filter chains, so it has no distribution to localize")
    leads = _leading(symptom.focus, chains)
    owners = [leads, *sorted(set(chains) - {leads})]
    stages = [stage for owner in owners for stage in chains[owner]]
    files = sorted({f for stage in stages for f in stage.log_files})

    gaps: list[str] = []
    if symptom.task_id is None:
        gaps.append(
            "no entity was named, so what comes back is every task the window holds -- "
            "a survey of what this chain does, not of what it did to one task"
        )
    # A filter stage carries no entry points, and pooling the module's junctions
    # would answer how work reaches anything in the file rather than how this
    # chain is run.  That approximation was reverted once already, so it is a
    # gap rather than a field.
    gaps.append(
        "the map cannot say what re-runs this chain: a filter stage has no entry point, "
        "and the triggers of the junctions in the same module are a different question"
    )
    # Summarised per helper rather than per tag.  These are true and they are
    # about a chain the evidence may well not choose, so five lines of them
    # would push the answer off the first screen for no extra information.
    unreadable: Counter = Counter(
        stage.owner for stage in stages if stage.criteria_tag and not stage.log_files
    )
    findings = [
        f"{short_owner(owner)} emits {count} tag(s) and its module declares no logger, so "
        "the map cannot say which file to read for them -- they surface in whichever "
        "broker called it"
        for owner, count in sorted(unreadable.items())
    ]

    return Strategy(
        symptom=symptom,
        map_id=code_map.map_id,
        derived_from=next(iter(stages)).derived_from if stages else "",
        observations=_brokerage_observations(files, owners, symptom.task_id),
        localization=Localization(
            chain=leads,
            log_files=files,
            cuts=sorted((_cut(stage) for stage in stages), key=lambda c: (c.owner, c.order)),
        ),
        findings=findings,
        gaps=gaps,
    )


async def derive(code_map: CodeMap, symptom: Symptom) -> Strategy:
    """Read the map and return what it has to say about *symptom*.

    Offline in the sense that matters: it reads the stored map and touches
    production not at all.  Everything that needs the deployment is expressed as
    an :class:`Observation` to be run later, so the plan can be inspected --
    and its model corrected -- before a single query goes out.

    Dispatches on the symptom's kind rather than on anything the caller passed
    alongside it.  The kind came off the vocabulary entry the description
    resolved to, which is what keeps the choice of derivation out of the user
    interface: a flag per symptom class would put the selection back in the
    hands of whoever phrased the question.
    """
    if symptom.kind == SYMPTOM_DISTRIBUTION:
        return await localize(code_map, symptom)

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
            keep_lines=(
                observation.keep_lines
                if observation.keep_lines is not None
                else (PROBE_KEEP_LINES if observation.role == PROBE else 0)
            ),
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
        # Deduplicated: one file now carries two probes -- the value line and
        # the arm's own -- and both being silent is one reason, said once.
        settled.because = "; ".join(dict.fromkeys(reasons))[:200]
    else:
        settled.verdict = ELIMINATED
        settled.because = "absent from every log that would carry the line"
    return settled


def _reasons_said(rows: list[evidence.Rejection]) -> list[str]:
    """What the log said as it dropped each one, most frequent first.

    Counted over lines rather than over sites, unlike the sites themselves: a
    reason repeated across passes is the same reason, and what matters about it
    is which of a stage's several wordings production actually used.  Trimmed,
    because one of these runs to two hundred characters of arithmetic.
    """
    counts = Counter(row.reason for row in rows if row.reason)
    return [f"{reason[:120]} ({seen}x)" for reason, seen in counts.most_common(4)]


def _by_tag(
    rejections: list[evidence.Rejection], here: set[str]
) -> dict[str, list[evidence.Rejection]]:
    """Rejections grouped by tag, from the file the chain actually wrote to."""
    grouped: dict[str, list[evidence.Rejection]] = {}
    for row in rejections:
        if here and row.log_file not in here:
            continue
        grouped.setdefault(row.tag, []).append(row)
    return grouped


def _settle_localization(strategy: Strategy, ev: evidence.Evidence) -> Localization:
    """Fill in how much of the list each step actually took.

    Everything numeric here comes from production; the map supplied the order,
    the condition and the place to read.  The chain is re-chosen from the
    evidence rather than kept from the focus, because a description naming one
    cut is a guess and the file the lines landed in is a fact.

    **Attribution when one file holds two chains.**  A task broker calls a job
    broker and hands over its own log slot, so both write to one file and the
    map cannot say so -- it records where a stage's own module writes.  A tag
    seen in a file is therefore attributed to the stages that carry it *and*
    name that file; where none does, every stage carrying it is named and the
    ambiguity is reported rather than resolved.
    """
    settled = strategy.localization.model_copy(deep=True)
    rejections, whole_cuts = evidence.observed_rejections(ev, strategy.symptom.task_id)
    survivors, whole_funnel = evidence.observed_survivors(ev, strategy.symptom.task_id)
    settled.sample = "complete" if whole_cuts and whole_funnel else "partial"

    seen_in = Counter(row.log_file for row in rejections) + Counter(
        row.log_file for row in survivors
    )
    if seen_in:
        settled.log_files = [file for file, _ in seen_in.most_common()]
    here = set(settled.log_files[:1])

    by_tag: dict[str, list[StageCut]] = {}
    for cut in settled.cuts:
        by_tag.setdefault(cut.tag, []).append(cut)
    unknown: Counter = Counter()
    for tag, rows in _by_tag(rejections, here).items():
        carrying = by_tag.get(tag, [])
        # The file is what separates two chains that emit the same tag.  Where
        # no stage carrying the tag names this file, every one of them is named
        # instead and the ambiguity is reported -- a helper that declares no
        # logger, or a broker that handed its log slot to the one it called.
        owned = [cut for cut in carrying if here & set(cut.log_files)] or carrying
        if not owned:
            unknown[tag] = len({row.site for row in rows})
            continue
        for cut in owned:
            cut.sites = sorted({row.site for row in rows})
            cut.reasons = _reasons_said(rows)

    # Every chain was carried this far so that the evidence could choose; now it
    # has, and the chains whose lines are nowhere near this file are not part of
    # the answer.  A cut with sites is kept whatever file it names -- that is
    # how a helper with no logger of its own stays in.
    if here:
        settled.cuts = [cut for cut in settled.cuts if cut.sites or (here & set(cut.log_files))]
    for cut in settled.cuts:
        if cut.sites:
            cut.verdict = SEEN
            cut.because = f"removed {len(cut.sites)} candidate(s)"
        elif settled.sample == "complete":
            cut.verdict = ELIMINATED
            cut.because = "this chain ran and never named it"
        else:
            cut.verdict = UNSETTLED
            cut.because = "the sample is partial, so nothing follows from not seeing it"
    settled.cuts.sort(key=lambda c: (-len(c.sites), c.owner, c.order))

    weight: Counter = Counter()
    for cut in settled.cuts:
        weight[cut.owner] += len(cut.sites)
    if weight and max(weight.values()):
        settled.chain = weight.most_common(1)[0][0]
    settled.chains_sharing_the_file = sorted({c.owner for c in settled.cuts} - {settled.chain})

    order = {cut.funnel_label: cut.order for cut in settled.cuts if cut.funnel_label}
    counts: dict[str, list[int]] = {}
    for row in survivors:
        if row.log_file in here or not here:
            counts.setdefault(row.label, []).append(row.count)
    settled.funnel = sorted(
        (
            FunnelStep(
                label=label,
                order=order.get(label),
                most=max(seen),
                fewest=min(seen),
                seen=len(seen),
            )
            for label, seen in counts.items()
        ),
        # Unmapped steps last: the map has no position for them, and putting
        # them at nought would claim one.
        key=lambda s: (s.order is None, s.order or 0, s.label),
    )
    mapped = [step for step in settled.funnel if step.order is not None]
    if mapped:
        settled.entered, settled.left = mapped[0].most, mapped[-1].most
        settled.passes = mapped[0].seen

    strategy.findings.extend(
        f"production logs {tag} against {sites} site(s) for this entity and the map has "
        "no stage for it"
        for tag, sites in sorted(unknown.items())
    )
    strategy.findings.extend(
        f"production counts a cut at {step.label!r} and the map has no step for it"
        for step in settled.funnel
        if step.order is None
    )
    if settled.chains_sharing_the_file:
        named = [settled.chain, *settled.chains_sharing_the_file]
        strategy.findings.append(
            f"{', '.join(short_owner(o) for o in named)} all have cuts in "
            f"{', '.join(sorted(here)) or 'this file'}, so a tag there can belong to either "
            "-- the map records where a stage's own module writes, not who delegated to it"
        )
    if settled.sample != "complete":
        strategy.gaps.append(
            "the sample was cut off at a bound, so no step can be said to have removed "
            "nothing -- only what was seen counts"
        )
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
    if settled.localization is not None:
        settled.localization = _settle_localization(settled, ev)
    diag = _recorded_message(ev, strategy.symptom)
    return name_the_arm(settled, diag) if diag else settled


def survivors(strategy: Strategy) -> list[Candidate]:
    """Candidates the evidence has not ruled out, most settled first."""
    order = {SEEN: 0, UNSETTLED: 1, UNASKABLE: 2, ELIMINATED: 3}
    return sorted(
        (c for c in strategy.candidates if c.verdict != ELIMINATED),
        key=lambda c: (order.get(c.verdict, 9), c.owner),
    )
