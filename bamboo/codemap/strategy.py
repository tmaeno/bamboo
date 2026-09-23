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

import ast
import logging
import math
import re
from collections import Counter
from typing import NamedTuple, Optional, Sequence

from bamboo.codemap import evidence, trace
from bamboo.codemap.evidence import GrepQuery
from bamboo.codemap.lookup import CodeMap
from bamboo.codemap.models import (
    ACTS,
    ANSWER_ABSENT,
    ANSWER_INCONCLUSIVE,
    ANSWER_NO_FILE,
    ANSWER_NOT_ASKED,
    ANSWER_SEEN,
    ELIMINATED,
    LEAD_CALLEE,
    LEAD_MAP,
    PASSTHROUGH_OUTCOME,
    READS_ONLY_AT_TOP,
    READS_ONLY_FOR_A_CALLER,
    REPORTS_DECISION,
    REPORTS_ROWS_CHANGED,
    SEEN,
    SELF_REPAIRING_TRIGGERS,
    SKELETON_PRINT,
    STOP_AMBIGUOUS,
    STOP_DESCENT,
    STOP_NO_WRITER,
    STOP_SHARED_TABLE,
    SYMPTOM_DISTRIBUTION,
    UNASKABLE,
    UNSETTLED,
    Candidate,
    CandidateBranch,
    EntityNode,
    FollowUp,
    FunnelStep,
    Handover,
    JunctionNode,
    Lead,
    Localization,
    LogSiteNode,
    MapTerm,
    Match,
    Observation,
    Reading,
    StageCut,
    Strategy,
    Symptom,
    outcome_excludes,
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
    return _exact_first(description, _mark_ties(matches))[:limit]


def _mark_ties(matches: list[Match]) -> list[Match]:
    """Say which entries scored the same, rather than letting the sort decide.

    The sort breaks a tie on the key, so ``-t1weight`` beats ``T1 weight check``
    because ``-`` sorts below ``T``.  That is not a reason, and the vocabulary
    ties often enough to matter: 15 of its 425 entries, asked by their own
    name, come back level with the next one.  Recorded on both sides, because
    a reader shown one of them has to know the other was its equal.
    """
    for index, match in enumerate(matches):
        match.tied_with = [
            other.term.key
            for position, other in enumerate(matches)
            if position != index and other.score == match.score
        ]
    return matches


def _exact_first(description: str, matches: list[Match]) -> list[Match]:
    """Put the entry the description *is* at the front of the ranking.

    Not a score adjustment.  The weighting divides by everything an entry
    carries, and a cut carries its step's words as well as its own, so the
    entry spelled exactly as asked can be outranked by a near-homograph that
    carries fewer words -- ``-t1weight`` (``t, 1, weight, check``) beat
    ``-t1_weight`` (``t, 1, weight, final, check``) when ``-t1_weight`` was
    what was typed, and the two are different stages of the same chain, 618
    lines apart.  Measured over the vocabulary: 23 of 425 entries do not rank
    first when asked by their own key.

    The rest of the ranking is left alone and the displaced entry stays in it.
    An exact key is a strong signal about which entry was meant; it is not
    evidence that the others were not.
    """
    said = description.strip()
    if not said:
        return matches
    exact = [m for m in matches if m.term.key == said]
    if not exact:
        return matches
    for match in exact:
        match.exact = True
    return exact + [m for m in matches if m.term.key != said]


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
    only settled at run time.  A tier-2 branch is not ruled out by the value,
    because "this one could have" is the honest answer for it -- unless its own
    text is a frame the value does not fit, which is the one case where the arm
    says so itself.  The junction-level half of the same rule is
    :meth:`~bamboo.codemap.lookup.CodeMap.producers_of`; this one has to agree
    with it, or a surviving junction would be offered with no arm to read.
    """
    stated = [b for b in junction.branches if b.outcome == observed]
    return stated or [
        b
        for b in junction.branches
        if b.tier == 2 and not outcome_excludes(b.outcome, observed)
    ]


def _candidate(junction: JunctionNode, observed: str) -> Candidate:
    stated = any(b.outcome == observed for b in junction.branches)
    return Candidate(
        owner=junction.owner,
        file=junction.anchor.file if junction.anchor else "",
        blob_sha=(junction.anchor.blob_sha or "") if junction.anchor else "",
        gloss_key=junction.gloss_key,
        tier=1 if stated else 2,
        dispatch=list(junction.dispatch),
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
        # One row per distinct handover, not per entry point.  A module that
        # both polls and reads a command row carries two entries for the one
        # construction, and they hand over the same thing -- the trigger is a
        # fact about arrival, and this is a fact about the data.
        handovers=list(
            {
                (entry.entry, entry.via, entry.reached_by): Handover(
                    entry=entry.entry,
                    via=entry.via or "",
                    reached_by=entry.reached_by,
                    fields=dict(entry.arg_binding),
                )
                for entry in junction.entry_points
                if entry.arg_binding
            }.values()
        ),
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
    # The reading follows the naming.  A match is proof that this arm decided,
    # so offering the other seventeen functions after saying which one wrote the
    # value would hand a reader the question the record already answered.
    settled.readings = _readings(settled.candidates, settled.observations)
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
    updated_by: list[str],
    selection_gates: list[str],
    writers: list[JunctionNode],
    carried_from: list[str],
    log_sites: dict[str, LogSiteNode],
    entities: list[EntityNode],
    changing: set[str],
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

    **A query where there is one, an update otherwise.**  Both act on rows by
    the value and only the first is somewhere the row could have been missed,
    so a query names the place to look wherever one exists.  Where none does --
    nine values in the corpus are reached by nothing but an update's predicate
    -- the update is still the actor, and dropping it would turn "this is the
    statement that has to match" into the much stronger "nothing acts on this
    at all".

    **Being selected is not the same as being acted on.**  Thirty of the
    hundred and fifty-eight values something selects are selected only by
    functions that settle no value and write no row.  Five of those are started
    by a trigger of their own -- metrics daemons, which read a status to average
    something and change nothing -- and sending a reader to ask one of them why
    it had not picked their row up is the verdict line pointed at a dead end.
    The other twenty-five are getters with no trigger, where what acts is the
    caller, so there the map says that and no more: claiming the row will not
    move would be the same mistake facing the other way.
    """
    selected = observed in selected_values
    # Junctions first, so that an owner which both settles a value and reads
    # one is answered from the node that carries the rest of its story.  Two
    # thirds of the readers the map names settle nothing, and looking only
    # among the writers left those with no file at all -- the query was named
    # and the log to check for it was not.
    by_owner: dict[str, JunctionNode | LogSiteNode] = dict(log_sites)
    by_owner.update({j.owner: j for j in writers})
    readers = [by_owner[o] for o in selected_by if o in by_owner]
    actors = readers or [by_owner[o] for o in updated_by if o in by_owner]
    reader_files = sorted({f for r in actors for f in r.observable_log_files()})
    # The actor's own triggers where it has any.  Most readers are proxy
    # methods the trigger slice reaches through a knight rather than directly,
    # so their entry points are empty -- and reading that as the answer says
    # "nothing reaches this subject", which is a stronger claim than the map
    # can make and, for ``pending``, the opposite of true.
    # Junctions only.  A log site settles nothing, so it has no entry points to
    # have -- said with a type test rather than a missing attribute, because
    # "this kind cannot answer that" and "this one happens not to" are
    # different facts and only the second should fall through to the writers.
    # A junction answers from its entry points and a log site from its
    # module's cadence.  Both are the same classification; only a junction has
    # an arm for an argument to be bound at, which is why the two are stored
    # differently and read together here.  Without the second half, a value
    # whose only re-evaluating reader settles nothing -- ``holding``, read by
    # ``copyArchive.main`` on a daemon cycle -- fell through to the writers
    # and was reported as reached by ``command, request`` alone, which sends
    # the reader to ask whether a command arrived.
    triggers = sorted(
        {
            entry.trigger
            for j in actors
            if isinstance(j, JunctionNode)
            for entry in j.entry_points
        }
        | {
            trigger
            for j in actors
            if isinstance(j, LogSiteNode)
            for trigger in j.triggers
        }
    ) or sorted({entry.trigger for j in writers for entry in j.entry_points})
    repairing = bool(set(triggers) & SELF_REPAIRING_TRIGGERS)
    # Judged over the owners the report *names*, not over ``actors``.  Those
    # are the ones that resolved to a node, and a reader with no log site of
    # its own resolves to nothing -- so a claim made from ``actors`` would be
    # about a subset while reading as though it were about the printed list.
    # ``changing`` is asked by name for the same reason it is global: the
    # subject's own writers are the only junctions ``by_owner`` holds, so nine
    # functions that settle some *other* subject's value look inert here, and
    # ``updateJobStatus`` called inert would be plainly false.
    named = list(selected_by) or list(updated_by)
    if not named or any(owner in changing for owner in named):
        reader_acts = ACTS
    elif all(
        isinstance(by_owner.get(owner), LogSiteNode) and by_owner[owner].triggers
        for owner in named
    ):
        reader_acts = READS_ONLY_AT_TOP
    else:
        # Every one of them inert, and at least one reached only as a callee or
        # not resolved at all.  The weaker statement covers both: one getter in
        # the set is enough for "the row is going nowhere" to be unsupported,
        # and a name the map could not place is not evidence of anything.
        reader_acts = READS_ONLY_FOR_A_CALLER
    # Kept apart from ``created_by`` being empty, which reads as "the map did
    # not look".  Three kinds of row in the corpus are changed here and made
    # somewhere this map has not read.
    creates_rows = any(entity.created_by for entity in entities) if entities else False
    created_by = sorted({owner for entity in entities for owner in entity.created_by})
    if selected_by:
        asks = f"{observed!r} is selected by {_readers_phrase(selected_by, reader_files)}"
    elif updated_by:
        asks = (
            f"no query selects on {observed!r}, but "
            f"{_readers_phrase(updated_by, reader_files)} updates rows holding it"
        )
    else:
        asks = f"a query selects on {observed!r}"
    carried_or_writers = ", ".join(carried_from) if carried_from else "the writers listed above"
    if selected and reader_acts == READS_ONLY_AT_TOP:
        # The value is selected and the row still goes nowhere, so the sentence
        # has to say both -- dropping to the unselected wording would leave the
        # "which query selects it" line above it unexplained.
        question = (
            f"{asks}, but that reader settles no value and writes no row, and nothing "
            "calls it -- so being selected leads nowhere and waiting will not move the "
            f"row: ask who wrote the step before it: {carried_or_writers}"
        )
    elif selected and repairing:
        # The clause about tables nothing writes belongs only where such a
        # table was named.  Folding table names into the map made twenty
        # subjects give up a gate they should never have had, and without this
        # the sentence went on referring to "those tables" after naming none.
        bounded = (
            "bounded by "
            + ", ".join(selection_gates)
            + ", and nothing in the map writes those tables"
            if selection_gates
            else "bounded by nothing this map can name"
        )
        question = (
            f"{asks} and a re-evaluating trigger reaches this "
            f"subject, so ask why it did not pick the row up -- its reach is {bounded}"
        )
    elif selected:
        # Where the answer is "ask whether the command arrived", say where
        # arriving would be.  A command reaching a service is a row appearing
        # in a table, and until the writing verbs were kept apart the map could
        # not name the statement that puts it there.
        question = (
            f"{asks}, but only {', '.join(triggers) or 'nothing the map recognises'} "
            "reaches it, and none of those re-evaluate -- so ask whether the "
            "command or message arrived, not which condition blocked it"
        )
        if created_by:
            question += (
                f"; a row of this kind is made by {', '.join(short_owner(o) for o in created_by)}"
            )
        elif creates_rows is False:
            question += (
                "; nothing in this map makes rows of this kind, so whatever does is "
                "outside what was read"
            )
    else:
        # Narrowed from "waiting will not move the row", which claimed more
        # than the computation behind it.  ``selected`` is true when a SQL
        # WHERE selects this value literally or an UPDATE acts on rows holding
        # it -- nothing else.  Code that reads the value and decides is
        # invisible to it: ``commandToHarvester`` sets ``to_skip`` when an
        # existing command holds this value, which stops the next write and is
        # unmistakably something acting on it.  That contradiction was in one
        # report with itself, eight lines apart, because the trace printed the
        # ``command_status in [...]`` test right underneath.  The population is
        # not cheap to measure -- resolving the receiver's spec class needs the
        # whole corpus -- so no count is claimed here and only the claim is
        # brought back inside what was looked at.
        question = (
            f"no query in the map selects on {observed!r} -- ask who wrote the step "
            f"before it: {carried_or_writers}"
        )
    if reader_acts == READS_ONLY_FOR_A_CALLER:
        # Appended rather than replacing the sentence: everything it says is
        # still true, and this only stops the named reader being read as the
        # thing that acts.  Its caller is where that question goes, and this
        # map resolves reach by name one hop, so naming it is not available
        # here -- saying so is better than implying the reader is the answer.
        question += (
            "; that reader settles no value and writes no row, so what acts on the "
            "row is whatever called it, which this line does not name"
        )
    return FollowUp(
        selected=selected,
        reader_acts=reader_acts,
        selected_by=list(selected_by),
        updated_by=list(updated_by),
        creates_rows=creates_rows,
        created_by=created_by,
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
    leading = _leading(symptom.focus, chains)
    owners = [leading, *sorted(set(chains) - {leading})]
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

    # Resolved here rather than when the rejection lines arrive: the value that
    # completes the next question comes from production, and by then the subject
    # list is not open any more.  The map's half is which field was tested; the
    # evidence's half is what it held.
    by_attr, by_class = _declared_by(await code_map.subjects())
    cuts = sorted((_cut(stage) for stage in stages), key=lambda c: (c.owner, c.order))
    ambiguous: list[Lead] = []
    for owner in owners:
        named, unsettled = _chain_reads(
            [c for stage in chains[owner] for c in stage.conditions], by_attr, by_class
        )
        ambiguous += unsettled
        for cut in cuts:
            if cut.owner == owner:
                cut.reads = named

    # What the map can say about the described cut before a single line is
    # read.  ``_leading`` picks the chain the tag sits in and stops there, and
    # the stage itself was then left to be found in a listing a hundred and
    # nine long -- so a description that resolved exactly still named no code.
    # The tag is written per rejected site by production, which is what makes
    # this a lookup rather than a search.
    emitting = await code_map.stage_for_tag(symptom.focus) if symptom.focus else []
    named = sorted((_cut(stage) for stage in emitting), key=lambda c: (c.owner, c.order))

    return Strategy(
        symptom=symptom,
        map_id=code_map.map_id,
        derived_from=next(iter(stages)).derived_from if stages else "",
        observations=_brokerage_observations(files, owners, symptom.task_id),
        localization=Localization(
            chain=leading,
            log_files=files,
            cuts=cuts,
            describes=symptom.focus if named else "",
            emitted_by=named,
        ),
        leads=_deduped(ambiguous),
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
    upstream = await code_map.carried_from(symptom.subject)
    carried = sorted(upstream)

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
            subject.updated_by.get(symptom.observed, []),
            subject.selection_gates,
            writers,
            carried,
            await code_map.log_sites(
                subject.selected_by.get(symptom.observed, [])
                + subject.updated_by.get(symptom.observed, [])
            ),
            await code_map.entities_for(subject.spec_class),
            await code_map.changing_functions(),
        ),
        # Map edges first, so that when both suppliers name one field the fold
        # in ``evaluate`` keeps the deterministic one.  Not folded here: doing
        # that before the surviving-candidate filter makes a lead's presence
        # depend on which candidate happened to be listed first.
        leads=(
            _leads(symptom, candidates, upstream, subject.selection_gates)
            + _joined(symptom, producers)
            + _consulted(
                symptom,
                producers,
                await code_map.selections_by_owner(),
                await code_map.entity_reads_by_owner(),
            )
        ),
        readings=_readings(candidates, asked),
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

    Four rules, all of them the asymmetry this layer is built on:

    * a line seen proves the writer wrote it, whatever the sample size;
    * a line not seen rules the writer out only where the answer was whole
      *and* the file is shown to carry that line at all;
    * a candidate no log names is not ruled out by anything -- being unable to
      look is not evidence;
    * a candidate no question names is not ruled out either -- not having
      asked and having asked and heard nothing are different facts.
    """
    settled = candidate.model_copy(deep=True)
    mine = [probe for probe in probes if candidate.owner in probe.settles]
    if not candidate.log_files and not mine:
        settled.verdict = UNASKABLE
        settled.because = "no log file names it"
        return settled
    if not mine:
        # Nobody asked.  Everything below reads a probe's silence, and with no
        # probe there is no silence to read: falling through leaves ``reasons``
        # empty and reaches the last branch, which rules the candidate out on
        # the strength of a question that was never put.  Not asked and asked
        # without an answer are different facts, and this is the one place the
        # derivation could confuse them into a confident wrong answer.
        settled.verdict = UNSETTLED
        settled.because = "no question about this one was put to production"
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


# ---------------------------------------------------------------------------
# Leads -- the one hop this derivation can license
# ---------------------------------------------------------------------------


def _reads(conditions: list[str]) -> dict[str, set[str]]:
    """Receiver -> the attributes *conditions* read off it.

    Parsed rather than matched, and the receiver kept.  The recognizer records
    a stage's inputs as bare leaf names, which is enough to say a backward walk
    continues and not enough to say where: ``status`` is declared by a dozen
    specs and ``tmpSiteSpec`` is the half that settles it.  Measured on the
    stored map, 77 receivers are dropped that way.

    Not shared with ``selection._identifiers`` despite doing similar work.
    That one lives in the PanDA recognizer plugin, and the read side is
    map-generic -- importing a plugin here would make every map's lookup depend
    on one map's extractor.  The duplication is the layer boundary, and it is
    the smaller cost.
    """
    found: dict[str, set[str]] = {}
    for condition in conditions:
        try:
            tree = ast.parse(condition.split("  [")[0], mode="eval")
        except SyntaxError:
            continue
        called = {node.func for node in ast.walk(tree) if isinstance(node, ast.Call)}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute) or node in called:
                continue
            receiver = node.value.id if isinstance(node.value, ast.Name) else ""
            found.setdefault(receiver, set()).add(node.attr)
    return found


def _declared_by(subjects: list) -> tuple[dict[str, list[str]], dict[str, set[str]]]:
    """Two indexes over the subject list: by attribute name, and by spec class."""
    by_attr: dict[str, list[str]] = {}
    by_class: dict[str, set[str]] = {}
    for subject in subjects:
        by_attr.setdefault(subject.attribute, []).append(subject.name)
        by_class.setdefault(subject.spec_class, set()).add(subject.attribute)
    return {a: sorted(n) for a, n in by_attr.items()}, by_class


def _read_names(
    receiver: str,
    attribute: str,
    reads: dict[str, set[str]],
    by_attr: dict[str, list[str]],
    by_class: dict[str, set[str]],
) -> tuple[str, list[str]]:
    """Which subject a read names, and the candidates when it cannot be said.

    The disambiguation is the map's own strongest idiom, turned around: a write
    site is attributed by the set of attributes touched on the receiver, because
    what a name is called says who called it while what is asked of it says what
    the code requires it to be.  The same holds for a read -- ``tmpSiteSpec`` is
    asked for ``maxwdir`` as well as ``status``, and only ``SiteSpec`` declares
    both.

    Where the subject list cannot separate the candidates the answer is the
    ambiguity, not a pick.  Choosing one would be the retrieval failure this
    design removes, reintroduced one hop further out and with no way for the
    reader to see it happened.
    """
    declaring = by_attr.get(attribute, [])
    if len(declaring) <= 1:
        return (declaring[0] if declaring else ""), declaring
    # Only attributes some subject declares can discriminate; the rest are
    # fields the promotion did not keep and say nothing about the class.
    asked = {name for name in reads.get(receiver, set()) if name in by_attr}
    fits = [name for name in declaring if asked <= by_class.get(name.rpartition(".")[0], set())]
    return (fits[0] if len(fits) == 1 else ""), declaring


#: ``due to status=offline (31x)`` -> ``status``, ``offline``.  The rejection
#: line carries the value that made the condition true, which is the only place
#: the next question's value can come from: the map knows the field and
#: production knows what it held.
_MEASURED = re.compile(r"\b(?P<name>[A-Za-z_]\w*)=(?P<value>[^\s,()]+)")


def _chain_reads(conditions: list[str], by_attr, by_class) -> tuple[dict[str, str], list[Lead]]:
    """What a chain's reads name, and a lead for each name it cannot settle.

    **Scoped to the whole chain, not to one stage, and measured that way.**  A
    stage's recorded conditions are the guards that dominate it rather than the
    test that names the field -- production's ``-status`` cut carries
    ``not sitePreAssigned``, which says nothing about a status.  Per stage there
    is one attribute per receiver, the structural test has nothing to work with
    and every shared name comes back ambiguous; over the function, ``taskSpec``
    is asked for ``cpuTimeUnit`` and ``walltimeUnit`` as well, which only
    ``JediTaskSpec`` declares.

    The function is the right scope for the same reason it is at build time:
    it is where a local name means one thing.  A chain reading ``.status`` off
    two receivers that resolve differently comes back ambiguous, which is the
    honest answer and the case the guard exists for.
    """
    reads = _reads(conditions)
    named: dict[str, str] = {}
    unsettled: list[Lead] = []
    for receiver, attributes in sorted(reads.items()):
        for attribute in sorted(attributes):
            subject, declaring = _read_names(receiver, attribute, reads, by_attr, by_class)
            if subject and named.get(attribute, subject) != subject:
                # Two receivers, two subjects, one name: nothing here can say
                # which of them a line mentioning that name was about.
                named.pop(attribute)
                unsettled.append(
                    Lead(
                        field=attribute,
                        stop=STOP_AMBIGUOUS,
                        why="this chain reads it off more than one kind of record",
                    )
                )
            elif subject:
                named[attribute] = subject
            elif len(declaring) > 1:
                unsettled.append(
                    Lead(
                        field=attribute,
                        stop=STOP_AMBIGUOUS,
                        why=f"{', '.join(declaring)} all declare it",
                    )
                )
    return named, unsettled


def _measured_leads(cut: StageCut) -> tuple[list[Lead], set[str]]:
    """The next question a rejection line already answered the value of.

    The strongest hop the system has, and the map alone could never make it:
    the map says the stage tested a field and production says what that field
    held.  Scoped to nothing -- the entity the task id names is a task and this
    question is about a site, so carrying it over would build a pattern that
    cannot match, which the eliminator would then read as silence.

    Also returns the names it could not tie to a subject, because that set is a
    measurement of the extraction rather than of the question.  Production
    reports ``status=test`` from a chain whose recorded conditions never read a
    status: the test is written into a flag and the message is emitted under
    the flag, so the predicate that names the field is not in the path
    condition.  Counted and reported instead of being reached for across
    chains, which would paper over a gap in the build with a guess in the read.
    """
    leads: list[Lead] = []
    unnamed: set[str] = set()
    for reason in cut.reasons:
        for found in _MEASURED.finditer(reason):
            subject = cut.reads.get(found.group("name"))
            if not subject:
                unnamed.add(found.group("name"))
                continue
            if any(lead.field == subject for lead in leads):
                continue
            leads.append(
                Lead(
                    field=subject,
                    symptom=Symptom(subject=subject, observed=found.group("value")),
                    why=f"{cut.tag or cut.funnel_label} said {found.group(0)}",
                    opened_by=cut.tag or cut.funnel_label,
                )
            )
    return leads, unnamed


def _consulted(
    symptom: Symptom,
    producers: list[JunctionNode],
    selections: dict[str, list[tuple[str, str]]],
    entity_reads: dict[str, list[str]],
) -> list[Lead]:
    """Leads from one hop along a ``self.<method>()`` call.

    The map's own edges reach the fields a value was *copied* from.  They do not
    reach the rows a value was *decided on*: an arm sends a task to
    ``exhausted`` because an aggregate over its jobs came out a certain way, and
    the write and the read of that aggregate are two methods with a call between
    them and no edge at all.  Measured over the junctions whose subject has a
    spec class, one hop opens a read of another entity for 138 of them; keeping
    that hop inside one file, as the first version did, opened 47.

    **Named by the entity, not by one of its columns.**  Two suppliers answer
    the same question at different resolutions: a predicate on a promoted
    attribute says which rows *and* what was asked of them, while an entity
    read says only which rows.  The second is the one that reaches
    ``getPandaIDsWithTask_JEDI``, whose only predicate is the join key, and
    which is exactly the helper that makes "waiting on jobs" true.  Keyed on
    the entity so the two fold into one lead per kind of row rather than one
    per column -- the hop is down to a population either way, and a reader
    given four names for one descent reads four descents.

    **Not the owner.**  The tempting version asks whether the junction's own
    function also selects on another entity, and it is unsound:
    ``selected_by`` records the function, not the statement, so a method
    dispatching several commands reads jobs in one arm and writes the task
    status in another.  Joining those through their shared owner is the same
    mistake that gave one junction fourteen entry points, and that let a funnel
    gate call a 4836-to-9669 split a majority.  A call is a real edge; sharing
    an enclosing function is not.

    Stops rather than continues.  The next question is about a population, not
    a row, and asking one is not something this derivation does -- naming it is
    a complete answer and a work item, which is how every other edge of the map
    is treated.
    """
    leads: list[Lead] = []
    here = symptom.subject.rpartition(".")[0]
    for junction in producers:
        for target in junction.calls:
            asked = target.rpartition("::")[2]
            # The column reading first, so that where one exists it is the
            # clause a reader gets: it says what the helper wanted to know,
            # which the entity alone cannot.
            detail: dict[str, str] = {}
            for subject, value in selections.get(target, ()):
                detail.setdefault(
                    subject.rpartition(".")[0], f"selects {subject}={value}"
                )
            for entity in entity_reads.get(target, ()):
                detail.setdefault(entity, f"reads {entity} rows")
            for entity, said in detail.items():
                # Another field of the same spec is a lateral read: the row is
                # the one already being asked about, so there is nothing to
                # descend to and calling it a descent would put a population
                # question where a value question belongs.
                if entity == here:
                    continue
                leads.append(
                    Lead(
                        field=entity,
                        stop=STOP_DESCENT,
                        why=(
                            f"{short_owner(junction.owner)} asks "
                            f"{asked}(), which {said}"
                        ),
                        opened_by=junction.owner,
                        source=LEAD_CALLEE,
                    )
                )
    return leads


def _joined(symptom: Symptom, producers: list[JunctionNode]) -> list[Lead]:
    """Leads from a query that asks for two kinds of row at once.

    The owner's *own* statement, which is the one place its reads can be used
    without the join this corpus keeps punishing.  Two facts about a function
    are not a relation between them -- that is what gave a junction fourteen
    entry points and a funnel gate a 4836-to-9669 "majority" -- but two tables
    in one ``FROM`` list are the corpus stating the relation itself.  Measured:
    reading the owner's statements together would open 113 junctions, and
    reading each statement on its own opens 31.

    What the six say is the question the map could not previously reach.
    ``prepareTasksToBeFinished_JEDI`` selects tasks against their datasets in a
    single statement, so asked why a task sits in ``finishing`` it can say
    *which datasets it is waiting on* rather than only that some helper it
    calls reads datasets somewhere.

    Reads only.  An update joining two tables states the same relation, but a
    descent follows the rows a decision was taken *from*, and following a write
    would walk forwards while claiming to step down.
    """
    here = symptom.subject.rpartition(".")[0]
    return [
        Lead(
            field=entity,
            stop=STOP_DESCENT,
            why=(
                f"{short_owner(junction.owner)} asks for {entity} rows in the "
                "same query as the rows it decides about"
            ),
            opened_by=junction.owner,
            source=LEAD_MAP,
        )
        for junction in producers
        for entity in junction.joined_entities
        if entity != here
    ]


def _leads(
    symptom: Symptom,
    candidates: list[Candidate],
    carried: dict[str, list[JunctionNode]],
    gates: list[str],
) -> list[Lead]:
    """Where a value symptom's answer continues, and where it stops.

    Only from what the map can say without observing anything new, which is two
    things.  A ``passthrough`` outcome says the value arrived unchanged, so the
    observed value is also the next question's -- a whole symptom, ready to ask.
    And a selection gate is a table the map only ever reads, which is a terminal
    of the walk and the one that mattered: a task sat in ``finishing`` because
    the watermark table bounding the rescue query had stopped being updated.

    A field that carries to itself is not a lead.  ``JediTaskSpec.status`` and
    ``oldStatus`` copy from each other and ``JobSpec.jobStatus`` from itself, so
    emitting those would hand back a loop dressed as progress -- and the caller
    cannot tell one from a real hop without already knowing the map.
    """
    leads: list[Lead] = []
    for candidate in candidates:
        opened: set[str] = set()
        for branch in candidate.branches:
            found = PASSTHROUGH_OUTCOME.match(branch.outcome or "")
            if not found:
                continue
            field = found.group("field")
            if field == symptom.subject or field in opened:
                continue
            opened.add(field)
            upstream = carried.get(field) or []
            leads.append(
                Lead(
                    field=field,
                    symptom=(
                        Symptom(
                            subject=field,
                            observed=symptom.observed,
                            task_id=symptom.task_id,
                        )
                        if upstream
                        else None
                    ),
                    stop="" if upstream else STOP_NO_WRITER,
                    why=(
                        f"{short_owner(candidate.owner)} copies the value from it, so it "
                        f"held {symptom.observed!r} too"
                    ),
                    opened_by=candidate.owner,
                )
            )
    leads += [
        Lead(
            field=gate,
            stop=STOP_SHARED_TABLE,
            why="it bounds which rows the queries selecting this value can see",
        )
        for gate in gates
    ]
    return leads


def visit_key(symptom: Symptom) -> str:
    """How a question is spelled when asking whether it has been asked before.

    The whole question, not the field.  ``JediTaskSpec.status`` is reached
    twice in most walks -- once as the symptom and once as what ``oldStatus``
    was copied from -- and those are the same question only if the value is
    the same too.  Keying on the field alone would cut a live path on the
    grounds that a different question about that field had already been asked.
    """
    if symptom.kind == SYMPTOM_DISTRIBUTION:
        return f"{SYMPTOM_DISTRIBUTION}:{symptom.focus}"
    return f"{symptom.subject}={symptom.observed}"


def next_question(strategy: Strategy, visited: set[str]) -> tuple[Optional[Lead], list[str]]:
    """The first lead worth taking, and the questions that came round again.

    Surviving leads only, and in the order the derivation put them: the map's
    own edges come before anything assembled here, so a deterministic hop is
    never passed over for a proposed one.

    Returns the repeats as well as the choice because they are an answer.
    ``status`` and ``oldStatus`` copy from each other, so a walk that merely
    skipped the repeat would stop looking like it had found a loop and start
    looking like it had run out of map -- and those call for opposite things
    from whoever reads the trace.
    """
    repeats: list[str] = []
    for lead in _deduped(_surviving_leads(strategy)):
        if lead.symptom is None:
            continue
        key = visit_key(lead.symptom)
        if key in visited:
            repeats.append(key)
            continue
        return lead, repeats
    return None, repeats


def _deduped(leads: list[Lead]) -> list[Lead]:
    """One lead per field, keeping the first that named it.

    Several stages of one chain test the same field, and a reader asked to hold
    twenty copies of "go and find out about ``status``" stops reading.

    **Always after the surviving-candidate filter, never before.**  Five writers
    of ``JediTaskSpec.oldStatus`` copy it from ``status``; folding them first
    keeps one, and if the evidence then rules that one out the field disappears
    although four surviving candidates still open it.  Collapsing before
    filtering makes a lead's presence depend on which candidate happened to be
    listed first, which is not a fact about the system.
    """
    seen: set[str] = set()
    kept: list[Lead] = []
    for lead in leads:
        if lead.field in seen:
            continue
        seen.add(lead.field)
        kept.append(lead)
    return kept


def _surviving_leads(strategy: Strategy) -> list[Lead]:
    """The leads whose opener the evidence has not ruled out.

    A walk that keeps descending from a branch production says did not fire is
    following a path the system did not take, and it would do so with all the
    confidence of the ones that did.
    """
    alive = {c.owner for c in strategy.candidates if c.verdict != ELIMINATED}
    if strategy.localization is not None:
        alive |= {
            cut.tag or cut.funnel_label
            for cut in strategy.localization.cuts
            if cut.verdict != ELIMINATED
        }
    return [lead for lead in strategy.leads if not lead.opened_by or lead.opened_by in alive]


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
    if rejections and not any(
        o.role == PROBE and o.verdict != ANSWER_NOT_ASKED for o in strategy.observations
    ):
        # The lines were found, but not by the questions derived here: a survey
        # of every broker log was already in the file and it contains them.
        # Worth saying, because that survey is the reason the sample is partial
        # -- it reaches its cap where a question naming one task would not.
        strategy.gaps.append(
            "these lines came from a wider question already in the evidence file rather "
            "than from the scoped ones derived here -- --fetch asks those, and only they "
            "come back whole enough for an absence to mean anything"
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
        # Only now can a cut's read become a whole question: the map said which
        # field the stage tested and the line says what it held.
        unnamed: set[str] = set()
        for cut in settled.localization.cuts:
            found, missed = _measured_leads(cut)
            settled.leads += found
            unnamed |= missed
        if unnamed:
            settled.gaps.append(
                f"{len(unnamed)} name(s) the rejection lines report a value for are not read "
                "by any condition this chain records, so the walk cannot say what they are "
                f"about: {', '.join(sorted(unnamed)[:8])}"
            )
    settled.leads = _deduped(_surviving_leads(settled))
    # Narrowed for the same reason the leads are: offering a reading of code the
    # evidence has ruled out sends a reader to look at a path the system did not
    # take, which is worse than offering nothing.
    settled.readings = _readings(survivors(settled), settled.observations)
    diag = _recorded_message(ev, strategy.symptom)
    return name_the_arm(settled, diag) if diag else settled


#: How far one sentence narrows the answer, worst first.  The four are
#: disjoint and tested in this order: a line no file carries cannot be asked at
#: all, a line two writers share cannot say which wrote it even when it is
#: found, and a line several arms of one function share settles the function
#: and not the arm.  Only the last is a question whose answer is an arm.
LINE_NO_FILE = "no file carries it"
LINE_ACROSS_WRITERS = "shared across writers"
LINE_ONE_FUNCTION = "settles the function, not the arm"
LINE_ONE_ARM = "names one arm"

DISCRIMINATION = (LINE_ONE_ARM, LINE_ONE_FUNCTION, LINE_ACROSS_WRITERS, LINE_NO_FILE)


class Sentence(NamedTuple):
    """One line production could print, and what it would pin down if seen.

    Deliberately the same shape for both cadences.  The map's arm sentences
    and the skeleton's lines are the two halves of the comparison this round
    exists to make, and scoring them with two functions would make any
    difference between them a fact about the two functions.
    """

    text: str
    owner: str
    line: Optional[int]
    files: Sequence[str]


def discrimination(sentences: Sequence[Sentence]) -> Counter:
    """How many of *sentences* narrow to an arm, a function, a set, or nothing.

    The whole set has to be passed at once: whether a line names one arm is not
    a property of that line but of how many other arms say the same thing, and
    a per-line answer would call every line unique.
    """
    owners: dict[str, set[str]] = {}
    arms: dict[str, set[tuple[str, Optional[int]]]] = {}
    for sentence in sentences:
        owners.setdefault(sentence.text, set()).add(sentence.owner)
        arms.setdefault(sentence.text, set()).add((sentence.owner, sentence.line))
    counted: Counter = Counter()
    for sentence in sentences:
        if not sentence.files:
            counted[LINE_NO_FILE] += 1
        elif len(owners[sentence.text]) > 1:
            counted[LINE_ACROSS_WRITERS] += 1
        elif len(arms[sentence.text]) > 1:
            counted[LINE_ONE_FUNCTION] += 1
        else:
            counted[LINE_ONE_ARM] += 1
    return counted


def skeleton_sentences(readings: list[Reading]) -> list[Sentence]:
    """The skeleton's side of the comparison, one entry per (line, arm) pair.

    Per pair rather than per row because the question ``discrimination`` asks
    is whether a sentence narrows to *an arm*, and a row two arms could both
    have printed is two chances for the same text to turn up.  A row no arm
    can be printed alongside keeps a ``None`` arm rather than being dropped:
    it is a line the function prints, and leaving it out would quietly
    flatter the count.
    """
    return [
        Sentence(row.pattern, entry.owner, arm, entry.log_files)
        for entry in readings
        for row in entry.skeleton
        if row.kind == SKELETON_PRINT and row.pattern
        for arm in (row.arms or [None])
    ]


def _readings(candidates: list[Candidate], observations: list[Observation]) -> list[Reading]:
    """The code to read, one entry per function rather than per candidate.

    Grouped because the sharing is real and asking twice about one function
    invites two answers about one piece of code: 497 junctions sit in 213
    functions, and one of them holds twenty.  The key is the map's, computed at
    build time, so the same question always selects the same text.

    A candidate the map cannot locate a function for is left out here and said
    out loud by the caller.  Five exist, all writes at module scope, where
    there is no enclosing function to hand over -- a fact about the code, not a
    hole in the reading.
    """
    # A record that names an arm has settled which one decided -- the message
    # and the branch were written in the same block -- so the reading narrows
    # to it.  The other direction is not available: naming none proves nothing,
    # and then every candidate is still worth reading.
    named = [c for c in candidates if c.named]
    if named:
        candidates = named

    scoped: dict[str, tuple[str, list[str]]] = {}
    for observation in observations:
        if observation.role != PROBE:
            continue
        for owner in observation.settles:
            pattern, files = scoped.setdefault(owner, (observation.pattern, []))
            if observation.log_file not in files:
                files.append(observation.log_file)

    grouped: dict[str, Reading] = {}
    for candidate in candidates:
        if not candidate.gloss_key:
            continue
        reading = grouped.get(candidate.gloss_key)
        if reading is None:
            pattern, files = scoped.get(candidate.owner, ("", []))
            reading = Reading(
                owner=candidate.owner,
                file=candidate.file,
                blob_sha=candidate.blob_sha,
                gloss_key=candidate.gloss_key,
                dispatch=list(candidate.dispatch),
                log_files=list(files),
                log_pattern=pattern,
            )
            grouped[candidate.gloss_key] = reading
        arms = candidate.named or candidate.branches
        for branch in arms:
            if branch.line is not None and branch.line not in reading.lines:
                reading.lines.append(branch.line)
            if branch.outcome and branch.outcome not in reading.outcomes:
                reading.outcomes.append(branch.outcome)
    for reading in grouped.values():
        reading.lines.sort()
    return sorted(grouped.values(), key=lambda r: r.owner)


def attach_traces(
    strategy: Strategy,
    roots: dict,
    classify=None,
) -> None:
    """Walk the source for each reading and record why those arms ran.

    A separate call rather than part of ``derive`` because it needs a source
    tree, and needing one is the difference between the two phases this whole
    design keeps apart: a derivation must be checkable before anything is
    read, and the reading is what the source is for.

    The handovers come from the candidates rather than from the readings,
    which group by function: the crossing is a property of the junction the
    map recorded, and two junctions in one function were reached the same way.

    *classify* names the interfaces, and is PanDA's rather than the walk's.
    Imported here rather than at the top for the layering ``factory`` already
    uses -- the walk is about Python, and which attribute means DDM is about
    this corpus.
    """
    if not roots or not strategy.readings:
        return
    if classify is None:
        from bamboo.codemap.panda.provenance import classify as classify_panda

        classify = classify_panda
    handovers: dict[str, list] = {}
    for candidate in strategy.candidates:
        if candidate.handovers:
            handovers.setdefault(candidate.owner, list(candidate.handovers))
    for entry in strategy.readings:
        if not entry.file or not entry.lines:
            continue
        walked = trace.walk(
            roots,
            file=entry.file,
            owner=entry.owner,
            lines=entry.lines,
            observed=strategy.symptom.observed or "",
            expected_sha=entry.blob_sha,
            handovers=handovers.get(entry.owner, ()),
            classify=classify,
        )
        entry.trace = walked.steps
        entry.trace_note = walked.note
        entry.skeleton = walked.skeleton
    _report_what_the_lines_cannot_separate(strategy)


def _report_what_the_lines_cannot_separate(strategy: Strategy) -> None:
    """Say which of the skeleton's lines would not settle an arm if they were seen.

    Before anything is asked, which is the half of this the map cannot do: a
    line two arms share is a question whose answer is already known to be
    ambiguous, and the reader is better served by being told that than by
    getting the answer and drawing an arm out of it.
    """
    counted = discrimination(skeleton_sentences(strategy.readings))
    vague = sum(counted[kind] for kind in DISCRIMINATION[1:])
    if not vague:
        return
    strategy.gaps.append(
        f"{vague} of the {sum(counted.values())} line(s) the tree says this code prints "
        f"would not settle which arm printed them: {counted[LINE_ONE_FUNCTION]} are "
        f"shared by arms of one function, {counted[LINE_ACROSS_WRITERS]} by more than "
        f"one writer, and {counted[LINE_NO_FILE]} land in no file the map names"
    )


def survivors(strategy: Strategy) -> list[Candidate]:
    """Candidates the evidence has not ruled out, most settled first."""
    order = {SEEN: 0, UNSETTLED: 1, UNASKABLE: 2, ELIMINATED: 3}
    return sorted(
        (c for c in strategy.candidates if c.verdict != ELIMINATED),
        key=lambda c: (order.get(c.verdict, 9), c.owner),
    )
