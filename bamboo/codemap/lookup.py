"""Read a stored Code Map back, in the vocabulary the map is written in.

This is the *use* half of the three cadences: ``build-map`` extracts, ``check-map``
says how far the extraction can be trusted, and an investigation only ever looks
things up.  Nothing here rebuilds, and nothing here reaches the source tree -- an
unattended ``analyze`` stays read-only, and a degraded answer is reported rather
than quietly repaired.

**Lookup, not search.**  That distinction is the reason the map exists.  Asking
the source navigator "why is this task pending?" means turning free text into grep
terms and ranking thirty candidates, which fails as `no_candidates`,
`too_many_candidates` or `irrelevant` -- all retrieval failures.  Asking the map
means naming a subject and getting its writers, and the answer does not depend on
having phrased the question well.

**Why the references are properties and not edges.**  A junction stores the
``name`` of the subject it writes, so the join is on a property.  Both work, and
the properties are what exists today -- see ``find_map_nodes``.  The one reference
this cannot express is a ``passthrough(...)`` outcome, which sits inside
``branches``: Neo4j has no nested values, so a junction's branches are stored as
JSON text and the chain is followed by decoding it here rather than by matching a
pattern in the query.  At ~1000 nodes that costs nothing, and it keeps the map
from declaring an edge kind before the backward walk needs one.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional, Type, TypeVar

from bamboo.codemap.models import (
    PASSTHROUGH_OUTCOME,
    SYMPTOM_DISTRIBUTION,
    SYMPTOM_VALUE,
    TERM_CHAIN,
    TERM_CUT,
    TERM_STEP,
    TERM_VALUE,
    BoundaryNode,
    EntityNode,
    FilterStageNode,
    JunctionNode,
    LogSiteNode,
    LoopCutNode,
    MapTerm,
    SubjectNode,
    Symptom,
    ValueEnumNode,
    outcome_excludes,
)
from bamboo.database.graph_database_client import GraphDatabaseClient
from bamboo.models.graph_element import BaseNode, NodeType

logger = logging.getLogger(__name__)

NodeT = TypeVar("NodeT", bound=BaseNode)

_MODELS: dict[Type[BaseNode], NodeType] = {
    SubjectNode: NodeType.SUBJECT,
    EntityNode: NodeType.ENTITY,
    JunctionNode: NodeType.JUNCTION_POINT,
    FilterStageNode: NodeType.FILTER_STAGE,
    LoopCutNode: NodeType.LOOP_CUT,
    LogSiteNode: NodeType.LOG_SITE,
    BoundaryNode: NodeType.BOUNDARY,
    ValueEnumNode: NodeType.VALUE_ENUM,
}


#: An outcome that names where the value came from rather than what it is.
#: Neither is something to be asked about by value -- nobody observes a task in
#: ``runtime(newStatus)``.
_DERIVED = re.compile(r"^(passthrough|runtime)\(")

#: Splits an identifier into the words a person would use for it:
#: ``AtlasProdJobBroker.py::doBrokerage`` -> prod, job, broker, do, brokerage.
_WORD = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z]+|[a-z]+|\d+")

#: What a chain is called when nobody names the file.  Hand-written, which this
#: design normally refuses -- but the alternative is that a description of this
#: symptom class matches nothing at all, because no identifier in the map
#: contains the words people use for it: the code says ``doBrokerage`` and a
#: person says "the jobs only went to two sites".  Six words, attached to the
#: one node kind whose name is never spoken, and counted as part of what the
#: resolver's measurement has to account for.
_BROKERAGE_WORDS = ("brokerage", "broker", "site", "sites", "distribution", "candidates")


def _decode(model: Type[NodeT], props: dict[str, Any]) -> NodeT:
    """Rebuild a node model from stored properties.

    The inverse of what the backend had to do on the way in, and derived from
    the same source of truth rather than from a second list of which fields are
    nested: whatever the model does not declare as text, but which came back as
    text, was JSON on the way in.  A hand-kept list of nested field names would
    be a third expression of a fact the model already states, free to drift from
    the encoder the moment a field is added.
    """
    decoded = dict(props)
    for name, field in model.model_fields.items():
        value = decoded.get(name)
        if not isinstance(value, str):
            continue
        if field.annotation in (str, Optional[str]):
            continue
        try:
            decoded[name] = json.loads(value)
        except (TypeError, ValueError):
            # Left as text: a field that is genuinely a string in a union, or a
            # value an older build wrote differently. Validation reports it
            # better than a guess here would.
            pass
    decoded.pop("valid_for", None)  # storage bookkeeping, not part of the model
    return model.model_validate(decoded)


def _short(owner: str) -> str:
    """``pandajedi/jedibrokerage/AtlasProdJobBroker.py::doBrokerage`` -> the last two names."""
    where, sep, method = owner.partition("::")
    return where.rsplit("/", 1)[-1].removesuffix(".py") + sep + method


def _words(*sources: str) -> list[str]:
    """The words *sources* are made of, lower-cased, in order, deduplicated.

    Identifiers are split on case and on punctuation so that ``lowmemory``,
    ``low memory`` and ``LowMemory`` all reduce to the same two words.  Order is
    kept because it reads better in a report; matching does not depend on it.
    """
    found: list[str] = []
    for source in sources:
        for word in _WORD.findall(source or ""):
            lowered = word.lower()
            if lowered not in found:
                found.append(lowered)
    return found


class CodeMap:
    """A stored map, read one question at a time.

    Holds no cache.  A run answers a handful of questions against ~1000 nodes,
    and a cache would have to be invalidated when a rebuild lands underneath it
    -- which is precisely the staleness the separate cadences exist to make
    visible rather than to paper over.
    """

    def __init__(
        self,
        graph_db: GraphDatabaseClient,
        map_id: str = "panda",
        version: Optional[str] = None,
    ) -> None:
        """
        Args:
            graph_db: A connected client.
            map_id:   Which map. Distribution-scoped, so ``panda`` covers both
                      ``pandaserver`` and ``pandajedi``.
            version:  Pin to one ``derived_from``.  ``None`` reads whatever is
                      stored, which is right until an incident's own code
                      version is known -- see :meth:`versions`.
        """
        self._db = graph_db
        self.map_id = map_id
        self.version = version

    async def _find(self, model: Type[NodeT], **match: Any) -> list[NodeT]:
        props = await self._db.find_map_nodes(
            _MODELS[model].value,
            self.map_id,
            {k: v for k, v in match.items() if v is not None} or None,
            self.version,
        )
        if self.version is not None:
            # A pin selects which nodes that build had; it cannot select which
            # build's *content* they carry, because the store merges on the
            # semantic signature and the later write wins.  So a node present in
            # both builds comes back with the later one's conditions.  Said out
            # loud rather than left to be discovered: reading a months-old
            # incident against edited conditions is the skew the version stamp
            # exists to make visible, not a thing to hide behind a pin.
            elsewhere = sum(1 for p in props if p.get("derived_from") != self.version)
            if elsewhere:
                logger.warning(
                    "%d of %d %s node(s) were last written by a build other than %s, "
                    "so their content is that build's",
                    elsewhere, len(props), _MODELS[model].value, self.version,
                )
        return [_decode(model, p) for p in props]

    async def versions(self) -> list[str]:
        """Every source version stored for this map, newest name last.

        Several coexist on purpose: an incident from months ago has to be read
        against the code that was running then, not the code on disk now.
        """
        subjects = await self._db.find_map_nodes(
            NodeType.SUBJECT.value, self.map_id, None, None
        )
        return sorted({s["derived_from"] for s in subjects if s.get("derived_from")})

    async def subject(self, name: str) -> Optional[SubjectNode]:
        """One subject by its qualified name, e.g. ``JediTaskSpec.status``.

        Qualified because the same attribute name means different things in
        different classes -- ``FileSpec.status`` and ``JediFileSpec.status`` are
        not the same field.
        """
        found = await self._find(SubjectNode, name=name)
        return found[0] if found else None

    async def subjects(self) -> list[SubjectNode]:
        """Every subject, for callers that need to resolve a name themselves."""
        return sorted(await self._find(SubjectNode), key=lambda s: s.name)

    async def writers_of(self, subject: str) -> list[JunctionNode]:
        """Every junction that settles *subject*.

        The map's central question in one call: *which code decides this
        value?*  The answer is a set rather than one place, and its size is the
        system's real fan-out -- ``JediTaskSpec.status`` has 37 writers -- not
        an artefact of how the map was built.  An expert faces the same 37; the
        difference is that this enumeration is complete and precomputed.

        Pruning them is the caller's job and is done from observation, which is
        why this deliberately does not rank.
        """
        return sorted(
            await self._find(JunctionNode, subject=subject),
            key=lambda j: j.owner,
        )

    async def producers_of(self, subject: str, outcome: str) -> list[JunctionNode]:
        """The junctions that can produce *outcome*, and only those.

        The first elimination, and the cheapest: an observed value rules out
        every writer whose branch table cannot produce it.  Tier-2 branches --
        writer known, value settled at run time -- are kept, because "this one
        could have" is the honest answer for them and dropping them would turn
        an incomplete candidate set into a confident wrong one.

        Kept, that is, unless the branch's own text says otherwise.  A tier-2
        outcome is sometimes a frame with holes in it, and a frame bounds what
        can come out: ``runtime(f'merge_{s}')`` cannot have written
        ``es_inaction`` however ``s`` resolved.  That is a syntactic fact about
        the branch, not a completeness claim about some other node, which is
        what keeps it on the safe side of the paragraph above -- see
        :func:`~bamboo.codemap.models.outcome_excludes`.

        The same rule lives in :func:`~bamboo.codemap.strategy._reaching`, one
        level down, because this answers *which junctions* and that answers
        *which arms of one*.  Both are read by the same report and a reader
        comparing them would see a junction offered with no arm to look at.
        """
        return [
            junction
            for junction in await self.writers_of(subject)
            if any(
                b.outcome == outcome
                or (b.tier == 2 and not outcome_excludes(b.outcome, outcome))
                for b in junction.branches
            )
        ]

    async def selection_gates_for(self, subject: str) -> list[str]:
        """Tables that bound which rows the queries selecting *subject* can see.

        The second half of "will anything pick this up".  ``selected_values``
        answers whether a query asks for the observed value; this answers
        whether the row is inside what that query can reach at all, and a task
        can fail the second while passing the first.  One did: it sat in
        ``finishing`` while the query that rescues orphaned commands ran every
        cycle, because ``JEDI_AUX_Status_MinTaskID`` had stopped being updated
        and its watermark had risen above the task's id.

        Every table named here is one nothing in the map writes, so its
        freshness is not something any branch condition can account for --
        an unbound boundary, and the first thing to check when a query that
        should have matched did not.
        """
        found = await self.subject(subject)
        return list(found.selection_gates) if found else []

    async def carried_from(self, subject: str) -> dict[str, list[JunctionNode]]:
        """Where *subject*'s value is copied from, one hop back.

        A ``passthrough`` outcome says the value arrived from somewhere else, so
        the question moves to that field rather than ending.  Returns the field
        each junction points at, mapped to the junctions that write *that*
        field -- an empty list means the walk stops there, which is an answer:
        the value came in from outside anything this map explains.
        """
        upstream: dict[str, list[JunctionNode]] = {}
        for junction in await self.writers_of(subject):
            for branch in junction.branches:
                carried = PASSTHROUGH_OUTCOME.match(branch.outcome or "")
                if not carried:
                    continue
                field = carried.group("field")
                if field not in upstream:
                    upstream[field] = await self.writers_of(field)
        return upstream

    async def selections_by_owner(self) -> dict[str, list[tuple[str, str]]]:
        """``{owner: [(subject, value), …]}`` -- which function selects on what.

        The map records selections on the subject, keyed by value and then by
        the function that reads them; this turns that inside out so a caller
        holding a function can ask what it consults.  Needed by the one hop out
        of a junction: ``setScoutJobData_JEDI`` writes ``exhausted`` on an
        aggregate it never reads itself, and the only way to the rows behind
        that aggregate is the helper it calls.

        Deliberately keyed by the whole ``module::method``, not the bare name.
        A bare name is not an identity in this corpus -- ``run`` is defined in
        every daemon script, and matching on it once gave a single junction
        fourteen entry points of which thirteen were wrong.  The caller's edge
        is a ``self.<method>()`` call, which is within one module by
        construction, so the qualified key is both stricter and free.
        """
        by_owner: dict[str, list[tuple[str, str]]] = {}
        for subject in await self.subjects():
            for value, owners in (subject.selected_by or {}).items():
                for owner in owners:
                    by_owner.setdefault(owner, []).append((subject.name, value))
        return by_owner

    async def entity_reads_by_owner(self) -> dict[str, list[str]]:
        """``{owner: [the entities whose rows it selects]}``.

        The same shape as :meth:`selections_by_owner` and used at the same
        place, for the half of the answer a value cannot carry.  A helper that
        selects a task's jobs on the join key alone names no value at all --
        ``SELECT PandaID FROM jobsActive4 WHERE jediTaskID=:jediTaskID`` -- and
        was read as consulting nothing, when it is the whole of what "waiting
        on jobs" means.

        Reads only ``read_by``.  A descent asks about the rows a decision was
        taken *from*; a function that updates those rows is a different claim
        and following it would walk forwards while claiming to walk down.
        """
        by_owner: dict[str, list[str]] = {}
        for entity in await self._find(EntityNode):
            for owner in entity.read_by:
                by_owner.setdefault(owner, []).append(entity.name)
        return {owner: sorted(names) for owner, names in by_owner.items()}

    async def changing_functions(self) -> set[str]:
        """Every function this map has seen change anything.

        The mirror of :meth:`entity_reads_by_owner`, and read for the one thing
        a read cannot say: whether a function named as a *reader* of a value is
        also somewhere the row could go on from.  A metrics daemon selects
        ``cancelled`` jobs to average a wait time and appears in none of these
        lists, so being selected by it leads nowhere -- which is the opposite of
        what a verdict built from "a query selects this value" alone told a
        reader.

        Two kinds of changing, unioned, because the question is whether
        anything happens here at all and either answers yes.  Rows: all three
        write verbs together, unlike ``read_by`` which is kept apart -- the
        distinction they carry, arriving and moving and leaving, is about rows
        of one kind and not about this.  Values: every junction owner, because
        settling a value *is* changing something and nine functions in this
        corpus settle one without an entity recording a write -- ``updateJob``
        and ``updateJobStatus`` among them, which called inert would be plainly
        false.
        """
        return {
            owner
            for entity in await self._find(EntityNode)
            for owner in (*entity.created_by, *entity.updated_by, *entity.deleted_by)
        } | {junction.owner for junction in await self._find(JunctionNode)}

    async def entities_for(self, qualifier: str) -> list[EntityNode]:
        """The kinds of row a subject's qualifier names.

        A subject is qualified by its spec class where one was learned and by
        its table otherwise, so both spellings have to resolve.  A class can
        name several: ``JobSpec`` covers ``jobsDefined4``, ``jobsActive4`` and
        ``jobsArchived4``, and for a row's *creation* that is the useful
        answer rather than an ambiguity -- a job row is made in exactly one of
        them, and which one is the fact.
        """
        folded = qualifier.lower()
        found = [
            entity
            for entity in await self._find(EntityNode)
            if entity.name == folded or (entity.spec_class or "") == qualifier
        ]
        return sorted(found, key=lambda entity: entity.name)

    async def chain(self, owner: str) -> list[FilterStageNode]:
        """One brokerage chain's stages, in the order the source runs them.

        Separate from junctions because brokerage never picks a site: it starts
        with every candidate and narrows about twenty-five times, so every stage
        runs and each removes some.  "The distribution is wrong" is answered by
        asking which stage threw the candidates away, not which branch fired.
        """
        return sorted(await self._find(FilterStageNode, owner=owner), key=lambda s: s.order)

    async def chains(self) -> dict[str, list[FilterStageNode]]:
        """Every chain, keyed by the function that holds it."""
        chains: dict[str, list[FilterStageNode]] = {}
        for stage in await self._find(FilterStageNode):
            chains.setdefault(stage.owner, []).append(stage)
        return {
            owner: sorted(stages, key=lambda s: s.order)
            for owner, stages in sorted(chains.items())
        }

    async def stage_for_tag(self, tag: str) -> list[FilterStageNode]:
        """The stages that emit ``criteria=-<tag>``, straight from a log line.

        Production writes the tag per rejected site, so this turns something
        already in hand into the code that put it there, with no search.
        """
        return await self._find(FilterStageNode, criteria_tag=tag)

    async def vocabulary(self) -> list[MapTerm]:
        """Everything this map can be asked about, with the symptom each becomes.

        The thing that makes a free-text description tractable without a
        search.  A source navigator turns "the jobs all went to two sites" into
        grep terms and ranks thirty files, and fails as no-candidates,
        too-many-candidates or irrelevant; against this the same description is
        matched into a set that is **closed and enumerable**.  Measured on the
        stored PanDA map: 296 ``(subject, value)`` pairs over 62 of 112
        subjects, 70 cut tags, 49 funnel steps, 5 chains -- 420 entries.

        Values come from both directions and are unioned.  A branch that states
        the value says the system can *produce* it; ``selected_values`` says
        some query *asks* for it.  Either makes the value something to be asked
        about, and a value only the second knows -- one settled at run time and
        never written as a literal -- would be missing from a list built from
        branches alone.
        """
        terms: list[MapTerm] = []
        writers: dict[str, list[JunctionNode]] = {}
        for junction in await self._find(JunctionNode):
            writers.setdefault(junction.subject, []).append(junction)
        for subject in await self.subjects():
            produced = {
                branch.outcome
                for junction in writers.get(subject.name, [])
                for branch in junction.branches
                if branch.tier == 1 and branch.outcome and not _DERIVED.match(branch.outcome)
            }
            for value in sorted(produced | set(subject.selected_values)):
                terms.append(
                    MapTerm(
                        kind=TERM_VALUE,
                        key=f"{subject.name}={value}",
                        words=_words(subject.attribute, subject.spec_class, value),
                        symptom=Symptom(
                            kind=SYMPTOM_VALUE, subject=subject.name, observed=value
                        ),
                    )
                )
        stages = await self._find(FilterStageNode)
        # A cut carries its step's words as well.  Production spells the reason
        # ``-lowmemory`` and a person says "memory", so matching the tag alone
        # would miss the wording everyone actually uses.
        labels: dict[str, str] = {s.criteria_tag: s.funnel_label for s in stages if s.criteria_tag}
        for tag in sorted(labels):
            terms.append(
                MapTerm(
                    kind=TERM_CUT,
                    key=tag,
                    words=_words(tag, labels[tag]),
                    symptom=Symptom(kind=SYMPTOM_DISTRIBUTION, focus=tag),
                )
            )
        for label in sorted({s.funnel_label for s in stages if s.funnel_label}):
            terms.append(
                MapTerm(
                    kind=TERM_STEP,
                    key=label,
                    words=_words(label),
                    symptom=Symptom(kind=SYMPTOM_DISTRIBUTION, focus=label),
                )
            )
        for owner in sorted({s.owner for s in stages}):
            terms.append(
                MapTerm(
                    kind=TERM_CHAIN,
                    key=owner,
                    # The directory and the extension are dropped, for the reason
                    # ``short_owner`` drops them from the report: they carry
                    # nothing a reader uses, and here they would be six words of
                    # ``pandajedi jedibrokerage py`` diluting the four that mean
                    # something.
                    words=_words(_short(owner), *_BROKERAGE_WORDS),
                    symptom=Symptom(kind=SYMPTOM_DISTRIBUTION, focus=owner),
                )
            )
        return terms

    async def log_files_for(self, subject: str) -> dict[str, dict[str, list[str]]]:
        """Which log to read for each writer of *subject*.

        The question the whole design is pointed at.  It cannot be answered from
        the package a junction lives in: JEDI opens its own TaskBuffer, so a
        ``db_proxy_mods`` write runs inside the JEDI process and lands in JEDI's
        log, while the same package's ``api/v1`` code lands in the server's.

        Two lists per writer, ``own`` and ``caller``, kept apart because they
        answer different questions and a caller often gives the better answer.
        Eleven of the eighteen junctions that can write ``pending`` are proxy
        methods whose ``own`` list is the same two files, which makes them look
        indistinguishable; their callers name nine different logs, and it is the
        caller that writes ``set task_status=`` in production.  Both empty means
        nothing in the source names a file -- reported, not guessed.
        """
        return {
            j.owner: {"own": list(j.log_files), "caller": list(j.caller_log_files)}
            for j in await self.writers_of(subject)
        }

    async def log_sites(self, owners: list[str]) -> dict[str, LogSiteNode]:
        """The log rows for *owners* that settle nothing themselves.

        Junctions are not included, and the caller should look there first:
        an owner that decides a value carries the same three fields on its
        junction, and returning it from both places is how one copy comes to
        disagree with the other.
        """
        if not owners:
            return {}
        wanted = set(owners)
        return {
            site.owner: site
            for site in await self._find(LogSiteNode)
            if site.owner in wanted
        }
