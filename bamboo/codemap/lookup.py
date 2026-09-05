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
    BoundaryNode,
    FilterStageNode,
    JunctionNode,
    SubjectNode,
    ValueEnumNode,
)
from bamboo.database.graph_database_client import GraphDatabaseClient
from bamboo.models.graph_element import BaseNode, NodeType

logger = logging.getLogger(__name__)

NodeT = TypeVar("NodeT", bound=BaseNode)

_MODELS: dict[Type[BaseNode], NodeType] = {
    SubjectNode: NodeType.SUBJECT,
    JunctionNode: NodeType.JUNCTION_POINT,
    FilterStageNode: NodeType.FILTER_STAGE,
    BoundaryNode: NodeType.BOUNDARY,
    ValueEnumNode: NodeType.VALUE_ENUM,
}

#: ``passthrough(JediTaskSpec.oldStatus)`` -> ``JediTaskSpec.oldStatus``.
_PASSTHROUGH = re.compile(r"^passthrough\((?P<field>[^)]+)\)$")


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
        """
        return [
            junction
            for junction in await self.writers_of(subject)
            if any(b.outcome == outcome or b.tier == 2 for b in junction.branches)
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
                carried = _PASSTHROUGH.match(branch.outcome or "")
                if not carried:
                    continue
                field = carried.group("field")
                if field not in upstream:
                    upstream[field] = await self.writers_of(field)
        return upstream

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
