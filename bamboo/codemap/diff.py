"""What changed between two builds of the same map.

Every other check compares the map with something else -- the source it came
from, or the logs the deployment writes.  Neither catches a threshold moving.
A comparison against the source agrees with whatever the source now says, and
a condition is not echoed to any log, so a build made from the wrong release
will explain a decision using a number that has since changed and sound
entirely convincing doing it.  Two builds compared against each other is the
only thing that shows it.

**Nodes are matched by semantic signature, never by position.**  That choice
was made when ``update_job`` moved from ``JobDispatcher.py:975`` to
``api/v1/pilot_api.py:332`` between two releases -- a key of file and line
would have called that one boundary two, or lost it.  Here the same decision
pays twice: matching by signature is what lets a moved-but-unchanged node be
reported as exactly that, which is the common case in a refactor and the one
that should generate no noise at all.

**Condition drift is the headline.**  A changed ``path_condition`` or filter
``conditions`` is the failure the gates are blind to, so it is separated from
everything else rather than listed among renamed fields and new columns.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from bamboo.codemap.models import MapFragment
from bamboo.models.graph_element import NodeType

# Keyed off the enum rather than spelled out, because spelling them out is how
# this went wrong once: the label is ``JunctionPoint`` and a hand-written
# "Junction" matched nothing, so junction branches were never compared and
# every junction reported as unchanged whatever had happened to it.
_JUNCTION = NodeType.JUNCTION_POINT.value

# Fields that describe what a node *does*, per kind.  Everything else is
# identity, provenance or position: ``anchor`` moves whenever a line is added
# above it, ``derived_from`` differs by definition, and ``references`` is a
# count that churns with edits elsewhere in the file.  Including any of them
# would bury the changes that matter in changes that never do.
CONTENT_FIELDS: dict[str, tuple[str, ...]] = {
    NodeType.VALUE_ENUM.value: ("namespace", "constant", "value", "comment"),
    NodeType.SUBJECT.value: (
        "spec_class",
        "qualifier_kind",
        "attribute",
        "criteria",
        "vocabulary",
        "selected_values",
    ),
    _JUNCTION: ("subject", "owner", "log_files", "attribution", "structural_subject"),
    NodeType.BOUNDARY.value: (
        "system",
        "kind",
        "transport",
        "interface",
        "carried_values",
        "handed_over",
        "operations",
        "access_conditions",
        "observable_values",
        "accepts_arbitrary",
        "version_binding",
        "resolution",
    ),
    NodeType.FILTER_STAGE.value: (
        "owner",
        "criteria_tag",
        "funnel_label",
        "order",
        "conditions",
        "inputs",
        "emits",
        "log_level",
        "log_files",
    ),
}

# The two fields that hold a decision's reasoning.  A change here is what no
# other check can see, so it is reported on its own.
DRIFT_FIELDS = frozenset({"path_condition", "conditions"})

# Long values are truncated for display; the point of a diff line is to say
# *that* something moved and roughly where, not to reproduce both versions.
_RENDER_LIMIT = 120


def _render(value: Any) -> str:
    """A stable, comparable rendering of a field value."""
    if isinstance(value, (list, tuple)):
        text = "[" + ", ".join(_render(item) for item in value) + "]"
    elif isinstance(value, BaseModel):
        text = _render(value.model_dump())
    elif isinstance(value, dict):
        text = "{" + ", ".join(f"{k}={_render(v)}" for k, v in sorted(value.items())) + "}"
    else:
        text = str(value)
    return text


def _abbreviate(text: str) -> str:
    return text if len(text) <= _RENDER_LIMIT else text[: _RENDER_LIMIT - 1] + "…"


class Change(BaseModel):
    """One behavioural difference in a node both builds contain."""

    node: str
    kind: str
    field: str
    before: str
    after: str

    @property
    def is_drift(self) -> bool:
        return self.field in DRIFT_FIELDS

    def render(self) -> str:
        return (
            f"{self.node} [{self.field}]\n"
            f"      - {_abbreviate(self.before)}\n"
            f"      + {_abbreviate(self.after)}"
        )


class MapDiff(BaseModel):
    """Everything that differs between two builds."""

    old_version: str
    new_version: str
    added: list[str] = Field(default_factory=list)
    removed: list[str] = Field(default_factory=list)
    changes: list[Change] = Field(default_factory=list)
    moved: list[str] = Field(default_factory=list)
    unchanged: int = 0

    def drift(self) -> list[Change]:
        """Changes to the reasoning behind a decision."""
        return [c for c in self.changes if c.is_drift]

    def other(self) -> list[Change]:
        return [c for c in self.changes if not c.is_drift]

    @property
    def identical(self) -> bool:
        return not (self.added or self.removed or self.changes)


def _key(node: Any) -> tuple[str, str]:
    """Identity: the node kind and its semantic signature.

    Kind is part of the key because the signatures are only unique within one
    -- a subject and the junction that writes it can share a name.
    """
    return (str(node.node_type.value), node.name)


def _index(fragment: MapFragment) -> dict[tuple[str, str], Any]:
    nodes: dict[tuple[str, str], Any] = {}
    for group in (
        fragment.value_enums,
        fragment.subjects,
        fragment.junctions,
        fragment.boundaries,
        fragment.filter_stages,
    ):
        for node in group:
            nodes[_key(node)] = node
    return nodes


def _branch_changes(name: str, old: Any, new: Any) -> list[Change]:
    """Compare a junction's branches, matched by outcome.

    Branch identity is ``(owner, outcome)`` -- the plan's choice, and the one
    that makes drift visible: a branch keeps its outcome while the condition
    that reaches it is edited, which is exactly the case a positional
    comparison would render as "everything after line 40 changed".
    """
    before = {b.outcome: b for b in old.branches}
    after = {b.outcome: b for b in new.branches}
    changes: list[Change] = []
    for outcome in sorted(set(before) - set(after)):
        changes.append(
            Change(node=name, kind=_JUNCTION, field="branch", before=outcome, after="—")
        )
    for outcome in sorted(set(after) - set(before)):
        changes.append(
            Change(node=name, kind=_JUNCTION, field="branch", before="—", after=outcome)
        )
    for outcome in sorted(set(before) & set(after)):
        for field in ("path_condition", "tier", "emits", "log_level"):
            old_value = _render(getattr(before[outcome], field))
            new_value = _render(getattr(after[outcome], field))
            if old_value != new_value:
                changes.append(
                    Change(
                        node=f"{name} → {outcome}",
                        kind=_JUNCTION,
                        field=field,
                        before=old_value,
                        after=new_value,
                    )
                )
    return changes


def _entry_changes(name: str, old: Any, new: Any) -> list[Change]:
    """Compare how a junction is reached.

    A junction losing its polled entry is not a cosmetic difference: it is the
    difference between a stall that clears itself and one that does not.
    """
    before = {(e.trigger, e.entry) for e in old.entry_points}
    after = {(e.trigger, e.entry) for e in new.entry_points}
    if before == after:
        return []
    return [
        Change(
            node=name,
            kind=_JUNCTION,
            field="entry_points",
            before=_render(sorted(before)),
            after=_render(sorted(after)),
        )
    ]


def _node_changes(kind: str, name: str, old: Any, new: Any) -> list[Change]:
    if kind not in CONTENT_FIELDS:
        # Silence here is the dangerous failure: a kind nobody listed compares
        # equal to itself no matter what changed, and the diff then reports a
        # rewritten node as unchanged.  A new node kind must add its fields.
        raise KeyError(f"{kind} has no content fields; add them to CONTENT_FIELDS")
    changes: list[Change] = []
    for field in CONTENT_FIELDS[kind]:
        old_value = _render(getattr(old, field, None))
        new_value = _render(getattr(new, field, None))
        if old_value != new_value:
            changes.append(
                Change(node=name, kind=kind, field=field, before=old_value, after=new_value)
            )
    if kind == _JUNCTION:
        changes.extend(_branch_changes(name, old, new))
        changes.extend(_entry_changes(name, old, new))
    return changes


def _anchor_ref(node: Any) -> Optional[str]:
    anchor = getattr(node, "anchor", None)
    return anchor.as_ref() if anchor else None


def compare(old: MapFragment, new: MapFragment) -> MapDiff:
    """Diff two builds of the same map.

    Both sides are built from source by the same plugin, so anything that
    differs here is a difference in the code -- which is what makes the result
    readable as "the release changed this" rather than "the extractor is
    unstable".
    """
    before = _index(old)
    after = _index(new)
    diff = MapDiff(old_version=old.derived_from, new_version=new.derived_from)

    for key in sorted(set(before) - set(after)):
        diff.removed.append(f"{key[0]} {key[1]}")
    for key in sorted(set(after) - set(before)):
        diff.added.append(f"{key[0]} {key[1]}")

    for key in sorted(set(before) & set(after)):
        kind, name = key
        changes = _node_changes(kind, name, before[key], after[key])
        if changes:
            diff.changes.extend(changes)
        elif _anchor_ref(before[key]) != _anchor_ref(after[key]):
            # The case the semantic signature exists for: the code moved and
            # says the same thing.  Counted rather than listed one by one,
            # because in a refactor this is most of the map.
            diff.moved.append(f"{kind} {name}")
        else:
            diff.unchanged += 1
    return diff
