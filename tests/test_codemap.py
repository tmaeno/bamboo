"""Code Map extraction, gates, and namespace-scoped storage.

The value-enumeration slice is the map's simplest: the code declares a name, a
value and usually a comment, all in a fixed shape, so nothing has to be
inferred.  These tests pin the parts that were *not* obvious and that real
PanDA source forced corrections to:

* what counts as one enumeration (a module can declare two, with overlapping
  values, and the shared name prefix is the only thing that separates them);
* that a constant nothing reads is not an index entry;
* that rebuilding a Code Map cannot take the incident graph with it.
"""

from __future__ import annotations

import ast
from collections import Counter
from unittest.mock import AsyncMock

import pytest

from bamboo.codemap import diff, evidence, gates
from bamboo.codemap.gitsource import blob_sha
from bamboo.codemap.models import (
    Anchor,
    BoundaryNode,
    Branch,
    EntryPoint,
    EnumerationWrite,
    FilterStageNode,
    JunctionNode,
    MapFragment,
    SourceModule,
    SubjectNode,
    ValueEnumNode,
)
from bamboo.codemap.panda import attribution, pathcond, promotion, sql, values
from bamboo.codemap.panda.recognizers import (
    alias,
    boundary,
    errorcode,
    logfile,
    progress,
    selection,
    sqlwrite,
    trigger,
)
from bamboo.models.graph_element import NodeType
from bamboo.scripts import check_map

MAP_ID = "panda"
VERSION = "panda-server-source 1.0.2"


def _module(source: str, rel: str = "pandaserver/taskbuffer/ErrorCode.py") -> SourceModule:
    """Build the parsed module a recognizer takes."""
    return SourceModule(
        package=rel.split("/")[0],
        rel_path=rel,
        tree=ast.parse(source),
        source=source,
        blob_sha=blob_sha(source),
    )


def _extract(source: str, rel: str = "pandaserver/taskbuffer/ErrorCode.py"):
    return errorcode.extract([_module(source, rel)], MAP_ID, VERSION)


# --------------------------------------------------------------------------- #
# extraction
# --------------------------------------------------------------------------- #


def test_extracts_name_value_and_comment():
    """The comment above a constant is its only human-readable meaning.

    ``ast`` throws comments away, so they are recovered from the source text;
    without them an index entry is a bare number with no explanation.
    """
    enums, _ = _extract("# error code\n\n# killed\nEC_Kill = 100\n")

    assert len(enums) == 1
    entry = enums[0]
    assert entry.constant == "EC_Kill"
    assert entry.value == 100
    assert entry.comment == "killed"
    assert entry.node_type is NodeType.VALUE_ENUM
    assert entry.map_id == MAP_ID
    assert entry.derived_from == VERSION


def test_comment_does_not_leak_across_a_blank_line():
    """A blank line ends one entry's comment; otherwise it absorbs its neighbour's."""
    enums, _ = _extract("# killed\nEC_Kill = 100\n\nEC_Transfer = 101\n")

    by_name = {e.constant: e for e in enums}
    assert by_name["EC_Kill"].comment == "killed"
    assert by_name["EC_Transfer"].comment is None


def test_prefix_separates_two_enumerations_in_one_module():
    """One module can declare two enumerations whose values overlap.

    ``EventServiceUtils`` defines event states *and* task types, so ``0`` means
    two different things in one file.  Python has no enum block here, so the
    shared prefix is how the code states the grouping -- keying on the module
    alone would make ``(namespace, value)`` ambiguous.
    """
    source = (
        "ST_ready = 0\nST_sent = 1\n"
        "TASK_NORMAL = 0\nTASK_EVENT_SERVICE = 1\n"
    )
    enums, _ = _extract(source, "pandaserver/taskbuffer/EventServiceUtils.py")

    namespaces = {e.constant: e.namespace for e in enums}
    assert namespaces["ST_ready"] != namespaces["TASK_NORMAL"]
    assert namespaces["ST_ready"] == namespaces["ST_sent"]
    assert namespaces["ST_ready"].endswith(".ST")
    # The display name stays module-scoped so it does not read as "...EC.EC_Kill".
    assert {e.name for e in enums} == {
        "taskbuffer.EventServiceUtils.ST_ready",
        "taskbuffer.EventServiceUtils.ST_sent",
        "taskbuffer.EventServiceUtils.TASK_NORMAL",
        "taskbuffer.EventServiceUtils.TASK_EVENT_SERVICE",
    }


def test_lone_prefixed_constant_is_not_given_its_own_namespace():
    """A prefix only groups when several constants share it."""
    enums, _ = _extract("MESSAGE_JSON = 'bad json'\nLATENCY = 'latency'\n",
                        "pandaserver/api/v1/common.py")

    assert {e.namespace for e in enums} == {"api.common"}


def test_declared_value_set_is_not_an_index_entry():
    """A collection literal is a vocabulary oracle, not a name<->value pair.

    Counting the two together makes an error-code module and a config module
    look like the same kind of file, which is what hid the distinction in the
    first place.
    """
    source = "EC_Kill = 100\nFINAL_TASK_STATUSES = ['done', 'failed']\n"
    enums, coverage = _extract(source)

    assert [e.constant for e in enums] == ["EC_Kill"]
    # Both are declarations, so both are candidates -- the value set shows up
    # as an unexplained candidate rather than vanishing from the denominator.
    assert coverage[0].candidates == 2
    assert coverage[0].explained == 1


def test_computed_constants_stay_out_of_the_denominator():
    """A computed value is not a failed extraction, it is a different statement."""
    source = "EC_Kill = 100\nTIMEOUT = compute_timeout()\nLIMIT = 3 * 60\n"
    enums, coverage = _extract(source)

    assert [e.constant for e in enums] == ["EC_Kill"]
    assert coverage[0].candidates == 1
    assert coverage[0].ratio == 1.0


def test_references_are_counted_across_the_whole_corpus():
    """A constant belongs in a decoding index only if something reads it."""
    definition = _module("EC_Kill = 100\nEC_Unused = 999\n")
    user = _module(
        "from x import ErrorCode\nif code == ErrorCode.EC_Kill:\n    pass\n",
        "pandaserver/dataservice/user.py",
    )
    enums, _ = errorcode.extract([definition, user], MAP_ID, VERSION)

    by_name = {e.constant: e for e in enums}
    assert by_name["EC_Kill"].references >= 1
    assert by_name["EC_Unused"].references == 0


# --------------------------------------------------------------------------- #
# gates
# --------------------------------------------------------------------------- #


def _enum(constant: str, value, namespace: str = "taskbuffer.ErrorCode.EC", references: int = 1):
    return ValueEnumNode(
        map_id=MAP_ID,
        derived_from=VERSION,
        name=f"{namespace}.{constant}",
        namespace=namespace,
        constant=constant,
        value=value,
        references=references,
        anchor=Anchor(package="pandaserver", file="x.py", line_start=1),
    )


def test_unreferenced_constants_are_flagged():
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        value_enums=[_enum("EC_Kill", 100), _enum("EC_Dead", 206, references=0)],
    )
    result = gates.value_enum_referenced(fragment)

    assert not result.passed
    assert result.checked == 2
    assert any("EC_Dead" in f for f in result.failures)


def test_reuse_across_namespaces_is_allowed():
    """Reused numbers are the reason the key carries a namespace at all."""
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        value_enums=[
            _enum("EC_Kill", 100, "taskbuffer.ErrorCode.EC"),
            _enum("EC_Watcher", 100, "jobdispatcher.ErrorCode.EC"),
        ],
    )
    assert gates.namespace_disambiguates(fragment).passed


def test_collision_inside_one_namespace_is_rejected():
    """Within one namespace a repeated value makes the key identify nothing."""
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        value_enums=[
            _enum("ST_ready", 0, "taskbuffer.EventServiceUtils"),
            _enum("TASK_NORMAL", 0, "taskbuffer.EventServiceUtils"),
        ],
    )
    result = gates.namespace_disambiguates(fragment)

    assert not result.passed
    assert "ST_ready" in result.failures[0]


# --------------------------------------------------------------------------- #
# identity
# --------------------------------------------------------------------------- #


def test_content_hash_ignores_the_anchor():
    """A gloss keyed on position would be thrown away by unrelated edits.

    Between two PanDA releases the pilot boundary moved file entirely while
    remaining the same boundary, so anything derived from a junction has to be
    keyed on what it means, not where it was found.
    """
    branches = [Branch(outcome="assigning", path_condition=["cloud is None"])]
    here = JunctionNode(
        map_id=MAP_ID, derived_from=VERSION, name="n", subject="JediTaskSpec.status",
        owner="m::f", branches=branches,
        anchor=Anchor(package="p", file="a.py", line_start=10),
    )
    moved = JunctionNode(
        map_id=MAP_ID, derived_from=VERSION, name="n", subject="JediTaskSpec.status",
        owner="m::f", branches=branches,
        anchor=Anchor(package="p", file="b.py", line_start=999),
    )
    assert here.content_hash() == moved.content_hash()


def test_content_hash_changes_when_the_condition_changes():
    common = {
        "map_id": MAP_ID,
        "derived_from": VERSION,
        "name": "n",
        "subject": "JediTaskSpec.status",
        "owner": "m::f",
    }
    before = JunctionNode(branches=[Branch(outcome="ready", path_condition=["a > 1"])], **common)
    after = JunctionNode(branches=[Branch(outcome="ready", path_condition=["a > 2"])], **common)

    assert before.content_hash() != after.content_hash()


# --------------------------------------------------------------------------- #
# storage namespace
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_store_fragment_replaces_only_its_own_version():
    """A rebuild clears this map at this version, never the incident graph.

    ``clear_all`` would take human-validated incident knowledge with it, and
    other versions must survive so an old incident can still be explained
    against the code that was running when it happened.
    """
    fragment = MapFragment(
        map_id=MAP_ID, derived_from=VERSION, value_enums=[_enum("EC_Kill", 100)]
    )
    graph_db = AsyncMock()
    graph_db.clear_map.return_value = 3

    from bamboo.codemap.store import store_fragment

    written = await store_fragment(fragment, graph_db)

    graph_db.clear_map.assert_awaited_once_with(MAP_ID, VERSION)
    graph_db.clear_all.assert_not_awaited()
    assert written["value_enums"] == 1
    assert graph_db.merge_map_node.await_count == 1


@pytest.mark.asyncio
async def test_store_fragment_can_keep_existing_versions():
    fragment = MapFragment(map_id=MAP_ID, derived_from=VERSION, value_enums=[_enum("EC_Kill", 100)])
    graph_db = AsyncMock()

    from bamboo.codemap.store import store_fragment

    await store_fragment(fragment, graph_db, replace_version=False)

    graph_db.clear_map.assert_not_awaited()


# --------------------------------------------------------------------------- #
# version stamping
# --------------------------------------------------------------------------- #


def test_blob_sha_matches_git_for_lf_source():
    """The hash equals ``git rev-parse HEAD:<path>`` for the same bytes.

    That equality is what lets a map built from a checkout be compared file by
    file with one built from the installed package.
    """
    # `git hash-object` on b"hello\n" -- the canonical example.
    assert blob_sha("hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"


def test_blob_sha_tracks_content_not_position():
    assert blob_sha("EC_Kill = 100\n") != blob_sha("EC_Kill = 101\n")
    assert blob_sha("EC_Kill = 100\n") == blob_sha("EC_Kill = 100\n")


def test_anchors_carry_the_content_hash():
    """Without it a rebuild cannot tell whether the enclosing file moved on."""
    enums, _ = _extract("# killed\nEC_Kill = 100\n")

    assert enums[0].anchor is not None
    assert enums[0].anchor.blob_sha == blob_sha("# killed\nEC_Kill = 100\n")


def test_describe_returns_none_outside_a_git_checkout(tmp_path):
    """Not being a checkout is a source form, not a failure.

    An installed distribution has no repository; the caller falls back to the
    identity that form does offer rather than the build refusing to run.
    """
    from bamboo.codemap.gitsource import describe

    assert describe(tmp_path) is None


def test_code_map_labels_are_distinct_from_incident_labels():
    """Label separation is what makes a namespace-scoped delete possible."""
    code_map = {NodeType.JUNCTION_POINT, NodeType.BOUNDARY, NodeType.SUBJECT, NodeType.VALUE_ENUM}
    incident = {NodeType.SYMPTOM, NodeType.CAUSE, NodeType.RESOLUTION, NodeType.PROCEDURE}

    assert not {n.value for n in code_map} & {n.value for n in incident}


# --------------------------------------------------------------------------- #
# boundaries
# --------------------------------------------------------------------------- #


_ENDPOINT = '''
@request_validation(_logger, secure=True, production=True, request_method="POST")
def update_job(req: PandaRequest, job_id: int, job_status: str, pilot_id: str = None):
    _logger.debug(f"update_job({job_id}, {job_status})")
    return None
'''


def _boundaries(source: str, rel: str = "pandaserver/api/v1/pilot_api.py"):
    return boundary.extract([_module(source, rel)], MAP_ID, VERSION)[0]


def test_endpoint_is_recognised_by_its_declaration():
    """The decorator marks the entry point; a ``req`` parameter only hints at it."""
    found = _boundaries(_ENDPOINT)

    assert len(found) == 1
    assert found[0].interface == "pilot_api::update_job"
    assert found[0].system == "pilot"
    assert found[0].node_type is NodeType.BOUNDARY


def test_request_object_is_not_a_carried_value():
    """``req`` is the transport, not something the far side supplies."""
    assert _boundaries(_ENDPOINT)[0].carried_values == ["job_id", "job_status", "pilot_id"]


def test_access_conditions_come_from_the_declaration():
    """These are the first place a request can be rejected.

    A command that appears to have vanished may have been turned away here,
    before any junction saw it -- so the boundary has to carry them.
    """
    assert _boundaries(_ENDPOINT)[0].access_conditions == {
        "secure": True,
        "production": True,
        "request_method": "POST",
    }


def test_only_logged_values_are_observable():
    """What crossed and was never written down cannot be recovered afterwards."""
    found = _boundaries(_ENDPOINT)[0]

    assert found.observable_values == ["job_id", "job_status"]
    assert "pilot_id" not in found.observable_values


def test_version_binding_is_picked_out_of_the_payload():
    """A system whose version moves independently cannot share this map's stamp."""
    assert _boundaries(_ENDPOINT)[0].version_binding == ["pilot_id"]


def test_endpoint_carrying_nothing_is_not_a_boundary():
    """An introspection endpoint is an entry point but nothing crosses it."""
    source = (
        '@request_validation(_logger, secure=True)\n'
        'def is_alive(req: PandaRequest):\n'
        '    return {"alive": True}\n'
    )
    assert _boundaries(source, "pandaserver/api/v1/system_api.py") == []


def test_arbitrary_kwargs_are_recorded_rather_than_ignored():
    """Listing no carried values here would claim nothing crosses -- the opposite."""
    source = (
        '@request_validation(_logger, secure=True)\n'
        'def get_attributes(req: PandaRequest, **kwargs):\n'
        '    return {}\n'
    )
    found = _boundaries(source, "pandaserver/api/v1/system_api.py")

    assert len(found) == 1
    assert found[0].accepts_arbitrary is True
    assert found[0].carried_values == []


def test_callers_without_their_own_release_cycle_are_not_separate_systems():
    """Crossing a trust boundary is not the same as crossing a version boundary."""
    found = _boundaries(_ENDPOINT, "pandaserver/api/v1/task_api.py")

    assert found[0].system == "client"


def test_ownership_gate_catches_a_parameter_the_signature_lacks():
    """The decorator and the signature must agree on which parameter carries the task."""
    source = (
        '@request_validation(_logger, task_owner=True, task_id_param="task_id")\n'
        'def kill(req: PandaRequest, jedi_task_id: int):\n'
        '    return None\n'
    )
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        boundaries=_boundaries(source, "pandaserver/api/v1/task_api.py"),
    )
    result = gates.boundary_ownership_param_declared(fragment)

    assert not result.passed
    assert "task_id" in result.failures[0]


# --------------------------------------------------------------------------- #
# progress: attribution and path conditions
# --------------------------------------------------------------------------- #

# Two classes declaring ``status`` is the situation the whole attribution layer
# exists for; ``jobStatus`` is the control, declared once and so never in doubt.
_SPECS = """
class JediTaskSpec(object):
    _attributes = ("jediTaskID", "status", "oldStatus")

    def statusToUpdateContents(cls):
        return ["defined", "ready"]

class JediFileSpec(object):
    _attributes = ("fileID", "status", "proc_status")

class FileSpec(object):
    _attributes = ("lfn", "status")

class JobSpec(object):
    _attributes = ("PandaID", "jobStatus")
"""


def _progress(source: str, rel: str):
    """Extract from *source* alongside the spec declarations it writes to."""
    modules = [
        _module(_SPECS, "pandaserver/taskbuffer/Specs.py"),
        _module(source, rel),
    ]
    subjects, junctions, coverage, _diag, _enum = progress.extract(modules, MAP_ID, VERSION)
    return subjects, [j for j in junctions if j.owner.startswith(rel)], coverage


def test_single_declaring_class_needs_no_object_type():
    """One declaring class settles the subject whatever the object expression is.

    ``self.job.jobStatus`` has an attribute, not a name, on the left -- there is
    no variable to resolve.  It does not matter: only ``JobSpec`` declares
    ``jobStatus``, so the write can only be to a ``JobSpec``.
    """
    source = "def f(self):\n    self.job.jobStatus = 'failed'\n"
    _subjects, junctions, _cov = _progress(source, "pandaserver/dataservice/adder_gen.py")

    assert [(j.subject, j.attribution) for j in junctions] == [("JobSpec.jobStatus", "certain")]


def test_constructor_call_attributes_the_write():
    """The code states the type outright, so nothing is guessed."""
    source = (
        "def f():\n"
        "    spec = JediFileSpec()\n"
        "    spec.status = 'ready'\n"
    )
    _subjects, junctions, _cov = _progress(source, "pandajedi/jedirefine/TaskRefinerBase.py")

    assert junctions[0].subject == "JediFileSpec.status"
    assert junctions[0].attribution == "certain"


def test_a_name_and_an_import_are_not_evidence():
    """Only the import points at ``JediFileSpec`` here, and that is not enough.

    Attributing from the variable's name -- narrowed by what the module
    imports -- used to settle writes like this one.  It was removed: across the
    whole of PanDA it decided two writes out of 278, and it was the only basis
    whose answers had to be marked as untrusted, since importing a class is not
    evidence that this particular variable holds one.

    The remedy for a shape like this is a type annotation upstream, not another
    rule here.
    """
    source = (
        "from pandaserver.taskbuffer.JediFileSpec import JediFileSpec\n"
        "def f(files):\n"
        "    for file in files:\n"
        "        file.status = 'ready'\n"
    )
    _subjects, junctions, _cov = _progress(source, "pandajedi/jediorder/JobGenerator.py")

    assert junctions[0].attribution == "unresolved"


def test_an_annotation_settles_what_the_name_could_not():
    """The intended remedy, and it lands in the strongest basis.

    One line of standard Python -- useful to mypy and a reader either way --
    replaces the inference rule that was removed.
    """
    source = (
        "from pandaserver.taskbuffer.JediFileSpec import JediFileSpec\n"
        "def f(file: JediFileSpec):\n"
        "    file.status = 'ready'\n"
    )
    _subjects, junctions, _cov = _progress(source, "pandajedi/jediorder/JobGenerator.py")

    assert junctions[0].subject == "JediFileSpec.status"
    assert junctions[0].attribution == "certain"


def test_importing_both_leaves_the_write_unresolved():
    """Where the evidence does not decide, the map says so rather than picking.

    The junction is still emitted: the writer is known even when the subject is
    not, which is what localize and prune work from.
    """
    source = (
        "from pandaserver.taskbuffer.FileSpec import FileSpec\n"
        "from pandaserver.taskbuffer.JediFileSpec import JediFileSpec\n"
        "def f(files):\n"
        "    for file in files:\n"
        "        file.status = 'ready'\n"
    )
    _subjects, junctions, _cov = _progress(source, "pandaserver/dataservice/adder_gen.py")

    assert junctions[0].attribution == "unresolved"
    assert junctions[0].subject == "?.status"
    assert junctions[0].branches[0].outcome == "ready"


def test_writes_to_attributes_no_spec_declares_are_not_junctions():
    """The subject universe is the spec declarations, not every attribute write.

    Without this bound the map fills with bookkeeping that can never be
    attributed and never belongs to a subject.
    """
    source = "def f(self):\n    self.plugin_flavor = 'atlas'\n"
    _subjects, junctions, coverage = _progress(source, "pandajedi/jedicore/Plugin.py")

    assert junctions == []
    assert coverage == []


def test_path_condition_records_the_negated_branch():
    """An ``else`` is a reason for the outcome, so it contributes ``not (...)``."""
    source = (
        "def f(self, taskSpec, taskBroken):\n"
        "    if taskBroken:\n"
        "        taskSpec.status = 'tobroken'\n"
        "    else:\n"
        "        taskSpec.status = 'finishing'\n"
    )
    _subjects, junctions, _cov = _progress(source, "pandajedi/jediorder/ContentsFeeder.py")

    by_outcome = {b.outcome: b.path_condition for b in junctions[0].branches}
    assert by_outcome["tobroken"] == ["taskBroken"]
    assert by_outcome["finishing"] == ["not (taskBroken)"]


def test_bare_name_condition_carries_the_expression_behind_it():
    """``if not allowed:`` names no predicate until the assignment is substituted.

    Real PanDA moved a command's acceptance test into a helper between
    releases, which is exactly this shape.
    """
    source = (
        "def f(self, taskSpec, comStr):\n"
        "    allowed = self._check_command_allowed(comStr)\n"
        "    if not allowed:\n"
        "        taskSpec.status = 'tobroken'\n"
    )
    _subjects, junctions, _cov = _progress(source, "pandaserver/taskbuffer/task_event_module.py")

    condition = junctions[0].branches[0].path_condition[0]
    assert "_check_command_allowed" in condition


def test_declared_subsets_do_not_gate_the_outcomes():
    """A purpose-built status list is a lower bound, not a vocabulary.

    ``statusToUpdateContents`` returns the statuses eligible for a content
    update, not every status a task can hold, so an outcome outside it is
    normal.  Reported for a human, never failed.
    """
    source = "def f(self, taskSpec):\n    taskSpec.status = 'finishing'\n"
    _subjects, junctions, _cov = _progress(source, "pandajedi/jediorder/ContentsFeeder.py")
    fragment = MapFragment(map_id=MAP_ID, derived_from=VERSION, junctions=junctions)

    assert all(r.gate != "outcome-in-vocabulary" for r in gates.run_all(fragment))


def test_unresolved_attributes_are_reported_where_they_concentrate():
    """A total would hide which attributes the map is thin on.

    Each row is a candidate for one line of annotation upstream, so naming the
    attribute is the whole point of the report.
    """
    source = (
        "def f(self, files):\n"
        "    for file in files:\n"
        "        file.status = 'ready'\n"
        "    self.job.jobStatus = 'failed'\n"
    )
    _subjects, junctions, _cov = _progress(source, "pandaserver/dataservice/adder_gen.py")
    fragment = MapFragment(map_id=MAP_ID, derived_from=VERSION, junctions=junctions)

    # ``jobStatus`` has a single declaring class, so only ``status`` is thin.
    assert [row[0] for row in gates.unresolved_attributes(fragment)] == ["status"]


def test_self_write_in_a_non_spec_class_is_not_a_spec_write():
    """``self.attr`` writes the enclosing class's own field, whatever it is named.

    A WatchDog's ``self.vo = "atlas"`` shares a name with a spec attribute and
    nothing else.  Recording it as an unresolved junction would put the
    WatchDog's bookkeeping in the map and leave it in the coverage denominator
    as a gap that can never close.
    """
    source = (
        "class AtlasTaskWithholderWatchDog(WatchDogBase):\n"
        "    def __init__(self):\n"
        "        self.vo = 'atlas'\n"
    )
    _subjects, junctions, coverage = _progress(
        source, "pandajedi/jedidog/AtlasTaskWithholderWatchDog.py"
    )

    assert junctions == []
    assert coverage == []


def test_subclass_of_a_spec_resolves_to_the_spec():
    """``PickleFileSpec(FileSpec)`` is real, and its ``self.status`` is a FileSpec's.

    The same hierarchy walk that drops the WatchDog write has to keep this one,
    which is why the rule reads the base classes instead of asking whether the
    enclosing class itself declares the attribute.
    """
    source = (
        "class PickleFileSpec(FileSpec):\n"
        "    def load(self):\n"
        "        self.status = 'ready'\n"
    )
    _subjects, junctions, _cov = _progress(source, "pandaserver/taskbuffer/PickleFileSpec.py")

    assert junctions[0].subject == "FileSpec.status"
    assert junctions[0].attribution == "certain"


def test_attribute_chain_resolves_from_its_own_usage():
    """``impl.taskSpec.status`` has no plain name on the left, and does not need one.

    Structural inference keys on the whole object expression, so a chain is
    read exactly like a variable.  What it never does is fall back to the
    chain's last *name* -- that was the naming heuristic, and it is gone.
    """
    source = (
        "def f(impl):\n"
        "    impl.taskSpec.status = 'staging'\n"
        "    use(impl.taskSpec.oldStatus)\n"
    )
    _subjects, junctions, _cov = _progress(source, "pandajedi/jediorder/TaskRefiner.py")

    assert junctions[0].subject == "JediTaskSpec.status"
    assert junctions[0].attribution == "structural"


def test_subscripted_object_stays_unresolved():
    """``self.dataset_map[name].status`` has no name to read; each container differs."""
    source = (
        "from pandaserver.taskbuffer.JediFileSpec import JediFileSpec\n"
        "class Adder:\n"
        "    def run(self):\n"
        "        self.dataset_map['x'].status = 'ready'\n"
    )
    _subjects, junctions, _cov = _progress(
        source, "pandaserver/dataservice/adder_atlas_plugin.py"
    )

    assert junctions[0].attribution == "unresolved"


def test_usage_settles_what_the_name_cannot():
    """The attributes touched on an object say more than what it was called.

    ``file`` fits ``FileSpec`` and ``JediFileSpec`` and the module imports
    neither, so naming is exhausted.  But only ``FileSpec`` declares every
    attribute the loop touches, and that is the answer without guessing.
    """
    source = (
        "class AdderGen:\n"
        "    def finalize(self):\n"
        "        for file in self.job.Files:\n"
        "            if file.lfn in self.merging:\n"
        "                file.status = 'merging'\n"
        "            self.log(file.GUID, file.fsize)\n"
    )
    _subjects, junctions, _cov = _progress(source, "pandaserver/dataservice/adder_gen.py")

    assert junctions[0].subject == "FileSpec.status"
    assert junctions[0].attribution == "structural"


def test_usage_contradicts_the_name_and_usage_is_taken():
    """A variable called ``fileSpec`` in a module importing ``JediFileSpec``…

    …that touches ``lfn``, which only ``FileSpec`` declares.  Naming would have
    answered ``JediFileSpec`` here and been wrong, which is the concrete reason
    that basis is gone rather than merely unprofitable.
    """
    source = (
        "from pandaserver.taskbuffer.JediFileSpec import JediFileSpec\n"
        "def f(files):\n"
        "    for fileSpec in files:\n"
        "        fileSpec.status = 'ready'\n"
        "        use(fileSpec.lfn)\n"
    )
    _subjects, junctions, _cov = _progress(source, "pandajedi/jediorder/JobGenerator.py")

    assert junctions[0].subject == "FileSpec.status"
    assert junctions[0].attribution == "structural"


def test_too_few_attributes_imply_nothing():
    """Touching only the attribute being written is not evidence of a class."""
    source = (
        "class Adder:\n"
        "    def run(self):\n"
        "        self.dataset_map['x'].status = 'running'\n"
    )
    _subjects, junctions, _cov = _progress(
        source, "pandaserver/dataservice/adder_atlas_plugin.py"
    )

    assert junctions[0].attribution == "unresolved"
    assert junctions[0].structural_subject is None


def test_gate_catches_declaration_and_usage_disagreeing():
    """Two independent readings of one fact; a split means one of them is wrong."""
    junction = JunctionNode(
        map_id=MAP_ID,
        derived_from=VERSION,
        name="j",
        subject="JediFileSpec.status",
        owner="pandaserver/dataservice/adder_gen.py::finalize",
        attribution="certain",
        structural_subject="FileSpec.status",
        branches=[Branch(outcome="ready")],
    )
    fragment = MapFragment(map_id=MAP_ID, derived_from=VERSION, junctions=[junction])
    result = gates.structural_attribution_agrees(fragment)

    assert not result.passed
    assert result.checked == 1
    assert "FileSpec.status (usage)" in result.failures[0]


def test_gate_skips_junctions_usage_cannot_speak_for():
    """No structural answer is silence, not disagreement."""
    junction = JunctionNode(
        map_id=MAP_ID,
        derived_from=VERSION,
        name="j",
        subject="JobSpec.jobStatus",
        owner="x.py::f",
        attribution="certain",
        structural_subject=None,
    )
    fragment = MapFragment(map_id=MAP_ID, derived_from=VERSION, junctions=[junction])
    result = gates.structural_attribution_agrees(fragment)

    assert result.passed
    assert result.checked == 0


# The adder idiom, as PanDA writes it: a method that appends its parameter, and
# a caller that hands it a freshly constructed spec.  Both containers are named
# ``Files`` and hold different classes, which is the whole point.
_ADDERS = """
class JobSpec(object):
    _attributes = ("PandaID", "jobStatus")
    __slots__ = _attributes + ("Files",)

    def addFile(self, file):
        self.Files.append(file)

    def load(self, states):
        for state in states:
            file_spec = FileSpec()
            self.addFile(file_spec)

class JediDatasetSpec(object):
    _attributes = ("datasetID", "status")
    __slots__ = _attributes + ("Files",)

    def addFile(self, fileSpec):
        self.Files.append(fileSpec)
"""

_JEDI_ADDER_CALLER = """
def refine(datasetSpec):
    use(datasetSpec.datasetID)
    fileSpec = JediFileSpec()
    datasetSpec.addFile(fileSpec)
"""


def _progress_multi(*sources: tuple[str, str]):
    """Extract with the spec fixtures plus several caller modules."""
    modules = [_module(_SPECS, "pandaserver/taskbuffer/Specs.py")]
    modules += [_module(text, rel) for text, rel in sources]
    subjects, junctions, coverage, _diag, _enum = progress.extract(modules, MAP_ID, VERSION)
    return subjects, junctions, coverage


def test_element_type_comes_from_what_was_put_in_the_container():
    """``for file in job.Files`` is settled by the ``FileSpec()`` that went in.

    PanDA never annotates ``addFile``, but it does construct the spec two lines
    before handing it over, and that is the same fact stated where the code
    happens to state it.
    """
    consumer = (
        "def finalize(job):\n"
        "    for file in job.Files:\n"
        "        file.status = 'merging'\n"
        "    job.jobStatus = 'merging'\n"      # pins ``job`` the way adder_gen does
    )
    _subjects, junctions, _cov = _progress_multi(
        (_ADDERS, "pandaserver/taskbuffer/JobSpec.py"),
        (consumer, "pandaserver/dataservice/adder_gen.py"),
    )
    written = [
        j for j in junctions if j.owner.endswith("::finalize") and j.subject.endswith(".status")
    ]

    assert written[0].subject == "FileSpec.status"
    assert written[0].attribution == "container"


def test_two_containers_named_files_hold_different_classes():
    """``JobSpec.Files`` and ``JediDatasetSpec.Files`` are the pair naming cannot split.

    Resolving them apart is what makes the container basis worth the hop: the
    variable is called ``file`` in both cases.
    """
    consumer = (
        "def walk(job, datasetSpec):\n"
        "    for file in job.Files:\n"
        "        file.status = 'merging'\n"
        "    for other in datasetSpec.Files:\n"
        "        other.status = 'ready'\n"
        "    use(job.jobStatus, datasetSpec.datasetID)\n"   # pins both holders
    )
    _subjects, junctions, _cov = _progress_multi(
        (_ADDERS, "pandaserver/taskbuffer/JobSpec.py"),
        (_JEDI_ADDER_CALLER, "pandajedi/jedirefine/TaskRefinerBase.py"),
        (consumer, "pandaserver/dataservice/adder_gen.py"),
    )
    subjects = {j.subject for j in junctions if j.owner.endswith("::walk")}

    assert subjects == {"FileSpec.status", "JediFileSpec.status"}


def test_container_attribute_need_not_be_a_declared_column():
    """``Files`` is an object-graph edge, not a DB column.

    ``__slots__ = _attributes + ("Files", ...)`` says so outright.  Requiring
    the container to be a declared attribute made the whole pass learn nothing.
    """
    modules = [
        _module(_SPECS, "pandaserver/taskbuffer/Specs.py"),
        _module(_ADDERS, "pandaserver/taskbuffer/JobSpec.py"),
    ]
    declarations = progress.spec_attributes(modules)
    attributor = attribution.SpecAttributor(declarations, attribution.class_bases(modules))
    attributor.learn_element_types(modules)
    attributor.learn_self_attributes(modules)

    assert "Files" not in declarations["JobSpec"]
    assert attributor._element_types[("JobSpec", "Files")] == {"FileSpec"}


def test_module_qualified_constructor_is_still_a_constructor():
    """``SiteSpec.SiteSpec()`` states the type as plainly as ``SiteSpec()``.

    PanDA imports the module and calls through it, so the callee is an
    attribute rather than a name.  Reading only names sent
    ``entity_module.getSiteInfo`` to structural inference for a type the code
    was stating outright, one attribute along.
    """
    source = (
        "from pandaserver.taskbuffer import FileSpec\n"
        "def get_file():\n"
        "    ret = FileSpec.FileSpec()\n"
        "    ret.status = 'ready'\n"
        "    return ret\n"
    )
    _subjects, junctions, _cov = _progress(
        source, "pandaserver/taskbuffer/db_proxy_mods/entity_module.py"
    )

    assert junctions[0].subject == "FileSpec.status"
    assert junctions[0].attribution == "certain"


def test_what_a_class_does_with_its_own_field_is_pooled_across_its_methods():
    """``self.taskSpec`` is one field of one class, so every method that touches
    it describes the same object.

    ``TaskRefinerBase`` is why: the method writing ``self.taskSpec.status``
    touches only ``jediTaskID`` beside it -- two attributes half the specs
    declare -- while the class touches twelve, which only ``JediTaskSpec`` has.
    That write is the sole producer of ``topreprocess``, so reading the method
    alone left a declared status looking unreachable.
    """
    source = (
        "class TaskRefinerBase(object):\n"
        "    def remember(self, taskSpec):\n"
        "        self.taskSpec.oldStatus = self.taskSpec.status\n"
        "    def refine(self):\n"
        "        self.taskSpec.status = 'topreprocess'\n"
    )
    _subjects, junctions, _cov = _progress(source, "pandajedi/jedirefine/TaskRefinerBase.py")
    written = [j for j in junctions if j.owner.endswith("::refine")]

    assert [(j.subject, j.attribution) for j in written] == [
        ("JediTaskSpec.status", "structural")
    ]


def test_a_local_name_is_not_pooled_across_methods():
    """A local is one thing only for as long as its function lasts -- two
    methods using ``spec`` need not mean the same kind of object, so the
    evidence for widening the scope is missing.  It stays unresolved, which is
    reported rather than guessed at.
    """
    source = (
        "class Refiner(object):\n"
        "    def remember(self, spec):\n"
        "        spec.oldStatus = 'ready'\n"
        "    def refine(self, spec):\n"
        "        spec.status = 'topreprocess'\n"
    )
    _subjects, junctions, _cov = _progress(source, "pandajedi/jedirefine/Other.py")
    written = [j for j in junctions if j.owner.endswith("::refine")]

    assert [j.attribution for j in written] == ["unresolved"]


def test_a_copy_holds_what_the_original_held():
    """``copy.copy(spec)`` preserves the type; that is semantics, not a guess."""
    source = (
        "import copy\n"
        "def clone():\n"
        "    lib_file_spec = FileSpec()\n"
        "    runFileSpec = copy.copy(lib_file_spec)\n"
        "    runFileSpec.status = 'ready'\n"
    )
    _subjects, junctions, _cov = _progress(source, "pandajedi/jediorder/JobGenerator.py")

    assert junctions[0].subject == "FileSpec.status"
    assert junctions[0].attribution == "certain"


# --------------------------------------------------------------------------- #
# progress: what the right-hand side settles
# --------------------------------------------------------------------------- #
#
# The slice used to read literal right-hand sides only, which left the map with
# two states for a write -- resolved or invisible -- when the model has a third
# the SQL slice was already using.  These pin the ladder that fixes it, and in
# particular the one rung that must not slip: ``passthrough`` names a place a
# value lives, so a local variable can never be its target.


def _outcomes(junctions):
    """``(outcome, tier)`` for every branch, in order."""
    return [(b.outcome, b.tier) for j in junctions for b in j.branches]


def test_both_arms_of_an_if_else_are_on_the_map():
    """The shape that showed the slice was wrong -- ``AtlasProdWatchDog.py:350``.

    Reading literals only, this junction said it always produces ``ready``.  The
    other arm is how a reassigned task returns to the status it held, so the
    reasoning could not offer "the restore did not fire" as a candidate at all.
    """
    source = (
        "def doActionForReassign(self, taskSpec):\n"
        "    if taskSpec.oldStatus in ['assigning', 'exhausted', None]:\n"
        "        taskSpec.status = 'ready'\n"
        "    else:\n"
        "        taskSpec.status = taskSpec.oldStatus\n"
    )
    _subjects, junctions, _cov = _progress(source, "pandajedi/jedidog/AtlasProdWatchDog.py")

    assert _outcomes(junctions) == [
        ("ready", 1),
        ("passthrough(JediTaskSpec.oldStatus)", 2),
    ]
    assert junctions[0].branches[1].path_condition == [
        "not (taskSpec.oldStatus in ['assigning', 'exhausted', None])"
    ]


def test_clearing_a_field_is_an_outcome():
    """``oldStatus = None`` decides the field as much as any word does.

    Spelled as the source spells it, which is also how a log line interpolating
    the field reads -- the map is compared against production text.
    """
    source = "def release(self, taskSpec):\n    taskSpec.oldStatus = None\n"
    _subjects, junctions, _cov = _progress(source, "pandajedi/jedidog/AtlasProdWatchDog.py")

    assert _outcomes(junctions) == [("None", 1)]


def test_a_local_is_resolved_by_reaching_definitions():
    """A guarded chain assigning the local carries its conditions to the write.

    The write's own guards come first because reaching it is necessary for any
    outcome; the assignment's guards say which one.
    """
    source = (
        "def refine(self, taskSpec, ok):\n"
        "    if taskSpec.nucleus:\n"
        "        newStatus = 'ready'\n"
        "    else:\n"
        "        newStatus = 'pending'\n"
        "    if ok:\n"
        "        taskSpec.status = newStatus\n"
    )
    _subjects, junctions, _cov = _progress(source, "pandajedi/jedirefine/TaskRefiner.py")

    assert _outcomes(junctions) == [("ready", 1), ("pending", 1)]
    assert [b.path_condition for b in junctions[0].branches] == [
        ["ok", "taskSpec.nucleus"],
        ["ok", "not (taskSpec.nucleus)"],
    ]


def test_an_unresolvable_local_is_a_run_time_outcome_not_a_passthrough():
    """``passthrough(X)`` claims the value lives in X and the walk continues
    there.  A local is a step in a computation, not a place a value lives, so a
    passthrough onto one is an edge whose far end cannot exist -- and the
    reference gate would not catch it, because it excuses a passthrough landing
    off the promoted set as a provenance terminal.
    """
    source = (
        "def refine(self, taskSpec, incoming):\n"
        "    newStatus = compute(incoming)\n"
        "    taskSpec.status = newStatus\n"
    )
    _subjects, junctions, _cov = _progress(source, "pandajedi/jedirefine/TaskRefiner.py")

    assert _outcomes(junctions) == [("runtime(newStatus)", 2)]


def test_a_qualified_field_name_is_a_passthrough():
    """``self.oldStatus = self.status`` is how ``oldStatus`` gets recorded -- the
    edge prune works from, and the first hop out of the flagship symptom."""
    source = (
        "class JediTaskSpec(object):\n"
        "    _attributes = ('jediTaskID', 'status', 'oldStatus')\n"
        "    def setOnHold(self):\n"
        "        self.oldStatus = self.status\n"
    )
    _subjects, junctions, _cov = _progress(source, "pandaserver/taskbuffer/JediTaskSpec.py")

    assert _outcomes(junctions) == [("passthrough(JediTaskSpec.status)", 2)]


def test_an_attribute_of_something_that_is_not_a_spec_is_a_run_time_outcome():
    """``self.status`` in a WatchDog is the WatchDog's own field: a name
    collision, not a place on the map.  Naming it as a passthrough source would
    point the walk at a node no map contains."""
    source = (
        "class AtlasProdWatchDog(object):\n"
        "    def doAction(self, taskSpec):\n"
        "        taskSpec.status = self.status\n"
    )
    _subjects, junctions, _cov = _progress(source, "pandajedi/jedidog/AtlasProdWatchDog.py")

    assert _outcomes(junctions) == [("runtime(self.status)", 2)]


def test_assembled_text_carries_its_own_search_key():
    """Free text is not a state, so there is no outcome to enumerate -- but the
    rendered expression keeps the literal frame, which is what a diagnostic seen
    in production is matched against.

    It is deliberately not copied into ``emits``: that field means the branch
    logs this line, which is true of a diagnostic and false of a computed
    ``lfn``, and no structural reading separates the two.
    """
    source = (
        "class JobSpec(object):\n"
        "    _attributes = ('PandaID', 'jobStatus', 'ddmErrorDiag')\n"
        "    def fail(self, n):\n"
        "        self.ddmErrorDiag = f'failed to get {n} files'\n"
    )
    _subjects, junctions, _cov = _progress(source, "pandaserver/taskbuffer/JobSpec.py")

    assert _outcomes(junctions) == [("runtime(f'failed to get {n} files')", 2)]
    assert junctions[0].branches[0].emits == []


def test_the_return_alias_shape_is_left_to_its_own_slice():
    """``taskSpec.status = self.getFinalTaskStatus(...)`` is resolved one hop up
    into a branch per returned value.  A second reading here would put "decided
    at run time" beside branches that say what the value is -- a weaker claim
    contradicting a stronger one at the same junction.
    """
    source = (
        "class PostProcessorBase(object):\n"
        "    def doPostProcess(self, taskSpec):\n"
        "        taskSpec.status = self.getFinalTaskStatus(taskSpec)\n"
    )
    _subjects, junctions, _cov = _progress(source, "pandajedi/jedipprocess/PostProcessorBase.py")

    assert junctions == []


# --------------------------------------------------------------------------- #
# sql-write: statements, table classes, bind writes
# --------------------------------------------------------------------------- #

# The flagship shape, cut down: one statement writing and another reading
# through the identically named bind, in one function.
_SQL_SOURCE = '''
class TaskModule:
    def updateTaskStatus(self, jediTaskID, taskStatus, broken):
        sqlU = f"UPDATE {panda_config.schemaJEDI}.JEDI_Tasks "
        sqlU += "SET status=:status,oldStatus=:oldStatus "
        sqlU += "WHERE jediTaskID=:jediTaskID "
        sqlL = f"UPDATE {panda_config.schemaJEDI}.JEDI_Tasks "
        sqlL += "SET lockedBy=NULL WHERE status=:status "
        varMap = {}
        varMap[":jediTaskID"] = jediTaskID
        if broken:
            varMap[":status"] = "tobroken"
        else:
            varMap[":status"] = "finishing"
        self.cur.execute(sqlU + comment, varMap)
'''


def _sql_extract(source: str, rel: str = "pandaserver/taskbuffer/db_proxy_mods/task_module.py"):
    modules = [_module(_SPECS, "pandaserver/taskbuffer/Specs.py"), _module(source, rel)]
    attributor = attribution.SpecAttributor(
        progress.spec_attributes(modules), attribution.class_bases(modules)
    )
    conflicts = attributor.learn_table_classes(modules)
    subjects, junctions, coverage, uncovered, _diag = sqlwrite.extract(
        modules, MAP_ID, VERSION, attributor
    )
    return subjects, junctions, coverage, uncovered, conflicts, attributor


def test_statement_is_reassembled_from_its_concatenation():
    """A statement is built by ``=`` then a run of ``+=``, interleaved with others."""
    module = _module(_SQL_SOURCE, "x.py")
    func = next(
        n for n in ast.walk(module.tree)
        if isinstance(n, ast.FunctionDef) and n.name == "updateTaskStatus"
    )
    text = sql.reconstruct(func, "sqlU")

    assert "UPDATE {}.JEDI_Tasks" in text
    assert "SET status=:status" in text
    # The other statement being built in the same function must not bleed in.
    assert "lockedBy" not in text


def test_a_bind_in_a_where_clause_is_not_a_write():
    """``:status`` is a write in one statement and a predicate in another.

    Both are in the same function, so the bind key alone cannot tell them
    apart; only the ``SET`` clause can.
    """
    writes = sql.writes("UPDATE {}.JEDI_Tasks SET lockedBy=NULL WHERE status=:status ")

    assert set(writes[0].columns) == {"lockedBy"}


def test_inline_values_are_kept_without_a_bind():
    """``stateChangeTime=CURRENT_DATE`` is a write whose value is in the statement."""
    writes = sql.writes("UPDATE {}.JEDI_Tasks SET status=:status,stateChangeTime=CURRENT_DATE ")

    assert writes[0].columns["status"] == sql.ColumnValue(kind="bind", text=":status")
    assert writes[0].columns["stateChangeTime"] == sql.ColumnValue(
        kind="expression", text="CURRENT_DATE"
    )


def test_a_column_copied_from_another_column_is_told_from_a_literal():
    """``SET status=oldStatus`` is a passthrough; ``SET status='ready'`` is not.

    The distinction has to survive the fact that both are bare-looking text on
    the right of an ``=``.  It is what turns "a task left pending" into an edge
    the backward walk can follow.
    """
    writes = sql.writes("UPDATE {}.JEDI_Tasks SET status=oldStatus,oldStatus=NULL ")

    assert writes[0].columns["status"] == sql.ColumnValue(kind="column", text="oldStatus")
    # ``NULL`` is bare-looking too, and it is neither a column nor unsettled:
    # the statement says exactly what the field will hold afterwards.
    assert writes[0].columns["oldStatus"] == sql.ColumnValue(kind="literal", text="None")

    literal = sql.writes("UPDATE ATLAS_PANDA.filesTable4 SET status='ready' ")
    assert literal[0].columns["status"] == sql.ColumnValue(kind="literal", text="ready")


def test_an_update_is_read_through_a_hint_and_a_table_alias():
    """``UPDATE /*+ index(tab ...) */ ... filesTable4 tab SET ...`` is one statement.

    Both forms occur on ``filesTable4`` and neither is exotic; without the
    allowance the statement reads as no statement at all, which is silent.
    """
    writes = sql.writes(
        "UPDATE /*+ index(tab FILESTABLE4_IDX) */ ATLAS_PANDA.filesTable4 tab "
        "SET status='ready' WHERE PandaID=:PandaID "
    )

    assert len(writes) == 1
    assert writes[0].table == "filesTable4"
    assert writes[0].columns["status"] == sql.ColumnValue(kind="literal", text="ready")


def test_a_statement_chosen_through_another_variable_is_still_read():
    """``sql = sqlTU`` then ``execute(sql, ...)`` -- and the alias carries the reason.

    ``reactivatePendingTasks_JEDI`` builds a timeout statement and a release
    statement unconditionally and picks between them in an ``if``/``else``, so
    reading only the statement loses which one ran and why.
    """
    source = (
        "def reactivate(self):\n"
        "    sqlTO = f'UPDATE {schema}.JEDI_Tasks '\n"
        "    sqlTO += 'SET status=:newStatus '\n"
        "    sqlTU = f'UPDATE {schema}.JEDI_Tasks '\n"
        "    sqlTU += 'SET status=oldStatus '\n"
        "    if timeout:\n"
        "        sql = sqlTO\n"
        "    else:\n"
        "        sql = sqlTU\n"
        "    self.cur.execute(sql + comment, varMap)\n"
    )
    func = ast.parse(source).body[0]

    statements = [w for text in sql.variants(func, "sql") for w in sql.writes(text)]

    assert [w.columns["status"].text for w in statements] == [":newStatus", "oldStatus"]


def test_a_statement_assembled_across_an_if_else_is_two_statements():
    """Folding the arms together builds a statement that cannot exist::

        UPDATE ... SET status=:status,SET status=oldStatus,...

    and the damage is not that it reads oddly: the second ``SET`` overwrote the
    first in the column map, so the bind arm of the write vanished.  That is the
    same shape as the attribute-slice if/else whose second arm was missing --
    one arm on the map, one lost.
    """
    source = (
        "def f(self, newTaskStatus):\n"
        "    sqlTU = 'UPDATE ATLAS_PANDA.JEDI_Tasks '\n"
        "    if newTaskStatus != 'dummy':\n"
        "        sqlTU += 'SET status=:status,'\n"
        "    else:\n"
        "        sqlTU += 'SET status=oldStatus,'\n"
        "    sqlTU += 'modificationTime=CURRENT_DATE WHERE jediTaskID=:jediTaskID '\n"
    )
    func = _func(source)

    statements = [w for text in sql.variants(func, "sqlTU") for w in sql.writes(text)]

    assert [w.columns["status"].kind for w in statements] == ["bind", "column"]
    assert [w.columns["status"].text for w in statements] == [":status", "oldStatus"]
    # The head is never dropped -- it is what the statement is.
    assert all(w.table == "JEDI_Tasks" for w in statements)


def test_two_independent_if_elses_give_every_combination():
    """``getTasksToExecCommand_JEDI`` picks the status arm and the oldStatus arm
    on unrelated tests, so all four statements are reachable."""
    source = (
        "def f(self, newTaskStatus, taskStatus):\n"
        "    sqlTU = 'UPDATE ATLAS_PANDA.JEDI_Tasks '\n"
        "    if newTaskStatus != 'dummy':\n"
        "        sqlTU += 'SET status=:status,'\n"
        "    else:\n"
        "        sqlTU += 'SET status=oldStatus,'\n"
        "    if taskStatus in ['paused']:\n"
        "        sqlTU += 'oldStatus=NULL,'\n"
        "    else:\n"
        "        sqlTU += 'oldStatus=status,'\n"
        "    sqlTU += 'modificationTime=CURRENT_DATE WHERE jediTaskID=:jediTaskID '\n"
    )
    func = _func(source)

    statements = [w for text in sql.variants(func, "sqlTU") for w in sql.writes(text)]

    assert [(w.columns["status"].text, w.columns["oldStatus"].text) for w in statements] == [
        (":status", "None"),
        (":status", "status"),
        ("oldStatus", "None"),
        ("oldStatus", "status"),
    ]


def test_an_optional_fragment_does_not_split_the_statement():
    """An ``if`` with no ``else`` makes the fragment optional, not alternative.

    Splitting on those too would double the count for every one of them, and
    the over-read is the safe direction: an extra clause is read, where a
    mis-split would attribute a value to a statement that never carries it.
    """
    source = (
        "def f(self, resetFrozenTime):\n"
        "    sqlTU = 'UPDATE ATLAS_PANDA.JEDI_Tasks SET status=:status,'\n"
        "    if resetFrozenTime:\n"
        "        sqlTU += 'frozenTime=NULL,'\n"
        "    sqlTU += 'modificationTime=CURRENT_DATE WHERE jediTaskID=:jediTaskID '\n"
    )
    func = _func(source)

    texts = sql.variants(func, "sqlTU")

    assert len(texts) == 1
    assert "frozenTime=NULL" in texts[0]


def test_table_class_is_inferred_from_the_column_names():
    """Nothing declares which spec a table holds; the columns give it away."""
    _s, _j, _c, _u, conflicts, attributor = _sql_extract(_SQL_SOURCE)

    assert conflicts == {}
    assert attributor.class_for_table("JEDI_Tasks") == "JediTaskSpec"


def test_the_statement_can_name_the_class_outright():
    """``INSERT INTO filesTable4 ({FileSpec.columnNames()})`` says it in the f-string.

    The stronger source, and the only one that settles ``filesTable4``: every
    ``UPDATE`` on it sets columns ``FileSpec`` and ``JediFileSpec`` both
    declare, so no column set ever separates the two.
    """
    source = (
        "def insert_file(self):\n"
        "    sqlF = f'INSERT INTO ATLAS_PANDA.filesTable4 ({FileSpec.columnNames()}) '\n"
        "    self.cur.execute(sqlF + comment, varMap)\n"
    )
    _s, _j, _c, _u, conflicts, attributor = _sql_extract(source)

    assert conflicts == {}
    assert attributor.class_for_table("filesTable4") == "FileSpec"


def test_bind_writes_become_branches_with_their_conditions():
    """The junction is anchored at the bind, where the value and its ``if`` are."""
    _s, junctions, _c, _u, _conf, _a = _sql_extract(_SQL_SOURCE)
    status = [j for j in junctions if j.subject == "JediTaskSpec.status"]

    assert len(status) == 1
    by_outcome = {b.outcome: b.path_condition for b in status[0].branches}
    assert by_outcome["tobroken"] == ["broken"]
    assert by_outcome["finishing"] == ["not (broken)"]
    assert status[0].attribution == "certain"


def test_a_value_decided_at_run_time_is_recorded_not_dropped():
    """The writer is known even when the value is not."""
    source = (
        "class M:\n"
        "    def f(self, newStatus):\n"
        "        sqlU = 'UPDATE ATLAS_PANDA.JEDI_Tasks SET status=:status,oldStatus=:oldStatus '\n"
        "        varMap = {}\n"
        "        varMap[':status'] = newStatus\n"
        "        self.cur.execute(sqlU + comment, varMap)\n"
    )
    _s, junctions, _c, _u, _conf, _a = _sql_extract(source)

    branch = junctions[0].branches[0]
    assert branch.tier == 2
    assert branch.outcome == "runtime(newStatus)"


def test_clearing_a_column_is_spelled_the_way_the_attribute_slice_spells_it():
    """``SET oldStatus=NULL`` and ``taskSpec.oldStatus = None`` are one fact.

    They reach the same subject, so two spellings would make the branch table
    offer two outcomes where the source has one -- and everything downstream
    compares an outcome to a value observed through Python.
    """
    sql_source = (
        "class M:\n"
        "    def f(self):\n"
        "        sqlU = 'UPDATE ATLAS_PANDA.JEDI_Tasks SET oldStatus=NULL '\n"
        "        self.cur.execute(sqlU + comment, {})\n"
    )
    _s, junctions, _c, _u, _conf, _a = _sql_extract(sql_source)
    from_sql = [(b.outcome, b.tier) for b in junctions[0].branches]

    attribute_source = (
        "from pandaserver.taskbuffer.Specs import JediTaskSpec\n"
        "class M:\n"
        "    def f(self):\n"
        "        taskSpec = JediTaskSpec()\n"
        "        taskSpec.oldStatus = None\n"
    )
    _s, attribute_junctions, _c = _progress(attribute_source, "pandajedi/jediorder/W.py")
    from_attribute = [(b.outcome, b.tier) for b in attribute_junctions[0].branches]

    assert from_sql == from_attribute == [("None", 1)]


def test_a_number_settles_a_column_as_firmly_as_a_quoted_string():
    """Quoting is a property of the type, not of how decided the value is."""
    source = (
        "class M:\n"
        "    def f(self):\n"
        "        sqlU = 'UPDATE ATLAS_PANDA.JEDI_Tasks SET coreCount=0 '\n"
        "        self.cur.execute(sqlU + comment, {})\n"
    )
    _s, junctions, _c, _u, _conf, _a = _sql_extract(source)

    assert [(b.outcome, b.tier) for b in junctions[0].branches] == [("0", 1)]


def test_what_the_database_itself_decides_is_a_branch_not_a_dropped_write():
    """The clock and the row's own contents are tier 2, on the same grounds a
    bind filled at run time is: the writer is known, and that is what prune
    reads.

    Dropping them was not neutral.  Promotion weighs a subject's literal writes
    against all of them, so a counter the source only ever increments looked,
    from its lone ``= 0``, like a field with a closed vocabulary.
    """
    source = (
        "class M:\n"
        "    def f(self):\n"
        "        sqlU = 'UPDATE ATLAS_PANDA.JEDI_Datasets SET nFiles=nFiles+:iFiles,'\n"
        "        sqlU += 'modificationTime=CURRENT_DATE '\n"
        "        self.cur.execute(sqlU + comment, {})\n"
    )
    _s, junctions, _c, _u, _conf, _a = _sql_extract(source)
    outcomes = {j.subject: [(b.outcome, b.tier) for b in j.branches] for j in junctions}

    # Table-qualified: neither column name belongs to one declaring class, so
    # the statement cannot say which spec owns the row and the table does.
    assert outcomes["JEDI_Datasets.nFiles"] == [("runtime(nFiles+:iFiles)", 2)]
    assert outcomes["JEDI_Datasets.modificationTime"] == [("runtime(CURRENT_DATE)", 2)]


def test_a_bind_filled_with_a_non_string_constant_is_settled():
    """``varMap[':frozenTime'] = None`` is decided, and ``runtime(None)`` --
    which is what the slice used to record -- says it is not."""
    source = (
        "class M:\n"
        "    def f(self):\n"
        "        sqlU = 'UPDATE ATLAS_PANDA.JEDI_Tasks SET frozenTime=:frozenTime '\n"
        "        varMap = {}\n"
        "        varMap[':frozenTime'] = None\n"
        "        self.cur.execute(sqlU + comment, varMap)\n"
    )
    _s, junctions, _c, _u, _conf, _a = _sql_extract(source)

    assert [(b.outcome, b.tier) for b in junctions[0].branches] == [("None", 1)]


def test_a_bind_filled_from_a_local_is_resolved_like_an_attribute_write_is():
    """The same reading the attribute slice makes, on the slice where most of
    PanDA's writes actually are: a knight decides and a proxy method binds.

    Both sets of guards are carried.  The ones reaching ``varMap[...] = local``
    say the write happened; the ones on the assignment that gave the local its
    value say which value -- and neither is derivable from the other's node.
    """
    source = (
        "class M:\n"
        "    def f(self, toSkip):\n"
        "        sqlU = 'UPDATE ATLAS_PANDA.JEDI_Tasks SET status=:status,oldStatus=:oldStatus '\n"
        "        if toSkip:\n"
        "            newStatus = 'scouted'\n"
        "        else:\n"
        "            newStatus = 'running'\n"
        "        if self.ready:\n"
        "            varMap = {}\n"
        "            varMap[':status'] = newStatus\n"
        "            self.cur.execute(sqlU + comment, varMap)\n"
    )
    _s, junctions, _c, _u, _conf, _a = _sql_extract(source)
    status = [j for j in junctions if j.subject == "JediTaskSpec.status"]

    assert [(b.outcome, b.tier) for b in status[0].branches] == [
        ("scouted", 1),
        ("running", 1),
    ]
    assert status[0].branches[0].path_condition == ["self.ready", "toSkip"]
    assert status[0].branches[1].path_condition == ["self.ready", "not (toSkip)"]


def test_a_bind_filled_from_a_declared_mapping_resolves_to_its_values():
    """``newTaskStatus = commandStatusMap[commandStr]["doing"]`` reaches the
    database through a bind, which is where the statuses only this mapping
    produces actually live."""
    source = (
        "class M:\n"
        "    def f(self, commandStr):\n"
        "        sqlU = 'UPDATE ATLAS_PANDA.JEDI_Tasks SET status=:status,oldStatus=:oldStatus '\n"
        "        varMap = {}\n"
        "        varMap[':status'] = JediTaskSpec.commandStatusMap()[commandStr]['doing']\n"
        "        self.cur.execute(sqlU + comment, varMap)\n"
    )
    modules = [
        _module(_SPECS, "pandaserver/taskbuffer/Specs.py"),
        _module(_COMMAND_MAP, "pandaserver/taskbuffer/JediTaskSpec.py"),
        _module(source, "pandaserver/taskbuffer/db_proxy_mods/task_module.py"),
    ]
    attributor = attribution.SpecAttributor(
        progress.spec_attributes(modules), attribution.class_bases(modules)
    )
    attributor.learn_table_classes(modules)
    _s, junctions, _c, _u, _d = sqlwrite.extract(modules, MAP_ID, VERSION, attributor)
    status = [j for j in junctions if j.subject == "JediTaskSpec.status"]

    assert {(b.outcome, b.tier) for b in status[0].branches} == {
        ("aborting", 1),
        ("finishing", 1),
        ("paused", 1),
    }


def test_a_table_with_no_spec_is_qualified_by_the_table():
    """A subject's key needs a qualifier that disambiguates, not a Python class.

    Requiring a spec class was doing filtering work it should not: it dropped
    ``ddm_endpoint.blacklisted``, the writer behind a blacklisted RSE, along
    with the bookkeeping.  What keeps bookkeeping out is promotion.
    """
    source = (
        "class M:\n"
        "    def f(self):\n"
        "        sqlU = 'UPDATE ATLAS_PANDA.ddm_endpoint SET blacklisted=:blacklisted '\n"
        "        varMap = {}\n"
        "        varMap[':blacklisted'] = 'Y'\n"
        "        self.cur.execute(sqlU + comment, varMap)\n"
    )
    subjects, junctions, _c, uncovered, _conf, _a = _sql_extract(source)

    assert junctions[0].subject == "ddm_endpoint.blacklisted"
    assert subjects[0].qualifier_kind == "table"
    assert uncovered == {"ddm_endpoint"}


def test_a_spec_backed_table_keeps_the_class_as_its_qualifier():
    """``jobsActive4`` and ``jobsArchived4`` are one JobSpec.jobStatus, not two.

    Keying on the table would split a subject across a job's lifetime, and
    would file an attribute write and a SQL write to the same field under
    different names.
    """
    source = (
        "class M:\n"
        "    def f(self):\n"
        "        sqlA = 'UPDATE ATLAS_PANDA.jobsActive4 SET jobStatus=:jobStatus,PandaID=:p '\n"
        "        varMap = {}\n"
        "        varMap[':jobStatus'] = 'running'\n"
        "        self.cur.execute(sqlA + comment, varMap)\n"
    )
    subjects, junctions, _c, _u, _conf, _a = _sql_extract(source)

    assert junctions[0].subject == "JobSpec.jobStatus"
    assert [s.qualifier_kind for s in subjects if s.attribute == "jobStatus"] == ["spec"]


# --------------------------------------------------------------------------- #
# promotion
# --------------------------------------------------------------------------- #


def _fragment_with(*subject_specs):
    """Build a fragment of subjects and their junctions for promotion tests."""
    subjects, junctions = [], []
    for qualifier, attribute, outcomes in subject_specs:
        name = f"{qualifier}.{attribute}"
        subjects.append(
            SubjectNode(
                map_id=MAP_ID,
                derived_from=VERSION,
                name=name,
                spec_class=qualifier,
                attribute=attribute,
            )
        )
        junctions.append(
            JunctionNode(
                map_id=MAP_ID,
                derived_from=VERSION,
                name=f"j:{name}",
                subject=name,
                owner="x.py::f",
                branches=[Branch(outcome=o, tier=t) for o, t in outcomes],
            )
        )
    return MapFragment(map_id=MAP_ID, derived_from=VERSION, subjects=subjects, junctions=junctions)


def test_a_predicate_against_literals_is_a_state_gate():
    """``WHERE t.status IN ('ready','running')`` gates another component."""
    modules = [_module("sql = \"SELECT x FROM t WHERE t.status IN ('ready','running') \"\n", "x.py")]

    assert promotion.gated_fields(modules)["status"] == 1


def test_a_predicate_against_bind_variables_is_a_lookup():
    """``WHERE PandaID=:PandaID`` selects a row, not a state.

    Without this ``lfn`` and ``jediTaskID`` rank first among subjects, which is
    how the narrowing was found.
    """
    modules = [
        _module('sql = "SELECT x FROM t WHERE PandaID=:PandaID "\n', "x.py"),
        _module('sql2 = "SELECT x FROM t WHERE fileID IN (:a,:b) "\n', "y.py"),
    ]
    gated = promotion.gated_fields(modules)

    assert gated["PandaID"] == 0
    assert gated["fileID"] == 0


def test_a_quote_inside_a_subquery_does_not_promote_the_outer_field():
    """``IN (SELECT ... WHERE x='y')`` -- the literal belongs to the subquery."""
    source = "sql = \"SELECT a FROM t WHERE jediTaskID IN (SELECT id FROM u WHERE type='x') \"\n"

    assert promotion.gated_fields([_module(source, "x.py")])["jediTaskID"] == 0


def test_two_literals_are_not_enough_to_close_a_set():
    """``jediTaskID`` has 528 writes of which a few are literal; that is not a set."""
    fragment = _fragment_with(
        ("JediTaskSpec", "jediTaskID", [("a", 1), ("b", 1)] + [("x", 2)] * 8),
        ("JediTaskSpec", "status", [("ready", 1), ("running", 1), ("done", 1)]),
    )
    criteria = promotion.criteria_for(fragment, Counter(), {})

    assert "JediTaskSpec.jediTaskID" not in criteria
    assert criteria["JediTaskSpec.status"] == ["3:closed-literal-set"]


def test_unpromoted_subjects_and_their_junctions_are_dropped():
    """An attribute nobody investigates is noise at both levels."""
    fragment = _fragment_with(
        ("JediTaskSpec", "status", [("ready", 1), ("done", 1)]),
        ("JobSpec", "modificationTime", [("now", 2)]),
    )
    dropped = promotion.apply(fragment, promotion.criteria_for(fragment, Counter(), {}))

    assert dropped == (1, 1)
    assert [s.name for s in fragment.subjects] == ["JediTaskSpec.status"]
    assert [j.subject for j in fragment.junctions] == ["JediTaskSpec.status"]


def test_an_unresolved_junction_survives_promotion():
    """It has no subject to judge, so dropping it would hide a reported gap."""
    fragment = _fragment_with(("JobSpec", "modificationTime", [("now", 2)]))
    fragment.junctions.append(
        JunctionNode(
            map_id=MAP_ID,
            derived_from=VERSION,
            name="j:?",
            subject="?.status",
            owner="x.py::f",
            attribution="unresolved",
            branches=[Branch(outcome="failed")],
        )
    )
    promotion.apply(fragment, promotion.criteria_for(fragment, Counter(), {}))

    assert [j.subject for j in fragment.junctions] == ["?.status"]


def test_a_column_copied_from_another_becomes_a_passthrough_branch():
    """``SET status=oldStatus`` is how a task leaves ``pending``.

    The outcome names the source as a *subject*, not as a column, so the
    backward walk has somewhere to go: ``passthrough(oldStatus)`` would be a
    string, ``passthrough(JediTaskSpec.oldStatus)`` is an edge.
    """
    source = (
        "class TaskModule:\n"
        "    def release(self, jediTaskID):\n"
        "        sqlTU = f'UPDATE {schema}.JEDI_Tasks '\n"
        "        sqlTU += 'SET status=oldStatus,oldStatus=NULL '\n"
        "        sqlTU += 'WHERE jediTaskID=:jediTaskID '\n"
        "        varMap = {}\n"
        "        self.cur.execute(sqlTU + comment, varMap)\n"
    )
    _s, junctions, _c, _u, _conf, _a = _sql_extract(source)

    branches = [b for j in junctions if j.subject == "JediTaskSpec.status" for b in j.branches]
    assert [(b.outcome, b.tier) for b in branches] == [
        ("passthrough(JediTaskSpec.oldStatus)", 2)
    ]


def test_a_conditionally_appended_fragment_carries_its_own_condition():
    """Reassembly flattens the ``if``; the fragment is what still holds it.

    ``getTasksToExecCommand_JEDI`` appends ``SET status=:status`` or ``SET
    status=oldStatus`` in the two arms of one test, so anchoring an inline value
    at the ``execute`` would report no reason at all.
    """
    source = (
        "class TaskModule:\n"
        "    def exec_command(self, newTaskStatus):\n"
        "        sqlTU = f'UPDATE {schema}.JEDI_Tasks '\n"
        "        if newTaskStatus != 'dummy':\n"
        "            sqlTU += 'SET status=:status,'\n"
        "        else:\n"
        "            sqlTU += 'SET status=oldStatus,'\n"
        "        sqlTU += 'oldStatus=NULL '\n"
        "        varMap = {}\n"
        "        varMap[':status'] = 'running'\n"
        "        self.cur.execute(sqlTU + comment, varMap)\n"
    )
    _s, junctions, _c, _u, _conf, _a = _sql_extract(source)

    branches = {
        b.outcome: b.path_condition
        for j in junctions
        if j.subject == "JediTaskSpec.status"
        for b in j.branches
    }
    assert branches["passthrough(JediTaskSpec.oldStatus)"] == ["not (newTaskStatus != 'dummy')"]


def test_promotion_follows_a_passthrough_into_its_source():
    """A subject a promoted one copies from is worth asking about too.

    Otherwise the first hop of the backward walk out of the flagship symptom
    points at a subject the map does not contain.
    """
    fragment = _fragment_with(
        ("JediTaskSpec", "status", [("ready", 1), ("done", 1)]),
        ("JediTaskSpec", "oldStatus", [("x", 2)]),
    )
    fragment.junctions[0].branches.append(
        Branch(outcome="passthrough(JediTaskSpec.oldStatus)", tier=2)
    )
    criteria = promotion.criteria_for(fragment, Counter(), {})
    assert "JediTaskSpec.oldStatus" not in criteria

    closed = promotion.close_over_passthrough(fragment, criteria)
    promotion.apply(fragment, closed)

    assert closed["JediTaskSpec.oldStatus"] == ["4:carried-into-a-promoted-subject"]
    assert {s.name for s in fragment.subjects} == {
        "JediTaskSpec.status",
        "JediTaskSpec.oldStatus",
    }


# --------------------------------------------------------------------------- #
# reaching definitions for one local
# --------------------------------------------------------------------------- #


def _func(source: str, name: str = "f"):
    tree = ast.parse(source)
    from bamboo.codemap.panda.pathcond import attach_parents

    attach_parents(tree)
    return next(
        n
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
    )


def test_a_guarded_chain_gives_each_value_its_own_condition():
    func = _func(
        "def f(self, spec):\n"
        "    if spec.status == 'tobroken':\n"
        "        status = 'broken'\n"
        "    elif spec.status == 'toabort':\n"
        "        status = 'aborted'\n"
        "    return status\n"
    )

    values = pathcond.literal_values(func, "status")

    assert [(literal, conditions) for literal, conditions, _line in values] == [
        ("broken", ["spec.status == 'tobroken'"]),
        (
            "aborted",
            ["not (spec.status == 'tobroken')", "spec.status == 'toabort'"],
        ),
    ]


def test_a_later_write_that_is_not_exclusive_becomes_a_condition():
    """The part dominating-guard analysis cannot see.  ``getFinalTaskStatus``
    decides a status in an if/elif chain and then rechecks twice at the end of
    the function, and either recheck can replace whatever the chain decided --
    so the chain's own guards are necessary and not sufficient.

    Comparing nesting depth or prefixes misses this: the rechecks sit at the
    top level, *shallower* than the branch they overwrite.
    """
    func = _func(
        "def f(self, spec):\n"
        "    if spec.status == 'toabort':\n"
        "        status = 'aborted'\n"
        "    else:\n"
        "        status = 'done'\n"
        "    if spec.is_hpo():\n"
        "        status = 'finished'\n"
        "    return status\n"
    )

    values = {
        literal: conditions
        for literal, conditions, _line in pathcond.literal_values(func, "status")
    }

    assert values["aborted"] == ["spec.status == 'toabort'", "not (spec.is_hpo())"]
    assert values["done"] == ["not (spec.status == 'toabort')", "not (spec.is_hpo())"]
    # The overwriting write itself has nothing after it.
    assert values["finished"] == ["spec.is_hpo()"]



def test_exclusive_siblings_do_not_negate_each_other():
    """Otherwise every branch of a chain carries the negation of every other:
    ``-dest_blacklisted`` came out requiring ``not (totalQueued >= limit)``, a
    condition with nothing to do with it."""
    func = _func(
        "def f(self, spec):\n"
        "    criteria = '-link_unusable'\n"
        "    if spec.blacklisted:\n"
        "        criteria = '-dest_blacklisted'\n"
        "    elif spec.queued >= spec.limit:\n"
        "        criteria = '-links_full'\n"
        "    return criteria\n"
    )

    values = {
        literal: conditions
        for literal, conditions, _line in pathcond.literal_values(func, "criteria")
    }

    assert values["-dest_blacklisted"] == ["spec.blacklisted"]
    assert values["-links_full"] == [
        "not (spec.blacklisted)",
        "spec.queued >= spec.limit",
    ]
    # The unconditional default carries both, because either replaces it.
    assert values["-link_unusable"] == [
        "not (spec.blacklisted)",
        "not (spec.queued >= spec.limit)",
    ]


# --------------------------------------------------------------------------- #
# declared mappings -- the value set behind a subscript
# --------------------------------------------------------------------------- #
#
# ``commandStatusMap()`` is the one declaration in the corpus that is complete
# rather than a sample: it *is* the command-to-status relation.  Two statuses
# PanDA declares exist nowhere else, so nothing else can account for them.

_COMMAND_MAP = """
class JediTaskSpec(object):
    _attributes = ("jediTaskID", "status", "oldStatus")

    def commandStatusMap(cls):
        return {
            "kill": {"doing": "aborting", "done": "toabort"},
            "finish": {"doing": "finishing", "done": "passed"},
            "pause": {"doing": "paused", "done": "dummy"},
        }

    commandStatusMap = classmethod(commandStatusMap)
"""


def _mappings(*sources: str):
    modules = [_module(text, f"pandaserver/taskbuffer/m{i}.py") for i, text in enumerate(sources)]
    return values.declared_mappings(modules)


def test_a_run_time_key_enumerates_the_level_a_stated_one_narrows_it():
    """``[commandStr]["done"]`` cannot say which command ran, but it can say the
    six statuses a completed command leaves behind -- and ``["kill"]["doing"]``
    resolves to exactly one."""
    mappings = _mappings(_COMMAND_MAP)
    func = _func(
        "def f(self, commandStr):\n"
        "    a = JediTaskSpec.commandStatusMap()[commandStr]['done']\n"
        "    b = JediTaskSpec.commandStatusMap()['kill']['doing']\n"
    )
    settle = values.resolver(mappings)
    assignments = [n for n in ast.walk(func) if isinstance(n, ast.Assign)]

    assert settle(assignments[0].value, func) == ["dummy", "passed", "toabort"]
    assert settle(assignments[1].value, func) == ["aborting"]


def test_a_local_holding_the_mapping_resolves_too():
    """``commandStatusMap = JediTaskSpec.commandStatusMap()`` sits at the top of
    the method that subscripts it a hundred lines further down."""
    mappings = _mappings(_COMMAND_MAP)
    func = _func(
        "def f(self, commandStr):\n"
        "    commandStatusMap = JediTaskSpec.commandStatusMap()\n"
        "    newTaskStatus = commandStatusMap[commandStr]['doing']\n"
    )
    settle = values.resolver(mappings)
    written = [n for n in ast.walk(func) if isinstance(n, ast.Assign)][1]

    assert settle(written.value, func) == ["aborting", "finishing", "paused"]


def test_a_mapping_with_one_computed_entry_resolves_to_nothing():
    """Elimination treats a short candidate list as complete, so a value set
    missing a member is worse than no value set: the caller then records an
    honest run-time branch instead of a closed set that is not closed."""
    mappings = _mappings(
        "class Spec(object):\n"
        "    def statusMap(cls):\n"
        "        return {'kill': 'aborting', 'finish': compute()}\n"
        "    statusMap = classmethod(statusMap)\n"
    )

    assert mappings == {}


def test_two_classes_disagreeing_on_a_name_drop_it():
    """A call site offers the method name and not the class, so where the name
    means two different mappings it cannot be told which one it reached."""
    mappings = _mappings(
        "class A(object):\n"
        "    def statusMap(cls):\n"
        "        return {'kill': 'aborting'}\n",
        "class B(object):\n"
        "    def statusMap(cls):\n"
        "        return {'kill': 'toabort'}\n",
    )

    assert mappings == {}


def test_a_stated_key_the_mapping_lacks_resolves_to_nothing():
    """Two parts of the source disagreeing is not this resolver's to settle."""
    settle = values.resolver(_mappings(_COMMAND_MAP))
    func = _func("def f(self):\n    a = JediTaskSpec.commandStatusMap()['resume']['done']\n")
    written = next(n for n in ast.walk(func) if isinstance(n, ast.Assign))

    assert settle(written.value, func) == []


def test_assembled_text_is_indexed_and_a_bare_literal_is_not():
    """The index answers "this message was seen, who wrote it" -- which needs a
    frame with holes.  An exact message can be found by searching the source for
    itself, so a bare literal is not a template and does not earn a row.
    """
    source = (
        "class JobSpec(object):\n"
        "    _attributes = ('PandaID', 'jobStatus', 'ddmErrorDiag')\n"
        "    def fail(self, n):\n"
        "        self.ddmErrorDiag = f'failed to get {n} files'\n"
        "        self.jobStatus = 'failed'\n"
    )
    modules = [
        _module(_SPECS, "pandaserver/taskbuffer/Specs.py"),
        _module(source, "pandaserver/taskbuffer/JobSpec.py"),
    ]
    _s, _j, _c, diagnostics, _enum = progress.extract(modules, MAP_ID, VERSION)

    assert [(d.template, d.field, d.form) for d in diagnostics] == [
        ("failed to get {} files", "JobSpec.ddmErrorDiag", "attribute")
    ]


def test_a_frame_of_nothing_but_holes_is_not_a_search_key():
    """``setErrDiag`` appends with ``f"{self.errorDialog} {diag}"``, whose frame
    is two holes and a space: true, and matching nothing.  A row that cannot be
    searched for is noise in an index that exists to be searched."""
    source = (
        "class JediTaskSpec(object):\n"
        "    _attributes = ('jediTaskID', 'status', 'errorDialog')\n"
        "    def setErrDiag(self, diag):\n"
        "        self.errorDialog = f'{self.errorDialog} {diag}'\n"
    )
    modules = [
        _module(_SPECS, "pandaserver/taskbuffer/Specs.py"),
        _module(source, "pandaserver/taskbuffer/JediTaskSpec.py"),
    ]
    _s, _j, _c, diagnostics, _enum = progress.extract(modules, MAP_ID, VERSION)

    assert diagnostics == []


def test_a_bound_template_is_filed_against_the_column_not_the_bind():
    """On its own the bind is called ``:errDiag`` and could belong to any
    statement in the method; which column it lands in is the join the SQL slice
    has already made, which is why the index is collected there.

    Qualified by the table here because a one-column statement cannot say which
    spec the row holds -- the same fallback a write gets, so the two agree.
    """
    source = (
        "class M:\n"
        "    def f(self, site):\n"
        "        sqlU = 'UPDATE ATLAS_PANDA.JEDI_Tasks SET errorDialog=:errDiag '\n"
        "        varMap = {}\n"
        "        varMap[':errDiag'] = f'site {site} is unknown'\n"
        "        self.cur.execute(sqlU + comment, varMap)\n"
    )
    modules = [
        _module(_SPECS, "pandaserver/taskbuffer/Specs.py"),
        _module(source, "pandaserver/taskbuffer/db_proxy_mods/task_module.py"),
    ]
    attributor = attribution.SpecAttributor(
        progress.spec_attributes(modules), attribution.class_bases(modules)
    )
    attributor.learn_table_classes(modules)
    _s, _j, _c, _u, diagnostics = sqlwrite.extract(modules, MAP_ID, VERSION, attributor)

    assert [(d.template, d.field, d.form) for d in diagnostics] == [
        ("site {} is unknown", "JEDI_Tasks.errorDialog", "bind")
    ]


def test_the_index_survives_promotion_dropping_the_field():
    """The whole point.  ``ddmErrorDiag`` satisfies no promotion criterion --
    correctly, since it has no value set to enumerate -- so its subject and its
    junctions are dropped.  The writes that assemble its text are still the
    answer to "who wrote this line", and an index makes no claim about the field
    that promotion could contradict.
    """
    source = (
        "class JobSpec(object):\n"
        "    _attributes = ('PandaID', 'jobStatus', 'ddmErrorDiag')\n"
        "    def fail(self, n, why):\n"
        "        self.ddmErrorDiag = f'failed to get {n} files'\n"
        "        self.ddmErrorDiag = why\n"
    )
    modules = [
        _module(_SPECS, "pandaserver/taskbuffer/Specs.py"),
        _module(source, "pandaserver/taskbuffer/JobSpec.py"),
    ]
    subjects, junctions, _c, diagnostics, _enum = progress.extract(modules, MAP_ID, VERSION)
    fragment = MapFragment(map_id=MAP_ID, derived_from=VERSION)
    fragment.subjects.extend(subjects)
    fragment.junctions.extend(junctions)
    fragment.diagnostics.extend(diagnostics)

    promotion.apply(fragment, promotion.criteria_for(fragment, Counter(), {}))

    assert "JobSpec.ddmErrorDiag" not in {s.name for s in fragment.subjects}
    assert "JobSpec.ddmErrorDiag" not in {j.subject for j in fragment.junctions}
    assert [d.template for d in fragment.diagnostics] == ["failed to get {} files"]


_ENUM_WRITER = (
    "class JobSpec(object):\n"
    "    _attributes = ('PandaID', 'jobStatus', 'taskBufferErrorCode', 'pilotErrorCode')\n"
    "    def kill(self):\n"
    "        self.taskBufferErrorCode = ErrorCode.EC_Kill\n"
    "        self.pilotErrorCode = 0\n"
)


def test_a_field_is_bound_to_the_enumeration_that_decodes_it():
    """``jobSpec.taskBufferErrorCode = ErrorCode.EC_Kill`` states both halves.

    ``errorcode`` says this binding is "not recoverable from the constant's
    location", and it is right about the location -- the write is where it is
    recoverable from, and the corpus has 79 of them.
    """
    modules = [
        _module(_SPECS, "pandaserver/taskbuffer/Specs.py"),
        _module(_ENUM_WRITER, "pandaserver/taskbuffer/JobSpec.py"),
    ]
    _s, _j, _c, _d, bindings = progress.extract(
        modules, MAP_ID, VERSION, {"EC_Kill": "taskbuffer.ErrorCode.EC"}
    )

    assert [(b.field, b.constant, b.namespace) for b in bindings] == [
        ("JobSpec.taskBufferErrorCode", "EC_Kill", "taskbuffer.ErrorCode.EC")
    ]


def test_the_binding_survives_promotion_dropping_the_field():
    """No criterion fires on an error-code field -- every write is tier 2, the
    right-hand side being a constant -- so promotion drops it, and rightly.

    "Why is this field 100?" is not the question; "what does 100 mean here?"
    is, and an index makes no claim promotion could contradict.
    """
    modules = [
        _module(_SPECS, "pandaserver/taskbuffer/Specs.py"),
        _module(_ENUM_WRITER, "pandaserver/taskbuffer/JobSpec.py"),
    ]
    subjects, junctions, _c, _d, bindings = progress.extract(
        modules, MAP_ID, VERSION, {"EC_Kill": "taskbuffer.ErrorCode.EC"}
    )
    fragment = MapFragment(map_id=MAP_ID, derived_from=VERSION)
    fragment.subjects.extend(subjects)
    fragment.junctions.extend(junctions)
    fragment.enumeration_writes.extend(bindings)

    promotion.apply(fragment, promotion.criteria_for(fragment, Counter(), {}))

    assert "JobSpec.taskBufferErrorCode" not in {s.name for s in fragment.subjects}
    assert [b.constant for b in fragment.enumeration_writes] == ["EC_Kill"]


def test_a_constant_two_enumerations_share_is_not_a_binding():
    """A name that decodes to two namespaces decodes nothing, so it is dropped
    rather than guessed at -- the plugin only hands over the unambiguous ones."""
    modules = [
        _module(_SPECS, "pandaserver/taskbuffer/Specs.py"),
        _module(_ENUM_WRITER, "pandaserver/taskbuffer/JobSpec.py"),
    ]
    _s, _j, _c, _d, bindings = progress.extract(modules, MAP_ID, VERSION, {})

    assert bindings == []


def test_the_attribute_slice_reads_a_subscript_of_a_declared_mapping():
    """``TaskCommando.py:178`` -- the write that accounts for ``passed``."""
    caller = (
        "def runImpl(self, tmpTaskSpec, commandStr):\n"
        "    if commandStr in ['kill', 'finish']:\n"
        "        tmpTaskSpec.status = JediTaskSpec.commandStatusMap()[commandStr]['done']\n"
    )
    _subjects, junctions, _cov = _progress_multi(
        (_COMMAND_MAP, "pandaserver/taskbuffer/JediTaskSpec.py"),
        (caller, "pandajedi/jediorder/TaskCommando.py"),
    )
    written = [j for j in junctions if j.owner.endswith("::runImpl")]

    assert _outcomes(written) == [("dummy", 1), ("passed", 1), ("toabort", 1)]


# --------------------------------------------------------------------------- #
# write alias
# --------------------------------------------------------------------------- #

_ON_HOLD = """
class JediTaskSpec(object):
    _attributes = ("jediTaskID", "status", "oldStatus")

    def setOnHold(self):
        if self.status in ["ready", "running"]:
            self.oldStatus = self.status
            self.status = "pending"
"""


def _alias_extract(*sources: tuple[str, str]):
    modules = [_module(_ON_HOLD, "pandaserver/taskbuffer/JediTaskSpec.py")]
    modules += [_module(text, rel) for text, rel in sources]
    attributor = attribution.SpecAttributor(
        progress.spec_attributes(modules), attribution.class_bases(modules)
    )
    return alias.extract(
        modules, MAP_ID, VERSION, progress.spec_attributes(modules), attributor
    )


def test_a_call_site_of_a_write_alias_is_a_junction():
    """The map must answer *why* ``pending``, not merely that ``setOnHold`` ran.

    Recorded only at the definition, the most contended value in the system has
    exactly one writer and no reason attached to it.
    """
    caller = (
        "def generate(self, taskSpec, inputChunk):\n"
        "    if not inputChunk.hasCandidates():\n"
        "        taskSpec.setOnHold()\n"
    )
    _subjects, junctions, _cov = _alias_extract((caller, "pandajedi/jediorder/JobGenerator.py"))

    assert [j.owner for j in junctions] == ["pandajedi/jediorder/JobGenerator.py::generate"]
    branch = junctions[0].branches[0]
    assert (junctions[0].subject, branch.outcome) == ("JediTaskSpec.status", "pending")
    assert branch.path_condition == [
        "not inputChunk.hasCandidates()",
        "self.status in ['ready', 'running']  [in setOnHold()]",
    ]


def test_the_aliases_own_guard_is_marked_as_its_own():
    """Reaching ``pending`` needs both conditions and they fail differently.

    A caller that never ran and a guard that refused the status are different
    diagnoses, and a flat conjunction cannot tell them apart.
    """
    caller = "def refine(self, taskSpec):\n    taskSpec.setOnHold()\n"
    _subjects, junctions, _cov = _alias_extract((caller, "pandajedi/jediorder/TaskRefiner.py"))

    assert junctions[0].branches[0].path_condition == [
        "self.status in ['ready', 'running']  [in setOnHold()]"
    ]


def test_only_literal_writes_make_a_method_an_alias():
    """Widened to any declared-attribute write, the rule matches serialization.

    On real PanDA that is 36 methods and 600 call sites led by ``__init__``,
    ``pack`` and ``setErrDiag`` -- none of which settle anything.
    """
    spec = (
        "class JediTaskSpec(object):\n"
        "    _attributes = ('jediTaskID', 'status', 'errorDialog')\n"
        "\n"
        "    def setErrDiag(self, diag):\n"
        "        self.errorDialog = diag\n"
    )
    modules = [_module(spec, "pandaserver/taskbuffer/JediTaskSpec.py")]

    assert alias.find_aliases(modules, progress.spec_attributes(modules)) == {}


# --------------------------------------------------------------------------- #
# return alias -- a helper that returns the value the caller writes
# --------------------------------------------------------------------------- #

_FINAL_STATUS = """
class JediTaskSpec(object):
    _attributes = ("jediTaskID", "status", "oldStatus")
"""

_POST_PROCESSOR = """
class PostProcessorBase(object):
    def doBasicPostProcess(self, taskSpec):
        taskSpec.status = self.getFinalTaskStatus(taskSpec)

    def getFinalTaskStatus(self, taskSpec, checkGoal=False):
        if taskSpec.status == 'tobroken':
            status = 'broken'
        elif taskSpec.status == 'toabort':
            status = 'aborted'
        else:
            status = 'done'
        if taskSpec.is_hpo_workflow():
            status = 'finished'
        if checkGoal:
            return True
        return status
"""


def _producer_extract(*sources: tuple[str, str]):
    modules = [_module(_FINAL_STATUS, "pandaserver/taskbuffer/JediTaskSpec.py")]
    modules += [_module(text, rel) for text, rel in sources]
    declarations = progress.spec_attributes(modules)
    attributor = attribution.SpecAttributor(declarations, attribution.class_bases(modules))
    return alias.extract_producers(modules, MAP_ID, VERSION, declarations, attributor)


def test_a_status_decided_inside_a_helper_reaches_the_map():
    """The gap gate nine found.  The attribute slice takes only literal
    right-hand sides, so ``taskSpec.status = self.getFinalTaskStatus(...)`` was
    not on the map at all -- and production put 104 tasks into ``aborted``,
    which is decided only inside that helper.
    """
    _subjects, junctions, _cov = _producer_extract(
        (_POST_PROCESSOR, "pandajedi/jedipprocess/PostProcessorBase.py")
    )

    junction = next(j for j in junctions if j.subject == "JediTaskSpec.status")
    outcomes = {b.outcome for b in junction.branches if b.tier == 1}

    assert outcomes == {"broken", "aborted", "done", "finished"}
    aborted = next(b for b in junction.branches if b.outcome == "aborted")
    # The helper's own guards, marked with where they came from, and the recheck
    # that can replace whatever the chain decided.
    assert aborted.path_condition == [
        "not (taskSpec.status == 'tobroken')  [in getFinalTaskStatus()]",
        "taskSpec.status == 'toabort'  [in getFinalTaskStatus()]",
        "not (taskSpec.is_hpo_workflow())  [in getFinalTaskStatus()]",
    ]


def test_an_unreadable_return_keeps_the_list_open():
    """``return True`` on the goal-checking path is a value this slice cannot
    follow, so the junction says so rather than letting the outcomes read as a
    closed set."""
    _subjects, junctions, _cov = _producer_extract(
        (_POST_PROCESSOR, "pandajedi/jedipprocess/PostProcessorBase.py")
    )

    junction = next(j for j in junctions if j.subject == "JediTaskSpec.status")

    assert [b.outcome for b in junction.branches if b.tier == 2] == [
        "runtime(getFinalTaskStatus())"
    ]


def test_the_callers_conditions_come_first_and_are_not_marked():
    """A reader who cannot tell the caller's guard from the helper's cannot tell
    a caller that never ran from a helper that decided otherwise."""
    caller = (
        "class AtlasProdPostProcessor(PostProcessorBase):\n"
        "    def doPostProcess(self, taskSpec):\n"
        "        if taskSpec.gshare != 'Test':\n"
        "            taskSpec.status = self.getFinalTaskStatus(taskSpec)\n"
    )
    _subjects, junctions, _cov = _producer_extract(
        (_POST_PROCESSOR, "pandajedi/jedipprocess/PostProcessorBase.py"),
        (caller, "pandajedi/jedipprocess/AtlasProdPostProcessor.py"),
    )

    junction = next(j for j in junctions if "AtlasProdPostProcessor" in j.owner)
    aborted = next(b for b in junction.branches if b.outcome == "aborted")

    assert aborted.path_condition[0] == "taskSpec.gshare != 'Test'"
    assert all("[in getFinalTaskStatus()]" in c for c in aborted.path_condition[1:])


def test_a_helper_that_computes_its_answer_is_not_a_producer():
    """``makeBuildJobParameters`` and ``getLargestAttemptNr`` settle nothing from
    a closed set, so the coverage row says these helpers compute rather than
    decide -- 2 of 7 sites in the corpus, and that is the honest number.

    The writer is still recorded, at tier 2.  This slice owns the whole
    ``self.helper()`` shape, so the attribute slice stays out of it entirely;
    dropping the sites this one cannot follow would put them on no slice at all.
    """
    source = (
        "class JobGenerator(object):\n"
        "    def generate(self, taskSpec):\n"
        "        taskSpec.status = self.computeStatus()\n"
        "    def computeStatus(self):\n"
        "        return compute(self.state)\n"
    )

    _subjects, junctions, coverage = _producer_extract(
        (source, "pandajedi/jediorder/JobGenerator.py")
    )

    assert [(b.outcome, b.tier) for j in junctions for b in j.branches] == [
        ("runtime(computeStatus())", 2)
    ]
    assert [(c.candidates, c.explained) for c in coverage] == [(1, 0)]


def test_only_a_bare_self_receiver_is_followed():
    """Widened to any call the shape matches forty-two sites of which two touch
    a subject; the rest are plugin lookups and config reads."""
    source = (
        "class Refiner(object):\n"
        "    def refine(self, taskSpec):\n"
        "        taskSpec.status = self.helper.getFinalTaskStatus(taskSpec)\n"
    )

    _subjects, junctions, coverage = _producer_extract(
        (_POST_PROCESSOR, "pandajedi/jedipprocess/PostProcessorBase.py"),
        (source, "pandajedi/jedirefine/TaskRefinerBase.py"),
    )

    assert not any("TaskRefinerBase" in j.owner for j in junctions)
    assert not any(c.file.endswith("TaskRefinerBase.py") for c in coverage)


# --------------------------------------------------------------------------- #
# shared-table boundaries
# --------------------------------------------------------------------------- #

_SCHEMAS = """
if "schemaPANDA" not in tmpSelf.__dict__:
    tmpSelf.__dict__["schemaPANDA"] = "ATLAS_PANDA"
if "schemaDEFT" not in tmpSelf.__dict__:
    tmpSelf.__dict__["schemaDEFT"] = "ATLAS_DEFT"
"""


def _channels(source: str, rel: str = "pandaserver/taskbuffer/db_proxy_mods/task_module.py"):
    modules = [
        _module(_SCHEMAS, "pandaserver/config/panda_config.py"),
        _module(source, rel),
    ]
    return boundary.extract_shared_tables(modules, MAP_ID, VERSION)


def test_the_schema_qualifier_says_whose_table_it_is():
    """``panda_config`` states which Oracle schema each name means.

    Read rather than assumed, so the boundary's interface is the schema an
    operator would type into a query.
    """
    modules = [_module(_SCHEMAS, "pandaserver/config/panda_config.py")]

    assert boundary.schema_names(modules) == {
        "PANDA": "ATLAS_PANDA",
        "DEFT": "ATLAS_DEFT",
    }


def test_a_foreign_schema_table_is_a_boundary_in_both_directions():
    """A shared table is a channel, and which way is broken is the first question.

    An endpoint is inbound only; ``PRODSYS_COMM`` carries commands out to DEFT
    and back, so recording one direction would describe half the channel.
    """
    source = (
        "class TaskModule:\n"
        "    def exec_command(self, jediTaskID):\n"
        "        sqlR = f'SELECT comm_task,comm_cmd FROM {panda_config.schemaDEFT}.PRODSYS_COMM '\n"
        "        sqlR += 'WHERE comm_owner=:comm_owner '\n"
        "        varMap = {}\n"
        "        self.cur.execute(sqlR + comment, varMap)\n"
        "    def send_command(self, jediTaskID):\n"
        "        sqlD = f'DELETE FROM {panda_config.schemaDEFT}.PRODSYS_COMM WHERE COMM_TASK=:t '\n"
        "        self.cur.execute(sqlD + comment, varMap)\n"
        "        sqlI = f'INSERT INTO {panda_config.schemaDEFT}.PRODSYS_COMM "
        "(COMM_TASK,COMM_CMD) VALUES (:t,:c) '\n"
        "        self.cur.execute(sqlI + comment, varMap)\n"
    )
    channels, _coverage = _channels(source)

    assert len(channels) == 1
    channel = channels[0]
    assert (channel.system, channel.interface) == ("deft", "ATLAS_DEFT.PRODSYS_COMM")
    assert channel.transport == "shared_table"
    assert channel.carried_values == ["comm_cmd", "comm_task"]
    assert channel.handed_over == ["COMM_CMD", "COMM_TASK"]
    # A DELETE next to an INSERT on a command table is the finding: a second
    # command silently replaces one that was never picked up.
    assert channel.operations == ["DELETE", "INSERT", "SELECT"]


def test_one_column_is_not_two_because_the_case_differs():
    """SQL identifiers are case-insensitive and PanDA writes both spellings."""
    source = (
        "class TaskModule:\n"
        "    def a(self):\n"
        "        s1 = f'UPDATE {panda_config.schemaDEFT}.T_TASK SET status=:s '\n"
        "        self.cur.execute(s1 + comment, varMap)\n"
        "    def b(self):\n"
        "        s2 = f'UPDATE {panda_config.schemaDEFT}.T_TASK SET STATUS=:s '\n"
        "        self.cur.execute(s2 + comment, varMap)\n"
    )
    channels, _coverage = _channels(source)

    assert channels[0].handed_over == ["status"]


def test_a_statement_spanning_two_schemas_is_left_alone():
    """A join no longer says which side of the boundary a column sits on."""
    source = (
        "class TaskModule:\n"
        "    def joined(self):\n"
        "        s = f'SELECT t.status FROM {panda_config.schemaDEFT}.T_TASK t, "
        "{panda_config.schemaPANDA}.JEDI_Tasks j '\n"
        "        self.cur.execute(s + comment, varMap)\n"
    )
    channels, coverage = _channels(source)

    assert channels == []
    assert coverage == []


def test_a_shared_table_is_not_reported_as_unlogged():
    """Asking whether a table was also logged inverts the question.

    Its values are queryable afterwards precisely because nobody had to write
    them down a second time.
    """
    fragment = MapFragment(map_id=MAP_ID, derived_from=VERSION)
    fragment.boundaries.append(
        BoundaryNode(
            map_id=MAP_ID,
            derived_from=VERSION,
            name="b:deft",
            system="deft",
            transport="shared_table",
            interface="ATLAS_DEFT.PRODSYS_COMM",
            carried_values=["comm_task", "comm_cmd"],
        )
    )

    assert gates.unobservable_boundaries(fragment) == []


# --------------------------------------------------------------------------- #
# entry points and triggers
# --------------------------------------------------------------------------- #

_KNIGHT = """
class TaskCommando:
    def start(self):
        while True:
            time.sleep(jedi_config.taskcommando.loopCycle)
            self.doAction()

    def doAction(self):
        tasks = self.taskBufferIF.getTasksToExecCommand_JEDI(vo, label, pid=self.pid)
"""

_PROXY = """
class TaskModule:
    def getTasksToExecCommand_JEDI(self, vo, prodSourceLabel, pid=None):
        sqlC = f'SELECT comm_task FROM {panda_config.schemaDEFT}.PRODSYS_COMM '
        self.cur.execute(sqlC + comment, varMap)
        self.markTask()

    def markTask(self):
        pass
"""


def _junction(owner: str, subject: str = "JediTaskSpec.status") -> JunctionNode:
    return JunctionNode(
        map_id=MAP_ID,
        derived_from=VERSION,
        name=f"j:{owner}",
        subject=subject,
        owner=owner,
    )


def _attach(*sources: tuple[str, str], junctions, tables=None):
    modules = [_module(text, rel) for text, rel in sources]
    return trigger.attach(junctions, modules, tables or {"PRODSYS_COMM"})


def test_a_sleeping_forever_loop_is_a_polled_entry():
    """A ``while True`` that sleeps starts something; a conditional loop retries.

    Accepting any sleeping loop matched 33 modules including ``Interaction``
    and ``ddm``, neither of which starts anything.
    """
    modules = [
        _module(_KNIGHT, "pandajedi/jediorder/TaskCommando.py"),
        _module(
            "def fetch(self):\n"
            "    while not done:\n"
            "        time.sleep(1)\n",
            "pandaserver/dataservice/ddm.py",
        ),
    ]
    kinds = trigger.classify(modules, set())

    assert kinds["pandajedi/jediorder/TaskCommando.py"] == {"polled"}
    assert "pandaserver/dataservice/ddm.py" not in kinds


def test_polling_for_a_foreign_row_is_also_a_command_entry():
    """It runs again next cycle; the command it failed to act on does not."""
    kinds = trigger.classify(
        [
            _module(_KNIGHT, "pandajedi/jediorder/TaskCommando.py"),
            _module(_PROXY, "pandaserver/taskbuffer/db_proxy_mods/task_module.py"),
        ],
        {"PRODSYS_COMM"},
    )

    assert kinds["pandajedi/jediorder/TaskCommando.py"] == {"polled", "command"}
    # The module that merely *contains* the read starts nothing.
    assert "pandaserver/taskbuffer/db_proxy_mods/task_module.py" not in kinds


def test_reach_follows_self_calls_past_the_door():
    """The door and the write are rarely the same method.

    ``add_main`` starts ``AdderGen.run``; the ``jobStatus`` writes are several
    ``self`` calls further in, and stopping at the door reported them as though
    nothing ran them.
    """
    junction = _junction("pandaserver/taskbuffer/db_proxy_mods/task_module.py::markTask")
    _attach(
        (_KNIGHT, "pandajedi/jediorder/TaskCommando.py"),
        (_PROXY, "pandaserver/taskbuffer/db_proxy_mods/task_module.py"),
        junctions=[junction],
    )

    assert {(e.trigger, e.via) for e in junction.entry_points} == {
        ("polled", "getTasksToExecCommand_JEDI"),
        ("command", "getTasksToExecCommand_JEDI"),
    }


def test_a_name_several_modules_implement_carries_no_edge():
    """``run`` is defined by every daemon, so following it invents callers.

    Before this, one ``datasetManager`` junction collected fourteen entry
    points, thirteen of them from daemons that have never heard of it.
    """
    daemon = "class D:\n    def run(self):\n        self.act()\n    def act(self):\n        pass\n"
    other = "class O:\n    def run(self):\n        pass\n"
    junction = _junction("pandaserver/daemons/scripts/first.py::act")
    _attach(
        (daemon, "pandaserver/daemons/scripts/first.py"),
        (other, "pandaserver/daemons/scripts/second.py"),
        junctions=[junction],
    )

    # Its own module is polled, so it is reached -- but not *by* the other
    # daemon, which shares only the method name.
    assert {e.entry for e in junction.entry_points} == {
        "pandaserver/daemons/scripts/first.py"
    }


def test_an_import_evidences_an_edge_a_shared_name_cannot():
    """``from ...adder_gen import AdderGen`` says which ``run`` is meant."""
    worker = (
        "class AdderGen:\n"
        "    def run(self):\n"
        "        self.finalize()\n"
        "    def finalize(self):\n"
        "        pass\n"
    )
    daemon = (
        "from pandaserver.dataservice.adder_gen import AdderGen\n"
        "def main():\n"
        "    AdderGen().run()\n"
    )
    other = "class O:\n    def run(self):\n        pass\n"
    junction = _junction("pandaserver/dataservice/adder_gen.py::finalize")
    _attach(
        (worker, "pandaserver/dataservice/adder_gen.py"),
        (daemon, "pandaserver/daemons/scripts/add_main.py"),
        (other, "pandaserver/daemons/scripts/other.py"),
        junctions=[junction],
    )

    assert {e.entry for e in junction.entry_points} == {
        "pandaserver/daemons/scripts/add_main.py"
    }


def test_a_facade_that_forwards_is_not_a_second_implementation():
    """``TaskBuffer`` and ``JediTaskBuffer`` are doors, not rival definitions."""
    facade = (
        "class TaskBuffer:\n"
        "    def markTask(self):\n"
        "        ret = proxy.markTask()\n"
        "        return ret\n"
    )
    modules = [
        _module(_PROXY, "pandaserver/taskbuffer/db_proxy_mods/task_module.py"),
        _module(facade, "pandaserver/taskbuffer/TaskBuffer.py"),
    ]

    assert (
        trigger.sole_definitions(modules)["markTask"]
        == "pandaserver/taskbuffer/db_proxy_mods/task_module.py"
    )


def test_entries_that_hand_over_different_arguments_are_reported():
    """An argument one entry omits is a guard that cannot fire on that path."""
    junction = _junction("x.py::f")
    junction.entry_points = [
        EntryPoint(trigger="polled", entry="JobGenerator.py", via="get", arg_binding={"minPriority": "p"}),
        EntryPoint(trigger="message", entry="msg.py", via="get", arg_binding={"target_tasks": "t"}),
    ]

    assert trigger.differing_arguments([junction]) == [
        (
            "JediTaskSpec.status",
            "x.py::f",
            {"JobGenerator.py": ["minPriority"], "msg.py": ["target_tasks"]},
        )
    ]


def test_a_subject_no_loop_reaches_does_not_repair_itself():
    """The question a stalled task actually asks: will waiting help?"""
    polled = _junction("a.py::f", "JediTaskSpec.status")
    polled.entry_points = [EntryPoint(trigger="polled", entry="a.py")]
    once = _junction("b.py::g", "T_TASK.vo")
    once.entry_points = [EntryPoint(trigger="command", entry="b.py")]

    assert trigger.fragile_subjects([polled, once]) == [("T_TASK.vo", ["command"])]


# --------------------------------------------------------------------------- #
# selection
# --------------------------------------------------------------------------- #

_BROKER = '''
class JobBrokerBase:
    def add_summary_message(self, old_list, new_list, message, tmp_log, msg_map):
        for site in msg_map:
            tmp_log.info(msg_map[site])
        tmp_log.info(f"{len(new_list)} candidates passed {message}")

class AtlasProdJobBroker(JobBrokerBase):
    def doBrokerage(self):
        newScanSiteList = []
        msg_map = {}
        for tmpSiteName in scanSiteList:
            if diskio_usage > diskio_limit and diskio_task > diskio_limit:
                msg_map[tmpSiteName] = f"  skip site={tmpSiteName} due to diskIO overload criteria=-diskIO"
            newScanSiteList.append(tmpSiteName)
        self.add_summary_message(oldScanSiteList, scanSiteList, "diskIO check", tmpLog, msg_map)
        newScanSiteList = []
        for tmpSiteName in scanSiteList:
            if taskSpec.ioIntensity > site_max:
                msg_map[tmpSiteName] = f"  skip site={tmpSiteName} since ioIntensity={taskSpec.ioIntensity} criteria=-max_io_intensity"
            newScanSiteList.append(tmpSiteName)
        self.add_summary_message(oldScanSiteList, scanSiteList, "IO intensity check", tmpLog, msg_map)
'''

_UNTAGGED = '''
class GenJobBroker:
    def doBrokerage(self):
        newScanSiteList = []
        for tmpSiteName in scanSiteList:
            if minDiskCount > tmpSiteSpec.maxwdir:
                tmpLog.debug(f"  skip {tmpSiteName} due to small scratch disk")
                continue
            newScanSiteList.append(tmpSiteName)
        scanSiteList = newScanSiteList
        tmpLog.debug(f"{len(scanSiteList)} candidates passed scratch disk check")
'''


def _selection(*sources: tuple[str, str]):
    modules = [_module(text, rel) for text, rel in sources]
    return selection.extract(modules, MAP_ID, VERSION)


def test_a_tagged_rejection_carries_its_condition():
    """The tag and the guard that reaches it are the whole content of a stage.

    ``criteria=-diskIO`` is what production logs carry per rejected site, so a
    stage keyed on it can be counted from the logs and explained from the map
    in one step.
    """
    stages, _cov, _gaps = _selection((_BROKER, "pandajedi/jedibrokerage/AtlasProdJobBroker.py"))

    diskio = next(s for s in stages if s.criteria_tag == "-diskIO")
    assert diskio.funnel_label == "diskIO check"
    assert diskio.conditions == [
        "diskio_usage > diskio_limit and diskio_task > diskio_limit"
    ]
    assert diskio.inputs == ["diskio_usage", "diskio_limit", "diskio_task"]


_VARIABLE_TAG = '''
class JobBrokerBase:
    def add_summary_message(self, old_list, new_list, message, tmp_log, msg_map):
        for site in msg_map:
            tmp_log.info(msg_map[site])

class AtlasProdJobBroker(JobBrokerBase):
    def doBrokerage(self, taskSpec, scanSiteList):
        nucleus = taskSpec.nucleus
        if nucleus:
            msg_map = {}
            for tmpPandaSiteName in scanSiteList:
                criteria = "-link_unusable"
                reason = ""
                if nucleus == tmpAtlasSiteName:
                    pass
                elif nucleus in unwritable_over_wan:
                    reason = "unwritable over WAN"
                    criteria = "-dest_blacklisted"
                elif totalQueued >= self.total_queue_threshold:
                    reason = "too many queued"
                    criteria = "-links_full"
                elif closeness == BLOCKED_LINK:
                    reason = "blocked link"
                tmpStr = f"  skip site={tmpPandaSiteName} due to {reason}"
                tmpStr += f": criteria={criteria}"
                msg_map[tmpPandaSiteName] = tmpStr
            self.add_summary_message(oldScanSiteList, scanSiteList, "link check", tmpLog, msg_map)
'''


def test_a_tag_assigned_to_a_variable_is_still_a_tag():
    """The blind spot ``tags-are-known`` found from the other side.

    ``AtlasProdJobBroker`` decides three of its reasons by assigning them to a
    variable and interpolating it later, so nothing in the message names a tag.
    Production logs ``criteria=-link_unusable`` and the map had no stage for it,
    while the funnel counted a cut at "link check" the slice could not explain --
    the same hole seen twice.
    """
    stages, _cov, gaps = _selection(
        (_VARIABLE_TAG, "pandajedi/jedibrokerage/AtlasProdJobBroker.py")
    )

    by_tag = {s.criteria_tag: s for s in stages}
    assert set(by_tag) == {"-link_unusable", "-dest_blacklisted", "-links_full"}
    assert {s.funnel_label for s in stages} == {"link check"}
    # Anchored where the reason is decided, not where the message is built.
    assert by_tag["-dest_blacklisted"].anchor.line_start < by_tag["-links_full"].anchor.line_start
    assert gaps == []


def test_a_default_tag_carries_what_would_have_replaced_it():
    """Reassignment is the one thing dominating-guard analysis cannot see.

    ``criteria = "-link_unusable"`` sits above the chain that overwrites it, so
    its own guard is necessary and not sufficient; left at that the map would
    claim this cut happens whenever the task has a nucleus.
    """
    stages, _cov, _gaps = _selection(
        (_VARIABLE_TAG, "pandajedi/jedibrokerage/AtlasProdJobBroker.py")
    )

    default = next(s for s in stages if s.criteria_tag == "-link_unusable")

    assert default.conditions == [
        "nucleus  [nucleus := taskSpec.nucleus]",
        "not (nucleus in unwritable_over_wan)",
        "not (totalQueued >= self.total_queue_threshold)",
    ]


def test_a_sibling_branch_is_not_treated_as_overwriting():
    """Comparing nesting depth looked right and was not: Python nests an ``elif``
    inside the previous ``if``'s ``orelse``, so a sibling is always deeper.  That
    put ``not (totalQueued >= limit)`` on the cut for a blacklisted destination,
    a condition with nothing to do with it."""
    stages, _cov, _gaps = _selection(
        (_VARIABLE_TAG, "pandajedi/jedibrokerage/AtlasProdJobBroker.py")
    )

    blacklisted = next(s for s in stages if s.criteria_tag == "-dest_blacklisted")

    assert blacklisted.conditions == [
        "nucleus  [nucleus := taskSpec.nucleus]",
        "not (nucleus == tmpAtlasSiteName)",
        "nucleus in unwritable_over_wan",
    ]


def test_a_variable_tag_promises_no_template():
    """``": criteria={}"`` is the tail of a message assembled across statements
    and is shared by every rejection in the file, so offering it as the line to
    look for would confirm nothing."""
    stages, _cov, _gaps = _selection(
        (_VARIABLE_TAG, "pandajedi/jedibrokerage/AtlasProdJobBroker.py")
    )

    assert all(s.emits == [] for s in stages)
    # The level still comes from the helper that logs the map.
    assert {s.log_level for s in stages} == {"info"}


def test_an_interpolated_criteria_that_is_not_a_bare_name_is_not_a_tag():
    """``f"start with criteria={str(criteria)}"`` is a progress log, not a
    rejection.  Requiring a bare name keeps the two other interpolated
    ``criteria=`` messages in the corpus out without naming them."""
    source = '''
class TaskEventModule:
    def send_command(self, criteria):
        tmpLog.debug(f"start with criteria={str(criteria)}")
'''

    stages, _cov, _gaps = _selection((source, "pandaserver/taskbuffer/task_event_module.py"))

    assert stages == []


def test_stage_order_follows_the_chain():
    """"Which step cut the candidates" is a question about position."""
    stages, _cov, _gaps = _selection((_BROKER, "pandajedi/jedibrokerage/AtlasProdJobBroker.py"))

    assert [(s.criteria_tag, s.order) for s in stages] == [
        ("-diskIO", 0),
        ("-max_io_intensity", 1),
    ]


def test_the_level_comes_from_the_helper_that_logs_it():
    """Most rejections are never logged where they are written.

    The broker fills a ``msg_map`` and hands it to ``add_summary_message``.
    Recording ``None`` would say the level is unknown when one hop settles it,
    and the level is what decides whether the observation exists in production.
    """
    stages, _cov, _gaps = _selection((_BROKER, "pandajedi/jedibrokerage/AtlasProdJobBroker.py"))

    assert {s.log_level for s in stages} == {"info"}


def test_a_named_step_with_no_tag_is_read_from_its_continues():
    """``GenJobBroker`` names eight steps and tags one.

    Without this, seven cuts in a production broker would be invisible -- and
    it is the file the chain matcher handles best, so a recognizer that only
    followed tags would have a hole exactly where the plan expected one.
    """
    stages, _cov, _gaps = _selection((_UNTAGGED, "pandajedi/jedibrokerage/GenJobBroker.py"))

    assert len(stages) == 1
    assert (stages[0].criteria_tag, stages[0].funnel_label) == ("", "scratch disk check")
    assert stages[0].conditions == ["minDiskCount > tmpSiteSpec.maxwdir"]
    assert stages[0].log_level == "debug"


def test_a_templated_step_name_is_not_a_step():
    """``f"{len(new_list)} candidates passed {message}"`` is the helper itself.

    The code templated the step name rather than naming one, so counting it as
    a step invents a cut inside the function that reports every other cut.
    """
    helper = (
        "class JobBrokerBase:\n"
        "    def add_summary_message(self, old_list, new_list, message, tmp_log, msg_map):\n"
        "        tmp_log.info(f'{len(new_list)} candidates passed {message}')\n"
    )
    stages, coverage, gaps = _selection((helper, "pandajedi/jedibrokerage/JobBrokerBase.py"))

    assert (stages, coverage, gaps) == ([], [], [])


def test_a_step_named_after_a_run_time_value_is_still_a_step():
    """A hole with words around it is a name; a hole alone is not.

    ``endpoint check with DISK_THRESHOLD={} TB`` is a step this broker runs
    1176 times in a day's logs, and requiring the name to be fixed left the
    funnel counting a cut the map could not place.
    """
    source = (
        "class B:\n"
        "    def runImpl(self):\n"
        "        for n in nucleusList:\n"
        "            if bad(n):\n"
        "                tmpLog.info(f'  skip nucleus={n} criteria=-space')\n"
        "                continue\n"
        "        tmpLog.info(f'{len(nucleusList)} candidates passed endpoint check "
        "with DISK_THRESHOLD={thr} TB')\n"
    )
    stages, _cov, gaps = _selection((source, "pandajedi/jedibrokerage/AtlasProdTaskBroker.py"))

    assert [s.funnel_label for s in stages] == ["endpoint check with DISK_THRESHOLD={} TB"]
    assert gaps == []


def test_a_step_reported_twice_is_one_step_named_by_its_funnel_line():
    """``AtlasProdTaskBroker`` writes its own funnel line and then files a
    summary entry, and at one step the two disagree on the name.

    Read as two steps the summary entry counts a cut whose every reason
    attached to the line above it.  The funnel line's name wins: it is the one
    production puts on a ``candidates passed`` line.
    """
    source = (
        "class B:\n"
        "    def runImpl(self):\n"
        "        for n in nucleusList:\n"
        "            if bad(n):\n"
        "                tmpLog.info(f'  skip nucleus={n} criteria=-space')\n"
        "                continue\n"
        "        tmpLog.info(f'{len(nucleusList)} candidates passed endpoint check "
        "with DISK_THRESHOLD={thr} TB')\n"
        "        self.add_summary_message(old, new, 'storage endpoint check')\n"
    )
    stages, _cov, gaps = _selection((source, "pandajedi/jedibrokerage/AtlasProdTaskBroker.py"))

    assert [(s.criteria_tag, s.funnel_label) for s in stages] == [
        ("-space", "endpoint check with DISK_THRESHOLD={} TB")
    ]
    assert gaps == []


def test_a_candidate_not_appended_is_cut_as_surely_as_one_skipped():
    """A loop can drop a candidate by skipping ahead or by keeping the
    survivors elsewhere, and the two are the same cut written differently.

    This is ``AtlasProdJobBroker``'s deferred "temporary problem check" -- 1111
    runs of it in a day's logs, and the reason the funnel could count
    candidates disappearing where the map had nothing to say.
    """
    source = (
        "class B:\n"
        "    def doBrokerage(self):\n"
        "        for tmpSiteName in scanSiteList:\n"
        "            if tmpSiteName in siteSkippedTmp:\n"
        "                msg_map[tmpSiteName] = siteSkippedTmp[tmpSiteName]\n"
        "            else:\n"
        "                newScanSiteList.append(tmpSiteName)\n"
        "        self.add_summary_message(old, new, 'temporary problem check', log, msg_map)\n"
    )
    stages, _cov, gaps = _selection((source, "pandajedi/jedibrokerage/AtlasProdJobBroker.py"))

    assert [(s.criteria_tag, s.funnel_label) for s in stages] == [
        ("", "temporary problem check")
    ]
    # The condition is the arm that does not append, positively: no negation to
    # compose, and no passthrough invented for a local that is not a place a
    # value lives.
    assert stages[0].conditions == ["tmpSiteName in siteSkippedTmp"]
    assert gaps == []


def test_one_tag_used_at_two_steps_stays_two_stages():
    """``criteria=-disk`` is emitted by both "disk check" and "Storage check".

    The tag alone is ambiguous within a chain; the funnel counter is what tells
    a reader which of the two a log line came from, so folding them together
    would lose one of the cuts.
    """
    source = (
        "class B:\n"
        "    def doBrokerage(self):\n"
        "        if a > b:\n"
        "            msg = f'skip criteria=-disk'\n"
        "        self.add_summary_message(old, new, 'disk check', log, msg_map)\n"
        "        if c > d:\n"
        "            msg = f'skip criteria=-disk'\n"
        "        self.add_summary_message(old, new, 'Storage check', log, msg_map)\n"
    )
    stages, _cov, _gaps = _selection((source, "pandajedi/jedibrokerage/AtlasProdJobBroker.py"))

    assert [(s.criteria_tag, s.funnel_label) for s in stages] == [
        ("-disk", "disk check"),
        ("-disk", "Storage check"),
    ]


def test_a_step_whose_reason_cannot_be_read_is_named():
    """The funnel will report the cut and the map has nothing to say about it."""
    source = (
        "class B:\n"
        "    def doBrokerage(self):\n"
        "        scanSiteList = [s for s in scanSiteList if keep(s)]\n"
        "        self.add_summary_message(old, new, 'link check', log, msg_map)\n"
    )
    _stages, _cov, gaps = _selection((source, "pandajedi/jedibrokerage/AtlasProdJobBroker.py"))

    assert gaps == [
        "pandajedi/jedibrokerage/AtlasProdJobBroker.py::doBrokerage "
        "counts a cut at 'link check' with no readable reason"
    ]


def test_a_field_a_filter_stage_gates_is_promoted():
    """Criterion 1 reads SQL predicates, and brokerage gates in Python.

    ``nucleus``, ``minRamCount`` and ``ioIntensity`` decide whether a site
    survives the chain without appearing in any ``WHERE``, so the first
    criterion cannot see them and the fifth restates it where it can.
    """
    fragment = _fragment_with(("JediTaskSpec", "ioIntensity", [("x", 2)]))
    assert promotion.criteria_for(fragment, Counter(), {}) == {}

    fragment.filter_stages.append(
        FilterStageNode(
            map_id=MAP_ID,
            derived_from=VERSION,
            name="f:-max_io_intensity",
            owner="AtlasProdJobBroker.py::doBrokerage",
            criteria_tag="-max_io_intensity",
            inputs=["ioIntensity", "max_io_intensity"],
        )
    )

    assert promotion.criteria_for(fragment, Counter(), {}) == {
        "JediTaskSpec.ioIntensity": ["5:gates-a-filter-stage"]
    }


def test_a_rejection_that_logs_none_of_what_it_tested_is_reported():
    """What a rejection did not log cannot be checked afterwards.

    ``criteria=-diskIO`` compares three numbers and logs only the site name;
    the numbers are on a separate line a different path emits.
    """
    fragment = MapFragment(map_id=MAP_ID, derived_from=VERSION)
    fragment.filter_stages.append(
        FilterStageNode(
            map_id=MAP_ID,
            derived_from=VERSION,
            name="f:-diskIO",
            owner="b.py::doBrokerage",
            criteria_tag="-diskIO",
            conditions=["diskio_usage > diskio_limit"],
            inputs=["diskio_usage", "diskio_limit"],
            emits=["  skip site={} due to diskIO overload criteria=-diskIO"],
        )
    )

    assert gates.unexplainable_rejections(fragment) == [
        ("-diskIO", ["diskio_usage", "diskio_limit"], "b.py::doBrokerage")
    ]


# --------------------------------------------------------------------------- #
# graph invariants
# --------------------------------------------------------------------------- #


def _graph(*, subjects, junctions):
    return MapFragment(
        map_id=MAP_ID, derived_from=VERSION, subjects=subjects, junctions=junctions
    )


def test_a_declared_status_nothing_writes_is_unreachable():
    """The sound direction of the comparison that was refuted the other way.

    Checking that every outcome sits inside a declared list fails on correct
    code -- those lists are purpose-built subsets.  Checking that every
    declared value is written somewhere is a genuine lower bound, and it could
    only be run once every write form was in.
    """
    fragment = _fragment_with(("JediTaskSpec", "status", [("defined", 1)]))
    fragment.subjects[0].vocabulary = ["defined", "aborting"]
    result = gates.declared_status_is_written(fragment)

    assert not result.passed
    assert result.checked == 2
    assert result.failures == [
        "JediTaskSpec.status declares 'aborting' and no writer produces it"
    ]


def test_a_run_time_writer_is_named_alongside_the_miss():
    """``commandStatusMap()[cmd]["doing"]`` produces 'aborting' with no literal.

    The writer is in the map and the value is not, which is a different thing
    from nothing writing it at all -- so the count goes in the failure rather
    than being left for the reader to work out.
    """
    fragment = _fragment_with(("JediTaskSpec", "status", [("runtime(newStatus)", 2)]))
    fragment.subjects[0].vocabulary = ["aborting"]

    assert gates.declared_status_is_written(fragment).failures == [
        "JediTaskSpec.status declares 'aborting' and no writer produces it "
        "(1 run-time writer(s) could)"
    ]


def test_every_junction_lands_on_a_subject_the_map_contains():
    """A backward walk that steps onto a missing node cannot report that it did."""
    fragment = _fragment_with(("JediTaskSpec", "status", [("ready", 1)]))
    fragment.junctions.append(
        JunctionNode(
            map_id=MAP_ID,
            derived_from=VERSION,
            name="j:ghost",
            subject="JediTaskSpec.ghost",
            owner="x.py::f",
            branches=[Branch(outcome="ready")],
        )
    )
    result = gates.map_references_resolve(fragment)

    assert not result.passed
    assert result.failures == ["x.py::f writes JediTaskSpec.ghost, which is not a subject"]


def test_a_class_declaring_columns_the_reader_cannot_parse_is_a_finding():
    """A declaration form nobody reads is silent in exactly the wrong way.

    ``WFDataSpec`` states its columns as ``AttributeWithType("status", str)``
    and derives the older name from that, so a reader taking only literal
    tuples matched the assignment and took nothing.  Twenty-four writes in one
    module went unattributed and ``unresolved`` swallowed them, because no gate
    was looking at a class that declares nothing.
    """
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        declaration_yields={"JobSpec (JobSpec.py)": 126, "WFDataSpec (workflow_base.py)": 0},
    )

    result = gates.spec_declarations_are_read(fragment)

    assert not result.passed
    assert result.checked == 2
    assert result.failures == [
        "WFDataSpec (workflow_base.py) declares columns and the extraction read none"
    ]


def test_the_typed_declaration_form_yields_the_same_column_names():
    """Both forms say the same thing, so both are read the same way."""
    source = (
        "class WFDataSpec:\n"
        "    attributes_with_types = (\n"
        "        AttributeWithType('data_id', int),\n"
        "        AttributeWithType('status', str),\n"
        "    )\n"
        "    attributes = tuple([a.attribute for a in attributes_with_types])\n"
    )

    declared = progress.spec_attributes([_module(source)])

    assert declared == {"WFDataSpec": {"data_id", "status"}}


def test_two_nodes_sharing_a_signature_are_reported_rather_than_one_being_lost():
    """The signature is the merge key, so a shared one is a node about to vanish.

    Found by storing the map for the first time: the build said 109 filter
    stages and Neo4j held 108.  ``AtlasProdJobBroker`` runs "temporary problem
    check" at two points guarded by ``hintForTB``, and keyed on the label alone
    they were one node -- so the map ended up claiming that cut is
    unconditional.  Neither side could see it; only the two counts together.
    """
    fragment = _fragment_with(("JediTaskSpec", "status", [("ready", 1)]))
    twin = fragment.junctions[0].model_copy(deep=True)
    twin.branches = [Branch(outcome="broken")]
    fragment.junctions.append(twin)

    result = gates.map_identities_are_distinct(fragment)

    assert not result.passed
    assert result.failures == [
        f"2 junction(s) share the signature {fragment.junctions[0].name}"
    ]


def test_a_map_whose_signatures_are_all_distinct_says_nothing():
    fragment = _fragment_with(("JediTaskSpec", "status", [("ready", 1)]))

    assert gates.map_identities_are_distinct(fragment).passed


def test_a_subject_nothing_writes_does_not_belong_in_the_map():
    fragment = _graph(
        subjects=[
            SubjectNode(
                map_id=MAP_ID,
                derived_from=VERSION,
                name="JediTaskSpec.status",
                spec_class="JediTaskSpec",
                attribute="status",
            )
        ],
        junctions=[],
    )

    assert gates.map_references_resolve(fragment).failures == [
        "JediTaskSpec.status is a subject nothing writes"
    ]


def test_promotion_does_not_follow_a_passthrough_to_a_node_that_is_absent():
    """``currentPriority`` is carried from ``taskPriority``, which nothing writes.

    Promoting the name anyway put a criterion on a node that did not exist --
    caught by the reference gate, which is what a gate over the assembly is
    for.
    """
    fragment = _fragment_with(("JediTaskSpec", "currentPriority", [("x", 2)]))
    fragment.junctions[0].branches.append(
        Branch(outcome="passthrough(JediTaskSpec.taskPriority)", tier=2)
    )
    closed = promotion.close_over_passthrough(
        fragment, {"JediTaskSpec.currentPriority": ["5:gates-a-filter-stage"]}
    )

    assert "JediTaskSpec.taskPriority" not in closed
    assert gates.carried_from_outside(fragment) == [
        ("JediTaskSpec.currentPriority", "JediTaskSpec.taskPriority")
    ]


def test_a_value_nothing_selects_on_is_reported_not_failed():
    """A terminal status is supposed to be a sink and nothing declares which."""
    fragment = _fragment_with(("JediTaskSpec", "status", [("ready", 1), ("broken", 1)]))
    fragment.subjects[0].selected_values = ["ready"]

    assert gates.unreachable_values(fragment) == [("JediTaskSpec.status", ["broken"])]


def test_a_subject_with_no_read_side_is_not_reported_as_all_sinks():
    """There the map has no question to answer, which is not an answer of no."""
    fragment = _fragment_with(("JediTaskSpec", "status", [("ready", 1)]))

    assert gates.unreachable_values(fragment) == []


def test_a_where_clause_says_which_values_something_acts_on():
    """The other half of the same statements, and the half that makes it a graph."""
    source = (
        "class TaskModule:\n"
        "    def find(self, vo):\n"
        "        varMap = {}\n"
        "        varMap[':oldStatus'] = 'pending'\n"
        "        sqlU = f'UPDATE {schema}.JEDI_Tasks '\n"
        "        sqlU += 'SET status=:status,oldStatus=NULL '\n"
        "        sqlU += \"WHERE status=:oldStatus AND vo IN ('atlas','test') \"\n"
        "        self.cur.execute(sqlU + comment, varMap)\n"
    )
    modules = [_module(_SPECS, "pandaserver/taskbuffer/Specs.py"), _module(source, "x.py")]
    attributor = attribution.SpecAttributor(
        progress.spec_attributes(modules), attribution.class_bases(modules)
    )
    attributor.learn_table_classes(modules)

    selected = sqlwrite.selected_values(modules, attributor)
    assert selected["JediTaskSpec.status"] == {"pending"}
    assert selected["JEDI_Tasks.vo"] == {"atlas", "test"}


def test_a_statement_built_with_str_format_is_read():
    """62 statements still use ``.format`` rather than an f-string.

    Missing them did not merely lose coverage: those are where JEDI selects
    tasks by status, so a dozen task states read as ones nothing selects on.
    """
    source = (
        "class TaskModule:\n"
        "    def find(self):\n"
        "        sqlR = 'SELECT jediTaskID FROM {0}.JEDI_Tasks tabT '.format(schemaJEDI)\n"
        "        sqlR += \"WHERE tabT.status='running' \"\n"
        "        self.cur.execute(sqlR + comment, varMap)\n"
    )
    func = ast.parse(source).body[0].body[0]

    assert sql.reconstruct(func, "sqlR").startswith(
        "SELECT jediTaskID FROM {}.JEDI_Tasks tabT "
    )
    assert sql.selected_literals(sql.reconstruct(func, "sqlR")) == [("status", "running")]


# ---------------------------------------------------------------------------
# Which log file holds the evidence
# ---------------------------------------------------------------------------


def test_a_logger_is_named_either_by_a_literal_or_by_the_module():
    """Two forms, and only two, across both packages.

    JEDI takes the module's own name and the server mostly spells one out,
    but both appear in both, so the shape is what is read rather than the
    package.
    """
    jedi = _module(
        'logger = PandaLogger().getLogger(__name__.split(".")[-1])\n',
        "pandajedi/jedibrokerage/AtlasProdJobBroker.py",
    )
    server = _module(
        '_logger = PandaLogger().getLogger("api_async_process")\n',
        "pandaserver/api/v1/async_process_api.py",
    )

    assert logfile.logger_name(jedi) == "AtlasProdJobBroker"
    assert logfile.logger_name(server) == "api_async_process"
    assert logfile.declared_files([jedi, server]) == {
        "pandajedi/jedibrokerage/AtlasProdJobBroker.py": "panda-AtlasProdJobBroker.log",
        "pandaserver/api/v1/async_process_api.py": "panda-api_async_process.log",
    }


def test_a_module_that_declares_no_logger_is_left_alone():
    """A wrong filename is worse than none: the query comes back empty, and
    empty reads as "production never emitted this"."""
    base = _module(
        "class JobBrokerBase:\n    pass\n",
        "pandajedi/jedibrokerage/JobBrokerBase.py",
    )

    assert logfile.logger_name(base) is None
    assert logfile.declared_files([base]) == {}


def test_a_mixin_inherits_the_log_files_of_what_mixes_it_in():
    """The proxy modules are the map's largest group and declare no logger.

    ``OraDBProxy.DBProxy`` mixes them in and ``JediDBProxy.DBProxy``
    subclasses that, so the same junction writes to the server's proxy log or
    JEDI's depending on which process ran it.  Both are named because which
    one is a runtime fact, not an ambiguity to be resolved.
    """
    mixin = _module(
        "class TaskStandaloneModule:\n    pass\n",
        "pandaserver/taskbuffer/db_proxy_mods/task_standalone_module.py",
    )
    ora = _module(
        '_logger = PandaLogger().getLogger("DBProxy")\n'
        "class DBProxy(task_standalone_module.TaskStandaloneModule):\n    pass\n",
        "pandaserver/taskbuffer/OraDBProxy.py",
    )
    jedi = _module(
        'logger = PandaLogger().getLogger(__name__.split(".")[-1])\n'
        "class DBProxy(OraDBProxy.DBProxy):\n    pass\n",
        "pandajedi/jedicore/JediDBProxy.py",
    )
    modules = [mixin, ora, jedi]

    inherited = logfile.inherited_files(modules, logfile.declared_files(modules))

    assert inherited[mixin.rel_path] == ["panda-DBProxy.log", "panda-JediDBProxy.log"]


def test_the_inheritance_walk_does_not_stop_at_one_link():
    """Stopping at the first would name the server's log and silently omit
    JEDI's -- the half most junctions actually run under."""
    mixin = _module("class Mixin:\n    pass\n", "p/mixin.py")
    middle = _module('l = PandaLogger().getLogger("Mid")\nclass Mid(Mixin):\n    pass\n', "p/middle.py")
    leaf = _module('l = PandaLogger().getLogger("Leaf")\nclass Leaf(Mid):\n    pass\n', "p/leaf.py")
    modules = [mixin, middle, leaf]

    inherited = logfile.inherited_files(modules, logfile.declared_files(modules))

    assert inherited["p/mixin.py"] == ["panda-Leaf.log", "panda-Mid.log"]


# ---------------------------------------------------------------------------
# Production evidence and the (ii) gates
# ---------------------------------------------------------------------------
#
# These pin the distinction the whole production half rests on: an empty grep
# result means three different things, and only one of them is a fact about
# PanDA.  Everything else here follows from getting that wrong being silent.

BROKER_LOG = "panda-AtlasProdJobBroker.log"
PROXY_LOG = "panda-DBProxy.log"


def _log_line(level: str, message: str, name: str = "JobBroker") -> str:
    """One line in PandaLogger's format.

    ``"%(asctime)s %(name)-12s: %(levelname)-8s %(message)s"`` -- the level is
    the token after the first ``": "``, which is what the parser anchors on.
    """
    return f"2026-08-30 12:00:01,123 {name:<12}: {level:<8} {message}"


def _sample(lines: list[str], filename: str = BROKER_LOG, **kwargs) -> evidence.GrepResult:
    """A result shaped the way the fetch path shapes one.

    ``matched`` and ``level_counts`` are filled in here for the same reason the
    fetch fills them: the level query keeps no lines, so a fixture that only
    set ``lines`` would be testing a result that cannot occur.
    """
    pattern = kwargs.pop("pattern", evidence.ANY_LINE_PATTERN)
    return evidence.GrepResult(
        query=evidence.GrepQuery(
            pattern=pattern,
            log_filename=filename,
            service=kwargs.pop("service", evidence.JEDI),
        ),
        machine=kwargs.pop("machine", "m1"),
        lines=lines,
        matched=len(lines),
        level_counts=dict(evidence._levels_in(lines)),
        return_code=kwargs.pop("return_code", 0),
        **kwargs,
    )


def _missing(filename: str = BROKER_LOG, machine: str = "m1") -> evidence.GrepResult:
    return _sample(
        [],
        filename,
        machine=machine,
        return_code=2,
        error=f"rg: /var/log/panda/{filename}: No such file or directory (os error 2)",
    )


def _evidence(*results: evidence.GrepResult) -> evidence.Evidence:
    return evidence.Evidence(fetched_at="2026-08-30T00:00:00+00:00", results=list(results))


def test_the_level_is_measured_per_log_file():
    """PanDA configures a level per logger, so a threshold taken from one file
    and applied to another would be a number established for something else."""
    ev = _evidence(
        _sample([_log_line("INFO", "a")], BROKER_LOG),
        _sample([_log_line("DEBUG", "b"), _log_line("INFO", "c")], PROXY_LOG),
    )

    assert evidence.effective_level(ev, log_filename=BROKER_LOG) == "INFO"
    assert evidence.effective_level(ev, log_filename=PROXY_LOG) == "DEBUG"
    assert evidence.effective_level(ev, log_filename="panda-absent.log") is None


def test_a_level_word_in_a_message_is_not_a_level():
    """The separator is what makes it a level, not the word.

    Counting the bare word would read a line that merely mentions DEBUG as
    proof that DEBUG is enabled, and so keep an observable production never
    emits.
    """
    ev = _evidence(_sample([_log_line("INFO", "restarting in DEBUG mode")]))

    assert evidence.level_histogram(ev) == Counter({"INFO": 1})


def test_a_continuation_line_carries_no_level():
    """Tracebacks span lines, and only the first one is formatted."""
    ev = _evidence(_sample([_log_line("ERROR", "boom"), "  File 'x.py', line 3"]))

    assert evidence.level_histogram(ev) == Counter({"ERROR": 1})


def test_an_empty_result_is_conclusive_only_when_the_tool_read_everything():
    """Three ways to come back empty, one of which is an answer.

    ``rg`` exits 1 for "no match" and 2 for "could not read the file", and a
    result over a megabyte is truncated.  A missing file and a cut sample look
    exactly like "production never emits this", so only exit 1 on a whole
    result licenses reading absence as evidence.
    """
    assert _sample([], return_code=1).conclusive
    assert not _missing().conclusive
    assert not _sample([_log_line("INFO", "a")], truncated=True).conclusive


def test_every_query_carries_its_bounds():
    """Unbounded is not an option: panda-DBProxy.log is six gigabytes and the
    processor buffers a matcher's whole output before storing a slice of it."""
    query = evidence.GrepQuery(pattern="x", log_filename=BROKER_LOG, service=evidence.JEDI)

    assert query.max_matches == evidence.DEFAULT_MAX_MATCHES
    assert query.tail_bytes == evidence.DEFAULT_TAIL_BYTES


def test_hitting_the_match_cap_reaches_the_gates_as_truncated():
    """The server sets truncated when it stops at the cap, and that is the
    whole point of sending one: a capped answer must not read as an absent one."""
    capped = _sample([_log_line("INFO", "a")], truncated=True, return_code=0)

    assert not capped.conclusive
    assert not _evidence(capped).conclusive(evidence.ANY_LINE_PATTERN)


def test_a_file_absent_everywhere_says_the_code_never_ran():
    """The strongest thing production says about the map.

    PandaLogger creates the file on first emit, so no file on any machine
    means that logger has never emitted -- not quiet, unexecuted.
    """
    ev = _evidence(_missing(machine="m1"), _missing(machine="m2"))

    assert ev.file_status(BROKER_LOG) == "absent"


def test_one_machine_having_the_file_makes_it_present():
    """The services run different knights; a union, not an intersection."""
    ev = _evidence(_missing(machine="m1"), _sample([_log_line("INFO", "a")], machine="m2"))

    assert ev.file_status(BROKER_LOG) == "present"
    assert len(ev.lines(evidence.ANY_LINE_PATTERN)) == 1


def test_another_kind_of_error_is_unknown_not_absent():
    """"Not authorized" must never be read as "the code never ran"."""
    ev = _evidence(_sample([], return_code=2, error="'me' is not authorized"))

    assert ev.file_status(BROKER_LOG) == "unknown"


def test_a_file_nobody_asked_about_is_unknown():
    ev = _evidence(_sample([_log_line("INFO", "a")], BROKER_LOG))

    assert ev.file_status(PROXY_LOG) == "unknown"


def test_a_read_error_becomes_an_error_not_an_empty_answer():
    """Exit 2 is usually a wrong filename; stderr turns that into a one-step fix."""
    query = evidence.GrepQuery(pattern="x", log_filename="panda-typo.log", service=evidence.JEDI)
    payload = {
        "expected_machines": ["m1"],
        "results": [
            {
                "machine_name": "m1",
                "result": "",
                "stderr": "panda-typo.log: No such file or directory",
                "return_code": 2,
            }
        ],
    }

    (result,) = evidence._results_from(query, payload)

    assert result.error == "panda-typo.log: No such file or directory"
    assert not result.conclusive


def test_a_machine_that_never_answered_is_recorded_as_silent():
    """Silence and "found nothing" are different facts about a machine."""
    query = evidence.GrepQuery(pattern="x", log_filename=BROKER_LOG, service=evidence.JEDI)
    payload = {
        "expected_machines": ["m1", "m2"],
        "results": [{"machine_name": "m1", "result": "hit\n", "return_code": 0}],
    }

    results = {r.machine: r for r in evidence._results_from(query, payload)}

    assert results["m1"].lines == ["hit"]
    assert results["m2"].error == "no result returned"


def test_evidence_round_trips_through_a_file(tmp_path):
    """Fetching and checking are separate steps, so the record has to survive
    the gap intact -- that is what lets the gates re-run offline."""
    ev = _evidence(_sample([_log_line("DEBUG", "a")], truncated=True))
    path = tmp_path / "nested" / "evidence.json"

    ev.save(path)
    back = evidence.Evidence.load(path)

    assert back.results[0].truncated
    assert evidence.effective_level(back, log_filename=BROKER_LOG) == "DEBUG"


def test_log_format_recognised_fails_when_no_line_carries_a_level():
    """A pattern that matches nothing looks exactly like a gate that passed.

    Every later production gate reads the log by matching against this
    format, so it is checked directly rather than assumed.
    """
    ev = _evidence(_sample(["something in another format"]))

    result = gates.log_format_recognised(ev)

    assert not result.passed
    assert "log format is not" in result.failures[0]


def test_log_format_recognised_names_a_query_that_did_not_run():
    """Not authorized, or a wrong filename, is not production disagreeing."""
    ev = _evidence(_sample([], return_code=2, error="'me' is not authorized"))

    result = gates.log_format_recognised(ev)

    assert not result.passed
    assert result.failures == ["jedi/m1: 'me' is not authorized"]


def test_log_format_recognised_passes_on_a_well_formed_sample():
    assert gates.log_format_recognised(_evidence(_sample([_log_line("INFO", "a")]))).passed


def _stage(tag: str, level: str, owner: str, files: list[str], **kwargs) -> FilterStageNode:
    return FilterStageNode(
        map_id=MAP_ID,
        derived_from=VERSION,
        name=f"f:{tag}",
        owner=owner,
        criteria_tag=tag,
        log_level=level,
        log_files=files,
        **kwargs,
    )


BROKER = "pandajedi/jedibrokerage/AtlasProdJobBroker.py::doBrokerage"


def test_an_observable_below_the_threshold_is_a_promise_the_map_cannot_keep():
    """A DEBUG line does not exist in a file whose logger runs at INFO.

    Worse than having no observable: a strategy would spend a step fetching
    a line that is never written.
    """
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        filter_stages=[
            _stage("-diskIO", "debug", BROKER, [BROKER_LOG]),
            _stage("-status", "info", BROKER, [BROKER_LOG]),
        ],
    )
    ev = _evidence(_sample([_log_line("INFO", "a")], BROKER_LOG))

    result = gates.observables_are_emitted(fragment, ev)

    assert result.checked == 2
    assert not result.passed
    assert result.failures == [
        f"-diskIO at {BROKER} emits at DEBUG but {BROKER_LOG} at INFO"
    ]


def test_emitting_in_either_candidate_file_is_enough():
    """The proxy mixins run under two processes; emitting in one makes the
    observable real, so an all-candidates rule is what the code needs."""
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        filter_stages=[_stage("-x", "debug", BROKER, [BROKER_LOG, PROXY_LOG])],
    )
    ev = _evidence(
        _sample([_log_line("INFO", "a")], BROKER_LOG),
        _sample([_log_line("DEBUG", "b")], PROXY_LOG),
    )

    assert gates.observables_are_emitted(fragment, ev).passed


def test_a_stage_whose_file_never_existed_is_not_a_logging_failure():
    """A path that never ran is a different finding from one that runs
    quietly, and conflating them would blame the map for the deployment."""
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        filter_stages=[_stage("-x", "debug", BROKER, [BROKER_LOG])],
    )
    ev = _evidence(_missing())

    emitted = gates.observables_are_emitted(fragment, ev)
    live = gates.code_paths_are_live(fragment, ev)

    assert emitted.failures == []
    # Says the file is on no machine rather than "no sample": the absence is a
    # finding of its own, and calling it missing evidence made one fact read as
    # two separate problems.
    assert emitted.inconclusive == [
        f"{BROKER_LOG} is on no machine, so -x at {BROKER} never ran here"
    ]
    assert not live.passed
    assert BROKER_LOG in live.failures[0]


def test_a_stage_the_source_gives_no_file_for_is_inconclusive():
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        filter_stages=[_stage("-x", "debug", BROKER, [])],
    )

    result = gates.observables_are_emitted(fragment, _evidence(_sample([_log_line("INFO", "a")])))

    assert result.passed
    assert result.inconclusive == [f"no log file is named in the source for -x at {BROKER}"]


def test_an_inconclusive_row_leads_with_its_reason():
    """The report summarises these rows by their first line, so the reason has
    to come before the anchor -- otherwise the summary keeps the part that says
    where and drops the part that says what."""
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        filter_stages=[_stage("-x", "debug", BROKER, [BROKER_LOG]), _stage("-y", "debug", BROKER, [])],
    )

    rows = gates.observables_are_emitted(fragment, _evidence(_missing())).inconclusive

    assert all(not row.startswith("-") for row in rows)


def test_code_paths_are_live_passes_when_every_file_is_there():
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        filter_stages=[_stage("-x", "info", BROKER, [BROKER_LOG])],
    )

    assert gates.code_paths_are_live(fragment, _evidence(_sample([_log_line("INFO", "a")]))).passed


def test_the_package_places_a_module_in_a_machine_group():
    """A first approximation only: JEDI opens its own TaskBuffer, so
    ``db_proxy_mods`` called from a knight logs to JEDI's files even though it
    lives in ``pandaserver``.  Sound for choosing which service declares a
    file, which is all it is used for."""
    assert evidence.service_for_module("pandajedi/jedibrokerage/x.py") == evidence.JEDI
    assert evidence.service_for_module("pandaserver/api/v1/pilot_api.py") == evidence.SERVER


def test_a_gate_summary_separates_issues_from_inconclusive():
    """Conflating them is the failure mode this whole half is built to avoid."""
    result = gates.GateResult(
        gate="g", passed=False, checked=3, failures=["a"], inconclusive=["b", "c"]
    )

    assert result.summary() == "[FAIL] g: 3 checked (1 issue(s), 2 inconclusive)"


# ---------------------------------------------------------------------------
# Comparing two builds
# ---------------------------------------------------------------------------
#
# The only check that can see a threshold move.  Comparing the map with its
# own source agrees with whatever the source now says, and a condition is
# never echoed to a log, so a build from the wrong release explains a decision
# with a number that has since changed -- convincingly.


def _stage_for_diff(tag: str, conditions: list[str], line: int, **kwargs) -> FilterStageNode:
    return FilterStageNode(
        map_id=MAP_ID,
        derived_from=kwargs.pop("version", VERSION),
        name=f"f:{tag}",
        owner="pandajedi/jedibrokerage/AtlasProdJobBroker.py::doBrokerage",
        criteria_tag=tag,
        conditions=conditions,
        anchor=Anchor(
            package="pandajedi",
            file="pandajedi/jedibrokerage/AtlasProdJobBroker.py",
            line_start=line,
            line_end=line,
        ),
        **kwargs,
    )


def _fragment_of(*nodes, version: str = VERSION) -> MapFragment:
    fragment = MapFragment(map_id=MAP_ID, derived_from=version)
    for node in nodes:
        if isinstance(node, FilterStageNode):
            fragment.filter_stages.append(node)
        elif isinstance(node, JunctionNode):
            fragment.junctions.append(node)
        elif isinstance(node, BoundaryNode):
            fragment.boundaries.append(node)
    return fragment


def test_a_node_that_only_moved_is_not_a_change():
    """The reason node identity is a signature and not a position.

    ``update_job`` moved file and line between two releases; a key built from
    those would have called one boundary two.  In a refactor this is most of
    the map, so it has to generate no noise.
    """
    old = _fragment_of(_stage_for_diff("-disk", ["a < b"], line=100))
    new = _fragment_of(_stage_for_diff("-disk", ["a < b"], line=140))

    result = diff.compare(old, new)

    assert result.changes == []
    assert result.moved == ["FilterStage f:-disk"]
    assert result.unchanged == 0


def test_an_edited_condition_is_reported_as_drift():
    """The failure no gate can see: the escape hatch is gone, and a map built
    from the older release would still offer it as the reason a site survived."""
    old = _fragment_of(
        _stage_for_diff("-disk", ["size < threshold and 'skip_RSE_check' not in catchall"], 100)
    )
    new = _fragment_of(_stage_for_diff("-disk", ["size < threshold"], 100))

    result = diff.compare(old, new)

    (drifted,) = result.drift()
    assert drifted.field == "conditions"
    assert "skip_RSE_check" in drifted.before
    assert "skip_RSE_check" not in drifted.after


def test_a_changed_interface_is_a_change_but_not_drift():
    """Both matter; only one of them silently rewrites an explanation."""
    def boundary(values):
        return BoundaryNode(
            map_id=MAP_ID,
            derived_from=VERSION,
            name="b:pilot:update_job",
            system="pilot",
            interface="pilot_api::update_job",
            carried_values=values,
        )

    result = diff.compare(
        _fragment_of(boundary(["job_id"])), _fragment_of(boundary(["job_id", "job_status"]))
    )

    assert result.drift() == []
    (change,) = result.other()
    assert change.field == "carried_values"


def _junction_for_diff(branches: list[Branch]) -> JunctionNode:
    return JunctionNode(
        map_id=MAP_ID,
        derived_from=VERSION,
        name="j:setStatus",
        subject="JediTaskSpec.status",
        owner="pandajedi/jediorder/ContentsFeeder.py::feed",
        branches=branches,
    )


def test_branches_are_matched_by_outcome_not_position():
    """A branch keeps its outcome while the condition reaching it is edited.

    Comparing by position would render an inserted branch as "everything
    below here changed" and bury the one edit that matters.
    """
    old = _fragment_of(
        _junction_for_diff(
            [
                Branch(outcome="ready", path_condition=["nFiles > 0"], order=0),
                Branch(outcome="pending", path_condition=["else"], order=1),
            ]
        )
    )
    new = _fragment_of(
        _junction_for_diff(
            [
                Branch(outcome="broken", path_condition=["corrupt"], order=0),
                Branch(outcome="ready", path_condition=["nFiles > 2"], order=1),
                Branch(outcome="pending", path_condition=["else"], order=2),
            ]
        )
    )

    result = diff.compare(old, new)

    drifted = result.drift()
    assert [c.node for c in drifted] == ["j:setStatus → ready"]
    assert drifted[0].before == "[nFiles > 0]"
    assert drifted[0].after == "[nFiles > 2]"
    assert any(c.field == "branch" and c.after == "broken" for c in result.other())


def test_losing_a_polled_entry_is_reported():
    """Not cosmetic: it is the difference between a stall that clears itself
    and one that does not."""
    def junction(entries):
        node = _junction_for_diff([Branch(outcome="ready", path_condition=[], order=0)])
        node.entry_points = entries
        return node

    polled = EntryPoint(trigger="polled", entry="jediorder/ContentsFeeder.py")
    message = EntryPoint(trigger="message", entry="jedimsgprocessor/feeder.py")

    result = diff.compare(
        _fragment_of(junction([polled, message])), _fragment_of(junction([message]))
    )

    (change,) = [c for c in result.changes if c.field == "entry_points"]
    assert "polled" in change.before
    assert "polled" not in change.after


def test_added_and_removed_nodes_are_named():
    old = _fragment_of(_stage_for_diff("-disk", ["a"], 100))
    new = _fragment_of(_stage_for_diff("-rse", ["a"], 100))

    result = diff.compare(old, new)

    assert result.removed == ["FilterStage f:-disk"]
    assert result.added == ["FilterStage f:-rse"]


def test_two_identical_builds_report_as_identical():
    old = _fragment_of(_stage_for_diff("-disk", ["a < b"], 100))
    new = _fragment_of(_stage_for_diff("-disk", ["a < b"], 100))

    result = diff.compare(old, new)

    assert result.identical
    assert result.unchanged == 1


# ---------------------------------------------------------------------------
# Reading absence correctly, and the gates that depend on it
# ---------------------------------------------------------------------------
#
# Seeing a line proves it is emitted however little of the log was read.  Not
# seeing one proves nothing unless everything the query matched came back.
# Every gate below turns on that asymmetry.


def _tag_sample(lines: list[str], filename: str = BROKER_LOG, **kwargs):
    return _sample(lines, filename, pattern=evidence.TAG_PATTERN, **kwargs)


def test_the_level_query_keeps_a_histogram_and_no_lines():
    """A 112 MB evidence file was 626,041 lines standing in for twenty rows."""
    (query,) = evidence.sample_queries({BROKER_LOG: evidence.JEDI})

    assert query.keep_lines == 0
    assert query.tail_bytes == evidence.LEVEL_TAIL_BYTES


def test_the_level_window_is_small_so_the_cap_is_not_reached():
    """``tail -c N | rg -m M`` returns the *first* M matches inside the window.

    A wide window with a cap therefore yields the oldest lines in it and comes
    back truncated, and a truncated sample cannot license "production does not
    emit this".  The window is what has to be small, not the cap.
    """
    assert evidence.LEVEL_TAIL_BYTES < evidence.DEFAULT_TAIL_BYTES
    assert evidence.LEVEL_MAX_MATCHES > evidence.DEFAULT_MAX_MATCHES


def test_a_kept_line_budget_does_not_hide_that_something_matched():
    """``matched`` is read wherever the question is "did anything match", so a
    query keeping no lines still answers it."""
    query = evidence.GrepQuery(
        pattern=evidence.ANY_LINE_PATTERN,
        log_filename=BROKER_LOG,
        service=evidence.JEDI,
        keep_lines=0,
    )
    payload = {
        "expected_machines": ["m1"],
        "results": [
            {
                "machine_name": "m1",
                "result": _log_line("INFO", "a") + "\n" + _log_line("DEBUG", "b") + "\n",
                "return_code": 0,
            }
        ],
    }

    (result,) = evidence._results_from(query, payload)

    assert result.lines == []
    assert result.matched == 2
    assert result.level_counts == {"INFO": 1, "DEBUG": 1}


def test_a_suppressed_observable_needs_a_complete_sample_to_fail():
    """The bug this exists to stop: a capped sample that happened to contain no
    DEBUG line would have been read as "production runs at INFO"."""
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        filter_stages=[_stage("-diskIO", "debug", BROKER, [BROKER_LOG])],
    )

    complete = gates.observables_are_emitted(
        fragment, _evidence(_sample([_log_line("INFO", "a")]))
    )
    partial = gates.observables_are_emitted(
        fragment, _evidence(_sample([_log_line("INFO", "a")], truncated=True))
    )

    assert not complete.passed
    assert partial.passed
    assert "sample incomplete" in partial.inconclusive[0]


def test_seeing_the_line_needs_no_complete_sample():
    """The other half of the asymmetry: positive evidence stands on its own."""
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        filter_stages=[_stage("-diskIO", "debug", BROKER, [BROKER_LOG])],
    )
    ev = _evidence(_sample([_log_line("DEBUG", "a")], truncated=True))

    result = gates.observables_are_emitted(fragment, ev)

    assert result.passed
    assert result.inconclusive == []


def test_a_tag_production_emits_with_no_stage_is_a_blind_spot():
    """The system naming a cut the map cannot explain.

    Positive evidence, so it fails the gate even from a partial read.
    """
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        filter_stages=[_stage("-disk", "info", BROKER, [BROKER_LOG])],
    )
    ev = _evidence(
        _tag_sample(
            [
                _log_line("INFO", "skip site=X criteria=-disk"),
                _log_line("INFO", "skip site=Y criteria=-newcut"),
            ],
            truncated=True,
        )
    )

    result = gates.tags_are_known(fragment, ev)

    assert not result.passed
    assert result.failures == ["production logs -newcut (1x) and the map has no stage for it"]


def test_a_tag_the_map_has_and_production_did_not_show_needs_a_complete_sample():
    """Otherwise the gate accuses the extraction of inventing a stage that is
    merely quiet."""
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        filter_stages=[
            _stage("-disk", "info", BROKER, [BROKER_LOG]),
            _stage("-rse", "info", BROKER, [BROKER_LOG]),
        ],
    )
    lines = [_log_line("INFO", "skip site=X criteria=-disk")]

    partial = gates.tags_are_known(fragment, _evidence(_tag_sample(lines, truncated=True)))
    complete = gates.tags_are_known(fragment, _evidence(_tag_sample(lines)))

    assert partial.inconclusive == []
    # Stays in ``inconclusive`` even when the sample is whole, and says "this
    # window" rather than "production never": writing the line needs the branch
    # to fire, so a quiet cut and a wrong tag look the same at any sample size.
    assert complete.inconclusive == [
        f"{BROKER_LOG}: the map has -rse and this window shows no cut using it"
    ]
    assert complete.passed
    assert complete.sample == gates.COMPLETE
    assert partial.sample == gates.PARTIAL


def _transition(task: str, status: str, stamp: str, filename: str = "panda-ContentsFeeder.log"):
    """One ``set task_status=`` line, as the knights write it."""
    return _sample(
        [
            _log_line(
                "INFO", f"<jediTaskID={task}> set task_status={status}", name="ContentsFeeder"
            )
        ],
        filename=filename,
        pattern=evidence.TRANSITION_PATTERN,
    ).model_copy(
        update={
            "lines": [
                f"2026-08-30 {stamp} ContentsFeeder: INFO     "
                f"<jediTaskID={task}> set task_status={status}"
            ]
        }
    )


def _task_status_fragment(outcomes, vocabulary=(), selected=(), runtime=()):
    branches = [Branch(outcome=o, tier=1) for o in outcomes]
    branches += [Branch(outcome=f"runtime({name})", tier=2) for name in runtime]
    return MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        subjects=[
            SubjectNode(
                map_id=MAP_ID,
                derived_from=VERSION,
                name=evidence.TRANSITION_SUBJECT,
                spec_class="JediTaskSpec",
                attribute="status",
                vocabulary=list(vocabulary),
                selected_values=list(selected),
            )
        ],
        junctions=[
            JunctionNode(
                map_id=MAP_ID,
                derived_from=VERSION,
                name="j",
                owner="pandajedi/jediorder/ContentsFeeder.py::feed",
                subject=evidence.TRANSITION_SUBJECT,
                branches=branches,
            )
        ],
    )


def test_a_task_history_is_rebuilt_across_files_and_machines():
    """The database keeps the current status and the one before it, so a
    sequence only exists in the log -- and one task's history is spread over the
    refiner, the generator and the post-processor, on whichever machine picked it
    up.  Ordering therefore comes from the timestamps, not from the order the
    lines came back in."""
    ev = _evidence(
        _transition("7", "running", "12:00:09,000", "panda-JobGenerator.log"),
        _transition("7", "defined", "12:00:00,000", "panda-TaskRefiner.log"),
        _transition("7", "ready", "12:00:05,000"),
        # Two knights logging the same value in a row is not a transition.
        _transition("7", "ready", "12:00:06,000"),
    )

    histories = evidence.observed_task_status(ev)

    assert [status for _stamp, status, _file in histories["7"]] == [
        "defined",
        "ready",
        "running",
    ]
    assert evidence.observed_pairs(histories) == Counter(
        {("defined", "ready"): 1, ("ready", "running"): 1}
    )


def test_a_departure_survives_a_gap_and_a_pair_does_not():
    """Seeing a task in one status and later in another proves it left the
    first, whether or not the step between was sampled.  The pair is what a gap
    invents, which is why one is evidence and the other a report."""
    histories = evidence.observed_task_status(
        _evidence(
            _transition("7", "defined", "12:00:00,000"),
            _transition("7", "running", "12:00:09,000"),
        )
    )

    assert evidence.observed_departures(histories) == Counter({"defined": 1})


def test_a_status_production_sets_and_no_branch_produces_is_a_finding():
    """What gate nine is for, and what it added over the offline check: a value
    the code declares and production sets, which no branch produces, cannot be a
    stale declaration any more."""
    fragment = _task_status_fragment(
        outcomes=["toabort"], vocabulary=["toabort", "aborted"], runtime=["newTaskStatus"]
    )
    ev = _evidence(
        _transition("7", "toabort", "12:00:00,000"),
        _transition("7", "aborted", "12:00:05,000"),
    )

    result = gates.transitions_are_explained(fragment, ev)

    assert not result.passed
    assert result.failures == [
        "production sets aborted (1x), the code declares it, and no branch in the map "
        "produces it (1 writer(s) of this subject decide the value at run time)"
    ]
    assert result.kind == gates.MAP_DEFECT


def test_an_undeclared_status_may_be_a_run_time_value():
    """Tier 2 is by design -- the writer is in the map and the outcome is not --
    so a value nothing declares is not evidence of a missing writer."""
    fragment = _task_status_fragment(outcomes=["defined"], runtime=["newTaskStatus"])
    ev = _evidence(
        _transition("7", "defined", "12:00:00,000"),
        _transition("7", "surprising", "12:00:05,000"),
    )

    result = gates.transitions_are_explained(fragment, ev)

    assert result.passed
    assert "may be a value one of the 1 run-time writer(s) computes" in result.inconclusive[0]


def test_a_status_nothing_selects_on_is_not_a_failure():
    """A task can be moved on by a junction that selects it by id or on a
    command, with no status predicate for the map to have missed."""
    fragment = _task_status_fragment(
        outcomes=["defined", "running"], selected=["defined"]
    )
    ev = _evidence(
        _transition("7", "defined", "12:00:00,000"),
        _transition("7", "running", "12:00:05,000"),
        _transition("7", "defined", "12:00:09,000"),
    )

    result = gates.transitions_are_explained(fragment, ev)

    assert result.passed
    assert any("moved tasks out of running" in row for row in result.inconclusive)


def test_conformance_passes_when_production_stays_inside_the_map():
    fragment = _task_status_fragment(
        outcomes=["defined", "ready", "running"], selected=["defined", "ready"]
    )
    ev = _evidence(
        _transition("7", "defined", "12:00:00,000"),
        _transition("7", "ready", "12:00:05,000"),
        _transition("7", "running", "12:00:09,000"),
    )

    result = gates.transitions_are_explained(fragment, ev)

    assert result.passed and result.failures == []
    assert result.checked == 3


def test_a_transposed_funnel_step_is_a_finding():
    """"Which step cut the candidates" is a question about position, so an
    order that disagrees makes every answer off by one step.

    Two traversals rather than one, because a chain that really runs its steps
    out of order does so every time, while a single reversal is what one wrap
    of a correctly-ordered chain produces.
    """
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        filter_stages=[
            _stage("-a", "info", BROKER, [BROKER_LOG], funnel_label="disk check", order=0),
            _stage("-b", "info", BROKER, [BROKER_LOG], funnel_label="memory check", order=1),
        ],
    )
    ev = _evidence(
        _sample(
            [
                _log_line("INFO", f"<jediTaskID={task}> {n} candidates passed {label}")
                for task in (1, 2)
                for n, label in ((100, "memory check"), (80, "disk check"))
            ],
            pattern=evidence.FUNNEL_PATTERN,
        )
    )

    result = gates.funnel_order_matches(fragment, ev)

    assert not result.passed
    assert "after" in result.failures[0]


def test_repeated_traversals_do_not_read_as_transpositions():
    """Brokerage walks the chain many times under one task and dataset -- once
    observed thirteen times -- and the log marks no boundary between them.

    Every attempt to cut the run into traversals cost false findings: read as
    one sequence 22, splitting on a return to the start 8, splitting on the
    size of the backward jump 2.  A majority needs no boundary, because each
    wrap is one reversed pair against many in order.
    """
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        filter_stages=[
            _stage("-a", "info", BROKER, [BROKER_LOG], funnel_label="status check", order=0),
            _stage("-b", "info", BROKER, [BROKER_LOG], funnel_label="backlog check", order=1),
        ],
    )
    # Three traversals of a two-step chain: the wrap from "backlog" back to
    # "status" is exactly the shape a size rule cannot tell from a swap.
    run = ["status check", "backlog check"] * 3
    ev = _evidence(
        _sample(
            [_log_line("INFO", f"<jediTaskID=1 datasetID=2> 5 candidates passed {label}") for label in run],
            pattern=evidence.FUNNEL_PATTERN,
        )
    )

    assert gates.funnel_order_matches(fragment, ev).passed


def test_two_chains_in_one_log_file_are_told_apart():
    """One log file is not one chain.  ``AtlasProdTaskBroker`` runs its own
    steps and then calls the job broker, whose steps are written through the log
    slot it was handed, so both chains land in the task broker's file.  Both
    begin with a step named ``status check``, so read as one chain every
    traversal contributed one pair in order and one reversed -- 4836 against
    4833 in production, a majority decided by nothing.

    The map cannot say which chains share a file; the evidence can, without
    segmenting anything.  A label only one chain uses names that chain, so the
    chains present are the ones whose unique labels appear, and a label shared
    by two of *those* is dropped.
    """
    task_log, job_log = "panda-AtlasProdTaskBroker.log", "panda-AtlasProdJobBroker.log"
    task_chain = "pandajedi/jedibrokerage/AtlasProdTaskBroker.py::runImpl"
    job_chain = "pandajedi/jedibrokerage/AtlasProdJobBroker.py::doBrokerage"
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        filter_stages=[
            _stage("-x", "info", task_chain, [task_log], funnel_label="status check", order=0),
            _stage("-y", "info", task_chain, [task_log], funnel_label="backlog check", order=1),
            _stage("-z", "info", task_chain, [task_log], funnel_label="endpoint check", order=2),
            _stage("-p", "info", job_chain, [job_log], funnel_label="status check", order=0),
            _stage("-q", "info", job_chain, [job_log], funnel_label="opportunistic check", order=1),
            _stage("-r", "info", job_chain, [job_log], funnel_label="memory check", order=2),
        ],
    )
    # What production shows, twice over: the task broker's three steps, then the
    # job broker's three, the second of which starts with the shared label.
    run = [
        "status check", "backlog check", "endpoint check",
        "status check", "opportunistic check", "memory check",
    ] * 3
    ev = _evidence(
        _sample(
            [
                _log_line("INFO", f"<jediTaskID=1> 5 candidates passed {label}")
                for label in run
            ],
            filename=task_log,
            pattern=evidence.FUNNEL_PATTERN,
        )
    )

    result = gates.funnel_order_matches(fragment, ev)

    assert result.passed, result.failures
    # Both chains were recognised as writing here, and only the shared label was
    # given up -- the unshared ones are still compared.
    assert result.checked > 0


def test_a_transposition_in_a_shared_file_is_still_found():
    """Dropping the ambiguous label must not turn the gate off: the steps only
    one of the two chains names are still compared."""
    task_log, job_log = "panda-AtlasProdTaskBroker.log", "panda-AtlasProdJobBroker.log"
    task_chain = "pandajedi/jedibrokerage/AtlasProdTaskBroker.py::runImpl"
    job_chain = "pandajedi/jedibrokerage/AtlasProdJobBroker.py::doBrokerage"
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        filter_stages=[
            _stage("-x", "info", task_chain, [task_log], funnel_label="status check", order=0),
            _stage("-y", "info", task_chain, [task_log], funnel_label="backlog check", order=1),
            _stage("-z", "info", task_chain, [task_log], funnel_label="endpoint check", order=2),
            _stage("-p", "info", job_chain, [job_log], funnel_label="status check", order=0),
        ],
    )
    # "endpoint" consistently before "backlog", against the map.
    run = ["status check", "endpoint check", "backlog check"] * 3
    ev = _evidence(
        _sample(
            [
                _log_line("INFO", f"<jediTaskID=1> 5 candidates passed {label}")
                for label in run
            ],
            filename=task_log,
            pattern=evidence.FUNNEL_PATTERN,
        )
    )

    result = gates.funnel_order_matches(fragment, ev)

    assert not result.passed
    assert "backlog check" in result.failures[0]
    # The chain is named, because a file can hold more than one.
    assert "runImpl" in result.failures[0]


def test_stages_sharing_a_step_count_as_one_position():
    """The funnel counts steps, not stages: ``AtlasAnalJobBroker`` rejects for
    two reasons under "disk check", and a position has to mean the step."""
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        filter_stages=[
            _stage("-a", "info", BROKER, [BROKER_LOG], funnel_label="disk check", order=0),
            _stage("-b", "info", BROKER, [BROKER_LOG], funnel_label="disk check", order=1),
            _stage("-c", "info", BROKER, [BROKER_LOG], funnel_label="memory check", order=2),
        ],
    )
    ev = _evidence(
        _sample(
            [
                _log_line("INFO", "<jediTaskID=1> 9 candidates passed disk check"),
                _log_line("INFO", "<jediTaskID=1> 8 candidates passed memory check"),
            ],
            pattern=evidence.FUNNEL_PATTERN,
        )
    )

    assert gates.funnel_order_matches(fragment, ev).passed


def test_an_early_exit_is_not_a_transposition():
    """The sample spans many tasks and a chain can exit early, so production
    shows a prefix or a gapped run of the map's order."""
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        filter_stages=[
            _stage("-a", "info", BROKER, [BROKER_LOG], funnel_label="disk check", order=0),
            _stage("-b", "info", BROKER, [BROKER_LOG], funnel_label="memory check", order=1),
            _stage("-c", "info", BROKER, [BROKER_LOG], funnel_label="pilot check", order=2),
        ],
    )
    ev = _evidence(
        _sample(
            [
                _log_line("INFO", "100 candidates passed disk check"),
                _log_line("INFO", "40 candidates passed pilot check"),
            ],
            pattern=evidence.FUNNEL_PATTERN,
        )
    )

    assert gates.funnel_order_matches(fragment, ev).passed


def _code_fragment() -> MapFragment:
    """A map that decodes ``taskBufferErrorCode`` and nothing else."""
    return MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        value_enums=[
            ValueEnumNode(
                map_id=MAP_ID,
                derived_from=VERSION,
                name="taskbuffer.ErrorCode.EC_Kill",
                namespace="taskbuffer.ErrorCode.EC",
                constant="EC_Kill",
                value=100,
            )
        ],
        enumeration_writes=[
            EnumerationWrite(
                map_id=MAP_ID,
                derived_from=VERSION,
                field="JobSpec.taskBufferErrorCode",
                constant="EC_Kill",
                namespace="taskbuffer.ErrorCode.EC",
                anchor=Anchor(package="pandaserver", file="x.py", line_start=1),
            )
        ],
    )


def _job_evidence(*jobs: dict) -> evidence.Evidence:
    return evidence.Evidence(
        fetched_at="2026-08-30T00:00:00+00:00",
        records=[evidence.JobRecords(task_id="7", jobs=list(jobs))],
        tasks_available=1,
    )


def test_a_code_the_index_cannot_decode_is_a_finding():
    """P2 means to use the index as a lookup, so a value it cannot name makes
    it answer wrongly rather than not at all."""
    ev = _job_evidence(
        {"PandaID": 1, "taskBufferErrorCode": 100},
        {"PandaID": 2, "taskBufferErrorCode": 999},
    )

    result = gates.error_codes_are_known(_code_fragment(), ev)

    assert not result.passed
    assert result.failures == [
        "production sets taskBufferErrorCode=999 (1x) and "
        "taskbuffer.ErrorCode.EC has no constant with that value"
    ]
    assert result.checked == 2


def test_the_field_decides_which_constant_a_value_means():
    """Why the binding exists at all.  ``100`` was recorded 338 times in one
    sample and meant three different things: ``EC_Kill`` in
    ``taskBufferErrorCode``, ``EC_Setupper`` in ``ddmErrorCode``, ``EC_Watcher``
    in ``jobDispatcherErrorCode``.  Without the field the index is a coin flip.
    """
    fragment = _code_fragment()
    fragment.value_enums.append(
        ValueEnumNode(
            map_id=MAP_ID,
            derived_from=VERSION,
            name="jobdispatcher.ErrorCode.EC_Watcher",
            namespace="jobdispatcher.ErrorCode.EC",
            constant="EC_Watcher",
            value=100,
        )
    )
    fragment.enumeration_writes.append(
        EnumerationWrite(
            map_id=MAP_ID,
            derived_from=VERSION,
            field="JobSpec.jobDispatcherErrorCode",
            constant="EC_Watcher",
            namespace="jobdispatcher.ErrorCode.EC",
            anchor=Anchor(package="pandaserver", file="Watcher.py", line_start=1),
        )
    )

    assert gates._decodes(fragment) == {
        "taskBufferErrorCode": {"taskbuffer.ErrorCode.EC"},
        "jobDispatcherErrorCode": {"jobdispatcher.ErrorCode.EC"},
    }
    # Both hold 100 and both decode, each through its own enumeration.
    ev = _job_evidence(
        {"PandaID": 1, "taskBufferErrorCode": 100, "jobDispatcherErrorCode": 100}
    )
    assert gates.error_codes_are_known(fragment, ev).passed


def test_a_field_the_map_binds_to_no_enumeration_is_not_checked():
    """``pilotErrorCode`` is the pilot's, and the map holds the pilot as a
    boundary on purpose.  Checking it against this index would report every
    value it carries as unknown, which says nothing about the index."""
    ev = _job_evidence({"PandaID": 1, "pilotErrorCode": 1099})

    result = gates.error_codes_are_known(_code_fragment(), ev)

    assert result.passed and result.checked == 0


def test_a_field_at_rest_is_not_a_coded_value():
    """An error-code field holds zero when there was no error, so counting it
    would make every index look as though it were missing the ordinary case."""
    ev = _job_evidence(
        {"PandaID": 1, "taskBufferErrorCode": 0},
        {"PandaID": 2, "taskBufferErrorCode": None},
    )

    result = gates.error_codes_are_known(_code_fragment(), ev)

    assert result.passed and result.checked == 0


def test_a_step_production_counts_and_the_map_lacks_is_a_finding():
    """The funnel counter's positive direction: candidates demonstrably went
    somewhere and the map has no step to name it.

    Sample-size independent -- the line proves the step -- which is why the
    other direction is not reported at all.
    """
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        filter_stages=[
            _stage("-a", "info", BROKER, [BROKER_LOG], funnel_label="disk check", order=0),
        ],
    )
    ev = _evidence(
        _sample(
            [
                _log_line("INFO", "100 candidates passed disk check"),
                _log_line("INFO", "80 candidates passed temporary problem check"),
            ],
            pattern=evidence.FUNNEL_PATTERN,
        )
    )

    result = gates.funnel_steps_are_known(fragment, ev)

    assert not result.passed
    assert result.failures == [
        "production counts a cut at 'temporary problem check' (1x) "
        "and the map has no step for it"
    ]
    # A step the map has and the window does not is never reported: reaching it
    # is what writes the line, so absence has two causes and no sample size
    # separates them.
    assert result.inconclusive == []


def test_a_step_named_after_a_run_time_value_is_matched_by_its_frame():
    """``AtlasProdTaskBroker`` names a step after a threshold it reads from
    configuration, so production writes two names for one step."""
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        filter_stages=[
            _stage(
                "-a",
                "info",
                BROKER,
                [BROKER_LOG],
                funnel_label="endpoint check with DISK_THRESHOLD={} TB",
                order=0,
            ),
        ],
    )
    ev = _evidence(
        _sample(
            [
                _log_line("INFO", "9 candidates passed endpoint check with DISK_THRESHOLD=10 TB"),
                _log_line("INFO", "4 candidates passed endpoint check with DISK_THRESHOLD=1000 TB"),
            ],
            pattern=evidence.FUNNEL_PATTERN,
        )
    )

    assert gates.funnel_steps_are_known(fragment, ev).passed


def test_one_reversed_observation_is_one_traversals_worth_of_noise():
    """Wrapping is not rare here -- pairs the map orders correctly still come
    back 4440 reversed against 6119 in order -- and every traversal that wraps
    contributes exactly one reversal.  A pair whose whole evidence is a single
    reversal is indistinguishable from one wrap, and has no majority in it.
    """
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        filter_stages=[
            _stage("-a", "info", BROKER, [BROKER_LOG], funnel_label="disk check", order=0),
            _stage("-b", "info", BROKER, [BROKER_LOG], funnel_label="memory check", order=1),
        ],
    )
    once = _evidence(
        _sample(
            [
                _log_line("INFO", "100 candidates passed memory check"),
                _log_line("INFO", "80 candidates passed disk check"),
            ],
            pattern=evidence.FUNNEL_PATTERN,
        )
    )
    assert gates.funnel_order_matches(fragment, once).passed

    # Seen again, it is no longer an anecdote.
    twice = _evidence(
        _sample(
            [
                _log_line("INFO", "<jediTaskID=1> 100 candidates passed memory check"),
                _log_line("INFO", "<jediTaskID=1> 80 candidates passed disk check"),
                _log_line("INFO", "<jediTaskID=2> 100 candidates passed memory check"),
                _log_line("INFO", "<jediTaskID=2> 80 candidates passed disk check"),
            ],
            pattern=evidence.FUNNEL_PATTERN,
        )
    )
    assert not gates.funnel_order_matches(fragment, twice).passed


def test_a_step_the_map_places_twice_cannot_testify_about_order():
    """``AtlasProdJobBroker`` runs "temporary problem check" early and returns
    when it was called for a task-brokerage hint, and otherwise runs it last.

    Order takes the first position, production mostly runs the other, and the
    chain is doing exactly what the map says -- so the label is dropped for the
    same reason a label two chains share is.
    """
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        filter_stages=[
            _stage("-a", "info", BROKER, [BROKER_LOG], funnel_label="temporary problem check", order=0),
            _stage("-b", "info", BROKER, [BROKER_LOG], funnel_label="IO check", order=1),
            _stage("-c", "info", BROKER, [BROKER_LOG], funnel_label="temporary problem check", order=2),
        ],
    )
    ev = _evidence(
        _sample(
            [
                _log_line("INFO", "100 candidates passed IO check"),
                _log_line("INFO", "80 candidates passed temporary problem check"),
            ],
            pattern=evidence.FUNNEL_PATTERN,
        )
    )

    result = gates.funnel_order_matches(fragment, ev)

    assert result.passed
    # Nothing left to compare: the other label is alone once the ambiguous one
    # is dropped, so the gate says so rather than passing on a pair it read.
    assert result.checked == 0


def test_a_template_seen_in_production_is_confirmed():
    """The only direction there is.

    A template missing from the log has two causes nothing distinguishes -- the
    wording moved on, or that branch did not fire in the window -- and writing
    the line requires the branch to fire, so no sample size separates them.  So
    only what was seen is returned.
    """
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        filter_stages=[
            _stage(
                "-disk",
                "info",
                BROKER,
                [BROKER_LOG],
                emits=["  skip site={} due to disk shortage criteria=-disk"],
                funnel_label="disk check",
            ),
            _stage(
                "-space",
                "info",
                BROKER,
                [BROKER_LOG],
                emits=["skip nucleus since disk shortage ({0} TB) criteria=-space"],
                funnel_label="space check",
            ),
        ],
    )
    ev = _evidence(
        _tag_sample([_log_line("INFO", "  skip site=X due to disk shortage criteria=-disk")])
    )

    assert gates.templates_confirmed(fragment, ev) == [f"disk check -disk ({BROKER_LOG})"]


def test_two_stages_sharing_a_tag_are_told_apart():
    """``AtlasAnalJobBroker`` emits ``-disk`` from both its scratch-disk and its
    storage-space check, so a row keyed on the tag alone printed one label twice
    and identified neither."""
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        filter_stages=[
            _stage(
                "-disk",
                "info",
                BROKER,
                [BROKER_LOG],
                emits=["skip site={} due to small scratch disk"],
                funnel_label="scratch disk check",
            ),
            _stage(
                "-disk",
                "info",
                BROKER,
                [BROKER_LOG],
                emits=["skip site={} since output endpoint undefined"],
                funnel_label="storage space check",
            ),
        ],
    )
    ev = _evidence(
        _tag_sample(
            [
                _log_line("INFO", "skip site=X due to small scratch disk criteria=-disk"),
                _log_line("INFO", "skip site=Y since output endpoint undefined criteria=-disk"),
            ]
        )
    )

    assert gates.templates_confirmed(fragment, ev) == [
        f"scratch disk check -disk ({BROKER_LOG})",
        f"storage space check -disk ({BROKER_LOG})",
    ]


def test_the_stem_is_the_longest_fixed_run_not_the_prefix():
    """``"  skip site={} due to disk shortage"`` begins with ten characters
    shared by half the file; what identifies the message is on the other side
    of the interpolation.  Taking the prefix skipped nearly every template."""
    assert gates._template_stem("  skip site={} due to disk shortage") == "due to disk shortage"
    # Below the floor, so not distinctive enough to search for.
    assert gates._template_stem("{} sites left") is None
    assert gates._template_stem("skip={}") is None
    assert gates._template_stem("{} of {}") is None


def test_returning_everything_and_writing_it_all_down_are_different():
    """Two halves of the trip.  A gate that searches the lines and concludes
    something is missing needs the stricter one, or trimming to keep_lines
    turns a kept sample into a false absence."""
    kept_all = evidence.GrepResult(
        query=evidence.GrepQuery(
            pattern=evidence.TAG_PATTERN, log_filename=BROKER_LOG, service=evidence.JEDI
        ),
        machine="m1",
        lines=["a", "b"],
        matched=2,
        return_code=0,
    )
    trimmed = kept_all.model_copy(update={"lines": ["a"]})

    assert kept_all.conclusive and kept_all.complete
    assert trimmed.conclusive and not trimmed.complete


def test_an_unconfirmed_template_is_not_reported_at_all():
    """It was reported once, as a ratio with the unconfirmed listed, and that is
    the half a reader cannot act on: a rare rejection reads as a stale template.
    Where drift *can* be detected is between two source trees -- ``diff-map``."""
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        filter_stages=[
            _stage(
                "-space",
                "info",
                BROKER,
                [BROKER_LOG],
                emits=["skip nucleus since disk shortage ({0} TB) criteria=-space"],
            )
        ],
    )
    trimmed = _tag_sample([_log_line("INFO", "something else criteria=-space")]).model_copy(
        update={"matched": 900, "truncated": True}
    )

    assert gates.templates_confirmed(fragment, _evidence(trimmed)) == []


# ---------------------------------------------------------------------------
# What the report says
#
# The gates were right and the report was not: it printed a deployment fact and
# an extraction miss with the same word, never said whether the sample it read
# was whole, and gave five incomparable counts the same label.  These pin the
# distinctions rather than the layout.
# ---------------------------------------------------------------------------


def _result(gate: str, **kwargs) -> gates.GateResult:
    return gates.GateResult(gate=gate, passed=not kwargs.get("failures"), **kwargs)


def test_a_deployment_difference_is_not_a_defect_and_does_not_fail_a_build():
    """The distinction the report exists to make.  ``code-paths-are-live``
    failing means the map is right and this deployment does not run that code;
    there is nothing to change, so it must not read as -- or exit like -- a
    genuine miss."""
    difference = _result(
        "code-paths-are-live", kind=gates.DEPLOYMENT_FACT, failures=["x is on no machine"]
    )
    defect = _result("tags-are-known", kind=gates.MAP_DEFECT, failures=["-newcut has no stage"])
    unusable = _result("log-format-recognised", kind=gates.CHECK_BROKEN, failures=["not authorized"])

    assert difference.verdict == "DIFFERS" and not difference.actionable
    assert defect.verdict == "FAIL" and defect.actionable
    assert unusable.verdict == "BROKEN" and unusable.actionable
    assert _result("funnel-order-matches").verdict == "PASS"


def test_the_real_gates_declare_which_kind_they_are():
    """Set on the gate rather than in the report, because which of the three a
    failure is depends on what the gate compared, not on how it is printed."""
    ev = _evidence(_sample([_log_line("DEBUG", "x")]), _missing("panda-Gone.log"))
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        filter_stages=[_stage("-x", "debug", BROKER, [BROKER_LOG])],
    )

    kinds = {r.gate: r.kind for r in gates.run_production(fragment, ev)}

    assert kinds["log-format-recognised"] == gates.CHECK_BROKEN
    assert kinds["code-paths-are-live"] == gates.DEPLOYMENT_FACT
    assert kinds["observables-are-emitted"] == gates.DEPLOYMENT_FACT
    assert kinds["tags-are-known"] == gates.MAP_DEFECT
    assert kinds["funnel-order-matches"] == gates.MAP_DEFECT


def test_every_count_carries_what_it_counts():
    """Query answers, log files, filter stages, tags and step pairs are not
    comparable quantities; printed as "checked" they invited the comparison."""
    ev = _evidence(_sample([_log_line("DEBUG", "x")]))
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        filter_stages=[_stage("-x", "debug", BROKER, [BROKER_LOG])],
    )

    units = {r.gate: r.unit for r in gates.run_production(fragment, ev)}

    assert units["code-paths-are-live"] == "log files"
    assert units["observables-are-emitted"] == "filter stages"
    assert "checked" not in units.values()
    assert "filter stages" in _result("observables-are-emitted", unit="filter stages").summary()


def test_the_sample_section_says_which_conclusions_are_load_bearing(capsys):
    """The worst of the report's problems was silence here: a gate that could
    not conclude anything looked exactly like one that checked and found
    nothing."""
    ev = _evidence(
        _sample([_log_line("DEBUG", "x")]),
        _tag_sample([_log_line("INFO", "skip criteria=-disk")], truncated=True),
    )

    check_map._report_sample(ev)
    out = capsys.readouterr().out

    assert "levels  complete" in out
    assert "tags    PARTIAL" in out
    assert "1 of 2 answer(s) hit a bound" in out
    # The reason a complete sample would not help either, said once.
    assert "branch has to fire to write" in out


def test_the_verdict_leads_with_the_worst_kind(capsys):
    """A check that did not run makes every other verdict unsafe to read, so it
    outranks a defect; a deployment difference asks for no change at all."""
    results = [
        _result(
            "code-paths-are-live",
            kind=gates.DEPLOYMENT_FACT,
            finding="the map describes code this deployment never runs",
            failures=["a"],
        ),
        _result(
            "tags-are-known",
            kind=gates.MAP_DEFECT,
            finding="the map is missing a cut production makes",
            failures=["b"],
        ),
    ]

    check_map._report_verdict(results)
    out = capsys.readouterr().out

    assert "1 map defect · 1 deployment difference" in out
    assert "the map is missing a cut production makes." in out


def test_a_clean_verdict_says_so(capsys):
    check_map._report_verdict([_result("funnel-order-matches")])

    assert "nothing to change" in capsys.readouterr().out


def test_the_level_table_collapses_to_the_file_that_differs(capsys):
    """Thirty rows whose whole payload was one sentence -- and the ``… 12 more``
    that truncated them was hiding one of the run's two findings."""
    ev = _evidence(
        _sample([_log_line("DEBUG", "x")], filename="panda-A.log"),
        _sample([_log_line("INFO", "y")], filename="panda-B.log"),
    )

    check_map._report_production(ev, [], [], top=10, full=False)
    out = capsys.readouterr().out

    assert "log level    1 at DEBUG, 1 at INFO" in out
    # Only the file that differs from the majority; the rest carry no
    # information one at a time.
    assert "not at DEBUG: panda-B.log" in out
    assert "panda-A.log" not in out


def test_full_puts_every_folded_row_back(capsys):
    ev = _evidence(_sample([_log_line("DEBUG", "x")], filename="panda-A.log"))

    check_map._report_production(ev, [], [f"disk check -disk ({BROKER_LOG})"], top=10, full=True)
    out = capsys.readouterr().out

    assert "panda-A.log" in out and "DEBUG=1" in out
    assert f"disk check -disk ({BROKER_LOG})" in out


def test_agreement_is_reported_one_sided(capsys):
    """No ratio: a line nobody saw may simply not have fired, and the ratio
    invited reading a rare rejection as a stale template."""
    check_map._report_production(_evidence(), [], ["a (f.log)", "b (f.log)"], top=10, full=False)
    out = capsys.readouterr().out

    assert "2 diagnostic line(s) of the map found verbatim" in out
    assert "one-sided" in out
    assert "/" not in out.split("agreement")[1].splitlines()[0]


def test_the_gate_table_names_the_question_and_marks_a_partial_sample(capsys):
    check_map._report_gates(
        [
            _result(
                "tags-are-known",
                checked=31,
                unit="observed tags",
                question="is every cut production makes in the map?",
                sample=gates.PARTIAL,
            ),
            _result("code-paths-are-live", checked=22, unit="log files", kind=gates.DEPLOYMENT_FACT),
        ]
    )
    out = capsys.readouterr().out

    assert "31 observed tags" in out
    assert "is every cut production makes in the map?" in out
    assert "[partial]" in out
    # A gate whose answer is not a sample gets no marker rather than a hedge.
    assert out.splitlines()[-1].count("[") == 0


def test_not_concluded_collapses_to_one_line_per_gate(capsys):
    check_map._report_unconcluded(
        [_result("observables-are-emitted", inconclusive=[f"row {i}" for i in range(7)])],
        full=False,
    )
    out = capsys.readouterr().out

    assert "not concluded (7)" in out
    assert "… 6 more (--full)" in out


def test_evidence_age_and_bound_sizes_are_readable():
    assert check_map._age("2026-08-30T00:00:00+00:00").endswith("ago")
    assert check_map._age("not a date") == "age unknown"
    assert check_map._size(64 * 1024 * 1024) == "64MB"
    assert check_map._size(1 << 30) == "1GB"
    assert (check_map._count(1, "map defect"), check_map._count(0, "map defect")) == (
        "1 map defect",
        "0 map defects",
    )


def test_a_clipped_row_keeps_the_half_that_says_what_happened():
    """Which is why the gate's rows lead with their reason."""
    row = f"{BROKER_LOG} is on no machine, so -x at {BROKER} never ran here"

    assert check_map._clip(row).startswith(f"{BROKER_LOG} is on no machine")
    assert check_map._clip("short") == "short"
