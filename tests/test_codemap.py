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
from unittest.mock import AsyncMock

import pytest

from bamboo.codemap import gates
from bamboo.codemap.gitsource import blob_sha
from bamboo.codemap.models import (
    Anchor,
    Branch,
    JunctionNode,
    MapFragment,
    SourceModule,
    ValueEnumNode,
)
from bamboo.codemap.panda import attribution
from bamboo.codemap.panda.recognizers import boundary, errorcode, progress
from bamboo.models.graph_element import NodeType

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
    subjects, junctions, coverage = progress.extract(modules, MAP_ID, VERSION)
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
    subjects, junctions, coverage = progress.extract(modules, MAP_ID, VERSION)
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
