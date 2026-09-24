"""Walking the source of one arm to say why it ran with the value it did.

Against snippets written to a temporary tree rather than against the corpus:
the walk is about Python, and a fixture that spells out the shape under test
says what is being claimed in a way a real 600-line function does not.  The
corpus checks that follow are in ``test_codemap_strategy``.

Nothing here reads the installed distribution and nothing here touches
production.
"""

from __future__ import annotations

import re
from pathlib import Path

from bamboo.codemap import models
from bamboo.codemap import trace as trace_mod
from bamboo.codemap.models import (
    STOP_CONFIG,
    STOP_EXTERNAL,
    STOP_PARAMETER,
    STOP_UPSTREAM,
    TRACE_BINDING,
    TRACE_HANDOVER,
    TRACE_LOOP,
    TRACE_UNBOUND,
    TRACE_WRITE,
    Handover,
)
from bamboo.codemap.panda import pathcond, provenance

PACKAGE = "pandajedi"


def _tree(tmp_path: Path, **files: str) -> dict[str, Path]:
    root = tmp_path / PACKAGE
    root.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (root / name).write_text(text)
    return {PACKAGE: root}


def _walked(roots, *, file, owner, lines, **kwargs):
    return trace_mod.walk(
        roots,
        file=f"{PACKAGE}/{file}",
        owner=f"{PACKAGE}/{file}::{owner}",
        lines=lines,
        classify=provenance.classify,
        **kwargs,
    )


def _walk(roots, *, file, owner, lines, **kwargs):
    """The two answers most of these tests are about: the steps and the note."""
    walked = _walked(roots, file=file, owner=owner, lines=lines, **kwargs)
    return walked.steps, walked.note


KNIGHT = '''\
class Feeder:
    def run(self):
        for spec in self.specs:
            broken = False
            try:
                meta = self.ddmIF.getDatasetMetaData(spec.name)
            except Exception:
                if errtype == Fatal:
                    broken = True
            if spec.empty():
                broken = True
            if broken:
                spec.status = "tobroken"
'''
BROKEN_LINE = 13


def test_the_arm_is_the_first_step_and_carries_its_own_guard(tmp_path):
    roots = _tree(tmp_path, feeder=KNIGHT)

    steps, note = _walk(roots, file="feeder", owner="run", lines=[BROKEN_LINE])

    assert note == ""
    first = steps[0]
    assert (first.kind, first.line, first.value) == (TRACE_WRITE, BROKEN_LINE, "'tobroken'")
    assert first.guards == ["broken"]
    # The loop is not an ``if``, so the path condition cannot mention it.
    assert any(u.startswith(pathcond.UNSEEN_LOOP) for u in first.unseen)


def test_a_flag_set_in_several_places_comes_back_as_several_steps(tmp_path):
    # The eager version had to fold this into one condition string per branch,
    # and that is where its loop rule and its length cap came from.  A
    # use-time answer may be a set, so three sites are three steps.
    roots = _tree(tmp_path, feeder=KNIGHT)

    steps, _ = _walk(roots, file="feeder", owner="run", lines=[BROKEN_LINE])

    bindings = [s for s in steps if s.kind == TRACE_BINDING and s.name == "broken"]
    assert sorted(s.value for s in bindings) == ["False", "True", "True"]


def test_a_handler_gets_the_terminal_of_what_the_try_body_called(tmp_path):
    # The handler ran because something above it raised, and naming what that
    # was is the only way the walk reaches "DDM said so" from an arm whose own
    # right-hand side is the constant True.
    roots = _tree(tmp_path, feeder=KNIGHT)

    steps, _ = _walk(roots, file="feeder", owner="run", lines=[BROKEN_LINE])

    raised = [s for s in steps if s.terminal == STOP_EXTERNAL]
    assert [s.detail for s in raised] == ["DDM / Rucio"]
    assert any(u.startswith(pathcond.UNSEEN_EXCEPT) for u in raised[0].unseen)


def test_a_guard_is_walked_as_well_as_the_value(tmp_path):
    roots = _tree(
        tmp_path,
        feeder='''\
class Feeder:
    def run(self):
        threshold = getConfigValue("jedi", "SCOUT_LIMIT")
        if self.count > threshold:
            spec.status = "exhausted"
''',
    )

    steps, _ = _walk(roots, file="feeder", owner="run", lines=[5])

    # The arm's own value is a literal, so everything below comes from the
    # guard: the threshold it compares against, and the attribute it reads.
    threshold = [s for s in steps if s.name == "threshold"]
    assert [(s.terminal, s.detail) for s in threshold] == [
        (STOP_CONFIG, "jedi.SCOUT_LIMIT")
    ]


def test_a_loop_target_is_followed_to_what_is_iterated(tmp_path):
    roots = _tree(
        tmp_path,
        commando='''\
class Worker:
    def runImpl(self):
        tasks = self.taskList.get(10)
        for task_id, command_map in tasks:
            command = command_map["command"]
            spec.status = command_map["done"]
''',
    )

    steps, _ = _walk(
        roots,
        file="commando",
        owner="runImpl",
        lines=[6],
        handovers=[
            Handover(
                entry=f"{PACKAGE}/commando",
                via="start",
                reached_by="dispatch",
                fields={"taskList": "taskList"},
            )
        ],
    )

    loop = [s for s in steps if s.kind == TRACE_LOOP]
    assert [(s.name, s.value) for s in loop] == [("command_map", "an element of tasks")]


def test_a_worker_field_crosses_to_the_function_that_built_it(tmp_path):
    # The edge no call graph has: the knight hands its rows over through a
    # constructor and a thread, so the arm's own module cannot reach them.
    roots = _tree(
        tmp_path,
        commando='''\
class Commando:
    def start(self):
        rows = self.taskBufferIF.getTasksToExecCommand_JEDI(vo, label)
        taskList = ListWithLock(rows)
        thr = CommandoThread(taskList)


class CommandoThread:
    def runImpl(self):
        tasks = self.taskList.get(10)
        spec.status = tasks[0]
''',
    )

    steps, _ = _walk(
        roots,
        file="commando",
        owner="runImpl",
        lines=[11],
        handovers=[
            Handover(
                entry=f"{PACKAGE}/commando",
                via="start",
                reached_by="dispatch",
                fields={"taskList": "taskList"},
            )
        ],
    )

    crossed = [s for s in steps if s.kind == TRACE_HANDOVER]
    assert [(s.name, s.value, s.owner.endswith("::start")) for s in crossed] == [
        ("self.taskList", "taskList", True)
    ]
    assert [(s.terminal, s.detail) for s in steps if s.terminal] == [
        (STOP_UPSTREAM, "getTasksToExecCommand_JEDI")
    ]


def test_a_field_no_handover_explains_stops_instead_of_guessing(tmp_path):
    roots = _tree(
        tmp_path,
        commando='''\
class Worker:
    def runImpl(self):
        spec.status = self.mystery
''',
    )

    steps, _ = _walk(roots, file="commando", owner="runImpl", lines=[3])

    stopped = [s for s in steps if s.name == "self.mystery"]
    assert [(s.kind, s.terminal) for s in stopped] == [(TRACE_UNBOUND, STOP_PARAMETER)]


def test_a_parameter_stops_at_the_call_site(tmp_path):
    roots = _tree(
        tmp_path,
        refiner='''\
class Refiner:
    def apply(self, new_status):
        spec.status = new_status
''',
    )

    steps, _ = _walk(roots, file="refiner", owner="apply", lines=[3])

    assert [(s.name, s.terminal) for s in steps if s.terminal] == [
        ("new_status", STOP_PARAMETER)
    ]


def test_a_name_bound_from_itself_does_not_loop(tmp_path):
    roots = _tree(
        tmp_path,
        refiner='''\
class Refiner:
    def apply(self):
        count = 0
        count = count + 1
        spec.status = count
''',
    )

    steps, note = _walk(roots, file="refiner", owner="apply", lines=[5])

    assert note == ""
    assert sum(1 for s in steps if s.name == "count" and s.kind == TRACE_BINDING) == 2


def test_the_walk_refuses_a_tree_the_map_was_not_built_from(tmp_path):
    # A map built from the installed 1.0.2 read against a checkout printed a
    # plausible line of the wrong version once already.  A trace is shaped
    # like an answer, so it is refused rather than warned about.
    roots = _tree(tmp_path, feeder=KNIGHT)

    steps, note = _walk(
        roots,
        file="feeder",
        owner="run",
        lines=[BROKEN_LINE],
        expected_sha="0" * 40,
    )

    assert steps == []
    assert "not the one that was mapped" in note


def test_a_budget_that_runs_out_is_recorded_rather_than_hidden(tmp_path):
    roots = _tree(tmp_path, feeder=KNIGHT)

    steps, note = _walk(
        roots,
        file="feeder",
        owner="run",
        lines=[BROKEN_LINE],
        budget=trace_mod.Budget(files=1, functions=1, steps=2),
    )

    assert len(steps) == 2
    assert "budget" in note


def test_an_imported_name_is_not_reported_as_unexplained(tmp_path):
    # ``JediTaskSpec`` and ``Interaction`` are named by half the conditions in
    # this corpus and the walk has nothing to say about either, so keeping
    # them costs one unexplained step per mention and buries the rest.
    roots = _tree(
        tmp_path,
        commando='''\
from pandajedi.jedicore import Interaction


class Worker:
    def runImpl(self):
        if tmpStat == Interaction.SC_SUCCEEDED:
            spec.status = "passed"
''',
    )

    steps, _ = _walk(roots, file="commando", owner="runImpl", lines=[7])

    assert "Interaction" not in {s.name for s in steps}
    assert "tmpStat" in {s.name for s in steps}


def test_a_call_handover_carries_a_parameter_back_to_the_caller(tmp_path):
    # The two kinds of handover are keyed differently -- an attribute of the
    # worker for a dispatch, a parameter for a call -- and both are the same
    # crossing as far as the walk is concerned.
    roots = _tree(
        tmp_path,
        refiner='''\
class Refiner:
    def apply(self, new_status):
        spec.status = new_status
''',
    )

    steps, _ = _walk(
        roots,
        file="refiner",
        owner="apply",
        lines=[3],
        handovers=[
            Handover(
                entry=f"{PACKAGE}/refiner",
                via="apply",
                reached_by="call",
                fields={"new_status": "taskSpec.oldStatus"},
            )
        ],
    )

    crossed = [s for s in steps if s.kind == TRACE_HANDOVER]
    assert [(s.name, s.value) for s in crossed] == [("new_status", "taskSpec.oldStatus")]


def test_a_row_read_through_the_cursor_comes_from_the_database(tmp_path):
    # The commonest ending in the corpus, 120 steps of 4852.  Without this it
    # reads as "chosen by whoever called or built this", which is true of the
    # cursor object and false of the row.
    roots = _tree(
        tmp_path,
        proxy='''\
class DBProxy:
    def getTask(self):
        self.cur.execute(sql, varMap)
        res = self.cur.fetchone()
        spec.status = res[0]
''',
    )

    steps, _ = _walk(roots, file="proxy", owner="getTask", lines=[5])

    assert [(s.name, s.terminal) for s in steps if s.terminal] == [
        ("res", STOP_UPSTREAM)
    ]


def test_calling_a_method_on_self_is_not_reading_a_field(tmp_path):
    # The value is what the call returns; the method's body is a hop
    # downwards this walk does not take.  Counted as a field it produced
    # thirty steps saying a caller chose ``self.getClobObj``.
    roots = _tree(
        tmp_path,
        proxy='''\
class DBProxy:
    def getTask(self):
        spec.status = self.decode(blob)
''',
    )

    steps, _ = _walk(roots, file="proxy", owner="getTask", lines=[3])

    assert "self.decode" not in {s.name for s in steps}
    assert "blob" in {s.name for s in steps}


def test_a_caught_exception_names_what_the_body_could_raise(tmp_path):
    roots = _tree(
        tmp_path,
        feeder='''\
class Feeder:
    def run(self):
        try:
            self.ddmIF.getDatasetMetaData(name)
        except Exception as exc:
            spec.status = str(exc)
''',
    )

    steps, _ = _walk(roots, file="feeder", owner="run", lines=[6])

    caught = [s for s in steps if s.name == "exc"]
    assert [(s.terminal, s.detail) for s in caught] == [(STOP_EXTERNAL, "DDM / Rucio")]


def test_a_field_assigned_in_the_same_function_is_read_there(tmp_path):
    # The one the `holding` walkthrough turned up: the arm reads
    # ``self.job_status`` and the assignment is eleven lines above it, and the
    # walk handed the question to the constructor without ever looking.
    roots = _tree(
        tmp_path,
        adder='''\
class Adder:
    def process(self):
        if self.job.isCancelled():
            self.job_status = "failed"
        spec.status = self.job_status
''',
    )

    steps, _ = _walk(roots, file="adder", owner="process", lines=[5])

    assert [(s.kind, s.line, s.value) for s in steps if s.name == "self.job_status"] == [
        (TRACE_BINDING, 4, "'failed'")
    ]


def test_a_field_assigned_in_a_sibling_method_is_read_in_that_method(tmp_path):
    # Two claims in one: the sibling is found at all, and its guard is the
    # one that dominates *it*.  Reporting ``reset``'s ``if`` as though it had
    # dominated ``process``'s arm would be a lie with the shape of an answer.
    roots = _tree(
        tmp_path,
        adder='''\
class Adder:
    def reset(self, why):
        if why is None:
            self.job_status = "failed"

    def process(self):
        spec.status = self.job_status
''',
    )

    steps, _ = _walk(roots, file="adder", owner="process", lines=[7])

    bound = [s for s in steps if s.name == "self.job_status"]
    assert [(s.kind, s.line, s.value) for s in bound] == [(TRACE_BINDING, 4, "'failed'")]
    assert bound[0].owner.endswith("::reset")
    assert bound[0].guards == ["why is None"]


def test_a_field_written_in_several_methods_comes_back_as_several_steps(tmp_path):
    # ``self.jobs`` is written in three methods of ``setupper_atlas_plugin``
    # and which of them ran last is a fact about the run.  Folding them would
    # answer a question the text cannot settle.
    roots = _tree(
        tmp_path,
        setupper='''\
class Setupper:
    def run(self):
        self.jobs = fetched

    def correct(self):
        self.jobs = corrected

    def emit(self):
        spec.status = self.jobs
''',
    )

    steps, _ = _walk(roots, file="setupper", owner="emit", lines=[9])

    assert sorted(s.value for s in steps if s.name == "self.jobs") == [
        "corrected",
        "fetched",
    ]


def test_a_constructor_assignment_carries_the_question_to_the_call_site(tmp_path):
    # Stage three's reason for existing: ``self.job_status = job_status`` is
    # what turns a worker's attribute into a question about whoever built it.
    roots = _tree(
        tmp_path,
        adder='''\
class Adder:
    def __init__(self, job_status):
        self.job_status = job_status

    def process(self):
        spec.status = self.job_status
''',
    )

    steps, _ = _walk(roots, file="adder", owner="process", lines=[6])

    assert ("self.job_status", "job_status") in [(s.name, s.value) for s in steps]
    assert ("job_status", STOP_PARAMETER) in [(s.name, s.terminal) for s in steps]


def test_a_named_interface_is_still_the_interface_and_not_its_constructor(tmp_path):
    # Stage one stays first, and this is the shape that makes it load-bearing.
    # A field a *value* reads is settled by ``classify`` before the walk ever
    # asks for it -- ``self.cur.fetchall()`` terminates where it stands.  It
    # is a field a *guard* reads that arrives here with nothing decided, and
    # answering that one from its ``__init__`` would replace "this came from
    # the database" with "this came from an argument": true of the handle and
    # false of the value.  Measured over the corpus, this reaches the field
    # reading 13 times for ``self.cur``, twice for ``self.ddmIF`` and once for
    # ``self.taskBufferIF``.
    roots = _tree(
        tmp_path,
        proxy='''\
class DBProxy:
    def __init__(self, conn):
        self.cur = conn.cursor()

    def getTask(self):
        if self.cur.rowcount > 0:
            spec.status = "done"
''',
    )

    steps, _ = _walk(roots, file="proxy", owner="getTask", lines=[7])

    assert [(s.name, s.terminal) for s in steps if s.name == "self.cur"] == [
        ("self.cur", STOP_UPSTREAM)
    ]
    assert not [s for s in steps if s.value == "conn.cursor()"]


def test_a_field_nothing_assigns_still_stops_rather_than_guessing(tmp_path):
    # The fallback has to survive the three stages above it, or "the map names
    # no handover for this" becomes unreachable and a genuinely unexplained
    # field reads as an empty answer.
    roots = _tree(
        tmp_path,
        commando='''\
class Worker:
    def runImpl(self):
        spec.status = self.mystery
''',
    )

    steps, _ = _walk(roots, file="commando", owner="runImpl", lines=[3])

    assert [(s.kind, s.terminal) for s in steps if s.name == "self.mystery"] == [
        (TRACE_UNBOUND, STOP_PARAMETER)
    ]


def test_a_field_a_closure_assigns_is_not_read_as_the_enclosing_guard(tmp_path):
    # A nested function runs when it is called, not where it is written, so
    # its guards never dominated the arm.  Left out rather than reported with
    # the wrong condition attached.
    roots = _tree(
        tmp_path,
        worker='''\
class Worker:
    def run(self):
        def later():
            self.flag = "set-later"
        self.flag = "set-here"
        spec.status = self.flag
''',
    )

    steps, _ = _walk(roots, file="worker", owner="run", lines=[6])

    assert [s.value for s in steps if s.name == "self.flag"] == ["'set-here'"]


def test_a_field_a_base_class_in_the_same_module_assigns_is_found(tmp_path):
    roots = _tree(
        tmp_path,
        plugin='''\
class Base:
    def prepare(self):
        self.mode = "scouting"

class Impl(Base):
    def run(self):
        spec.status = self.mode
''',
    )

    steps, _ = _walk(roots, file="plugin", owner="run", lines=[7])

    assert [s.value for s in steps if s.name == "self.mode"] == ["'scouting'"]


def test_a_name_bound_by_a_with_statement_is_not_called_unbound(tmp_path):
    # ``with self.proxyPool.get() as proxy:`` is how every TaskBuffer method
    # reaches the database, and reaching definitions do not cover the form, so
    # the walk used to say the function does not bind a name it plainly binds.
    roots = _tree(
        tmp_path,
        buffer='''\
class Buffer:
    def store(self):
        with self.pool.get() as proxy:
            spec.status = proxy
''',
    )

    steps, _ = _walk(roots, file="buffer", owner="store", lines=[4])

    held = [s for s in steps if s.name == "proxy"]
    assert [s.kind for s in held] == [TRACE_BINDING]
    assert held[0].value == "self.pool.get()"


def test_a_name_bound_by_a_walrus_is_not_called_unbound(tmp_path):
    roots = _tree(
        tmp_path,
        feeder='''\
class Feeder:
    def run(self):
        if (staging := self.count_staging()) > 0:
            spec.status = staging
''',
    )

    steps, _ = _walk(roots, file="feeder", owner="run", lines=[4])

    held = [s for s in steps if s.name == "staging"]
    assert [s.kind for s in held] == [TRACE_BINDING]
    assert held[0].value == "self.count_staging()"


def test_a_module_level_constant_is_named_rather_than_stopped_at(tmp_path):
    # A fact reachable forward from the anchor belongs to the trace, and a
    # constant in the same file is as reachable as one a line above the arm.
    roots = _tree(
        tmp_path,
        broker='''\
SKIP_TYPES = ["prod_test"]

class Broker:
    def schedule(self):
        spec.status = SKIP_TYPES
''',
    )

    steps, _ = _walk(roots, file="broker", owner="schedule", lines=[5])

    held = [s for s in steps if s.name == "SKIP_TYPES"]
    assert [s.kind for s in held] == [TRACE_BINDING]
    assert held[0].value == "['prod_test']"


def test_a_name_nothing_in_the_module_binds_still_stops(tmp_path):
    roots = _tree(
        tmp_path,
        broker='''\
class Broker:
    def schedule(self):
        spec.status = mystery
''',
    )

    steps, _ = _walk(roots, file="broker", owner="schedule", lines=[3])

    assert [s.kind for s in steps if s.name == "mystery"] == [TRACE_UNBOUND]


def test_a_binding_from_a_nested_function_says_it_may_not_run_here(tmp_path):
    # ``assigned_expressions`` is shared with the build and walks into nested
    # defs, so the row is kept -- for a daemon whose work lives in inner
    # functions it is the only answer -- but it is not this function's own.
    roots = _tree(
        tmp_path,
        daemon='''\
class Daemon:
    def main(self):
        def later():
            end_status = "deleted"
            spec.status = end_status
''',
    )

    steps, _ = _walk(roots, file="daemon", owner="main", lines=[5])

    held = [s for s in steps if s.name == "end_status"]
    assert [s.value for s in held] == ["'deleted'"]
    assert any("nested-def" in entry for entry in held[0].unseen)


def test_a_binding_from_the_functions_own_body_is_not_marked(tmp_path):
    roots = _tree(
        tmp_path,
        daemon='''\
class Daemon:
    def main(self):
        end_status = "deleted"
        spec.status = end_status
''',
    )

    steps, _ = _walk(roots, file="daemon", owner="main", lines=[4])

    held = [s for s in steps if s.name == "end_status"]
    assert [s.value for s in held] == ["'deleted'"]
    assert not any("nested-def" in entry for entry in held[0].unseen)


# ---------------------------------------------------------------------------
# The line production prints for this arm
# ---------------------------------------------------------------------------


def _skeleton(roots, *, file, owner, lines, **kwargs):
    return _walked(roots, file=file, owner=owner, lines=lines, **kwargs).skeleton


def _printed(roots, *, file, owner, lines, **kwargs):
    return [
        row
        for row in _skeleton(roots, file=file, owner=owner, lines=lines, **kwargs)
        if row.kind == models.SKELETON_PRINT
    ]


def test_the_line_is_anchored_either_side_of_the_hole_the_value_fills(tmp_path):
    """What the map's shared sentence cannot do.

    ``line_shape`` anchors on the literal before the *first* hole, whichever
    hole that is.  Here the value goes in the second, and for 80 of the 182
    subjects that get a probe at all the first is the wrong one.
    """
    roots = _tree(
        tmp_path,
        refiner='''\
class Refiner:
    def apply(self, taskSpec):
        taskSpec.status = "exhausted"
        logger.debug(f"task {taskSpec.taskID} set to {taskSpec.status} by the goal check")
''',
    )

    (row,) = _printed(
        roots, file="refiner", owner="apply", lines=[3], observed="exhausted"
    )

    assert row.line == 4
    assert row.arms == [3]
    assert row.hole == "taskSpec.status"
    assert row.because == "the arm writes it"
    assert row.value == r"task\ [^\ ]*\ set\ to\ exhausted\ by\ the\ goal\ check"
    assert row.pattern == r"task\ [^\ ]*\ set\ to\ [^\ ]*\ by\ the\ goal\ check"


def test_a_hole_is_closed_against_the_literal_that_follows_it(tmp_path):
    """The defect this round opens with.

    ``set\\ .*=None`` matched ``set task_status=pending oldTask=False ...``:
    the run crossed the field name, its value and two more fields to reach an
    ``=None`` belonging to nothing in the statement.  The literal after the
    hole says where the hole has to stop.
    """
    roots = _tree(
        tmp_path,
        proxy='''\
class Proxy:
    def reassign(self, spec):
        spec.cloud = None
        logger.debug(f"reassigning to set {target}={value} for this task now")
''',
    )

    (row,) = _printed(roots, file="proxy", owner="reassign", lines=[3], observed="None")

    assert row.pattern == r"reassigning\ to\ set\ [^=]*=[^\ ]*\ for\ this\ task\ now"
    assert ".*" not in row.pattern
    assert not re.search(row.pattern, "set task_status=pending oldTask=False with (True, 1)")


def test_a_value_with_no_literal_after_it_gets_a_right_hand_anchor(tmp_path):
    """``newPrio=100`` matched ``newPrio=1000``.

    With nothing after the hole there is no literal to say the value ended,
    and a question that cannot miss is read by the eliminator as one that was
    answered.
    """
    roots = _tree(
        tmp_path,
        proxy='''\
class Proxy:
    def bump(self, spec):
        spec.currentPriority = newPrio
        logger.debug(f"raising the task priority to newPrio={newPrio}")
''',
    )

    (row,) = _printed(roots, file="proxy", owner="bump", lines=[3], observed="100")

    assert row.value.endswith(r"newPrio=100(?![\w.-])")
    assert re.search(row.value, "raising the task priority to newPrio=100")
    assert not re.search(row.value, "raising the task priority to newPrio=1000")


def test_a_message_with_too_little_literal_is_refused_and_says_so(tmp_path):
    """Refused rather than rendered, and said rather than left blank.

    ``to\\ None`` is seven characters of literal; it is not a sentence, and
    handing it to a reader to grep with is handing them a wrong conclusion.
    A row that silently lost its pattern would read as *production does not
    print here*, which is a different and also wrong thing to say.
    """
    roots = _tree(
        tmp_path,
        proxy='''\
class Proxy:
    def run(self, spec):
        spec.status = "deleting"
        logger.debug(f"to {spec.status}")
''',
    )

    (row,) = _printed(roots, file="proxy", owner="run", lines=[3], observed="deleting")

    assert row.pattern == ""
    assert row.value == ""
    assert row.refused == "too little literal (2 chars)"


def test_the_line_the_arm_writes_the_value_of_is_matched_too(tmp_path):
    """An assignment is one fact said from two ends, and production reports
    either end about as often."""
    roots = _tree(
        tmp_path,
        refiner='''\
class Refiner:
    def apply(self, taskSpec):
        newStatus = "exhausted"
        taskSpec.status = newStatus
        logger.debug(f"moving the task over to {newStatus} right now")
''',
    )

    (row,) = _printed(
        roots, file="refiner", owner="apply", lines=[4], observed="exhausted"
    )

    assert row.hole == "newStatus"
    assert row.value == r"moving\ the\ task\ over\ to\ exhausted\ right\ now"


def test_a_line_from_a_branch_the_arm_excludes_is_not_reachable_from_it(tmp_path):
    """The ``else`` of the arm's own ``if`` cannot have printed for this arm.

    Both lines are in the skeleton, because both are lines this function
    prints and a reader laying a region against it needs to see both.  What
    the arm's exclusion settles is which of them can have been printed
    *alongside the arm*, and that is what ``arms`` carries.
    """
    roots = _tree(
        tmp_path,
        refiner='''\
class Refiner:
    def apply(self, taskSpec):
        if taskSpec.useJumbo:
            taskSpec.status = "exhausted"
            logger.debug(f"jumbo task has been set to {taskSpec.status} here")
        else:
            taskSpec.status = "exhausted"
            logger.debug(f"plain task has been set to {taskSpec.status} here")
''',
    )

    rows = _printed(roots, file="refiner", owner="apply", lines=[4], observed="exhausted")

    jumbo = next(row for row in rows if "jumbo" in row.text)
    plain = next(row for row in rows if "plain" in row.text)
    assert jumbo.arms == [4]
    assert plain.arms == []


def test_a_line_before_a_continue_the_arm_is_past_is_not_reachable_from_it(tmp_path):
    """The exclusion a path condition alone cannot see.

    ``if X: ...; continue`` puts ``not (X)`` on everything after it, and only
    :func:`pathcond.enclosing_guards` knows that -- without it the skipped
    block's line reads as compatible with the write below, and the arm gets
    handed a sentence printed on the path it did not take.  Found in the
    corpus, in ``setTobeDeletedToDis``.
    """
    roots = _tree(
        tmp_path,
        proxy='''\
class Proxy:
    def setTobeDeleted(self, dsList):
        for tmpDS in dsList:
            if tmpDS.status == 'deleting':
                logger.debug(f"skipping {tmpDS.name} since its status={tmpDS.status}")
                continue
            tmpDS.status = 'deleting'
            logger.debug(f"now setting the status to {tmpDS.status} for {tmpDS.name}")
''',
    )

    rows = _printed(
        roots, file="proxy", owner="setTobeDeleted", lines=[7], observed="deleting"
    )

    skipped = next(row for row in rows if "skipping" in row.text)
    written = next(row for row in rows if "now setting" in row.text)
    assert skipped.arms == []
    assert written.arms == [7]


def test_a_hole_with_no_literal_beside_it_is_not_an_anchor(tmp_path):
    """``.*deleting.*`` matches every line in the file that says the word, and
    a question that cannot miss is read by the eliminator as one that was
    answered."""
    roots = _tree(
        tmp_path,
        proxy='''\
class Proxy:
    def run(self, spec):
        spec.status = "deleting"
        logger.debug(f"the dataset in question here{spec.name}{spec.status}")
''',
    )

    (row,) = _printed(roots, file="proxy", owner="run", lines=[3], observed="deleting")

    assert row.value == ""
    assert row.pattern == r"the\ dataset\ in\ question\ here"


def test_a_hole_reaching_the_value_is_used_only_where_the_arm_names_none(tmp_path):
    """Second best, and said so: a hole the arm names is a fact about this
    write, where one merely holding the value is a fact about the function."""
    roots = _tree(
        tmp_path,
        proxy='''\
class Proxy:
    def run(self, spec):
        newStatus = "deleting"
        spec.status = self.decide()
        logger.debug(f"we are about to move it over to {newStatus} now")
''',
    )

    (row,) = _printed(roots, file="proxy", owner="run", lines=[4], observed="deleting")

    assert row.hole == "newStatus"
    assert row.because == "newStatus is set to this value above"


def test_the_arms_own_hole_wins_over_one_that_merely_holds_the_value(tmp_path):
    roots = _tree(
        tmp_path,
        proxy='''\
class Proxy:
    def run(self, spec):
        newStatus = "deleting"
        spec.status = newStatus
        logger.debug(f"we are planning to use {newStatus} very shortly")
        logger.debug(f"we have now moved it to {spec.status} at long last")
''',
    )

    rows = _printed(roots, file="proxy", owner="run", lines=[4], observed="deleting")

    assert [row.because for row in rows] == ["the arm writes it", "the arm writes it"]
    assert sorted(row.hole for row in rows) == ["newStatus", "spec.status"]


def test_no_observed_value_means_no_value_in_any_hole(tmp_path):
    """Half the question is the value, and inventing one would put a pattern in
    front of a reader that nothing in the record supports.

    The skeleton still comes back.  A line that reports nothing about the
    value still says the code around it ran, which is the whole reason this
    round stopped requiring every row to name one.
    """
    roots = _tree(
        tmp_path,
        refiner='''\
class Refiner:
    def apply(self, taskSpec):
        taskSpec.status = "exhausted"
        logger.debug(f"the task has been set to {taskSpec.status} by the goal check")
''',
    )

    (row,) = _printed(roots, file="refiner", owner="apply", lines=[3])

    assert row.value == ""
    assert row.pattern == r"the\ task\ has\ been\ set\ to\ [^\ ]*\ by\ the\ goal\ check"


def test_a_tree_that_is_not_the_maps_has_no_skeleton(tmp_path):
    """The same refusal the steps get, for the same reason: a plausible line
    computed from the wrong release reads exactly like an answer."""
    roots = _tree(
        tmp_path,
        refiner='''\
class Refiner:
    def apply(self, taskSpec):
        taskSpec.status = "exhausted"
        logger.debug(f"the task has been set to {taskSpec.status} by the goal check")
''',
    )

    walked = _walked(
        roots,
        file="refiner",
        owner="apply",
        lines=[3],
        observed="exhausted",
        expected_sha="0" * 40,
    )

    assert walked.skeleton == []
    assert walked.note == "the walk was not run: this tree is not the one that was mapped"


def test_a_subtree_with_no_line_and_no_arm_is_left_out(tmp_path):
    """A skeleton is not the source again.

    A 540-line function whose every branch is reproduced buries the handful of
    rows a reader came for, which is the failure this round is elsewhere
    fixing in the report.
    """
    roots = _tree(
        tmp_path,
        proxy='''\
class Proxy:
    def run(self, spec):
        if spec.quiet:
            spec.counter += 1
            spec.other = self.decide()
        if spec.loud:
            spec.status = "deleting"
            logger.debug(f"we have now set the status to {spec.status} here")
''',
    )

    rows = _skeleton(roots, file="proxy", owner="run", lines=[7], observed="deleting")

    assert [row.text for row in rows if row.kind == models.SKELETON_BRANCH] == [
        "if spec.loud:"
    ]
    assert [row.line for row in rows if row.kind == models.SKELETON_ARM] == [7]


def test_the_nesting_says_which_rows_could_have_been_printed_together(tmp_path):
    """The part a flat list of lines cannot carry.

    Two rows under one ``if`` were printed together or not at all; two rows
    either side of an ``else`` cannot both have been.  A reader with a grepped
    region reads that off the indentation, so the headers are the artefact and
    not decoration.
    """
    roots = _tree(
        tmp_path,
        proxy='''\
class Proxy:
    def run(self, spec):
        try:
            if spec.loud:
                logger.debug("about to start working on this dataset")
                spec.status = "deleting"
        except Exception:
            logger.debug("failed while working on this dataset here")
''',
    )

    rows = _skeleton(roots, file="proxy", owner="run", lines=[6], observed="deleting")

    shape = [
        (row.kind, row.text if row.kind == models.SKELETON_BRANCH else row.line, row.depth)
        for row in rows
    ]
    assert shape == [
        (models.SKELETON_BRANCH, "try:", 0),
        (models.SKELETON_BRANCH, "if spec.loud:", 1),
        (models.SKELETON_PRINT, 5, 2),
        (models.SKELETON_ARM, 6, 2),
        (models.SKELETON_BRANCH, "except Exception:", 0),
        (models.SKELETON_PRINT, 8, 1),
    ]


def test_an_assignment_the_call_cannot_reach_is_not_one_of_its_messages(tmp_path):
    """``_logged_arguments`` hands back every assignment to the name.

    ``log.debug(msg_str)`` resolved one hop through a local gets all five
    assignments to ``msg_str`` in the function, and four of them are in
    branches the call cannot be reached from.  The build cannot narrow it --
    a stored answer has to hold for every caller -- but at use time it can:
    an assignment below the call cannot have run before it, and one the
    call's own path condition contradicts cannot have run on the way to it.
    """
    roots = _tree(
        tmp_path,
        proxy='''\
class Proxy:
    def run(self, spec):
        if spec.loud:
            msg_str = "the first branch decided to stop here"
        else:
            msg_str = "the second branch decided to stop here"
        logger.debug(msg_str)
        spec.status = "deleting"
        msg_str = "this one is assigned below the call above"
''',
    )

    rows = _printed(roots, file="proxy", owner="run", lines=[8], observed="deleting")

    assert [row.pattern for row in rows] == [
        r"the\ first\ branch\ decided\ to\ stop\ here",
        r"the\ second\ branch\ decided\ to\ stop\ here",
    ]



def test_a_shown_block_keeps_the_header_that_explains_it(tmp_path):
    """An ``except`` with no ``try`` over it, and an ``else`` with no condition.

    Dropping the headers of blocks that hold nothing produced both, and the
    nesting is the thing a reader is here for -- a header that explains a
    shown block is not decoration.  The rule that a statement holding neither
    a line nor an arm is left out entirely still applies; this is about the
    blocks *inside* one that is shown.
    """
    roots = _tree(
        tmp_path,
        proxy='''\
class Proxy:
    def run(self, spec):
        try:
            spec.counter += 1
        except Exception:
            logger.debug("failed while working on this dataset here")
        if spec.quiet:
            spec.other = self.decide()
        else:
            spec.status = "deleting"
''',
    )

    rows = _skeleton(roots, file="proxy", owner="run", lines=[10], observed="deleting")

    assert [row.text for row in rows if row.kind == models.SKELETON_BRANCH] == [
        "try:",
        "except Exception:",
        "if spec.quiet:",
        "else:",
    ]


# ---------------------------------------------------------------------------
# What a site has to be able to reach
# ---------------------------------------------------------------------------


def test_a_binding_below_the_arm_is_not_offered_as_an_explanation(tmp_path):
    """Pruning is by elimination, so a site that cannot have run on the way
    does not merely add noise -- it dilutes "one of these two" into "one of
    these three" and sends a reader down a path the code did not take."""
    roots = _tree(
        tmp_path,
        m="""
def f(spec):
    status = "new"
    if spec.kind == "merge":
        status = "merging"
    spec.status = status
    status = "done"
    return status
""",
    )

    steps, _note = _walk(roots, file="m", owner="f", lines=[6])

    listed = {(step.line, step.value) for step in steps if step.name == "status"}
    assert listed == {(3, "'new'"), (5, "'merging'")}


def test_a_binding_below_the_arm_inside_a_loop_that_holds_it_stays(tmp_path):
    """The next iteration reaches it.  Without this half the position rule is
    not a refinement but an error -- and it is the half ``reachable_calls`` has
    been missing since it was written."""
    roots = _tree(
        tmp_path,
        m="""
def f(items):
    token = "first"
    for item in items:
        spec.status = token
        token = item.next_token
""",
    )

    steps, _note = _walk(roots, file="m", owner="f", lines=[5])

    listed = {step.line for step in steps if step.name == "token"}
    assert listed == {3, 6}


def test_a_name_bound_only_below_the_arm_says_which(tmp_path):
    """"Not bound in this function" would be false and silence would leave the
    guard unexplained, so the step names the third possibility."""
    roots = _tree(
        tmp_path,
        m="""
def f(spec):
    if later == "x":
        spec.status = "done"
    later = "x"
""",
    )

    steps, _note = _walk(roots, file="m", owner="f", lines=[4])

    (step,) = [s for s in steps if s.name == "later"]
    assert step.kind == TRACE_UNBOUND
    assert "only below the arms" in step.detail
    assert step.terminal != STOP_PARAMETER


def test_a_message_assigned_at_the_foot_of_a_loop_still_reaches_its_call(tmp_path):
    """``reachable_calls`` dropped it on the plain position test, although
    every iteration but the first reaches it -- three rows came back over the
    41-case sample when the loop half was added."""
    roots = _tree(
        tmp_path,
        m="""
def f(items, log):
    message = "starting the first pass over the list"
    for item in items:
        log.debug(message)
        spec.status = "done"
        message = f"finished the pass over {item} in the list"
""",
    )

    walked = _walked(roots, file="m", owner="f", lines=[6])

    printed = [
        row.text
        for row in walked.skeleton
        if row.kind == models.SKELETON_PRINT and row.pattern
    ]
    assert any("finished the pass over" in one for one in printed), printed
    assert any("starting the first pass" in one for one in printed), printed


def test_a_message_the_walk_cannot_resolve_is_refused_rather_than_dropped(tmp_path):
    """A bare name bound only by an unpacking resolves to nothing, and the call
    used to leave no row at all -- where the same call holding an expression
    leaves a refusal.  A reader lays the skeleton over a grep of the log
    region, so a line there and not here reads as certainty."""
    roots = _tree(
        tmp_path,
        m="""
def f(api, log):
    out, err = api.list_datasets()
    if out is None:
        log.error(f"failed to list the datasets with {err}")
    else:
        log.debug(out)
        spec.status = "done"
""",
    )

    walked = _walked(roots, file="m", owner="f", lines=[8])

    refused = {
        row.line: row.refused
        for row in walked.skeleton
        if row.kind == models.SKELETON_PRINT and row.refused
    }
    assert refused == {7: "the message is not a literal this walk can render"}


def test_an_unpacked_binding_says_which_slot_it_took(tmp_path):
    """``tmpStat, taskSpec = get(...)`` rendered as ``taskSpec = get(...)`` says
    the call returns the spec, when it returns a pair whose first element is
    the status deciding whether the second means anything."""
    roots = _tree(
        tmp_path,
        m="""
def f(db):
    status, spec, extra = db.get_task()
    job.state = spec
""",
    )

    steps, _note = _walk(roots, file="m", owner="f", lines=[4])

    (step,) = [s for s in steps if s.name == "spec"]
    assert step.value == "db.get_task()[1]"


def test_a_starred_target_leaves_the_slot_unsaid(tmp_path):
    """Where the position of everything after the star depends on how long the
    value is, no slot is better than a guessed one."""
    roots = _tree(
        tmp_path,
        m="""
def f(db):
    first, *rest, spec = db.get_task()
    job.state = spec
""",
    )

    steps, _note = _walk(roots, file="m", owner="f", lines=[4])

    (step,) = [s for s in steps if s.name == "spec"]
    assert step.value == "db.get_task()"
