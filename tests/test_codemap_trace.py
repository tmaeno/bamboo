"""Walking the source of one arm to say why it ran with the value it did.

Against snippets written to a temporary tree rather than against the corpus:
the walk is about Python, and a fixture that spells out the shape under test
says what is being claimed in a way a real 600-line function does not.  The
corpus checks that follow are in ``test_codemap_strategy``.

Nothing here reads the installed distribution and nothing here touches
production.
"""

from __future__ import annotations

from pathlib import Path

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


def _walk(roots, *, file, owner, lines, **kwargs):
    return trace_mod.walk(
        roots,
        file=f"{PACKAGE}/{file}",
        owner=f"{PACKAGE}/{file}::{owner}",
        lines=lines,
        classify=provenance.classify,
        **kwargs,
    )


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
