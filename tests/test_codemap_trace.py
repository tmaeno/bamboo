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
