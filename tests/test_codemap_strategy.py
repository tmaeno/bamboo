"""Deriving an investigation from a stored Code Map.

Run against the in-memory example backend rather than mocks, for the reason
``test_codemap_lookup`` gives: a mock hands back whatever the test put in and
proves nothing about the encoding.  Here it also proves something about the
reading -- the strategy is built from decoded nodes, so a field that does not
survive storage would show up as a wrong verdict rather than as a type error.

Nothing here touches production.  Evidence is constructed from the queries the
strategy itself asks for, which keeps the fixtures honest about the shape a
real answer arrives in: per machine, with an exit code, and with the two ways of
being empty kept apart.
"""

from __future__ import annotations

import pytest

from bamboo.codemap import evidence as evidence_mod
from bamboo.codemap import strategy as strategy_mod
from bamboo.codemap.evidence import Evidence, GrepQuery, GrepResult
from bamboo.codemap.lookup import CodeMap
from bamboo.codemap.models import (
    ELIMINATED,
    SEEN,
    UNASKABLE,
    UNSETTLED,
    Anchor,
    Branch,
    Emit,
    EntryPoint,
    JunctionNode,
    MapFragment,
    SubjectNode,
    Symptom,
)
from bamboo.codemap.store import store_fragment
from bamboo.database.backends.examples.in_memory_backend import InMemoryGraphBackend

MAP_ID = "panda"
VERSION = "panda-server-source 1.0.2"
SUBJECT = "JediTaskSpec.status"

KNIGHT_LOG = "panda-ContentsFeeder.log"
OTHER_LOG = "panda-TaskCommando.log"
PROXY_LOGS = ["panda-DBProxy.log", "panda-JediDBProxy.log"]


def _subject(
    selected: list[str] | None = None,
    gates: list[str] | None = None,
    name: str = SUBJECT,
    selected_by: dict[str, list[str]] | None = None,
) -> SubjectNode:
    spec_class, _, attribute = name.rpartition(".")
    return SubjectNode(
        map_id=MAP_ID,
        derived_from=VERSION,
        name=name,
        spec_class=spec_class,
        attribute=attribute,
        selected_values=selected or [],
        selected_by=selected_by or {},
        selection_gates=gates or [],
    )


def _junction(
    owner: str,
    *branches: Branch,
    log_files: list[str] | None = None,
    caller_log_files: list[str] | None = None,
    owns_logger: bool = True,
    triggers: tuple[str, ...] = (),
    subject: str = SUBJECT,
) -> JunctionNode:
    return JunctionNode(
        map_id=MAP_ID,
        derived_from=VERSION,
        name=JunctionNode.make_name(MAP_ID, subject, owner),
        subject=subject,
        owner=owner,
        branches=list(branches),
        log_files=log_files or [],
        caller_log_files=caller_log_files or [],
        owns_logger=owns_logger,
        entry_points=[
            EntryPoint(trigger=trigger, entry=owner.split("::")[0]) for trigger in triggers
        ],
        anchor=Anchor(package="pandajedi", file=owner.split("::")[0], line_start=1),
    )


async def _map(fragment: MapFragment) -> CodeMap:
    backend = InMemoryGraphBackend()
    await backend.connect()
    await store_fragment(fragment, backend)
    return CodeMap(backend, map_id=MAP_ID)


def _result(query: GrepQuery, matched: int = 0, truncated: bool = False,
            missing: bool = False, lines: list[str] | None = None) -> GrepResult:
    if missing:
        return GrepResult(
            query=query,
            machine="m1",
            return_code=2,
            error="rg: panda-x.log: No such file or directory (os error 2)",
        )
    return GrepResult(
        query=query,
        machine="m1",
        matched=matched,
        lines=lines or [],
        truncated=truncated,
        return_code=0 if matched else 1,
    )


def _evidence(strategy, decide) -> Evidence:
    """Answer every query *strategy* asks for, as *decide* says.

    Built from the strategy's own queries so the patterns line up exactly.  A
    fixture that invented its own pattern would pass while the code asked
    something else entirely, which is the failure this whole layer is about.
    """
    results = []
    for query in strategy_mod.queries(strategy):
        role = (
            strategy_mod.CONTROL
            if query.pattern == evidence_mod.TRANSITION_PATTERN
            else strategy_mod.PROBE
        )
        results.append(_result(query, **decide(role, query.log_filename, query.service)))
    return Evidence(fetched_at="2026-09-05T00:00:00+00:00", results=results)


def _found(role, filename, service):
    """Every file carries the line, and the probe matches everywhere."""
    return {"matched": 1, "lines": ["... set task_status=pending"]}


def _quiet(role, filename, service):
    """Every file carries the line, and the probe matches nowhere."""
    return {"matched": 1} if role == strategy_mod.CONTROL else {"matched": 0}


# ---------------------------------------------------------------------------
# Deriving
# ---------------------------------------------------------------------------


async def test_every_junction_that_could_produce_the_value_is_a_candidate():
    """The fan-out is the system's, and the point is that it is complete.

    An expert asked why a task is pending faces the same set; what the map adds
    is that the enumeration is exhaustive and precomputed, so nothing is ranked
    away before the evidence has had a chance to rule it out.
    """
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        subjects=[_subject()],
        junctions=[
            _junction("a.py::f", Branch(outcome="pending"), log_files=[KNIGHT_LOG]),
            _junction("b.py::g", Branch(outcome="runtime(newStatus)", tier=2)),
            _junction("c.py::h", Branch(outcome="ready")),
        ],
    )
    strategy = await strategy_mod.derive(
        await _map(fragment), Symptom(subject=SUBJECT, observed="pending", task_id="1")
    )

    assert [(c.owner, c.tier) for c in strategy.candidates] == [("a.py::f", 1), ("b.py::g", 2)]


async def test_a_proxy_is_probed_at_its_callers_log_and_not_its_own():
    """The guard that stops eleven candidates being ruled out at once.

    ``panda-DBProxy.log`` is a true statement about where the mixin's own
    output lands and a false one about where a line saying it fired appears:
    the knight that called it writes that.  Probing the proxy file would come
    back empty however often the junction runs.
    """
    proxy = _junction(
        "db_proxy_mods/task_complex_module.py::getTasksToExecCommand_JEDI",
        Branch(outcome="runtime(newStatus)", tier=2),
        log_files=PROXY_LOGS,
        caller_log_files=[OTHER_LOG],
        owns_logger=False,
    )
    knight = _junction(
        "jediorder/ContentsFeeder.py::feed", Branch(outcome="pending"), log_files=[KNIGHT_LOG]
    )
    assert proxy.observable_log_files() == [OTHER_LOG]
    assert knight.observable_log_files() == [KNIGHT_LOG]


async def test_a_candidate_no_log_names_is_reported_rather_than_guessed_at():
    """Being unable to look is not evidence, and it is not silence either.

    This is how ``makeTaskPending_JEDI`` came to light: a junction that writes
    ``pending`` outright, inheriting its files from the proxies that mix it in
    and reached by nothing -- its only call site is commented out.  Falling back
    to the inherited files would have turned that into a confident elimination.
    """
    orphan = _junction(
        "db_proxy_mods/task_standalone_module.py::makeTaskPending_JEDI",
        Branch(outcome="pending"),
        log_files=PROXY_LOGS,
        owns_logger=False,
    )
    fragment = MapFragment(
        map_id=MAP_ID, derived_from=VERSION, subjects=[_subject()], junctions=[orphan]
    )
    strategy = await strategy_mod.derive(
        await _map(fragment), Symptom(subject=SUBJECT, observed="pending", task_id="1")
    )

    assert strategy.candidates[0].log_files == []
    assert strategy.observations == []
    assert "makeTaskPending_JEDI" in strategy.findings[0]
    assert "nothing the map recognises reaches it" in strategy.findings[0]


async def test_one_question_per_log_file_not_per_candidate():
    """Candidates share files, and asking the same file twice buys nothing."""
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        subjects=[_subject()],
        junctions=[
            _junction("a.py::f", Branch(outcome="pending"), log_files=[KNIGHT_LOG]),
            _junction("b.py::g", Branch(outcome="pending"), log_files=[KNIGHT_LOG]),
        ],
    )
    strategy = await strategy_mod.derive(
        await _map(fragment), Symptom(subject=SUBJECT, observed="pending", task_id="42")
    )

    probes = [o for o in strategy.observations if o.role == strategy_mod.PROBE]
    assert [o.log_file for o in probes] == [KNIGHT_LOG]
    assert probes[0].settles == ["a.py::f", "b.py::g"]
    # Both machine groups, because the package does not decide which runs it.
    assert probes[0].services == list(evidence_mod.SERVICES)
    assert "jediTaskID=42[ >]" in probes[0].pattern


async def test_without_an_entity_there_is_no_control_and_nothing_can_be_ruled_out():
    """A pattern scoped to nothing cannot separate "not this row" from "never".

    Reported as a gap rather than absorbed: the honest answer is that this run
    can confirm a writer is live and can eliminate none.
    """
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        subjects=[_subject()],
        junctions=[_junction("a.py::f", Branch(outcome="pending"), log_files=[KNIGHT_LOG])],
    )
    strategy = await strategy_mod.derive(
        await _map(fragment), Symptom(subject=SUBJECT, observed="pending")
    )

    assert all(o.role == strategy_mod.PROBE for o in strategy.observations)
    assert any("rules nothing out" in gap for gap in strategy.gaps)

    settled = strategy_mod.evaluate(strategy, _evidence(strategy, _quiet))
    assert settled.candidates[0].verdict == UNSETTLED


async def test_a_subject_with_no_known_line_shape_is_a_gap_not_a_weaker_answer():
    """A writer that logs nothing about the value can be enumerated and
    explained but not observed.  Saying so names the next thing to record;
    returning an empty observation list quietly would not."""
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        subjects=[_subject(name="JobSpec.jobStatus")],
        junctions=[
            _junction(
                "a.py::f",
                Branch(outcome="holding"),
                log_files=[KNIGHT_LOG],
                subject="JobSpec.jobStatus",
            )
        ],
    )
    strategy = await strategy_mod.derive(
        await _map(fragment),
        Symptom(subject="JobSpec.jobStatus", observed="holding", task_id="1"),
    )

    assert strategy.candidates
    assert strategy.observations == []
    assert any("records no diagnostic line" in gap for gap in strategy.gaps)


async def test_an_entity_of_the_wrong_kind_is_a_gap_rather_than_a_query():
    """The prefix that scopes a query to one row names a task.

    A job is tagged ``PandaID``.  Building a ``jediTaskID=`` pattern for one
    asks production something that cannot match, and an answer that cannot
    match is a silence -- which is exactly what the eliminator reads as
    evidence.  Refused rather than asked.
    """
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        subjects=[_subject(name="JobSpec.jobStatus")],
        junctions=[
            _junction(
                "a.py::f",
                Branch(
                    outcome="holding",
                    emits=[Emit(template="set job status to {}", log_level="info")],
                ),
                log_files=[KNIGHT_LOG],
                subject="JobSpec.jobStatus",
            )
        ],
    )
    strategy = await strategy_mod.derive(
        await _map(fragment),
        Symptom(subject="JobSpec.jobStatus", observed="holding", task_id="1"),
    )

    assert strategy.observations == []
    assert any("rows are not" in gap for gap in strategy.gaps)


async def test_the_control_asks_about_the_probes_own_sentence():
    """A control about a different line answers a question nobody asked.

    It licenses the probe's silence, so it has to be the same sentence with the
    entity and the value taken out.  While the shape was a constant the control
    was that constant too, which was right for one subject and would have been
    silently wrong for every other.
    """
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        subjects=[_subject(selected=["pending"])],
        junctions=[
            _junction(
                "jediorder/ContentsFeeder.py::feed",
                Branch(
                    outcome="pending",
                    emits=[Emit(template="moved task to {}", log_level="info")],
                ),
                log_files=[KNIGHT_LOG],
                triggers=("polled",),
            ),
        ],
    )
    strategy = await strategy_mod.derive(
        await _map(fragment), Symptom(subject=SUBJECT, observed="pending", task_id="42")
    )
    controls = [o for o in strategy.observations if o.role == strategy_mod.CONTROL]

    assert [o.pattern for o in controls] == [r"moved\ task\ to\ "]


async def test_a_subject_the_map_does_not_hold_is_refused_with_what_it_does_hold():
    fragment = MapFragment(map_id=MAP_ID, derived_from=VERSION, subjects=[_subject()])
    code_map = await _map(fragment)

    with pytest.raises(LookupError, match=SUBJECT):
        await strategy_mod.derive(
            code_map, Symptom(subject="JobSpec.nosuch", observed="x", task_id="1")
        )


# ---------------------------------------------------------------------------
# The follow-up: will anything move the row on
# ---------------------------------------------------------------------------


async def test_a_selected_value_with_a_polled_trigger_points_at_what_bounds_the_query():
    """The case the whole selection-gate slice exists for.

    A task sat in ``finishing`` while the query that rescues orphaned commands
    ran every cycle: it selected that value, a loop re-evaluated, and the row
    was still invisible because ``JEDI_AUX_Status_MinTaskID`` had stopped being
    updated and its watermark had risen above the task's id.  "Which junction
    wrote it" would have led nowhere.
    """
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        subjects=[_subject(selected=["finishing"], gates=["JEDI_AUX_Status_MinTaskID"])],
        junctions=[
            _junction(
                "a.py::f",
                Branch(outcome="finishing"),
                log_files=[KNIGHT_LOG],
                triggers=("polled",),
            )
        ],
    )
    strategy = await strategy_mod.derive(
        await _map(fragment), Symptom(subject=SUBJECT, observed="finishing", task_id="7")
    )

    follow = strategy.follow_up
    assert follow.selected is True
    assert follow.self_repairing is True
    assert follow.selection_gates == ["JEDI_AUX_Status_MinTaskID"]
    assert "JEDI_AUX_Status_MinTaskID" in follow.question


async def test_the_query_that_selects_the_value_is_named_with_the_log_it_writes_to():
    """"Something selects it" is half an answer; exactly one query does.

    The reader is what an investigation goes and reads, and where it is also a
    writer the map already knows its log file -- so the follow-up can say where
    to look rather than only that looking is worthwhile.
    """
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        subjects=[
            _subject(
                selected=["finishing"],
                gates=["JEDI_AUX_Status_MinTaskID"],
                selected_by={"finishing": ["jediorder/TaskCommando.py::run"]},
            )
        ],
        junctions=[
            _junction(
                "jediorder/TaskCommando.py::run",
                Branch(outcome="finishing"),
                log_files=[OTHER_LOG],
                triggers=("command",),
            ),
        ],
    )
    strategy = await strategy_mod.derive(
        await _map(fragment), Symptom(subject=SUBJECT, observed="finishing", task_id="42")
    )

    assert strategy.follow_up.selected_by == ["jediorder/TaskCommando.py::run"]
    assert strategy.follow_up.reader_log_files == [OTHER_LOG]
    assert "TaskCommando" in strategy.follow_up.question


async def test_a_reader_the_map_holds_no_log_for_is_named_without_one():
    """Being unable to say where to look is a finding, not a reason to guess.

    Sixty of the corpus's readers are pure readers -- they select on a value and
    settle nothing, so they are not junctions and no log file is attached to
    them.  Naming one with an invented file would be worse than naming it with
    none.
    """
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        subjects=[
            _subject(
                selected=["pending"],
                selected_by={"pending": ["taskbuffer/db_proxy_mods/entity_module.py::calc"]},
            )
        ],
        junctions=[
            _junction(
                "jediorder/ContentsFeeder.py::feed",
                Branch(outcome="pending"),
                log_files=[KNIGHT_LOG],
                triggers=("polled",),
            ),
        ],
    )
    strategy = await strategy_mod.derive(
        await _map(fragment), Symptom(subject=SUBJECT, observed="pending", task_id="42")
    )

    assert strategy.follow_up.selected_by == [
        "taskbuffer/db_proxy_mods/entity_module.py::calc"
    ]
    assert strategy.follow_up.reader_log_files == []


async def test_a_reader_with_no_entry_point_does_not_empty_the_triggers():
    """The reader's triggers replace the writers' only where it has some.

    Most readers are proxy methods, which nothing in the trigger slice reaches
    directly -- a knight calls them.  Taking their empty set as the answer
    turned ``pending`` from self-repairing into "nothing reaches this subject",
    which is a stronger claim than the map can make and the opposite of true.
    """
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        subjects=[
            _subject(
                selected=["pending"],
                selected_by={"pending": ["taskbuffer/db_proxy_mods/task_module.py::reactivate"]},
            )
        ],
        junctions=[
            # The reader is a junction, and no trigger reaches it directly.
            _junction(
                "taskbuffer/db_proxy_mods/task_module.py::reactivate",
                Branch(outcome="pending"),
                log_files=PROXY_LOGS,
                owns_logger=False,
                caller_log_files=[KNIGHT_LOG],
            ),
            _junction(
                "jediorder/ContentsFeeder.py::feed",
                Branch(outcome="pending"),
                log_files=[KNIGHT_LOG],
                triggers=("polled",),
            ),
        ],
    )
    strategy = await strategy_mod.derive(
        await _map(fragment), Symptom(subject=SUBJECT, observed="pending", task_id="42")
    )

    assert strategy.follow_up.triggers == ["polled"]
    assert strategy.follow_up.self_repairing is True


async def test_a_selected_value_reached_only_by_a_message_asks_whether_it_arrived():
    """Nothing re-evaluates, so the question is delivery, not conditions."""
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        subjects=[_subject(selected=["pending"])],
        junctions=[
            _junction(
                "a.py::f", Branch(outcome="pending"), log_files=[KNIGHT_LOG], triggers=("message",)
            )
        ],
    )
    strategy = await strategy_mod.derive(
        await _map(fragment), Symptom(subject=SUBJECT, observed="pending", task_id="7")
    )

    assert strategy.follow_up.self_repairing is False
    assert "arrived" in strategy.follow_up.question


async def test_a_value_nothing_selects_sends_the_question_one_step_back():
    """Waiting will not help, so the useful question is who wrote the step before.

    ``carried_from`` is what makes that actionable rather than a shrug: the
    passthrough outcome names the field the value came from.
    """
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        subjects=[_subject(selected=["ready"]), _subject(name="JediTaskSpec.oldStatus")],
        junctions=[
            _junction(
                "a.py::f",
                Branch(outcome="pending"),
                Branch(outcome="passthrough(JediTaskSpec.oldStatus)", tier=2),
                log_files=[KNIGHT_LOG],
                triggers=("polled",),
            )
        ],
    )
    strategy = await strategy_mod.derive(
        await _map(fragment), Symptom(subject=SUBJECT, observed="pending", task_id="7")
    )

    assert strategy.follow_up.selected is False
    assert strategy.follow_up.carried_from == ["JediTaskSpec.oldStatus"]
    assert "JediTaskSpec.oldStatus" in strategy.follow_up.question


# ---------------------------------------------------------------------------
# Evaluating against production
# ---------------------------------------------------------------------------


async def _two_candidates() -> "strategy_mod.Strategy":
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        subjects=[_subject(selected=["pending"])],
        junctions=[
            _junction(
                "jediorder/ContentsFeeder.py::feed",
                Branch(outcome="pending"),
                log_files=[KNIGHT_LOG],
                triggers=("polled",),
            ),
            _junction(
                "jediorder/TaskCommando.py::run",
                Branch(outcome="pending"),
                log_files=[OTHER_LOG],
                triggers=("polled",),
            ),
        ],
    )
    return await strategy_mod.derive(
        await _map(fragment), Symptom(subject=SUBJECT, observed="pending", task_id="42")
    )


async def test_a_line_seen_confirms_the_writer_whatever_the_sample_size():
    """The asymmetry this layer is built on, in its easy direction."""
    strategy = await _two_candidates()

    def decide(role, filename, service):
        if role == strategy_mod.CONTROL:
            return {"matched": 5}
        if filename == KNIGHT_LOG:
            # Truncated on purpose: a bound does not weaken a positive.
            return {"matched": 1, "lines": ["set task_status=pending"], "truncated": True}
        return {"matched": 0}

    settled = strategy_mod.evaluate(strategy, _evidence(strategy, decide))
    verdicts = {c.owner: c.verdict for c in settled.candidates}

    assert verdicts["jediorder/ContentsFeeder.py::feed"] == SEEN
    assert verdicts["jediorder/TaskCommando.py::run"] == ELIMINATED
    assert strategy_mod.survivors(settled)[0].verdict == SEEN


async def test_a_confirmed_writer_whose_write_races_is_confirmed_as_the_decider():
    """A log line proves what the code decided, not what the row became.

    ``UPDATE ... SET status=:status WHERE ... AND status IN (:old_...)`` changes
    nothing when another writer moved the row first, and says nothing about it
    -- the count comes back zero and most callers discard it.  The candidate
    stays confirmed, because it did decide the value; what changes is that the
    confirmation no longer settles whether the row took it.
    """
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        subjects=[_subject(selected=["pending"])],
        junctions=[
            _junction(
                "jediorder/ContentsFeeder.py::feed",
                Branch(outcome="pending", row_precondition=["status IN (:old_1)"]),
                log_files=[KNIGHT_LOG],
                triggers=("polled",),
            ),
        ],
    )
    strategy = await strategy_mod.derive(
        await _map(fragment), Symptom(subject=SUBJECT, observed="pending", task_id="42")
    )

    assert strategy.candidates[0].row_precondition == ["status IN (:old_1)"]

    settled = strategy_mod.evaluate(strategy, _evidence(strategy, _found))
    confirmed = settled.candidates[0]

    assert confirmed.verdict == SEEN
    assert "conditional on the row" in confirmed.because


async def test_an_unconditional_write_is_confirmed_without_the_caveat():
    """The caveat is a fact about the statement, not decoration on every verdict."""
    strategy = await _two_candidates()

    settled = strategy_mod.evaluate(strategy, _evidence(strategy, _found))

    assert all(not c.row_precondition for c in settled.candidates)
    assert all("conditional on the row" not in c.because for c in settled.candidates)


async def test_a_silent_file_that_never_carries_the_line_rules_nothing_out():
    """The guard that makes the whole eliminator safe.

    ``panda-DBProxy.log`` answers nothing for ``set task_status=`` however often
    a junction under it fires.  Without the control, that silence eliminates
    every candidate reached through such a file at once -- and then the
    survivors look unanimous, which is the one failure elimination cannot
    recover from.
    """
    strategy = await _two_candidates()

    def decide(role, filename, service):
        if role == strategy_mod.CONTROL and filename == OTHER_LOG:
            return {"matched": 0}  # this file does not speak this sentence
        if role == strategy_mod.CONTROL:
            return {"matched": 5}
        return {"matched": 0}

    settled = strategy_mod.evaluate(strategy, _evidence(strategy, decide))
    verdicts = {c.owner: c.verdict for c in settled.candidates}
    reasons = {c.owner: c.because for c in settled.candidates}

    assert verdicts["jediorder/ContentsFeeder.py::feed"] == ELIMINATED
    assert verdicts["jediorder/TaskCommando.py::run"] == UNSETTLED
    assert "says nothing" in reasons["jediorder/TaskCommando.py::run"]


async def test_a_truncated_silence_is_not_a_silence():
    """A cut sample and an empty one look identical, so they are kept apart."""
    strategy = await _two_candidates()

    def decide(role, filename, service):
        if role == strategy_mod.CONTROL:
            return {"matched": 5}
        return {"matched": 0, "truncated": filename == OTHER_LOG}

    settled = strategy_mod.evaluate(strategy, _evidence(strategy, decide))
    verdicts = {c.owner: c.verdict for c in settled.candidates}

    assert verdicts["jediorder/ContentsFeeder.py::feed"] == ELIMINATED
    assert verdicts["jediorder/TaskCommando.py::run"] == UNSETTLED


async def test_the_service_that_does_not_have_the_file_is_dropped_not_counted():
    """Both groups are asked precisely because the map cannot say which runs it.

    One of them reporting the file missing is the expected shape of a right
    answer; counting it as an inconclusive would make every question
    unanswerable and quietly disable the eliminator.
    """
    strategy = await _two_candidates()

    def decide(role, filename, service):
        if service == evidence_mod.SERVER:
            return {"missing": True}
        if role == strategy_mod.CONTROL:
            return {"matched": 5}
        return {"matched": 0}

    settled = strategy_mod.evaluate(strategy, _evidence(strategy, decide))

    assert all(c.verdict == ELIMINATED for c in settled.candidates)


async def test_a_file_no_machine_has_means_that_code_never_ran_here():
    """The one negative production can prove.

    A log file is created on its logger's first emit, so its absence is not
    conditioned on any branch firing -- unlike every other silence here.
    """
    strategy = await _two_candidates()

    settled = strategy_mod.evaluate(
        strategy, _evidence(strategy, lambda role, filename, service: {"missing": True})
    )

    assert all(c.verdict == ELIMINATED for c in settled.candidates)


async def test_evidence_that_answers_none_of_these_questions_settles_nothing():
    """An evidence file collected for another symptom carries no answer to this
    one, and reading its silence would rule everything out at once."""
    strategy = await _two_candidates()
    empty = Evidence(fetched_at="2026-09-05T00:00:00+00:00")

    settled = strategy_mod.evaluate(strategy, empty)

    assert all(c.verdict == UNSETTLED for c in settled.candidates)
    assert all(o.verdict == "not_asked" for o in settled.observations)


async def test_a_candidate_with_no_log_is_never_ruled_out_by_the_others_answers():
    strategy = await strategy_mod.derive(
        await _map(
            MapFragment(
                map_id=MAP_ID,
                derived_from=VERSION,
                subjects=[_subject(selected=["pending"])],
                junctions=[
                    _junction(
                        "jediorder/ContentsFeeder.py::feed",
                        Branch(outcome="pending"),
                        log_files=[KNIGHT_LOG],
                    ),
                    _junction(
                        "base/PostProcessorBase.py::doBasicPostProcess",
                        Branch(outcome="pending"),
                        owns_logger=False,
                    ),
                ],
            )
        ),
        Symptom(subject=SUBJECT, observed="pending", task_id="42"),
    )

    settled = strategy_mod.evaluate(strategy, _evidence(strategy, _found))
    verdicts = {c.owner: c.verdict for c in settled.candidates}

    assert verdicts["jediorder/ContentsFeeder.py::feed"] == SEEN
    assert verdicts["base/PostProcessorBase.py::doBasicPostProcess"] == UNASKABLE
    assert len(strategy_mod.survivors(settled)) == 2


# ---------------------------------------------------------------------------
# The arm, not only the junction
# ---------------------------------------------------------------------------

_SCOUT = "taskbuffer/db_proxy_mods/task_utils_module.py::setScoutJobData_JEDI"


def _arm(reason: str, condition: str, line: int) -> Branch:
    """One arm of the scout check, as the map records it."""
    return Branch(
        outcome="exhausted",
        path_condition=[condition],
        line=line,
        tags=["action=set_exhausted", f"reason={reason}"],
        emits=[
            Emit(
                template=f"#ATM #KV action=set_exhausted reason={reason} measured {{}}",
                log_level="info",
                log_files=PROXY_LOGS,
                reports="decision",
            )
        ],
    )


async def _six_arms(diag: str | None = None):
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        subjects=[_subject(selected=["exhausted"])],
        junctions=[
            _junction(
                _SCOUT,
                _arm("scout_cpuTime", "scoutData['cpuTime'] > thr", 1239),
                _arm("low_efficiency", "not high_io_intensity", 1423),
                _arm("low_success_rate", "extraInfo['successRate'] < rate", 1449),
                owns_logger=False,
            ),
            _junction(
                "jediorder/TaskCommando.py::runImpl",
                Branch(outcome="runtime(newTaskStatus)", tier=2),
                log_files=[OTHER_LOG],
            ),
        ],
    )
    return await strategy_mod.derive(
        await _map(fragment),
        Symptom(subject=SUBJECT, observed="exhausted", task_id="7", observed_diag=diag),
    )


async def test_the_arms_of_one_junction_are_not_pooled_into_one_reason():
    """Six arms send a task to ``exhausted`` and differ only in the reason they
    give.  Pooling their conditions answers "why" with the union of six answers,
    which was right for ``pending`` -- where fourteen of eighteen candidates
    have no condition at all -- and hides the question entirely here."""
    strategy = await _six_arms()

    scout = next(c for c in strategy.candidates if c.owner == _SCOUT)
    assert [b.tags[1] for b in scout.branches] == [
        "reason=scout_cpuTime",
        "reason=low_efficiency",
        "reason=low_success_rate",
    ]
    assert [b.line for b in scout.branches] == [1239, 1423, 1449]


async def test_the_message_the_record_carries_names_one_arm():
    """The evidence with no window: ``errordialog`` is a column, so one call
    returns it whole, and PanDA writes the reason into the same text it logs."""
    strategy = await _six_arms(
        "#ATM #KV action=set_exhausted reason=low_efficiency lowest CPU efficiency 23 is less than 50"
    )

    scout = next(c for c in strategy.candidates if c.owner == _SCOUT)
    assert [b.tags[1] for b in scout.named] == ["reason=low_efficiency"]
    assert scout.verdict == SEEN
    assert scout.because == "the record's own message names this branch"


async def test_naming_one_arm_rules_out_no_other_junction():
    """The field holds the *last* message written to it, so a later junction may
    have overwritten it.  A match is proof and a silence is not evidence, which
    is the same asymmetry as everywhere else here and arrived at for a different
    reason."""
    strategy = await _six_arms(
        "#ATM #KV action=set_exhausted reason=low_efficiency measured 23"
    )

    others = [c for c in strategy.candidates if c.owner != _SCOUT]
    assert [c.verdict for c in others] == [UNSETTLED]
    assert len(strategy_mod.survivors(strategy)) == 2


async def test_a_reason_is_not_matched_by_a_message_naming_a_different_one():
    strategy = await _six_arms("#ATM #KV action=set_exhausted reason=low_success_rate 3")

    scout = next(c for c in strategy.candidates if c.owner == _SCOUT)
    assert [b.tags[1] for b in scout.named] == ["reason=low_success_rate"]


async def test_a_tagged_arm_is_asked_for_in_the_file_its_own_line_lands_in():
    """The shared head belongs to the other writers entirely: six arms write the
    value and none of their lines interpolates it.  The emit carries its own
    files because the arm logs in the proxy's file while the value line is
    written by the caller."""
    strategy = await _six_arms()

    tagged = [
        o
        for o in strategy.observations
        if o.role == strategy_mod.PROBE and "set_exhausted" in o.pattern
    ]
    assert {o.log_file for o in tagged} == set(PROXY_LOGS)
    assert all(_SCOUT in o.settles for o in tagged)
    assert all(o.pattern.startswith("jediTaskID=7[ >]") for o in tagged)
    # One control per tagged probe, paired by pattern rather than by file: the
    # file also carries the value line, which is a different sentence.
    controls = {
        o.control_for for o in strategy.observations if o.role == strategy_mod.CONTROL
    }
    assert {o.pattern for o in tagged} <= controls


async def test_production_carrying_the_tagged_line_settles_the_arm():
    strategy = await _six_arms()

    results = []
    for query in strategy_mod.queries(strategy):
        # Only the one arm's line is in production, which is what makes the
        # answer say which arm rather than which junction.
        matched = 1 if "low_efficiency" in query.pattern else 0
        results.append(
            _result(query, matched=matched, lines=["... reason=low_efficiency"] if matched else [])
        )
    settled = strategy_mod.evaluate(
        strategy, Evidence(fetched_at="2026-09-05T00:00:00+00:00", results=results)
    )

    scout = next(c for c in settled.candidates if c.owner == _SCOUT)
    assert scout.verdict == SEEN
    assert [b.tags[1] for b in scout.named] == ["reason=low_efficiency"]


async def test_the_record_fetched_with_the_logs_names_the_arm_too():
    """The same reading whichever route the message arrives by.  Fetched
    alongside the greps rather than instead of them: the record holds the last
    message written to the field, so it confirms an arm and never rules one
    out, and the log line is what survives a later junction overwriting it."""
    strategy = await _six_arms()
    ev = Evidence(
        fetched_at="2026-09-05T00:00:00+00:00",
        results=[_result(q, matched=0) for q in strategy_mod.queries(strategy)],
        tasks=[
            evidence_mod.TaskRecord(
                task_id="7",
                fields={
                    "status": "exhausted",
                    "errordialog": "#ATM #KV action=set_exhausted reason=low_efficiency 23 < 50",
                },
            )
        ],
    )

    settled = strategy_mod.evaluate(strategy, ev)

    scout = next(c for c in settled.candidates if c.owner == _SCOUT)
    assert [b.tags[1] for b in scout.named] == ["reason=low_efficiency"]
    assert scout.verdict == SEEN


async def test_a_record_for_another_task_settles_nothing():
    strategy = await _six_arms()
    ev = Evidence(
        fetched_at="2026-09-05T00:00:00+00:00",
        tasks=[
            evidence_mod.TaskRecord(
                task_id="8",
                fields={"errordialog": "#ATM action=set_exhausted reason=low_efficiency"},
            )
        ],
    )

    settled = strategy_mod.evaluate(strategy, ev)

    assert all(not c.named for c in settled.candidates)


async def _prose_arms(diag: str | None = None):
    """Two refusals in one chain, telling themselves apart in prose only."""
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        subjects=[_subject(selected=["exhausted"])],
        junctions=[
            _junction(
                "db_proxy_mods/task_complex_module.py::retryTask_JEDI",
                Branch(
                    outcome="exhausted",
                    path_condition=["attempts >= task_max_attempt"],
                    line=5025,
                    messages=["no longer continue because too many task attempts more than {}"],
                ),
                Branch(
                    outcome="exhausted",
                    path_condition=["rate >= max_failed_hep_score_rate"],
                    line=5057,
                    messages=["no longer continue because failed/total HEP score rate ({}) exceeds {}"],
                ),
                log_files=[OTHER_LOG],
            ),
        ],
    )
    return await strategy_mod.derive(
        await _map(fragment),
        Symptom(subject=SUBJECT, observed="exhausted", task_id="7", observed_diag=diag),
    )


async def test_an_untagged_frame_names_the_arm_when_it_is_the_only_one():
    """Of thirty tasks found in exhausted with a message on the record, none
    carried a tag: they are retry refusals, which write prose.  Matching the
    frame is weaker evidence and the only evidence there is."""
    strategy = await _prose_arms(
        "no longer continue because failed/total HEP score rate (0.914) exceeds 0.9"
    )

    retry = strategy.candidates[0]
    assert [b.line for b in retry.named] == [5057]
    assert retry.verdict == SEEN


async def test_a_frame_two_arms_share_names_neither():
    """A frame is the author's wording, not a contract: two arms can have one.
    Saying nothing is the failure this should have, rather than saying the
    wrong one."""
    strategy = await _prose_arms("no longer continue because ")

    assert strategy.candidates[0].named == []
    assert strategy.candidates[0].verdict != SEEN


async def test_a_frame_is_matched_inside_a_longer_message():
    """errorDialog is appended to, so a frame that had to account for the whole
    field would match none of the records carrying two messages."""
    strategy = await _prose_arms(
        "task was retried; no longer continue because too many task attempts more than 5 "
        "are forbidden. No further retries are accepted."
    )

    assert [b.line for b in strategy.candidates[0].named] == [5025]
