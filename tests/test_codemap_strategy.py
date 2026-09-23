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
from bamboo.codemap import models
from bamboo.codemap import strategy as strategy_mod
from bamboo.codemap.evidence import Evidence, GrepQuery, GrepResult
from bamboo.codemap.lookup import CodeMap
from bamboo.codemap.models import (
    ELIMINATED,
    SEEN,
    SYMPTOM_DISTRIBUTION,
    UNASKABLE,
    UNSETTLED,
    Anchor,
    Branch,
    Emit,
    EntityNode,
    EntryPoint,
    FilterStageNode,
    JunctionNode,
    LogSiteNode,
    MapFragment,
    MapTerm,
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
    updated_by: dict[str, list[str]] | None = None,
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
        updated_by=updated_by or {},
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
    gloss_key: str | None = None,
    calls: list[str] | None = None,
    joined_entities: list[str] | None = None,
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
        gloss_key=gloss_key if gloss_key is not None else f"key-{owner}",
        calls=calls or [],
        joined_entities=joined_entities or [],
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
    # The role comes off the strategy's own observations rather than being
    # guessed from the pattern: an arm's control is not the transition pattern,
    # so guessing called it a probe and a fixture could never make it speak.
    roles = {
        (o.log_file, o.pattern): o.role for o in strategy.observations
    }
    results = []
    for query in strategy_mod.queries(strategy):
        role = roles.get((query.log_filename, query.pattern), strategy_mod.PROBE)
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


async def test_a_framed_arm_is_not_offered_as_a_way_the_value_was_reached():
    """The same rule as ``producers_of``, one level down, and it has to agree.

    ``producers_of`` answers which junctions, this answers which arms of one,
    and both are printed by the same report.  A junction kept by its free arm
    must be shown with that arm and not the bounded one, or the report offers
    a place to read that cannot produce what was observed.
    """
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        subjects=[_subject()],
        junctions=[
            _junction(
                "mixed.py::f",
                Branch(outcome="runtime(f'merge_{s}')", tier=2),
                Branch(outcome="runtime(newStatus)", tier=2),
                log_files=[KNIGHT_LOG],
            ),
            _junction("framed.py::g", Branch(outcome="runtime(f'merge_{s}')", tier=2)),
        ],
    )
    strategy = await strategy_mod.derive(
        await _map(fragment), Symptom(subject=SUBJECT, observed="pending", task_id="1")
    )

    assert [c.owner for c in strategy.candidates] == ["mixed.py::f"]
    assert [b.outcome for b in strategy.candidates[0].branches] == ["runtime(newStatus)"]


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
            _junction(
                "a.py::f", _tagged("pending", "reason=same", KNIGHT_LOG), log_files=[KNIGHT_LOG]
            ),
            _junction(
                "b.py::g", _tagged("pending", "reason=same", KNIGHT_LOG), log_files=[KNIGHT_LOG]
            ),
        ],
    )
    strategy = await strategy_mod.derive(
        await _map(fragment), Symptom(subject=SUBJECT, observed="pending", task_id="42")
    )

    probes = [o for o in strategy.observations if o.role == strategy_mod.PROBE]
    assert [o.log_file for o in probes] == [KNIGHT_LOG]
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


async def test_a_writer_that_names_no_decision_is_not_a_capability_gap():
    """Deriving no question here is the division of labour, not a shortfall.

    A probe used to be built from the head the most writers share, and a
    subject whose writers shared none came back as a capability gap -- the map
    apologising for a question it could not phrase.  What a function prints is
    now reported by the walk, line by line and per arm, and which of those to
    put to production is the reader's choice.  So the absence of a question
    here is not something missing, and dressing it as a gap would send a reader
    looking for the wrong repair.
    """
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
    assert not any("no question can be put to production" in gap for gap in strategy.gaps)


async def test_an_entity_of_the_wrong_kind_does_not_get_the_task_prefix():
    """The prefix that scopes a query to one row names a task.

    A job is tagged ``PandaID``.  Putting ``jediTaskID=`` in front of a pattern
    about one asks production something that cannot match, and an answer that
    cannot match is a silence -- which is exactly what the eliminator reads as
    evidence.  The question is still asked, unscoped; what is withheld is the
    prefix, and with it the control that would license a silence.
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
                    tags=["reason=held"],
                    emits=[
                        Emit(
                            template="set job status to {} reason=held",
                            log_level="info",
                            log_files=[KNIGHT_LOG],
                        )
                    ],
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

    probes = [o for o in strategy.observations if o.role == strategy_mod.PROBE]
    assert [o.pattern for o in probes] == ["reason=held"]
    assert not any("jediTaskID" in o.pattern for o in strategy.observations)
    # No control either: it exists to license the silence of a scoped pattern,
    # and there is nothing to license when the pattern was never narrowed.
    assert not [o for o in strategy.observations if o.role == strategy_mod.CONTROL]


async def test_the_control_asks_about_the_probes_own_sentence():
    """A control about a different line answers a question nobody asked.

    It licenses the probe's silence, so it has to be the same sentence with the
    entity taken out -- the probe scoped to one row, the control asking whether
    the file carries that line at all.
    """
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        subjects=[_subject(selected=["pending"])],
        junctions=[
            _junction(
                "jediorder/ContentsFeeder.py::feed",
                _tagged("pending", "reason=fed", KNIGHT_LOG),
                log_files=[KNIGHT_LOG],
                triggers=("polled",),
            ),
        ],
    )
    strategy = await strategy_mod.derive(
        await _map(fragment), Symptom(subject=SUBJECT, observed="pending", task_id="42")
    )
    probes = [o for o in strategy.observations if o.role == strategy_mod.PROBE]
    controls = [o for o in strategy.observations if o.role == strategy_mod.CONTROL]

    assert [o.pattern for o in controls] == ["reason=fed"]
    assert controls[0].control_for == probes[0].pattern
    assert probes[0].pattern.endswith(controls[0].pattern)


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


async def test_a_run_time_candidate_says_where_its_value_comes_from():
    """Why a candidate whose value nothing states is on the list at all.

    The expression was already in the report, a hundred and eighty lines down
    in ``code to read``.  A reader who stops at the candidate listing -- which
    is most of the point of there being one -- saw four rows and no way to tell
    which of them could have been theirs.  A stated value needs no such line
    and does not get one.
    """
    import click
    import click.testing

    from bamboo.scripts.derive_strategy import _report_candidates

    strategy = strategy_mod.Strategy(
        symptom=Symptom(subject=SUBJECT, observed="finishing"),
        map_id=MAP_ID,
        derived_from=VERSION,
        candidates=[
            models.Candidate(
                owner="a.py::stated",
                tier=1,
                branches=[models.CandidateBranch(outcome="finishing", tier=1)],
            ),
            models.Candidate(
                owner="b.py::copied",
                tier=2,
                branches=[
                    models.CandidateBranch(outcome="passthrough(JobSpec.jobStatus)", tier=2)
                ],
            ),
        ],
    )
    runner = click.testing.CliRunner()
    command = click.Command("x", callback=lambda: _report_candidates(strategy, 10, False, False))
    output = runner.invoke(command).output

    assert "value from passthrough(JobSpec.jobStatus)" in output
    stated, copied = output.split("b.py::copied")[0], output.split("b.py::copied")[1]
    assert "value from" not in stated
    assert "value from" in copied


_METRICS = "daemons/scripts/metric_collector.py::analy_pmerge_jobs_wait_time"
_GETTER_READER = "taskbuffer/db_proxy_mods/task_standalone_module.py::checkDuplication_JEDI"


def _read_only(owner: str, triggers: tuple[str, ...] = ()) -> LogSiteNode:
    """A function the map records for its output and for nothing it settles."""
    return LogSiteNode(
        map_id=MAP_ID,
        derived_from=VERSION,
        name=owner,
        owner=owner,
        log_files=[OTHER_LOG],
        triggers=list(triggers),
        owns_logger=True,
    )


async def _selected_by(reader: LogSiteNode, entities: list[EntityNode] | None = None):
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        entities=entities or [],
        subjects=[
            _subject(selected=["finishing"], selected_by={"finishing": [reader.owner]})
        ],
        junctions=[
            _junction(
                "jediorder/ContentsFeeder.py::feed",
                Branch(outcome="finishing"),
                log_files=[KNIGHT_LOG],
                triggers=("polled",),
            ),
        ],
        log_sites=[reader],
    )
    return await strategy_mod.derive(
        await _map(fragment), Symptom(subject=SUBJECT, observed="finishing", task_id="42")
    )


async def test_a_reader_that_changes_nothing_and_nothing_calls_leaves_the_row_where_it_is():
    """Being selected and being acted on are not the same claim.

    A metrics daemon selects ``cancelled`` jobs to average a wait time and
    writes nothing back.  Reading "a query selects this value" as "something
    will pick this row up" sent the verdict to ask that daemon why it had not.
    """
    strategy = await _selected_by(_read_only(_METRICS, triggers=("polled",)))

    assert strategy.follow_up.selected is True
    assert strategy.follow_up.reader_acts == models.READS_ONLY_AT_TOP
    assert "did not pick the row up" not in strategy.follow_up.question
    assert "waiting will not move the row" in strategy.follow_up.question


async def test_a_reader_with_no_trigger_is_not_said_to_leave_the_row_where_it_is():
    """The same emptiness, and the opposite conclusion would be just as wrong.

    A getter settles nothing and writes nothing either, and what acts on the
    row is whoever called it.  This map resolves reach by name one hop, so the
    caller is not available to name here -- saying so beats implying the getter
    is the answer, in either direction.
    """
    strategy = await _selected_by(_read_only(_GETTER_READER))

    assert strategy.follow_up.selected is True
    assert strategy.follow_up.reader_acts == models.READS_ONLY_FOR_A_CALLER
    assert "whatever called it" in strategy.follow_up.question
    assert "waiting will not move the row" not in strategy.follow_up.question


async def test_a_reader_that_writes_a_row_somewhere_still_counts_as_acting():
    """Settling no value and touching no row are different facts.

    A proxy that only ever UPDATEs satisfies the first, and calling it inert on
    that alone would put nine of this corpus's functions -- ``updateJobStatus``
    among them -- in a state the map plainly contradicts elsewhere.
    """
    strategy = await _selected_by(
        _read_only(_GETTER_READER),
        entities=[
            EntityNode(
                name="jobsactive4",
                map_id=MAP_ID,
                derived_from=VERSION,
                tables=["jobsActive4"],
                updated_by=[_GETTER_READER],
            )
        ],
    )

    assert strategy.follow_up.reader_acts == models.ACTS
    assert "did not pick the row up" in strategy.follow_up.question


async def test_a_value_only_an_update_acts_on_names_the_update_and_not_a_query():
    """The nine values in the corpus no query ever asks for.

    Dropping the update once the verbs came apart would turn "this statement
    has to match" into "nothing acts on this at all" -- the strongest claim the
    follow-up can make, and here a false one.  Named as an update, because a
    query is somewhere the row could have been missed and an update is not.
    """
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        subjects=[
            _subject(
                selected=["finishing"],
                updated_by={"finishing": ["jediorder/TaskCommando.py::run"]},
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

    assert strategy.follow_up.selected == 1
    assert strategy.follow_up.selected_by == []
    assert strategy.follow_up.updated_by == ["jediorder/TaskCommando.py::run"]
    assert strategy.follow_up.reader_log_files == [OTHER_LOG]
    assert "no query selects" in strategy.follow_up.question
    assert "updates rows holding it" in strategy.follow_up.question


async def test_a_query_wins_over_an_update_that_acts_on_the_same_value():
    """Both act on the row; only one is a place it could have been missed."""
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        subjects=[
            _subject(
                selected=["finishing"],
                selected_by={"finishing": ["jediorder/TaskCommando.py::run"]},
                updated_by={"finishing": ["jedidog/AtlasProdWatchDog.py::run"]},
            )
        ],
        junctions=[
            _junction(
                "jediorder/TaskCommando.py::run",
                Branch(outcome="finishing"),
                log_files=[OTHER_LOG],
                triggers=("command",),
            ),
            _junction(
                "jedidog/AtlasProdWatchDog.py::run",
                Branch(outcome="finishing"),
                log_files=[KNIGHT_LOG],
                triggers=("polled",),
            ),
        ],
    )
    strategy = await strategy_mod.derive(
        await _map(fragment), Symptom(subject=SUBJECT, observed="finishing", task_id="42")
    )

    assert strategy.follow_up.reader_log_files == [OTHER_LOG]
    assert "is selected by" in strategy.follow_up.question


async def test_a_reader_the_map_holds_no_log_for_is_named_without_one():
    """Being unable to say where to look is a finding, not a reason to guess.

    A reader that settles nothing carries its files on a log site rather than
    on a junction; where the build could resolve none there is no site, and
    naming an invented file would be worse than naming none.
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


def _tagged(outcome: str, tag: str, log_file: str, **rest) -> Branch:
    """A branch that names its own decision, which is what is asked about now.

    The probe built from the head the writers share is gone, so a fixture that
    wants a question put to production has to give the branch the thing the
    surviving family reads: a tag, and an emit that writes it.
    """
    return Branch(
        outcome=outcome,
        tags=[tag],
        emits=[
            Emit(
                template=f"set task_status={{}} {tag}",
                log_level="info",
                log_files=[log_file],
            )
        ],
        **rest,
    )


async def _two_candidates() -> "strategy_mod.Strategy":
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        subjects=[_subject(selected=["pending"])],
        junctions=[
            _junction(
                "jediorder/ContentsFeeder.py::feed",
                _tagged("pending", "reason=fed", KNIGHT_LOG),
                log_files=[KNIGHT_LOG],
                triggers=("polled",),
            ),
            _junction(
                "jediorder/TaskCommando.py::run",
                _tagged("pending", "reason=commanded", OTHER_LOG),
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
                _tagged(
                    "pending",
                    "reason=fed",
                    KNIGHT_LOG,
                    row_precondition=["status IN (:old_1)"],
                ),
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


async def test_a_candidate_no_question_was_put_about_is_not_ruled_out():
    """The one path that could produce a confident wrong answer.

    A junction whose arms print nothing gets no probe, but it still names log
    files, so it walked past the guard above and reached the bottom of
    ``_settle`` with an empty list of reasons -- which used to read as "absent
    from every log that would carry the line".  Measured over the stored map,
    every one of its 306 eliminations was of this kind.
    """
    other = "FileSpec.status"
    strategy = await strategy_mod.derive(
        await _map(
            MapFragment(
                map_id=MAP_ID,
                derived_from=VERSION,
                subjects=[_subject(selected=["finished"], name=other)],
                junctions=[
                    _junction(
                        "taskbuffer/db_proxy_mods/misc.py::updateInFiles",
                        Branch(outcome="finished"),
                        log_files=[KNIGHT_LOG],
                        subject=other,
                    )
                ],
            )
        ),
        Symptom(subject=other, observed="finished", task_id="42"),
    )
    assert not [o for o in strategy.observations if o.role == strategy_mod.PROBE]

    settled = strategy_mod.evaluate(strategy, Evidence(fetched_at="2026-09-05T00:00:00+00:00"))

    assert [c.verdict for c in settled.candidates] == [UNSETTLED]
    assert settled.candidates[0].because == "no question about this one was put to production"


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
                        _tagged("pending", "reason=fed", KNIGHT_LOG),
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


# ---------------------------------------------------------------------------
# Resolving a description into a symptom
# ---------------------------------------------------------------------------


def _term(kind: str, key: str, words: list[str], symptom: Symptom) -> MapTerm:
    return MapTerm(kind=kind, key=key, words=words, symptom=symptom)


def test_a_rare_word_outweighs_one_every_entry_carries():
    """Counting matched words makes ``check`` worth what ``lowmemory`` is worth,
    and ``check`` ends forty of the forty-nine funnel steps.  The weight comes
    from the vocabulary in hand rather than from a list someone maintains, so
    it moves on its own as the map grows."""
    terms = [
        _term("step", "memory check", ["memory", "check"], Symptom(kind=SYMPTOM_DISTRIBUTION, focus="memory check")),
        _term("step", "status check", ["status", "check"], Symptom(kind=SYMPTOM_DISTRIBUTION, focus="status check")),
        _term("step", "disk check", ["disk", "check"], Symptom(kind=SYMPTOM_DISTRIBUTION, focus="disk check")),
        _term("step", "network check", ["network", "check"], Symptom(kind=SYMPTOM_DISTRIBUTION, focus="network check")),
    ]

    ranked = strategy_mod.resolve("sites dropped on memory check", terms)

    assert ranked[0].term.key == "memory check"
    assert ranked[0].score > ranked[1].score


def test_a_description_naming_two_things_comes_back_as_two():
    """Picking one silently is ``too_many_candidates`` answered as though it
    were one candidate -- the retrieval failure the map exists to remove, put
    back one layer up."""
    terms = [
        _term("value", "JediTaskSpec.status=pending", ["jedi", "task", "spec", "status", "pending"], Symptom(subject="JediTaskSpec.status", observed="pending")),
        _term("value", "JobSpec.jobStatus=pending", ["job", "spec", "status", "pending"], Symptom(subject="JobSpec.jobStatus", observed="pending")),
    ]

    ranked = strategy_mod.resolve("stuck in pending", terms)

    assert len(ranked) == 2
    assert all(m.words == ["pending"] for m in ranked)


def test_a_description_saying_nothing_the_map_knows_resolves_to_nothing():
    """An empty answer is the honest one.  A best match out of words the
    vocabulary never used would be a guess wearing a score."""
    terms = [
        _term("value", "JediTaskSpec.status=pending", ["jedi", "task", "spec", "status", "pending"], Symptom(subject="JediTaskSpec.status", observed="pending")),
    ]

    assert strategy_mod.resolve("the cluster is on fire", terms) == []


def test_the_key_that_was_typed_wins_over_a_near_homograph():
    """``-t1weight`` and ``-t1_weight`` are two real stages of one chain, 618
    lines apart in ``AtlasProdJobBroker``.  Asked for the second by name, the
    weighting returned the first: a cut carries its step's words as well as
    its own, so the entry that says more has the larger denominator and loses.
    Measured over the vocabulary before this, 23 of 425 entries did not rank
    first when asked by their own key."""
    terms = [
        _term("cut", "-t1weight", ["t", "1", "weight", "check"], Symptom(kind=SYMPTOM_DISTRIBUTION, focus="-t1weight")),
        _term("cut", "-t1_weight", ["t", "1", "weight", "final", "check"], Symptom(kind=SYMPTOM_DISTRIBUTION, focus="-t1_weight")),
        _term("step", "T1 weight check", ["t", "1", "weight", "check"], Symptom(kind=SYMPTOM_DISTRIBUTION, focus="T1 weight check")),
    ]

    ranked = strategy_mod.resolve("-t1_weight", terms)

    assert ranked[0].term.key == "-t1_weight"
    assert ranked[0].exact
    # The one it displaced is still offered: an exact key says which entry was
    # meant, not that the others were not.
    assert "-t1weight" in [m.term.key for m in ranked[1:]]
    assert ranked[0].score < ranked[1].score


def test_an_exact_key_does_not_disturb_a_description_that_is_not_one():
    """The weighting is left alone.  Only a description that *is* a key is
    treated as one, so free text ranks exactly as it did."""
    terms = [
        _term("cut", "-t1weight", ["t", "1", "weight", "check"], Symptom(kind=SYMPTOM_DISTRIBUTION, focus="-t1weight")),
        _term("cut", "-t1_weight", ["t", "1", "weight", "final", "check"], Symptom(kind=SYMPTOM_DISTRIBUTION, focus="-t1_weight")),
    ]

    ranked = strategy_mod.resolve("T1 weight", terms)

    assert [m.term.key for m in ranked] == ["-t1weight", "-t1_weight"]
    assert not any(m.exact for m in ranked)


def test_entries_that_score_the_same_say_so():
    """The sort breaks a tie on the key, so ``-`` beating ``T`` decides which
    of two equals is printed as the answer.  Nothing about the question said
    so, and the vocabulary ties often enough to matter."""
    terms = [
        _term("cut", "-io", ["io"], Symptom(kind=SYMPTOM_DISTRIBUTION, focus="-io")),
        _term("step", "IO check", ["io"], Symptom(kind=SYMPTOM_DISTRIBUTION, focus="IO check")),
    ]

    ranked = strategy_mod.resolve("io", terms)

    assert ranked[0].score == ranked[1].score
    assert ranked[0].tied_with == [ranked[1].term.key]
    assert ranked[1].tied_with == [ranked[0].term.key]


def test_an_exact_key_ranked_below_the_cut_is_still_found():
    """``-blacklist`` did not reach the top three of the real vocabulary: its
    own entry carries its stage's words (``storage space check``) while four
    longer entries carry three apiece, so each of them accounts for more of
    itself than ``-blacklist`` does.  Spelled out here with the corpus's own
    word lists, because a promotion that only reordered the rows already shown
    would not have reached this one."""
    terms = [
        _term("cut", "-blacklist", ["blacklist", "storage", "space", "check"], Symptom(kind=SYMPTOM_DISTRIBUTION, focus="-blacklist")),
        _term("cut", "-read_lan_blacklist", ["read", "lan", "blacklist"], Symptom(kind=SYMPTOM_DISTRIBUTION, focus="-read_lan_blacklist")),
        _term("cut", "-read_wan_blacklist", ["read", "wan", "blacklist"], Symptom(kind=SYMPTOM_DISTRIBUTION, focus="-read_wan_blacklist")),
        _term("cut", "-write_lan_blacklist", ["write", "lan", "blacklist"], Symptom(kind=SYMPTOM_DISTRIBUTION, focus="-write_lan_blacklist")),
        _term("cut", "-write_wan_blacklist", ["write", "wan", "blacklist"], Symptom(kind=SYMPTOM_DISTRIBUTION, focus="-write_wan_blacklist")),
    ]

    ranked = strategy_mod.resolve("-blacklist", terms, limit=2)

    assert ranked[0].term.key == "-blacklist"
    assert ranked[0].exact
    # The claim this test is really making: without the promotion it is not
    # merely second, it is off the end of the list the caller asked for.
    assert ranked[0].score < ranked[1].score


def test_the_entity_is_read_apart_from_the_question():
    """An id says which row, not which question, so it is not matched against
    the vocabulary -- where eight digits would score against every numeric
    fragment in it."""
    assert strategy_mod.entity_in("task 52249469 went nowhere") == "52249469"
    # Two different ids name no single row, and guessing which is meant is
    # exactly the kind of pick this layer refuses to make.
    assert strategy_mod.entity_in("52249469 and 52357008") is None
    assert strategy_mod.entity_in("no id here") is None


# ---------------------------------------------------------------------------
# The second symptom class: which step of a chain threw the candidates away
# ---------------------------------------------------------------------------

PROD_JOB_LOG = "panda-AtlasProdJobBroker.log"
PROD_JOB = "pandajedi/jedibrokerage/AtlasProdJobBroker.py::doBrokerage"
PROD_TASK = "pandajedi/jedibrokerage/AtlasProdTaskBroker.py::runImpl"


def _filter_stage(
    owner: str,
    tag: str,
    label: str,
    order: int,
    files: list[str] | None = None,
    condition: str = "site.status != 'online'",
) -> FilterStageNode:
    return FilterStageNode(
        map_id=MAP_ID,
        derived_from=VERSION,
        name=FilterStageNode.make_name(MAP_ID, owner, f"{label}|{tag}"),
        owner=owner,
        criteria_tag=tag,
        funnel_label=label,
        order=order,
        conditions=[condition],
        log_files=[] if files is None else files,
        anchor=Anchor(package="pandajedi", file=owner.split("::")[0], line_start=400 + order),
    )


def _skip(task: str, site: str, tag: str, why: str = "due to status=offline") -> str:
    return f"2026-09-06 04:10:24,902 panda.log.x: INFO     <jediTaskID={task}>   skip site={site} {why} criteria={tag}"


def _passed(task: str, count: int, label: str) -> str:
    return f"2026-09-06 04:10:24,902 panda.log.x: INFO     <jediTaskID={task}> {count} candidates passed {label}"


async def _localized(stages, lines, task="52249469", focus=None, whole=False):
    """Derive a distribution strategy and answer its questions with *lines*."""
    code_map = await _map(
        MapFragment(map_id=MAP_ID, derived_from=VERSION, filter_stages=stages)
    )
    strategy = await strategy_mod.derive(
        code_map,
        Symptom(kind=SYMPTOM_DISTRIBUTION, focus=focus or stages[0].criteria_tag, task_id=task),
    )
    results = []
    for query in strategy_mod.queries(strategy):
        mine = [
            line
            for line in lines.get(query.log_filename, [])
            if ("criteria=" in line) == (evidence_mod.TAG_PATTERN in query.pattern)
        ]
        results.append(
            _result(query, matched=len(mine), lines=mine, truncated=not whole and bool(mine))
        )
    return strategy_mod.evaluate(
        strategy, Evidence(fetched_at="2026-09-05T00:00:00+00:00", results=results)
    )


async def test_a_described_cut_names_the_stage_before_anything_is_asked():
    """The chain alone was not an answer.

    ``_leading`` picks the chain the tag sits in and stops there, and the
    ranked listing only ever holds stages the evidence measured -- so with no
    evidence the report named no code at all, and with it the stage sat as deep
    as forty-seventh in a listing of a hundred and nine.  Production writes the
    tag per rejected site, which makes this a lookup.
    """
    stages = [
        _filter_stage(PROD_TASK, "-status", "status check", 0, ["panda-AtlasProdTaskBroker.log"]),
        _filter_stage(PROD_JOB, "-status", "status check", 0, [PROD_JOB_LOG]),
        _filter_stage(PROD_JOB, "-lowmemory", "memory check", 4, [PROD_JOB_LOG],
                      condition="siteSpec.maxrss < minRamCount"),
    ]
    code_map = await _map(
        MapFragment(map_id=MAP_ID, derived_from=VERSION, filter_stages=stages)
    )

    strategy = await strategy_mod.localize(
        code_map, Symptom(kind=SYMPTOM_DISTRIBUTION, focus="-lowmemory")
    )
    local = strategy.localization

    assert local.describes == "-lowmemory"
    (named,) = local.emitted_by
    assert (named.owner, named.order, named.line) == (PROD_JOB, 4, 404)
    assert named.conditions == ["siteSpec.maxrss < minRamCount"]
    assert named.log_files == [PROD_JOB_LOG]
    # Every chain stays in play: naming the stage is not narrowing to it.
    assert len(local.cuts) == len(stages)


async def test_a_tag_two_chains_emit_names_both_stages():
    """Which of them it was is the file's answer and comes later, so both are
    named rather than one being chosen here."""
    stages = [
        _filter_stage(PROD_TASK, "-status", "status check", 0, ["panda-AtlasProdTaskBroker.log"]),
        _filter_stage(PROD_JOB, "-status", "status check", 0, [PROD_JOB_LOG]),
    ]
    code_map = await _map(
        MapFragment(map_id=MAP_ID, derived_from=VERSION, filter_stages=stages)
    )

    strategy = await strategy_mod.localize(
        code_map, Symptom(kind=SYMPTOM_DISTRIBUTION, focus="-status")
    )

    assert [c.owner for c in strategy.localization.emitted_by] == sorted([PROD_TASK, PROD_JOB])


async def test_a_chain_named_as_the_focus_names_no_stage():
    """A chain is not a tag, and saying "every stage in it" would be a listing
    rather than a localization."""
    stages = [_filter_stage(PROD_JOB, "-status", "status check", 0, [PROD_JOB_LOG])]
    code_map = await _map(
        MapFragment(map_id=MAP_ID, derived_from=VERSION, filter_stages=stages)
    )

    strategy = await strategy_mod.localize(
        code_map, Symptom(kind=SYMPTOM_DISTRIBUTION, focus=PROD_JOB)
    )

    assert strategy.localization.emitted_by == []
    assert strategy.localization.describes == ""


async def test_the_evidence_chooses_the_chain_and_the_focus_only_opens_the_answer():
    """A description naming one cut is a guess; the file the lines landed in is
    a fact.  Narrowing on the focus first cost exactly what it was meant to
    prevent -- a focus resolved to the task broker's ``-job`` dropped the job
    broker's stages, and its rejections were then attributed to the task
    broker's ``status check``, condition and all."""
    stages = [
        _filter_stage(PROD_TASK, "-status", "status check", 0, ["panda-AtlasProdTaskBroker.log"],
                      condition="taskSpec.nucleus in siteMapper.nuclei"),
        _filter_stage(PROD_TASK, "-job", "job check", 6, ["panda-AtlasProdTaskBroker.log"]),
        _filter_stage(PROD_JOB, "-status", "status check", 0, [PROD_JOB_LOG],
                      condition="not sitePreAssigned"),
    ]
    strategy = await _localized(
        stages,
        {PROD_JOB_LOG: [_skip("52249469", "SITE_A", "-status"), _skip("52249469", "SITE_B", "-status")]},
        focus="-job",
    )

    assert strategy.localization.chain == PROD_JOB
    (cut,) = [c for c in strategy.localization.cuts if c.sites]
    assert cut.owner == PROD_JOB
    assert cut.conditions == ["not sitePreAssigned"]


async def test_a_cut_is_weighed_in_candidates_not_in_log_lines():
    """Brokerage re-runs for a task many times over a window, so a line count is
    a count of passes multiplied by an effect and ranks by how often the task
    was brokered."""
    stages = [_filter_stage(PROD_JOB, "-status", "status check", 0, [PROD_JOB_LOG])]
    strategy = await _localized(
        stages,
        {PROD_JOB_LOG: [_skip("52249469", "SITE_A", "-status")] * 5
         + [_skip("52249469", "SITE_B", "-status")]},
    )

    (cut,) = strategy.localization.cuts
    assert cut.sites == ["SITE_A", "SITE_B"]
    assert cut.verdict == SEEN


async def test_the_reason_the_line_gave_is_kept_with_the_cut():
    """The text between the site and the tag carries the measured values that
    made the condition true, which is what turns a condition into a reason --
    and it is the value at the moment of the decision, not now."""
    stages = [_filter_stage(PROD_JOB, "-status", "status check", 0, [PROD_JOB_LOG])]
    strategy = await _localized(
        stages,
        {PROD_JOB_LOG: [
            _skip("52249469", "SITE_A", "-status", "due to status=brokeroff"),
            _skip("52249469", "SITE_B", "-status", "due to status=brokeroff"),
            _skip("52249469", "SITE_C", "-status", "due to status=test"),
        ]},
    )

    assert strategy.localization.cuts[0].reasons == [
        "due to status=brokeroff (2x)",
        "due to status=test (1x)",
    ]


async def test_a_partial_sample_rules_no_step_out():
    """An answer cut off at a bound is indistinguishable from a step that never
    fired, so only what was seen counts."""
    stages = [
        _filter_stage(PROD_JOB, "-status", "status check", 0, [PROD_JOB_LOG]),
        _filter_stage(PROD_JOB, "-lowmemory", "memory check", 14, [PROD_JOB_LOG]),
    ]
    strategy = await _localized(
        stages, {PROD_JOB_LOG: [_skip("52249469", "SITE_A", "-status")]}
    )

    verdicts = {c.tag: c.verdict for c in strategy.localization.cuts}
    assert verdicts == {"-status": SEEN, "-lowmemory": UNSETTLED}
    assert strategy.localization.sample == "partial"
    assert any("cut off at a bound" in gap for gap in strategy.gaps)


async def test_a_whole_sample_does_rule_out_a_step_that_took_nothing():
    """The other direction, and the only thing that licenses it."""
    stages = [
        _filter_stage(PROD_JOB, "-status", "status check", 0, [PROD_JOB_LOG]),
        _filter_stage(PROD_JOB, "-lowmemory", "memory check", 14, [PROD_JOB_LOG]),
    ]
    strategy = await _localized(
        stages,
        {PROD_JOB_LOG: [_skip("52249469", "SITE_A", "-status"), _passed("52249469", 9, "status check")]},
        whole=True,
    )

    verdicts = {c.tag: c.verdict for c in strategy.localization.cuts}
    assert verdicts == {"-status": SEEN, "-lowmemory": ELIMINATED}
    assert strategy.localization.sample == "complete"


async def test_the_funnel_keeps_the_count_and_is_read_in_the_map_s_order():
    """The step the log reports last is not the step the chain runs last: the
    chain re-runs, and over 51,128 adjacent pairs in one file 11% of them show
    the count going up.  So the order comes from the map and the passes are
    aggregated rather than guessed apart."""
    stages = [
        _filter_stage(PROD_JOB, "-status", "status check", 0, [PROD_JOB_LOG]),
        _filter_stage(PROD_JOB, "-shortwalltime", "walltime check", 19, [PROD_JOB_LOG]),
    ]
    strategy = await _localized(
        stages,
        {PROD_JOB_LOG: [
            _passed("52249469", 40, "walltime check"),
            _passed("52249469", 151, "status check"),
            _passed("52249469", 2, "walltime check"),
            _passed("52249469", 150, "status check"),
        ]},
    )

    funnel = strategy.localization.funnel
    assert [(f.label, f.most, f.fewest, f.seen) for f in funnel] == [
        ("status check", 151, 150, 2),
        ("walltime check", 40, 2, 2),
    ]
    assert (strategy.localization.entered, strategy.localization.left) == (151, 40)
    assert strategy.localization.passes == 2


async def test_a_step_production_counts_that_the_map_has_no_stage_for_is_a_finding():
    """Positive evidence, so it holds whatever the sample size: the map is
    missing a cut production makes."""
    stages = [_filter_stage(PROD_JOB, "-status", "status check", 0, [PROD_JOB_LOG])]
    strategy = await _localized(
        stages,
        {PROD_JOB_LOG: [
            _skip("52249469", "SITE_A", "-nucleus"),
            _passed("52249469", 9, "endpoint check with DISK_THRESHOLD=10 TB"),
        ]},
    )

    assert any("-nucleus" in finding for finding in strategy.findings)
    assert any("endpoint check" in finding for finding in strategy.findings)


async def test_another_task_s_lines_are_not_this_task_s_answer():
    """The wrapper's prefix is what attributes a line, and a rejection message
    can carry other numbers of its own."""
    stages = [_filter_stage(PROD_JOB, "-status", "status check", 0, [PROD_JOB_LOG])]
    strategy = await _localized(
        stages,
        {PROD_JOB_LOG: [_skip("52249469", "MINE", "-status"), _skip("52357008", "THEIRS", "-status")]},
    )

    assert strategy.localization.cuts[0].sites == ["MINE"]


# ---------------------------------------------------------------------------
# The one entry point
# ---------------------------------------------------------------------------


def test_the_command_refuses_to_be_given_the_question_twice():
    """One way in.  A description is resolved against the map's vocabulary and a
    subject names an entry outright; accepting both would leave it to the
    command to decide which the caller meant, which is the choice this layer
    exists to take away from whoever phrased the question."""
    from click.testing import CliRunner

    from bamboo.scripts.derive_strategy import main

    both = CliRunner().invoke(
        main, ["--describe", "x", "--subject", "JediTaskSpec.status", "--observed", "pending"]
    )
    neither = CliRunner().invoke(main, [])
    half = CliRunner().invoke(main, ["--subject", "JediTaskSpec.status"])

    assert both.exit_code != 0 and "either --describe" in both.output
    assert neither.exit_code != 0
    assert half.exit_code != 0 and "only meaningful together" in half.output


# ---------------------------------------------------------------------------
# Leads -- where the answer goes next
# ---------------------------------------------------------------------------


async def test_a_carried_value_opens_the_next_question_with_the_value_intact():
    """The one hop the map can license without observing anything new.

    ``status = oldStatus`` says the value arrived unchanged, so the observed
    value is also the answer to the next question.  That is what makes the lead
    a whole symptom rather than a field to go and look at -- the caller can put
    it straight back in, which is the contract the loop runs on.
    """
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        subjects=[_subject(), _subject(name="JediTaskSpec.oldStatus")],
        junctions=[
            _junction(
                "jedidog/AtlasProdWatchDog.py::doActionForReassign",
                Branch(outcome="passthrough(JediTaskSpec.oldStatus)", tier=2),
                log_files=[KNIGHT_LOG],
            ),
            _junction(
                "jediorder/TaskRefiner.py::runImpl",
                Branch(outcome="finishing"),
                log_files=[OTHER_LOG],
                subject="JediTaskSpec.oldStatus",
            ),
        ],
    )
    strategy = await strategy_mod.derive(
        await _map(fragment), Symptom(subject=SUBJECT, observed="finishing", task_id="7")
    )

    carried = [lead for lead in strategy.leads if lead.field == "JediTaskSpec.oldStatus"]
    assert len(carried) == 1
    assert carried[0].symptom is not None
    assert carried[0].symptom.subject == "JediTaskSpec.oldStatus"
    assert carried[0].symptom.observed == "finishing"
    assert carried[0].symptom.task_id == "7"
    assert carried[0].stop == ""


async def test_a_carried_value_with_no_upstream_writer_stops_and_says_so():
    """Reaching the edge of the map is an answer, not a failure to find one.

    The value came in from outside anything this map explains, which is exactly
    what an unbound boundary is -- a complete statement and a work item at the
    same time.
    """
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        subjects=[_subject()],
        junctions=[
            _junction(
                "a.py::f",
                Branch(outcome="passthrough(JediTaskSpec.oldStatus)", tier=2),
                log_files=[KNIGHT_LOG],
            )
        ],
    )
    strategy = await strategy_mod.derive(
        await _map(fragment), Symptom(subject=SUBJECT, observed="finishing", task_id="7")
    )

    carried = [lead for lead in strategy.leads if lead.field == "JediTaskSpec.oldStatus"]
    assert carried and carried[0].symptom is None
    assert carried[0].stop == models.STOP_NO_WRITER


async def test_the_walk_does_not_lead_back_to_the_question_it_is_answering():
    """``JediTaskSpec.status`` and ``oldStatus`` carry from each other, and
    ``JobSpec.jobStatus`` carries from itself.  Emitting those as leads would
    hand the caller a loop dressed as progress, and the caller cannot tell it
    from a real hop without already knowing the map."""
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        subjects=[_subject()],
        junctions=[
            _junction(
                "a.py::f",
                Branch(outcome=f"passthrough({SUBJECT})", tier=2),
                log_files=[KNIGHT_LOG],
            )
        ],
    )
    strategy = await strategy_mod.derive(
        await _map(fragment), Symptom(subject=SUBJECT, observed="finishing", task_id="7")
    )

    assert [lead.field for lead in strategy.leads] == []


async def test_a_gate_the_map_only_reads_is_a_lead_that_stops():
    """The table that bounded the query nobody could see past.  It is a
    terminal of the walk rather than a junction, and saying which terminal is
    the answer -- ``JEDI_AUX_Status_MinTaskID`` had stopped being updated."""
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        subjects=[_subject(selected=["finishing"], gates=["JEDI_AUX_Status_MinTaskID"])],
        junctions=[_junction("a.py::f", Branch(outcome="finishing"), log_files=[KNIGHT_LOG])],
    )
    strategy = await strategy_mod.derive(
        await _map(fragment), Symptom(subject=SUBJECT, observed="finishing", task_id="7")
    )

    gate = [lead for lead in strategy.leads if lead.field == "JEDI_AUX_Status_MinTaskID"]
    assert gate and gate[0].stop == models.STOP_SHARED_TABLE
    assert gate[0].symptom is None


async def test_eliminating_a_candidate_takes_its_leads_with_it():
    """A walk that keeps descending from a branch the evidence ruled out is
    following a path the system did not take.  This is the same asymmetry the
    candidates carry, applied one step further out."""
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        subjects=[_subject(), _subject(name="JediTaskSpec.oldStatus")],
        junctions=[
            _junction(
                "jediorder/ContentsFeeder.py::feed",
                _tagged(
                    "passthrough(JediTaskSpec.oldStatus)",
                    "reason=fed",
                    KNIGHT_LOG,
                    tier=2,
                ),
                log_files=[KNIGHT_LOG],
            ),
            _junction(
                "jediorder/TaskRefiner.py::runImpl",
                Branch(outcome="pending"),
                log_files=[OTHER_LOG],
                subject="JediTaskSpec.oldStatus",
            ),
        ],
    )
    code_map = await _map(fragment)
    strategy = await strategy_mod.derive(
        code_map, Symptom(subject=SUBJECT, observed="pending", task_id="1")
    )
    assert [lead.field for lead in strategy.leads] == ["JediTaskSpec.oldStatus"]

    settled = strategy_mod.evaluate(strategy, _evidence(strategy, _quiet))

    assert [c.verdict for c in settled.candidates] == [ELIMINATED]
    assert settled.leads == []


async def test_a_cut_names_the_field_it_tested_and_the_value_the_line_reported():
    """The strongest hop in the system, and the evidence supplies it.

    A rejection line carries the measured value that made the condition true --
    ``due to status=offline`` -- so the next question is not "go and look at the
    site status" but a whole symptom with the value already in it.  The map
    alone could never have said ``offline``.
    """
    stages = [
        _filter_stage(
            PROD_JOB,
            "-status",
            "status check",
            0,
            [PROD_JOB_LOG],
            condition="tmpSiteSpec.status not in ('online',) and tmpSiteSpec.maxwdir",
        )
    ]
    code_map = await _map(
        MapFragment(
            map_id=MAP_ID,
            derived_from=VERSION,
            subjects=[
                _subject(name="SiteSpec.status"),
                _subject(name="SiteSpec.maxwdir"),
                _subject(name="JediTaskSpec.status"),
            ],
            junctions=[
                _junction(
                    "configurator.py::run",
                    Branch(outcome="offline"),
                    log_files=[OTHER_LOG],
                    subject="SiteSpec.status",
                )
            ],
            filter_stages=stages,
        )
    )
    strategy = await strategy_mod.derive(
        code_map, Symptom(kind=SYMPTOM_DISTRIBUTION, focus="-status", task_id="52249469")
    )
    results = []
    for query in strategy_mod.queries(strategy):
        mine = [_skip("52249469", "SITE_A", "-status")] if evidence_mod.TAG_PATTERN in query.pattern else []
        results.append(_result(query, matched=len(mine), lines=mine))
    settled = strategy_mod.evaluate(
        strategy, Evidence(fetched_at="2026-09-05T00:00:00+00:00", results=results)
    )

    site = [lead for lead in settled.leads if lead.field == "SiteSpec.status"]
    assert site, [lead.field for lead in settled.leads]
    assert site[0].symptom is not None
    assert site[0].symptom.subject == "SiteSpec.status"
    assert site[0].symptom.observed == "offline"
    # The task id is the *task*'s, and this question is about a site.
    assert site[0].symptom.task_id is None


async def test_a_name_several_subjects_declare_is_reported_rather_than_picked():
    """``status`` is declared by a dozen specs, and the record keeps only the
    leaf of each read -- the receiver that would settle it was thrown away when
    the stage was stored.  Choosing one anyway is the retrieval failure this
    design exists to remove, so the ambiguity is the answer."""
    stages = [
        _filter_stage(
            PROD_JOB, "-status", "status check", 0, [PROD_JOB_LOG],
            condition="tmpSiteSpec.status != 'online'",
        )
    ]
    code_map = await _map(
        MapFragment(
            map_id=MAP_ID,
            derived_from=VERSION,
            subjects=[_subject(name="SiteSpec.status"), _subject(name="JobSpec.status")],
            filter_stages=stages,
        )
    )
    strategy = await strategy_mod.derive(
        code_map, Symptom(kind=SYMPTOM_DISTRIBUTION, focus="-status", task_id="52249469")
    )

    ambiguous = [lead for lead in strategy.leads if lead.stop == models.STOP_AMBIGUOUS]
    assert ambiguous and ambiguous[0].field == "status"
    assert "SiteSpec.status" in ambiguous[0].why and "JobSpec.status" in ambiguous[0].why


async def test_a_lead_survives_while_any_surviving_candidate_still_opens_it():
    """Five writers of ``oldStatus`` copy it from ``status``.  Folding them to
    one before the evidence has spoken makes the field's presence depend on
    which candidate happened to be listed first: rule that one out and the
    whole hop disappears, although four others still open it."""
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        subjects=[_subject(), _subject(name="JediTaskSpec.oldStatus")],
        junctions=[
            _junction(
                "a.py::ruled_out",
                _tagged(
                    "passthrough(JediTaskSpec.oldStatus)",
                    "reason=fed",
                    KNIGHT_LOG,
                    tier=2,
                ),
                log_files=[KNIGHT_LOG],
            ),
            _junction(
                "b.py::no_log_to_ask",
                Branch(outcome="passthrough(JediTaskSpec.oldStatus)", tier=2),
                log_files=PROXY_LOGS,
                owns_logger=False,
            ),
            _junction(
                "c.py::upstream",
                Branch(outcome="pending"),
                log_files=[OTHER_LOG],
                subject="JediTaskSpec.oldStatus",
            ),
        ],
    )
    strategy = await strategy_mod.derive(
        await _map(fragment), Symptom(subject=SUBJECT, observed="pending", task_id="1")
    )
    assert [lead.opened_by for lead in strategy.leads] == ["a.py::ruled_out", "b.py::no_log_to_ask"]

    settled = strategy_mod.evaluate(strategy, _evidence(strategy, _quiet))

    ruled_out = [c for c in settled.candidates if c.verdict == ELIMINATED]
    assert [c.owner for c in ruled_out] == ["a.py::ruled_out"]
    assert [lead.opened_by for lead in settled.leads] == ["b.py::no_log_to_ask"]


def test_a_continuing_lead_is_printed_as_the_command_that_asks_it():
    """The loop's contract, and the only part of it a reader can check.

    Describing the next question in prose would leave nobody able to take the
    hop, and leave this layer free to emit one that cannot be asked.  Printing
    it as the invocation makes the two the same thing.
    """
    import click

    from bamboo.scripts.derive_strategy import _report_leads

    strategy = strategy_mod.Strategy(
        symptom=Symptom(subject=SUBJECT, observed="finishing", task_id="7"),
        map_id=MAP_ID,
        derived_from=VERSION,
        leads=[
            models.Lead(
                field="JEDI_AUX_Status_MinTaskID",
                stop=models.STOP_SHARED_TABLE,
                why="it bounds which rows the query can see",
            ),
            models.Lead(
                field="JediTaskSpec.oldStatus",
                symptom=Symptom(
                    subject="JediTaskSpec.oldStatus", observed="finishing", task_id="7"
                ),
                why="the writer copies the value from it",
            ),
        ],
    )
    runner = click.testing.CliRunner()
    command = click.Command("x", callback=lambda: _report_leads(strategy, 5, False))
    output = runner.invoke(command).output

    # Continuing first: one can be asked now, the other is where it ran out.
    assert output.index("ask   ") < output.index("stops ")
    assert (
        "--subject JediTaskSpec.oldStatus --observed finishing --task 7" in output
    ), output


# ---------------------------------------------------------------------------
# What code the answer says to read
# ---------------------------------------------------------------------------


async def test_the_reading_is_per_function_not_per_arm():
    """One entry however many arms it holds.

    The sharing is not incidental: 497 junctions sit in 213 functions and one
    of them holds twenty.  Asking about each arm separately reads the same text
    over again and invites two answers about one piece of code.
    """
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        subjects=[_subject()],
        junctions=[
            _junction("jediorder/A.py::run", Branch(outcome="pending", line=10), gloss_key="same"),
            _junction("jediorder/A.py::run2", Branch(outcome="pending", line=40), gloss_key="same"),
        ],
    )
    strategy = await strategy_mod.derive(
        await _map(fragment), Symptom(subject=SUBJECT, observed="pending", task_id="7")
    )

    assert len(strategy.candidates) == 2
    assert len(strategy.readings) == 1
    assert strategy.readings[0].lines == [10, 40]


async def test_an_arm_with_no_line_of_its_own_is_shown_as_silent():
    """Said rather than left out.

    Of 1046 branches the map can point at 988 in the source and 484 carry a
    line production prints, so 43% can be put side by side.  Inventing a
    pattern for the rest would turn a known silence into an empty query, and an
    empty query is what this design reads as evidence.
    """
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        subjects=[_subject()],
        junctions=[
            _junction(
                "jedipprocess/PostProcessorBase.py::doPreCheck",
                Branch(outcome="pending", line=272),
                owns_logger=False,
            )
        ],
    )
    strategy = await strategy_mod.derive(
        await _map(fragment), Symptom(subject=SUBJECT, observed="pending", task_id="7")
    )

    assert [r.silent for r in strategy.readings] == [True]


async def test_the_reading_follows_the_arm_the_record_names():
    """A match is proof that this arm decided, so the others stop being offered.

    The other direction is not available: a record naming none proves nothing,
    because the field holds the last message written to it.
    """
    strategy = await _six_arms(
        diag="#ATM #KV action=set_exhausted reason=scout_cpuTime measured 900"
    )

    assert len(strategy.candidates) == 2
    assert len(strategy.readings) == 1
    assert strategy.readings[0].owner == _SCOUT
    assert strategy.readings[0].lines == [1239]


async def test_a_ruled_out_candidate_takes_its_reading_with_it():
    """Reading code the evidence ruled out sends a reader down a path the system
    did not take, which is worse than offering nothing."""
    strategy = await _two_candidates()
    before = len(strategy.readings)

    def decide(role, filename, service):
        if role == strategy_mod.CONTROL:
            return {"matched": 5}
        if filename == KNIGHT_LOG:
            return {"matched": 1, "lines": ["set task_status=pending"]}
        return {"matched": 0}

    settled = strategy_mod.evaluate(strategy, _evidence(strategy, decide))
    verdicts = {c.owner: c.verdict for c in settled.candidates}

    assert before == 2
    assert verdicts["jediorder/TaskCommando.py::run"] == ELIMINATED
    assert [r.owner for r in settled.readings] == ["jediorder/ContentsFeeder.py::feed"]


# ---------------------------------------------------------------------------
# Lead source 2: the helper a junction calls
# ---------------------------------------------------------------------------

_GETTER = "taskbuffer/db_proxy_mods/task_utils_module.py::getScoutJobData_JEDI"
_JOB_STATUS = "JobSpec.jobStatus"
_JOB = "JobSpec"
_JOB_FETCH = "taskbuffer/db_proxy_mods/task_standalone_module.py::getPandaIDsWithTask_JEDI"


async def _scout_calling(
    calls: list[str] | None = None,
    reader: str = _GETTER,
    read_subject: str = _JOB_STATUS,
    entities: list[EntityNode] | None = None,
    subjects: list | None = None,
):
    """The scout junction, and a helper of the same module that reads jobs."""
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        entities=entities or [],
        subjects=subjects
        if subjects is not None
        else [
            _subject(selected=["exhausted"]),
            _subject(
                name=read_subject,
                selected=["finished"],
                selected_by={"finished": [reader]},
            ),
        ],
        junctions=[
            _junction(
                _SCOUT,
                _arm("scout_cpuTime", "scoutData['cpuTime'] > thr", 1239),
                owns_logger=False,
                caller_log_files=[KNIGHT_LOG],
                calls=[_GETTER] if calls is None else calls,
            )
        ],
    )
    return await strategy_mod.derive(
        await _map(fragment),
        Symptom(subject=SUBJECT, observed="exhausted", task_id="7"),
    )


async def test_a_query_asking_for_two_kinds_of_row_opens_the_second():
    """The one place a junction's own reads can be used without the bad join.

    Sharing a function proves nothing -- that mistake gave a junction fourteen
    entry points -- but a single ``FROM`` list is the corpus stating the
    relation.  ``prepareTasksToBeFinished_JEDI`` selects tasks against their
    datasets in one statement, so it can say which datasets the task waits on.
    """
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        subjects=[_subject(selected=["finishing"])],
        junctions=[
            _junction(
                "taskbuffer/db_proxy_mods/task_complex_module.py"
                "::prepareTasksToBeFinished_JEDI",
                Branch(outcome="finishing"),
                joined_entities=["JediDatasetSpec"],
            )
        ],
    )
    strategy = await strategy_mod.derive(
        await _map(fragment), Symptom(subject=SUBJECT, observed="finishing", task_id="7")
    )

    lead = next(lead for lead in strategy.leads if lead.field == "JediDatasetSpec")
    assert lead.stop == models.STOP_DESCENT
    # Deterministic: the map holds the statement, so this is not a hypothesis
    # the way a hop along a call is.
    assert lead.source == models.LEAD_MAP
    assert "same query as the rows it decides about" in lead.why


async def test_the_junctions_own_entity_is_not_a_descent_from_itself():
    """A statement reading two tables of the same kind of row says nothing
    about where to go next -- the rows are the ones already being asked about,
    and calling that a descent puts a population question in a value's place."""
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        subjects=[_subject(selected=["finishing"])],
        junctions=[
            _junction(
                "taskbuffer/db_proxy_mods/task_complex_module.py::rescue",
                Branch(outcome="finishing"),
                joined_entities=["JediTaskSpec"],
            )
        ],
    )
    strategy = await strategy_mod.derive(
        await _map(fragment), Symptom(subject=SUBJECT, observed="finishing", task_id="7")
    )

    assert not [lead for lead in strategy.leads if lead.field == "JediTaskSpec"]


async def test_a_helper_the_junction_calls_opens_the_row_it_selects_on():
    """The scout arm reads an aggregate over jobs, and the map has no edge for
    it: ``setScoutJobData_JEDI`` writes ``exhausted`` and
    ``getScoutJobData_JEDI`` selects ``JobSpec.jobStatus='finished'``, with only
    the call between them.  One hop along ``self.<method>()`` is what turns the
    question into one about jobs, and nothing else in the map does."""
    strategy = await _scout_calling()

    lead = next(lead for lead in strategy.leads if lead.field == _JOB)
    assert lead.source == models.LEAD_CALLEE
    assert lead.stop == models.STOP_DESCENT
    assert "getScoutJobData_JEDI" in lead.why
    # Named by the entity, said by the column: the descent is to a population
    # either way, and which column was asked about is what says why.
    assert f"selects {_JOB_STATUS}=finished" in lead.why
    assert lead.opened_by == _SCOUT


async def test_a_helper_that_names_no_promoted_value_still_opens_its_rows():
    """The half a pair of class and attribute cannot carry.

    ``getPandaIDsWithTask_JEDI`` selects a task's jobs on the join key alone, so
    no promoted attribute appears in its predicates and the map read it as
    consulting nothing -- when fetching those rows is the whole of what "the
    finish is waiting on jobs" means.
    """
    strategy = await _scout_calling(
        calls=[_JOB_FETCH],
        subjects=[_subject(selected=["exhausted"])],
        entities=[
            EntityNode(
                name="JobSpec",
                map_id=MAP_ID,
                derived_from=VERSION,
                tables=["jobsActive4"],
                read_by=[_JOB_FETCH],
            )
        ],
    )

    lead = next(lead for lead in strategy.leads if lead.field == _JOB)
    assert lead.stop == models.STOP_DESCENT
    assert "getPandaIDsWithTask_JEDI(), which reads JobSpec rows" in lead.why


async def test_two_columns_of_one_entity_open_one_descent():
    """The descent is to a population, so naming it once per column reads as
    several descents where the map is making one claim."""
    strategy = await _scout_calling(
        subjects=[
            _subject(selected=["exhausted"]),
            _subject(
                name=_JOB_STATUS, selected=["finished"], selected_by={"finished": [_GETTER]}
            ),
            _subject(
                name="JobSpec.jobSubStatus",
                selected=["es_discard"],
                selected_by={"es_discard": [_GETTER]},
            ),
        ],
        entities=[
            EntityNode(
                name="JobSpec",
                map_id=MAP_ID,
                derived_from=VERSION,
                tables=["jobsActive4"],
                read_by=[_GETTER],
            )
        ],
    )

    assert [lead.field for lead in strategy.leads] == [_JOB]


async def test_a_function_that_only_writes_an_entity_opens_nothing():
    """A descent asks about the rows a decision was read *from*.  Following a
    writer instead walks forwards while calling itself a step down."""
    strategy = await _scout_calling(
        calls=[_JOB_FETCH],
        subjects=[_subject(selected=["exhausted"])],
        entities=[
            EntityNode(
                name="JobSpec",
                map_id=MAP_ID,
                derived_from=VERSION,
                tables=["jobsActive4"],
                updated_by=[_JOB_FETCH],
            )
        ],
    )

    assert [lead.field for lead in strategy.leads] == []


async def test_sharing_an_owner_with_a_reader_is_not_a_call():
    """The unsound join this corpus has already punished twice.

    ``selected_by`` records the function, not the statement, so a 300-line
    method that handles several commands reads jobs in one arm and writes the
    task status in another -- and joining the two facts through their shared
    owner calls that a relation.  Only a call edge counts."""
    strategy = await _scout_calling(calls=[], reader=_SCOUT)

    assert [lead.field for lead in strategy.leads] == []


async def test_a_helper_of_the_same_name_elsewhere_opens_nothing():
    """The join is on the whole target, never the bare name.

    A call may cross a module boundary, but which module it lands in was
    settled when the map was built and is spelt out in the target.  Matching
    the name instead is the rule that gave one junction fourteen entry points,
    thirteen of them wrong.
    """
    strategy = await _scout_calling(
        reader="jedirefine/TaskRefinerBase.py::getScoutJobData_JEDI"
    )

    assert [lead.field for lead in strategy.leads] == []


async def test_a_helper_reading_the_same_entity_is_not_a_descent():
    """Reading another field of the same spec is a lateral read, not a step down
    to the rows underneath -- and calling it one would put a population question
    where a value question belongs."""
    strategy = await _scout_calling(read_subject="JediTaskSpec.oldStatus")

    assert [lead.field for lead in strategy.leads] == []


async def test_a_ruled_out_candidate_takes_its_callee_lead_with_it():
    """Same rule as the map's own leads: a walk that keeps descending from an
    arm production says did not fire follows a path the system did not take."""
    strategy = await _scout_calling()

    def decide(role, filename, service):
        if role == strategy_mod.CONTROL:
            return {"matched": 5}
        return {"matched": 0}

    settled = strategy_mod.evaluate(strategy, _evidence(strategy, decide))

    assert [c.verdict for c in settled.candidates] == [ELIMINATED]
    assert [lead.field for lead in settled.leads] == []


async def test_the_map_leads_say_they_came_from_the_map():
    """Provenance is on every lead, not only the proposed ones.  Without it a
    trace cannot tell a deterministic edge from a hypothesis, and the map's
    coverage reads better than it is."""
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        subjects=[
            _subject(gates=["JEDI_AUX_Status_MinTaskID"]),
            _subject(name="JediTaskSpec.oldStatus"),
        ],
        junctions=[
            _junction(
                "jedidog/AtlasProdWatchDog.py::doActionForReassign",
                Branch(outcome="passthrough(JediTaskSpec.oldStatus)", tier=2),
                log_files=[KNIGHT_LOG],
            ),
            _junction(
                "jediorder/TaskRefiner.py::runImpl",
                Branch(outcome="finishing"),
                log_files=[OTHER_LOG],
                subject="JediTaskSpec.oldStatus",
            ),
        ],
    )
    strategy = await strategy_mod.derive(
        await _map(fragment), Symptom(subject=SUBJECT, observed="finishing", task_id="7")
    )

    assert len(strategy.leads) == 2
    assert {lead.source for lead in strategy.leads} == {models.LEAD_MAP}


# ---------------------------------------------------------------------------
# The walk: one derivation per hop, driven from outside
# ---------------------------------------------------------------------------


async def _two_field_map(loops_back: bool = False) -> CodeMap:
    """``status`` carries from ``oldStatus``, which has a writer of its own."""
    # The real shape: ``JediTaskSpec.recordOldStatus`` does ``self.oldStatus =
    # self.status``, a junction of its own rather than another arm of the one
    # that writes a literal.
    back = (
        [
            _junction(
                "taskbuffer/JediTaskSpec.py::recordOldStatus",
                Branch(outcome="passthrough(JediTaskSpec.status)", tier=2),
                log_files=[OTHER_LOG],
                subject="JediTaskSpec.oldStatus",
            )
        ]
        if loops_back
        else []
    )
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        subjects=[_subject(), _subject(name="JediTaskSpec.oldStatus")],
        junctions=[
            _junction(
                "jedidog/AtlasProdWatchDog.py::doActionForReassign",
                Branch(outcome="passthrough(JediTaskSpec.oldStatus)", tier=2),
                log_files=[KNIGHT_LOG],
            ),
            _junction(
                "jediorder/TaskRefiner.py::runImpl",
                Branch(outcome="finishing"),
                log_files=[OTHER_LOG],
                subject="JediTaskSpec.oldStatus",
            ),
            *back,
        ],
    )
    return await _map(fragment)


async def test_the_walk_takes_the_second_hop_without_being_asked_again():
    """The point of the whole round.  A symptom is a path, not a lookup: the map
    named ``oldStatus`` as where ``finishing`` came from, and until now a person
    had to retype that as the next question.  The evidence chooses the next hop;
    the description only ever chose the first."""
    from bamboo.scripts.derive_strategy import walk

    investigation = await walk(
        await _two_field_map(),
        Symptom(subject=SUBJECT, observed="finishing", task_id="7"),
        budget=3,
    )

    assert [hop.symptom.subject for hop in investigation.hops] == [
        SUBJECT,
        "JediTaskSpec.oldStatus",
    ]
    assert investigation.hops[1].symptom.observed == "finishing"
    assert investigation.hops[1].symptom.task_id == "7"
    assert investigation.hops[1].opened == "JediTaskSpec.oldStatus"
    assert investigation.hops[1].source == models.LEAD_MAP


async def test_a_question_already_asked_stops_the_walk_and_is_named():
    """``status`` and ``oldStatus`` copy from each other, so the walk comes back
    to where it started.  That is a property of the system and not a failure, so
    it is reported as one -- a walk without the set would go round for ever, and
    one that silently dropped the repeat would look like a dead end."""
    from bamboo.scripts.derive_strategy import walk

    investigation = await walk(
        await _two_field_map(loops_back=True),
        Symptom(subject=SUBJECT, observed="finishing", task_id="7"),
        budget=5,
    )

    assert len(investigation.hops) == 2
    assert investigation.stopped == models.WALK_ASKED
    assert investigation.cycles == [f"{SUBJECT}=finishing"]


async def test_the_budget_is_what_stops_a_walk_nobody_is_watching():
    """An unattended run reaches production once per hop, so the hop count is
    the only thing standing in for the person who would otherwise stop it."""
    from bamboo.scripts.derive_strategy import walk

    investigation = await walk(
        await _two_field_map(loops_back=True),
        Symptom(subject=SUBJECT, observed="finishing", task_id="7"),
        budget=1,
    )

    assert len(investigation.hops) == 1
    assert investigation.stopped == models.WALK_BUDGET


async def test_a_walk_whose_leads_all_stop_says_which_terminal_it_reached():
    """Reaching the edge of the map is the answer.  Naming the category it
    stopped in is what makes it one, rather than an absence of one."""
    from bamboo.scripts.derive_strategy import walk

    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        subjects=[_subject(gates=["JEDI_AUX_Status_MinTaskID"])],
        junctions=[
            _junction(
                "jediorder/TaskRefiner.py::runImpl",
                Branch(outcome="finishing"),
                log_files=[OTHER_LOG],
            )
        ],
    )
    investigation = await walk(
        await _map(fragment),
        Symptom(subject=SUBJECT, observed="finishing", task_id="7"),
        budget=3,
    )

    assert len(investigation.hops) == 1
    assert investigation.stopped == models.WALK_TERMINAL
    assert [lead.stop for lead in investigation.hops[0].strategy.leads] == [
        models.STOP_SHARED_TABLE
    ]


async def test_the_walk_settles_each_hop_before_choosing_the_next():
    """Evidence goes between the hops, not around them.  A walk that derived
    every hop first would follow leads opened by arms production had already
    ruled out -- which is the bug the surviving-candidate filter exists for, one
    level up."""
    from bamboo.scripts.derive_strategy import walk

    seen: list[str] = []

    def settle(strategy):
        seen.append(strategy.symptom.subject)
        return strategy

    investigation = await walk(
        await _two_field_map(),
        Symptom(subject=SUBJECT, observed="finishing", task_id="7"),
        budget=3,
        settle=settle,
    )

    assert seen == [SUBJECT, "JediTaskSpec.oldStatus"]
    assert len(investigation.hops) == 2


async def test_a_recorded_edge_is_listed_before_an_assembled_one():
    """``_deduped`` keeps the first lead naming a field, so the order the
    derivation builds them in decides which supplier wins when both name one.
    A hypothesis taken in place of an edge the extraction actually recorded is
    the map reporting less than it knows."""
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        subjects=[
            _subject(),
            _subject(name="JediTaskSpec.oldStatus"),
            _subject(
                name=_JOB_STATUS,
                selected=["finished"],
                selected_by={"finished": [_GETTER]},
            ),
        ],
        junctions=[
            _junction(
                _SCOUT,
                Branch(outcome="passthrough(JediTaskSpec.oldStatus)", tier=2),
                calls=[_GETTER],
                log_files=[KNIGHT_LOG],
            ),
            _junction(
                "jediorder/TaskRefiner.py::runImpl",
                Branch(outcome="finishing"),
                log_files=[OTHER_LOG],
                subject="JediTaskSpec.oldStatus",
            ),
        ],
    )
    strategy = await strategy_mod.derive(
        await _map(fragment), Symptom(subject=SUBJECT, observed="finishing", task_id="7")
    )

    assert [lead.source for lead in strategy.leads] == [
        models.LEAD_MAP,
        models.LEAD_CALLEE,
    ]


async def test_the_report_says_which_leads_were_not_recorded_edges():
    """Provenance has to reach the page.  A reader who cannot tell an edge the
    extraction found from one assembled out of a call reads the map's coverage
    as better than it is, which is the measurement P3 depends on."""
    import click.testing

    from bamboo.scripts.derive_strategy import _report_leads

    strategy = strategy_mod.Strategy(
        symptom=Symptom(subject=SUBJECT, observed="exhausted"),
        map_id=MAP_ID,
        derived_from=VERSION,
        leads=[
            models.Lead(field="JediTaskSpec.oldStatus", why="copied", source=models.LEAD_MAP),
            models.Lead(
                field=_JOB_STATUS,
                stop=models.STOP_DESCENT,
                why="asks getScoutJobData_JEDI()",
                source=models.LEAD_CALLEE,
            ),
        ],
    )
    command = click.Command("x", callback=lambda: _report_leads(strategy, 5, False))
    out = click.testing.CliRunner().invoke(command).output

    assert "[callee]" in out
    assert "[map]" not in out  # the default supplier is not worth a badge on every line


async def test_two_helpers_reaching_one_entity_are_both_named():
    """Folding the destination is for reading; folding the openers hides the answer.

    Several helpers reach one kind of row and which one did is all that tells
    them apart: ``runImpl`` asks ``reassignShare`` in one arm and
    ``getPandaIDsWithTask_JEDI`` in another, and keeping only the first says
    the task is reassigning when it is waiting on its jobs.
    """
    import click.testing

    from bamboo.scripts.derive_strategy import _report_leads

    strategy = strategy_mod.Strategy(
        symptom=Symptom(subject=SUBJECT, observed="finishing"),
        map_id=MAP_ID,
        derived_from=VERSION,
        leads=[
            models.Lead(
                field=_JOB,
                stop=models.STOP_DESCENT,
                why="runImpl asks reassignShare(), which selects JobSpec.jobStatus=activated",
                source=models.LEAD_CALLEE,
            ),
            models.Lead(
                field=_JOB,
                stop=models.STOP_DESCENT,
                why="runImpl asks getPandaIDsWithTask_JEDI(), which reads JobSpec rows",
                source=models.LEAD_CALLEE,
            ),
        ],
    )
    command = click.Command("x", callback=lambda: _report_leads(strategy, 5, False))
    out = click.testing.CliRunner().invoke(command).output

    assert out.count("stops JobSpec") == 1
    assert "reassignShare" in out
    assert "getPandaIDsWithTask_JEDI" in out


async def test_a_reader_that_settles_nothing_is_still_told_where_it_logs():
    """The question the follow-up exists to answer, for two thirds of readers.

    ``getTasksToExecCommand_JEDI`` selects ``finishing`` every cycle and writes
    no value, so it owns no junction; looking the reader up among the writers
    found nothing and the follow-up named the query without naming a log.  The
    file is a caller's, which is the same asymmetry a junction already states.
    """
    owner = "taskbuffer/db_proxy_mods/task_complex_module.py::getTasksToExecCommand_JEDI"
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        subjects=[
            _subject(
                selected=["finishing"],
                gates=["JEDI_AUX_Status_MinTaskID"],
                selected_by={"finishing": [owner]},
            )
        ],
        junctions=[
            _junction(
                "jediorder/ContentsFeeder.py::feed",
                Branch(outcome="finishing"),
                log_files=[KNIGHT_LOG],
                triggers=("polled",),
            ),
        ],
        log_sites=[
            LogSiteNode(
                map_id=MAP_ID,
                derived_from=VERSION,
                name=owner,
                owner=owner,
                log_files=["panda-DBProxy.log", "panda-JediDBProxy.log"],
                caller_log_files=[OTHER_LOG],
                owns_logger=False,
            )
        ],
    )
    strategy = await strategy_mod.derive(
        await _map(fragment), Symptom(subject=SUBJECT, observed="finishing", task_id="42")
    )

    assert strategy.follow_up.selected_by == [owner]
    # Its own file and the caller's.  A query writes its diagnostics into the
    # file it inherits, so leaving that out -- the rule a junction follows,
    # where the line in question is the knight's -- would say nothing about
    # the query can be seen anywhere.
    assert strategy.follow_up.reader_log_files == [
        "panda-DBProxy.log",
        "panda-JediDBProxy.log",
        OTHER_LOG,
    ]
    # A log site settles nothing and so carries no entry point.  Reading its
    # empty set as the answer would say nothing reaches the subject.
    assert strategy.follow_up.triggers == ["polled"]


async def test_a_candidate_says_who_constructed_the_worker_that_holds_it():
    """The one step forward from the anchor that reading the arm cannot take.

    ``runImpl`` works on ``self.taskList`` and nothing in its module says
    where that came from -- the knight filled it through a constructor and a
    thread, with no call in between.  Shown on the candidate rather than only
    stored, because a fact the derivation holds and never prints is one the
    investigation cannot use.
    """
    owner = "jediorder/TaskCommando.py::runImpl"
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        subjects=[_subject(selected=["passed"])],
        junctions=[
            _junction(
                owner,
                Branch(outcome="passed"),
                log_files=[KNIGHT_LOG],
                triggers=("polled",),
            ),
        ],
    )
    fragment.junctions[0].entry_points = [
        EntryPoint(
            trigger="polled",
            entry="jediorder/TaskCommando.py",
            via="start",
            arg_binding={"taskList": "taskList", "taskBufferIF": "self.taskBufferIF"},
            reached_by="dispatch",
        )
    ]
    strategy = await strategy_mod.derive(
        await _map(fragment), Symptom(subject=SUBJECT, observed="passed", task_id="42")
    )

    handed = [c for c in strategy.candidates if c.owner == owner][0].handovers
    assert [(h.via, h.reached_by) for h in handed] == [("start", "dispatch")]
    assert handed[0].fields["taskList"] == "taskList"


async def test_an_entry_that_hands_over_nothing_is_not_listed_as_one():
    """Empty means two different things and neither is 'handed nothing'.

    A call binding no keyword passed its arguments positionally; listing it
    would make that look like a path that omits them.
    """
    owner = "jediorder/ContentsFeeder.py::feed"
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        subjects=[_subject(selected=["pending"])],
        junctions=[
            _junction(owner, Branch(outcome="pending"), log_files=[KNIGHT_LOG]),
        ],
    )
    fragment.junctions[0].entry_points = [
        EntryPoint(trigger="message", entry="jedimsgprocessor/feeder.py", via="process")
    ]
    strategy = await strategy_mod.derive(
        await _map(fragment), Symptom(subject=SUBJECT, observed="pending", task_id="42")
    )

    assert [c for c in strategy.candidates if c.owner == owner][0].handovers == []


async def test_two_triggers_on_one_construction_are_one_handover():
    """``TaskCommando`` both polls and reads a command row, and built one worker.

    Two entry points, one construction.  The trigger is a fact about how
    control arrives and the handover is a fact about the data, so repeating
    the second per trigger printed the same line twice.
    """
    owner = "jediorder/TaskCommando.py::runImpl"
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        subjects=[_subject(selected=["passed"])],
        junctions=[_junction(owner, Branch(outcome="passed"), log_files=[KNIGHT_LOG])],
    )
    fragment.junctions[0].entry_points = [
        EntryPoint(
            trigger=trigger,
            entry="jediorder/TaskCommando.py",
            via="start",
            arg_binding={"taskList": "taskList"},
            reached_by="dispatch",
        )
        for trigger in ("command", "polled")
    ]
    strategy = await strategy_mod.derive(
        await _map(fragment), Symptom(subject=SUBJECT, observed="passed", task_id="42")
    )

    candidate = [c for c in strategy.candidates if c.owner == owner][0]
    assert len(candidate.handovers) == 1
    assert sorted(candidate.triggers) == ["command", "polled"]


# --------------------------------------------------------------------------- #
# the use-time walk, wired to the readings the map chose
# --------------------------------------------------------------------------- #

_WORKER = '''\
class Commando:
    def start(self):
        rows = self.taskBufferIF.getTasksToExecCommand_JEDI(vo, label)
        taskList = ListWithLock(rows)
        thr = CommandoThread(taskList)


class CommandoThread:
    def runImpl(self):
        tasks = self.taskList.get(10)
        spec.status = tasks[0]
'''


def _traceable(tmp_path, *, blob_sha: str = "", handovers=None):
    root = tmp_path / "pandajedi"
    root.mkdir(parents=True, exist_ok=True)
    (root / "commando.py").write_text(_WORKER)
    candidate = models.Candidate(
        owner="pandajedi/commando.py::runImpl",
        file="pandajedi/commando.py",
        blob_sha=blob_sha,
        gloss_key="k",
        outcome="finishing",
        handovers=handovers or [],
    )
    strategy = models.Strategy(
        symptom=Symptom(subject=SUBJECT, observed="finishing"),
        map_id=MAP_ID,
        derived_from=VERSION,
        candidates=[candidate],
        readings=[
            models.Reading(
                owner=candidate.owner,
                file=candidate.file,
                blob_sha=blob_sha,
                gloss_key="k",
                lines=[11],
            )
        ],
    )
    return strategy, {"pandajedi": root}


def test_the_walk_uses_the_handover_the_candidate_carries(tmp_path):
    # The readings group by function, so the crossing has to be looked up from
    # the candidates -- it is a property of the junction the map recorded.
    strategy, roots = _traceable(
        tmp_path,
        handovers=[
            models.Handover(
                entry="pandajedi/commando.py",
                via="start",
                reached_by="dispatch",
                fields={"taskList": "taskList"},
            )
        ],
    )

    strategy_mod.attach_traces(strategy, roots)

    steps = strategy.readings[0].trace
    assert strategy.readings[0].trace_note == ""
    assert [s.terminal for s in steps if s.terminal] == [models.STOP_UPSTREAM]


def test_without_the_handover_the_walk_stops_at_the_worker_attribute(tmp_path):
    strategy, roots = _traceable(tmp_path)

    strategy_mod.attach_traces(strategy, roots)

    assert [s.terminal for s in strategy.readings[0].trace if s.terminal] == [
        models.STOP_PARAMETER
    ]


def test_a_tree_the_map_was_not_built_from_is_not_walked(tmp_path):
    strategy, roots = _traceable(tmp_path, blob_sha="0" * 40)

    strategy_mod.attach_traces(strategy, roots)

    assert strategy.readings[0].trace == []
    assert "not the one that was mapped" in strategy.readings[0].trace_note


def test_no_source_tree_leaves_the_reading_as_coordinates(tmp_path):
    strategy, _roots = _traceable(tmp_path)

    strategy_mod.attach_traces(strategy, {})

    assert strategy.readings[0].trace == []
    assert strategy.readings[0].trace_note == ""


async def test_a_value_nothing_re_evaluates_says_where_a_row_of_that_kind_is_made():
    """The verdict told a reader to go and ask whether a command arrived, and
    could not say where a command arriving would be.

    A command arriving *is* a row appearing in a table.  The map read the verb
    that says so and pooled it into "written", so the answer existed in the
    source and nowhere in the map.  With the verbs apart it can name the
    function whose INSERT brings the row into existence.
    """
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        subjects=[
            _subject(
                name="async_results.status",
                selected=["running"],
                selected_by={"running": ["taskbuffer/async_request_module.py::recover"]},
            )
        ],
        junctions=[
            _junction(
                "taskbuffer/async_request_module.py::recover",
                Branch(outcome="running"),
                log_files=[OTHER_LOG],
                triggers=("command",),
            ),
        ],
        entities=[
            EntityNode(
                name="async_results",
                map_id=MAP_ID,
                derived_from=VERSION,
                tables=["async_results"],
                created_by=["taskbuffer/async_request_module.py::claim_async_result"],
                read_by=["taskbuffer/async_request_module.py::recover"],
            )
        ],
    )
    strategy = await strategy_mod.derive(
        await _map(fragment),
        Symptom(subject="async_results.status", observed="running", task_id="42"),
    )

    assert strategy.follow_up.created_by == [
        "taskbuffer/async_request_module.py::claim_async_result"
    ]
    assert "claim_async_result" in strategy.follow_up.question


async def test_a_row_this_map_never_creates_says_so_rather_than_naming_nobody():
    """Silence and "nothing here makes these rows" are different answers.

    Three kinds of row in the corpus are changed here and created elsewhere.
    Reporting that as an empty list reads as "the map did not look".
    """
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        subjects=[
            _subject(
                name="async_results.status",
                selected=["running"],
                selected_by={"running": ["taskbuffer/async_request_module.py::recover"]},
            )
        ],
        junctions=[
            _junction(
                "taskbuffer/async_request_module.py::recover",
                Branch(outcome="running"),
                log_files=[OTHER_LOG],
                triggers=("command",),
            ),
        ],
        entities=[
            EntityNode(
                name="async_results",
                map_id=MAP_ID,
                derived_from=VERSION,
                tables=["async_results"],
                updated_by=["taskbuffer/async_request_module.py::recover"],
                read_by=["taskbuffer/async_request_module.py::recover"],
            )
        ],
    )
    strategy = await strategy_mod.derive(
        await _map(fragment),
        Symptom(subject="async_results.status", observed="running", task_id="42"),
    )

    assert strategy.follow_up.created_by == []
    assert strategy.follow_up.creates_rows is False


async def test_a_reach_bounded_by_nothing_does_not_go_on_to_mention_those_tables():
    """Folding table names dropped twenty subjects' gates, and the sentence
    kept its trailing clause about tables it no longer named."""
    fragment = MapFragment(
        map_id=MAP_ID,
        derived_from=VERSION,
        subjects=[
            _subject(
                selected=["finishing"],
                selected_by={"finishing": ["jediorder/TaskCommando.py::run"]},
            )
        ],
        junctions=[
            _junction(
                "jediorder/TaskCommando.py::run",
                Branch(outcome="finishing"),
                log_files=[OTHER_LOG],
                triggers=("polled",),
            ),
        ],
    )
    strategy = await strategy_mod.derive(
        await _map(fragment), Symptom(subject=SUBJECT, observed="finishing", task_id="42")
    )

    assert strategy.follow_up.self_repairing is True
    assert "bounded by nothing this map can name" in strategy.follow_up.question
    assert "those tables" not in strategy.follow_up.question


async def test_the_argument_list_goes_once_the_walk_has_crossed():
    """The fields are input to the walk, not an answer for a reader.

    Measured over the vocabulary: 491 lines and 103,127 characters of them,
    41% repeated verbatim inside one report, and of the 4,620 fields printed
    the walk drew on 232.  Once it has crossed, its ``handover`` step says the
    same thing about the one name that mattered, with the expression actually
    supplied.
    """
    import click
    import click.testing

    from bamboo.scripts.derive_strategy import _report_candidates

    handover = models.Handover(
        entry="jediorder/TaskCommando.py",
        via="start",
        reached_by=models.ARRIVES_BY_DISPATCH,
        fields={"taskList": "taskList", "pid": "self.pid", "ddmIF": "self.ddmIF"},
    )
    strategy = strategy_mod.Strategy(
        symptom=Symptom(subject=SUBJECT, observed="finishing"),
        map_id=MAP_ID,
        derived_from=VERSION,
        candidates=[
            models.Candidate(
                owner="jediorder/TaskCommando.py::runImpl",
                tier=2,
                branches=[models.CandidateBranch(outcome="finishing", tier=2)],
                handovers=[handover],
            )
        ],
        readings=[
            models.Reading(
                owner="jediorder/TaskCommando.py::runImpl",
                trace=[models.TraceStep(kind=models.TRACE_HANDOVER, name="self.taskList")],
            )
        ],
    )
    runner = click.testing.CliRunner()
    command = click.Command("x", callback=lambda: _report_candidates(strategy, 10, False, False))

    assert "built by" not in runner.invoke(command).output


async def test_the_argument_list_stays_when_nothing_walked():
    """The only line that says who filled ``self.taskList``.

    No call runs between the handover and the arm, so a reader without a
    source tree cannot recover it from the arm's own module either -- and
    without a tree there is no ``handover`` step to carry it instead.
    """
    import click
    import click.testing

    from bamboo.scripts.derive_strategy import _report_candidates

    strategy = strategy_mod.Strategy(
        symptom=Symptom(subject=SUBJECT, observed="finishing"),
        map_id=MAP_ID,
        derived_from=VERSION,
        candidates=[
            models.Candidate(
                owner="jediorder/TaskCommando.py::runImpl",
                tier=2,
                branches=[models.CandidateBranch(outcome="finishing", tier=2)],
                handovers=[
                    models.Handover(
                        entry="jediorder/TaskCommando.py",
                        via="start",
                        reached_by=models.ARRIVES_BY_DISPATCH,
                        fields={"taskList": "taskList"},
                    )
                ],
            )
        ],
        readings=[models.Reading(owner="jediorder/TaskCommando.py::runImpl")],
    )
    runner = click.testing.CliRunner()
    command = click.Command("x", callback=lambda: _report_candidates(strategy, 10, False, False))

    assert "built by TaskCommando.py::start with taskList=taskList" in runner.invoke(command).output


async def test_the_handover_step_says_who_handed_it_over():
    """Computed all along and printed only for terminal steps.

    So the one line that names the caller was being dropped while the
    candidate block above repeated the whole argument list.
    """
    import click
    import click.testing

    from bamboo.scripts.derive_strategy import _report_reading

    strategy = strategy_mod.Strategy(
        symptom=Symptom(subject=SUBJECT, observed="finishing"),
        map_id=MAP_ID,
        derived_from=VERSION,
        readings=[
            models.Reading(
                owner="jediorder/TaskCommando.py::runImpl",
                trace=[
                    models.TraceStep(
                        kind=models.TRACE_HANDOVER,
                        name="self.taskList",
                        value="res[iRows:iRows + nRows]",
                        detail="handed over by start, reached by dispatch",
                    )
                ],
            )
        ],
    )
    runner = click.testing.CliRunner()
    command = click.Command("x", callback=lambda: _report_reading(strategy, {}, 10, False))

    assert "from   handed over by start, reached by dispatch" in runner.invoke(command).output


async def test_the_follow_up_claims_only_what_it_looked_at():
    """"Anything acts on this value" was wider than the thing computed.

    ``selected`` is true when a SQL WHERE picks the value out literally or an
    UPDATE acts on rows holding it, and nothing else.  Code that reads the
    value and decides is invisible to it -- ``commandToHarvester`` sets
    ``to_skip`` when an existing command holds this value, which stops the
    next write.  That report contradicted itself eight lines apart, because
    the trace printed the ``command_status in [...]`` test underneath.
    """
    strategy = await strategy_mod.derive(
        await _map(
            MapFragment(
                map_id=MAP_ID,
                derived_from=VERSION,
                subjects=[_subject(selected=[])],
                junctions=[
                    _junction(
                        "taskbuffer/db_proxy_mods/misc.py::update",
                        Branch(outcome="finishing"),
                        log_files=[KNIGHT_LOG],
                    )
                ],
            )
        ),
        Symptom(subject=SUBJECT, observed="finishing", task_id="42"),
    )

    assert strategy.follow_up.selected is False
    assert "no query in the map selects on 'finishing'" in strategy.follow_up.question
    assert "waiting will not move the row" not in strategy.follow_up.question


async def test_the_gap_counts_the_skeletons_lines_not_the_arms():
    """The count basis moved with the artefact.

    ``predicted`` produced one object per (arm, line) and the skeleton
    produces one row per line, counted once per arm it could have been
    printed alongside -- so the two are still like for like, and a row no arm
    can be printed with keeps a ``None`` arm rather than being dropped.
    """
    reading = models.Reading(
        owner="a.py::run",
        log_files=[KNIGHT_LOG],
        skeleton=[
            models.SkeletonLine(kind=models.SKELETON_BRANCH, line=1, text="if x:"),
            models.SkeletonLine(
                kind=models.SKELETON_PRINT, line=2, pattern="one", arms=[3, 4]
            ),
            models.SkeletonLine(kind=models.SKELETON_PRINT, line=5, pattern="two", arms=[]),
            models.SkeletonLine(kind=models.SKELETON_PRINT, line=6, pattern="", arms=[3]),
            models.SkeletonLine(kind=models.SKELETON_ARM, line=3, text="x = 1"),
        ],
    )

    sentences = strategy_mod.skeleton_sentences([reading])

    assert [(s.text, s.line) for s in sentences] == [("one", 3), ("one", 4), ("two", None)]
    counted = strategy_mod.discrimination(sentences)
    assert counted[strategy_mod.LINE_ONE_FUNCTION] == 2
    assert counted[strategy_mod.LINE_ONE_ARM] == 1


def test_evaluating_keeps_the_skeleton_the_walk_already_built(tmp_path):
    """The walk reads a tree; ``evaluate`` reads a file of answers.

    ``evaluate`` rebuilds the readings so that a candidate the evidence ruled
    out stops being offered, and rebuilding threw away what the walk had put
    there.  The reader then got coordinates back from a run that had already
    computed the text, and the only way to get it again was to walk the same
    tree twice.
    """
    strategy, roots = _traceable(tmp_path)
    strategy_mod.attach_traces(strategy, roots)
    assert strategy.readings[0].skeleton, "fixture walks nothing"
    before = strategy.readings[0].skeleton

    settled = strategy_mod.evaluate(
        strategy, Evidence(fetched_at="2026-09-05T00:00:00+00:00")
    )

    assert [row.model_dump() for row in settled.readings[0].skeleton] == [
        row.model_dump() for row in before
    ]
    assert settled.readings[0].trace == strategy.readings[0].trace
    assert settled.readings[0].trace_note == strategy.readings[0].trace_note


async def test_a_function_whose_lines_the_walk_found_is_not_reported_as_silent():
    """"Silent" meant the map had no pattern, which stopped meaning what it said.

    While a question was derived from the head the writers share, an empty
    ``log_pattern`` did mean nothing about this code is printed.  With that
    question gone the field is empty for most arms, and the walk has meanwhile
    read the tree and listed every line the function prints.  Reporting those
    as silent tells a reader there is nothing to grep for at the exact moment
    the map is holding a list of things to grep for.
    """
    import click
    import click.testing

    from bamboo.scripts.derive_strategy import _report_reading

    strategy = strategy_mod.Strategy(
        symptom=Symptom(subject=SUBJECT, observed="finishing"),
        map_id=MAP_ID,
        derived_from=VERSION,
        readings=[
            models.Reading(
                owner="jediorder/TaskCommando.py::runImpl",
                log_files=[KNIGHT_LOG],
                skeleton=[
                    models.SkeletonLine(
                        kind=models.SKELETON_PRINT,
                        line=4,
                        pattern=r"done\ with\ ",
                        arms=[3],
                        value=r"done\ with\ finishing",
                    ),
                    models.SkeletonLine(
                        kind=models.SKELETON_PRINT,
                        line=9,
                        pattern=r"gave\ up",
                        arms=[3, 7],
                    ),
                ],
            )
        ],
    )
    runner = click.testing.CliRunner()
    output = runner.invoke(
        click.Command("x", callback=lambda: _report_reading(strategy, {}, 10, False))
    ).output

    assert "silent" not in output
    # What is handed over, and what the reader is left to choose.
    assert "2 line(s)" in output
    assert "1 name(s) one arm" in output
    assert "1 carr" in output


async def test_a_function_that_prints_nothing_is_still_reported_as_silent():
    """The word has to keep meaning something, or the previous test is a way of
    never saying it."""
    import click
    import click.testing

    from bamboo.scripts.derive_strategy import _report_reading

    strategy = strategy_mod.Strategy(
        symptom=Symptom(subject=SUBJECT, observed="finishing"),
        map_id=MAP_ID,
        derived_from=VERSION,
        readings=[models.Reading(owner="jediorder/TaskCommando.py::runImpl")],
    )
    runner = click.testing.CliRunner()
    output = runner.invoke(
        click.Command("x", callback=lambda: _report_reading(strategy, {}, 10, False))
    ).output

    assert "silent" in output
