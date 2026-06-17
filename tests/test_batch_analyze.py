"""Tests for the `bamboo batch-analyze` command.

The command's value is amortizing startup across many tasks, so the key
behaviours to lock in are: deps are built once, every task produces one result
file, a failing task is isolated (sidecar + non-zero exit) without aborting the
batch, and input validation rejects an empty invocation. Live services (Neo4j /
Qdrant / LLM) are mocked — functional analysis is covered elsewhere.
"""

from __future__ import annotations

import json
import logging
from unittest import mock

import pytest
from click.testing import CliRunner

from bamboo.scripts.batch_analyze import main


@pytest.fixture(autouse=True)
def _isolate_root_logging():
    """Restore root-logger handlers/level after each test.

    Invoking the command runs ``setup_logging()``, which adds a handler to the
    root logger; without this, that handler leaks into later tests that assert on
    global logging state (e.g. test_narration).
    """
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    try:
        yield
    finally:
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)


class _FakeDB:
    """Async DB client stub: connect/close are no-ops, counted for assertions."""

    def __init__(self) -> None:
        self.connects = 0
        self.closes = 0

    async def connect(self) -> None:
        self.connects += 1

    async def close(self) -> None:
        self.closes += 1


class _FakeDeps:
    def __init__(self) -> None:
        self.graph_db = _FakeDB()
        self.vector_db = _FakeDB()
        self.mcp_client = None
        self.reasoning_navigator = None


class _FakeResult:
    def __init__(self, task_id: int) -> None:
        self.task_id = task_id
        self.root_cause = "disk full"
        self.confidence = 0.9
        self.unmatched_symptoms: list[str] = []
        self.email_content: str | None = None

    def model_dump_json(self, indent: int = 2) -> str:
        return json.dumps({"task_id": self.task_id, "root_cause": self.root_cause}, indent=indent)


def _write_task(d, name: str, task_id: int) -> None:
    (d / name).write_text(json.dumps({"jediTaskID": task_id, "errorDialog": "boom"}))


def test_help_lists_options():
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0
    for option in ("--input-dir", "--task-id", "--output-dir", "--concurrency", "--drafts-dir"):
        assert option in result.output


def test_requires_input_dir_or_task_id(tmp_path):
    """Neither --input-dir nor --task-id → friendly usage error, exit 2."""
    result = CliRunner().invoke(main, ["--output-dir", str(tmp_path / "out")])
    assert result.exit_code == 2
    assert "--input-dir" in result.output and "--task-id" in result.output


def test_batch_writes_one_result_per_task_and_isolates_failures(tmp_path):
    """Deps built once; good tasks get result files; a failing task is isolated."""
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    out_dir = tmp_path / "out"
    _write_task(in_dir, "a.json", 111)  # ok
    _write_task(in_dir, "b.json", 222)  # raises
    _write_task(in_dir, "c.json", 333)  # ok

    fake_deps = _FakeDeps()

    async def fake_analyze_one(deps, task_dict, external_dict=None, **kwargs):
        assert deps is fake_deps  # the single, shared deps bundle
        tid = task_dict["jediTaskID"]
        if tid == 222:
            raise RuntimeError("kaboom")
        return _FakeResult(tid), {"hints": []}, "EMAIL BODY", None

    with mock.patch("bamboo.agents.helpers.deps.build_deps", return_value=fake_deps) as build, \
         mock.patch("bamboo.scripts.analyze_task.analyze_one", side_effect=fake_analyze_one):
        result = CliRunner().invoke(
            main, ["--input-dir", str(in_dir), "--output-dir", str(out_dir)]
        )

    # One failed task → non-zero exit, but the batch completed the others.
    assert result.exit_code == 1, result.output

    # Deps built exactly once and connected/closed once (amortization invariant).
    build.assert_called_once()
    assert fake_deps.graph_db.connects == 1 and fake_deps.graph_db.closes == 1
    assert fake_deps.vector_db.connects == 1 and fake_deps.vector_db.closes == 1

    # Successful tasks named by jediTaskID; failed task gets an error sidecar.
    assert (out_dir / "111.json").exists()
    assert (out_dir / "333.json").exists()
    assert not (out_dir / "222.json").exists()
    assert (out_dir / "b.error.json").exists()
    err = json.loads((out_dir / "b.error.json").read_text())
    assert "kaboom" in err["error"]


def test_all_success_exits_zero(tmp_path):
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    out_dir = tmp_path / "out"
    _write_task(in_dir, "only.json", 999)

    async def fake_analyze_one(deps, task_dict, external_dict=None, **kwargs):
        return _FakeResult(task_dict["jediTaskID"]), {"hints": []}, "EMAIL", None

    with mock.patch("bamboo.agents.helpers.deps.build_deps", return_value=_FakeDeps()), \
         mock.patch("bamboo.scripts.analyze_task.analyze_one", side_effect=fake_analyze_one):
        result = CliRunner().invoke(
            main, ["--input-dir", str(in_dir), "--output-dir", str(out_dir)]
        )

    assert result.exit_code == 0, result.output
    assert (out_dir / "999.json").exists()
