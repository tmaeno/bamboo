"""Tests for `deploy/batch/entrypoint.sh`'s argument dispatch.

The script boots Neo4j + Qdrant + Ollama, so its workloads are not testable here. Its
*dispatch* is, and completely: every path that does not run a workload — usage, the
migration errors for the removed `run`/`batch`, unknown tokens, and the argument guards —
exits before `do_setup` allocates or starts anything. Reaching them is side-effect free
(the code executed is env exports, `log`, and the `/app/.env` probe), so this covers the
wiring most likely to rot, plus a `bash -n` parse of the whole file.

Each assertion also checks that no service was started, which is what makes these cheap:
a regression that boots the stack before failing would blow the timeout instead of
silently making the suite slow.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ENTRYPOINT = Path(__file__).resolve().parents[1] / "deploy" / "batch" / "entrypoint.sh"

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None, reason="bash not available"
)

# Markers do_setup emits before it can block on anything slow.
_BOOT_MARKERS = ("starting neo4j", "ports:", "restoring Neo4j dump")


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [shutil.which("bash") or "bash", str(ENTRYPOINT), *args],
        capture_output=True,
        text=True,
        timeout=60,
    )


def _assert_nothing_booted(proc: subprocess.CompletedProcess[str]) -> None:
    for marker in _BOOT_MARKERS:
        assert marker not in proc.stderr, f"do_setup ran: {marker!r} in stderr"


def test_script_parses() -> None:
    """A syntax error would only surface when a batch job runs; catch it here."""
    proc = subprocess.run(
        [shutil.which("bash") or "bash", "-n", str(ENTRYPOINT)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr


# --- no argument: print usage, do not guess a workload -------------------------------


def test_no_subcommand_prints_usage_and_fails() -> None:
    """`batch-analyze` is not the only batch command, so nothing is implied."""
    proc = _run()
    assert proc.returncode != 0
    assert "no subcommand given" in proc.stderr
    assert "Usage:" in proc.stderr
    assert proc.stdout == ""  # usage goes to stderr when it accompanies an error
    _assert_nothing_booted(proc)


@pytest.mark.parametrize("flag", ["help", "--help", "-h"])
def test_help_prints_usage_to_stdout(flag: str) -> None:
    proc = _run(flag)
    assert proc.returncode == 0
    assert "Usage:" in proc.stdout
    assert "batch-analyze" in proc.stdout
    assert "exec <cmd…>" in proc.stdout
    _assert_nothing_booted(proc)


# --- removed / unknown tokens --------------------------------------------------------


@pytest.mark.parametrize("removed", ["run", "batch"])
def test_removed_subcommands_point_at_batch_analyze(removed: str) -> None:
    """Both used to mean batch-analyze; an old wrapper must fail loudly, not silently."""
    proc = _run(removed)
    assert proc.returncode != 0
    assert f"'{removed}' no longer exists" in proc.stderr
    assert "batch-analyze" in proc.stderr
    _assert_nothing_booted(proc)


def test_unknown_subcommand_points_at_exec() -> None:
    proc = _run("frobnicate")
    assert proc.returncode != 0
    assert "unknown subcommand 'frobnicate'" in proc.stderr
    assert "exec frobnicate" in proc.stderr
    _assert_nothing_booted(proc)


def test_options_are_no_longer_forwarded_to_batch_analyze() -> None:
    """`--task-id 123` used to imply a batch-analyze run; now it must name one."""
    proc = _run("--task-id", "123")
    assert proc.returncode != 0
    assert "unknown option '--task-id'" in proc.stderr
    assert "batch-analyze --task-id" in proc.stderr
    _assert_nothing_booted(proc)


# --- argument guards ----------------------------------------------------------------


def test_exec_without_command_fails_before_boot() -> None:
    proc = _run("exec")
    assert proc.returncode != 0
    assert "exec needs a command" in proc.stderr
    _assert_nothing_booted(proc)


def test_exec_rejects_a_nonexistent_command_before_boot() -> None:
    """Resolving argv up front keeps a typo from costing a full stack boot."""
    proc = _run("exec", "no_such_binary_xyz")
    assert proc.returncode != 0
    assert "not an executable: 'no_such_binary_xyz'" in proc.stderr
    _assert_nothing_booted(proc)


def test_shell_rejects_arguments_and_points_at_exec() -> None:
    """`shell` used to swallow args, silently ignoring the command it was handed."""
    proc = _run("shell", "bamboo", "verify")
    assert proc.returncode != 0
    assert "'shell' takes no arguments" in proc.stderr
    assert "exec bamboo verify" in proc.stderr
    _assert_nothing_booted(proc)
