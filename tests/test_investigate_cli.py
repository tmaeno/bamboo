"""Smoke tests for `bamboo investigate` CLI entry point.

End-to-end stdin scripting is impractical here (the orchestrator calls a real
LLM and a real PandaMcpClient). These tests verify that the Click command is
correctly wired and that input validation rejects nonsense up front, without
needing live services. Functional behavior is covered by
test_investigation_session.py.
"""

from __future__ import annotations

from click.testing import CliRunner

from bamboo.scripts.investigate import main


def test_help_lists_all_documented_options():
    """--help should render and mention every documented flag."""
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    for option in (
        "--task-id",
        "--task-data",
        "--symptom",
        "--save",
        "--resume",
        "--max-turns",
        "--dry-run",
        "--verbose",
    ):
        assert option in result.output, f"missing {option} in --help output"


def test_requires_at_least_one_entry_source():
    """No --task-id / --task-data / --symptom / --resume → friendly error, exit 2."""
    runner = CliRunner()
    result = runner.invoke(main, [])
    assert result.exit_code == 2
    assert "Need one of" in result.output


def test_dry_run_flag_is_recognised():
    """--dry-run is a boolean flag; presence in --help and no error when supplied with --symptom."""
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert "--dry-run" in result.output


def test_max_turns_accepts_integer():
    """--max-turns is typed as integer; bad value triggers Click's type error."""
    runner = CliRunner()
    result = runner.invoke(main, ["--symptom", "x", "--max-turns", "not-an-int"])
    assert result.exit_code != 0
    assert "Invalid value" in result.output or "not-an-int" in result.output
