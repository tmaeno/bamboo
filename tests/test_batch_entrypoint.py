"""Tests for `deploy/batch/entrypoint.sh`'s argument dispatch and sandbox resolution.

The script boots Neo4j + Qdrant + Ollama, so its workloads are not testable here. Two
things around them are.

*Dispatch*, completely: every path that does not run a workload — usage, the migration
errors for the removed `run`/`batch`, unknown tokens, and the argument guards — exits
before `do_setup` allocates or starts anything. Reaching them is side-effect free (the
code executed is env exports, `log`, and the `/app/.env` probe), so this covers the wiring
most likely to rot, plus a `bash -n` parse of the whole file.

*Sandbox resolution* (`BAMBOO_SANDBOX`), which by design runs at the very top of
`do_setup`. Those tests therefore do enter `do_setup`, but only its first few lines: they
give it a scratch dir of their own and a sandbox whose `models/` has no
`bamboo-model.json`, so the existing `LLM_MODEL` guard stops each run a few lines later —
before the KB restore, and long before any service starts. The tarballs hold empty
directories, since resolve_sandbox only reads directory names.

Every assertion also checks that no service was started, which is what makes these cheap:
a regression that boots the stack before failing would blow the timeout instead of
silently making the suite slow.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

ENTRYPOINT = Path(__file__).resolve().parents[1] / "deploy" / "batch" / "entrypoint.sh"

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None, reason="bash not available"
)

# Markers do_setup emits before it can block on anything slow.
_BOOT_MARKERS = ("starting neo4j", "ports:", "restoring Neo4j dump")

# The subset the sandbox tests assert on: they intentionally reach do_setup's first lines,
# so "ports:" is expected there, but nothing may touch the KB or start a service.
_SERVICE_MARKERS = (
    "restoring Neo4j dump",
    "starting neo4j",
    "starting qdrant",
    "starting ollama",
)


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


def _assert_no_service_started(proc: subprocess.CompletedProcess[str]) -> None:
    for marker in _SERVICE_MARKERS:
        assert marker not in proc.stderr, (
            f"got past sandbox resolution: {marker!r} in stderr"
        )


# The sandbox tests deliberately end inside do_setup, either at the `${LLM_MODEL:?…}` guard
# or — with no `python` on PATH for the free-port helper — a couple of lines earlier. Neither
# is an explicit `exit`, and bash 3.2 (macOS /bin/bash) does not carry the status of a `set -u`
# / `${var:?}` abort through an EXIT trap: $? reads 0 there and the process exits 0, where
# bash 5.x (what the image ships) exits 1. So those tests assert on stderr and never on
# returncode; `die`-based rejections propagate everywhere and do assert it.
_HAS_PYTHON = shutil.which("python") is not None


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


# --- BAMBOO_SANDBOX: the staged inputs as one archive --------------------------------


def _make_sandbox(tmp_path: Path, *components: str, wrapper: str | None = None) -> Path:
    """Tar up `components` as empty dirs, the way the guide's recipe does it.

    `tar czf … -C sandbox .` is what the docs tell users to run, so the archive members
    are `./models` &c — mirroring that here is the point, since a `./` prefix is exactly
    the sort of thing a path-matching bug would trip over.
    """
    stage = tmp_path / "stage"
    (stage / wrapper if wrapper else stage).mkdir(parents=True, exist_ok=True)
    for component in components:
        ((stage / wrapper if wrapper else stage) / component).mkdir(parents=True)
    tgz = tmp_path / "sandbox.tgz"
    with tarfile.open(tgz, "w:gz") as tf:
        tf.add(stage, arcname=".")
    return tgz


def _run_sandbox(
    sandbox: Path, work: Path, *args: str, **env: str
) -> subprocess.CompletedProcess[str]:
    """Run the entrypoint against `sandbox` in a deliberately minimal environment.

    Only PATH is inherited: a developer's own OLLAMA_MODELS / HF_HOME / LLM_MODEL would
    otherwise decide what the "keeping …" assertions below are actually asserting. `setup`
    is the subcommand used because it goes straight to do_setup with no argv of its own to
    resolve, so these tests don't depend on `bamboo` being on PATH.
    """
    return subprocess.run(
        [shutil.which("bash") or "bash", str(ENTRYPOINT), *(args or ("setup",))],
        capture_output=True,
        text=True,
        timeout=60,
        env={
            "PATH": os.environ.get("PATH", ""),
            "BAMBOO_SANDBOX": str(sandbox),
            "BAMBOO_WORK": str(work),
            **env,
        },
    )


def test_sandbox_rejects_a_path_that_is_neither_file_nor_directory(
    tmp_path: Path,
) -> None:
    """A typo'd or undelivered sandbox must fail here, not as a confusing miss later."""
    proc = _run_sandbox(tmp_path / "not-delivered.tgz", tmp_path)
    assert proc.returncode != 0
    assert "is neither a file nor a directory" in proc.stderr
    _assert_nothing_booted(proc)


def test_sandbox_without_any_component_is_rejected(tmp_path: Path) -> None:
    """Wrong archive: report it against its own top level, and say nothing about mounts.

    The "keeping …" lines are checked to be absent on purpose — printing them before
    failing would read as if the sandbox had been accepted.
    """
    proc = _run_sandbox(_make_sandbox(tmp_path, "junk", "other"), tmp_path)
    assert proc.returncode != 0
    assert "holds none of models kb embeddings in at its top level" in proc.stderr
    assert "junk" in proc.stderr and "other" in proc.stderr
    assert "keeping" not in proc.stderr
    _assert_no_service_started(proc)


def test_sandbox_maps_every_component(tmp_path: Path) -> None:
    proc = _run_sandbox(
        _make_sandbox(tmp_path, "models", "kb", "embeddings", "in"), tmp_path
    )
    for component in ("models", "kb", "embeddings", "in"):
        assert f"-> {tmp_path}" in proc.stderr
        assert f"/sandbox/{component}" in proc.stderr, f"{component} not re-pointed"
    assert "keeping" not in proc.stderr
    _assert_no_service_started(proc)


@pytest.mark.skipif(
    not _HAS_PYTHON, reason="entrypoint.sh needs `python` to get this far"
)
def test_the_sandbox_models_path_feeds_the_model_derivation(tmp_path: Path) -> None:
    """Re-pointing OLLAMA_MODELS must actually redirect what reads it.

    The `${LLM_MODEL:?…}` guard names the manifest it looked for, so its message is the
    cheapest proof that the derivation downstream of resolve_sandbox followed the sandbox
    rather than /models.
    """
    proc = _run_sandbox(_make_sandbox(tmp_path, "models", "kb"), tmp_path)
    assert "/sandbox/models/bamboo-model.json" in proc.stderr
    _assert_no_service_started(proc)


def test_partial_sandbox_keeps_the_mounts_it_does_not_supply(tmp_path: Path) -> None:
    """A kb-only archive must compose with mounted /models + /embeddings, not replace them."""
    proc = _run_sandbox(_make_sandbox(tmp_path, "kb"), tmp_path)
    assert f"kb         -> {tmp_path}" in proc.stderr
    assert "no models/ — keeping OLLAMA_MODELS=/models" in proc.stderr
    assert "no embeddings/ — keeping HF_HOME=/embeddings" in proc.stderr
    assert "no in/ — keeping IN_DIR=/in" in proc.stderr
    _assert_no_service_started(proc)


def test_sandbox_tolerates_one_wrapper_directory(tmp_path: Path) -> None:
    """`tar czf sandbox.tgz sandbox/` is the other way people build these."""
    proc = _run_sandbox(
        _make_sandbox(tmp_path, "models", "kb", wrapper="sandbox"), tmp_path
    )
    assert "descending into" in proc.stderr
    assert "/sandbox/sandbox/models" in proc.stderr
    assert "/sandbox/sandbox/kb" in proc.stderr
    _assert_no_service_started(proc)


def test_sandbox_directory_is_used_without_extraction(tmp_path: Path) -> None:
    """An already-extracted sandbox costs no second copy of a multi-GB tree."""
    extracted = tmp_path / "extracted"
    for component in ("models", "kb", "embeddings"):
        (extracted / component).mkdir(parents=True)
    proc = _run_sandbox(extracted, tmp_path)
    assert f"already-extracted directory {extracted}" in proc.stderr
    assert "extracting" not in proc.stderr
    assert f"models     -> {extracted}/models" in proc.stderr
    _assert_no_service_started(proc)


def test_batch_analyze_builds_its_argv_after_the_boot() -> None:
    """A sandbox `in/` only reaches `--input-dir` if IN_DIR is expanded *after* do_setup.

    Asserting on source order rather than behaviour is deliberate: proving it functionally
    needs a booted Neo4j + Qdrant + Ollama, and the failure mode — reverting to
    `cmd_exec bamboo batch-analyze --input-dir "${IN_DIR}" …`, whose argv is expanded
    before the boot — is silent, a job that reads an empty /in and reports success. If a
    refactor moves these lines, that is exactly when this should be looked at again.
    """
    body = (
        ENTRYPOINT.read_text().partition("cmd_batch_analyze() {")[2].partition("\n}")[0]
    )
    assert "--input-dir" in body, "cmd_batch_analyze no longer wires --input-dir"
    assert body.index("_boot") < body.index("--input-dir"), (
        "IN_DIR is expanded before do_setup runs — a sandbox in/ would be ignored"
    )


def test_without_the_env_var_no_sandbox_logic_runs(tmp_path: Path) -> None:
    """The three-mount path must be untouched: opt-in means silent when unset."""
    proc = subprocess.run(
        [shutil.which("bash") or "bash", str(ENTRYPOINT), "setup"],
        capture_output=True,
        text=True,
        timeout=60,
        env={"PATH": os.environ.get("PATH", ""), "BAMBOO_WORK": str(tmp_path)},
    )
    assert "sandbox" not in proc.stderr
    _assert_no_service_started(proc)


def test_do_setup_does_not_create_the_output_dir(tmp_path: Path) -> None:
    """`bamboo batch-analyze` creates its own --output-dir, and it is the only writer.

    Creating it in do_setup instead made every subcommand — `exec bamboo verify`, `shell`,
    `setup` — depend on a writable /out they never touch, which aborts the whole boot on
    the read-only rootfs of a rootless Apptainer run.
    """
    out = tmp_path / "never-created"
    proc = _run_sandbox(
        _make_sandbox(tmp_path, "models", "kb"), tmp_path, BAMBOO_OUT=str(out)
    )
    assert not out.exists(), "do_setup created OUT_DIR"
    _assert_no_service_started(proc)
