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


# --- gpu-check / LD_LIBRARY_PATH -----------------------------------------------------
#
# `gpu-check` boots Ollama alone, and its first act is to print the library search path it
# will hand `ollama serve`. That runs before the `${LLM_MODEL:?…}` guard, so pointing
# OLLAMA_MODELS at a directory with no manifest reaches the interesting output and then
# stops — no ollama binary needed, nothing started.
#
# What is being pinned down is the ordering rule: Ollama's own lib dirs must come FIRST,
# because its cuda_v<N>/libcudart is found via RUNPATH $ORIGIN and the linker searches
# RUNPATH *after* LD_LIBRARY_PATH — so a site CUDA dir on the path silently wins otherwise
# (ALRB's /alrb/cuda/lib64 did, which is what this whole path exists for), and Ollama then
# falls back to the CPU without failing anything the readiness probe checks.


def _fake_ollama_lib(tmp_path: Path, *cuda_dirs: str) -> Path:
    """A stand-in for /usr/local/lib/ollama, which tests cannot create."""
    root = tmp_path / "ollama-lib"
    root.mkdir(parents=True, exist_ok=True)
    for name in cuda_dirs:
        (root / name).mkdir()
    return root


def _run_gpu_check(tmp_path: Path, **env: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [shutil.which("bash") or "bash", str(ENTRYPOINT), "gpu-check"],
        capture_output=True,
        text=True,
        timeout=60,
        env={
            "PATH": os.environ.get("PATH", ""),
            "BAMBOO_WORK": str(tmp_path),
            "OLLAMA_MODELS": str(tmp_path / "no-such-models"),
            **env,
        },
    )


_LD_MARKER = "LD_LIBRARY_PATH (for ollama):"


def _ollama_ld_path(proc: subprocess.CompletedProcess[str]) -> str:
    for line in proc.stderr.splitlines():
        if _LD_MARKER in line:
            return line.partition(_LD_MARKER)[2].strip()
    raise AssertionError(f"no {_LD_MARKER!r} line in:\n{proc.stderr}")


def test_gpu_check_is_offered_and_takes_no_arguments() -> None:
    assert "gpu-check" in _run("help").stdout
    proc = _run("gpu-check", "--verbose")
    assert proc.returncode != 0
    assert "'gpu-check' takes no arguments" in proc.stderr
    _assert_nothing_booted(proc)


@pytest.mark.skipif(
    not _HAS_PYTHON, reason="entrypoint.sh needs `python` to get this far"
)
def test_ollama_lib_dirs_are_prepended_and_the_caller_path_is_preserved(
    tmp_path: Path,
) -> None:
    """Ollama's dirs first, every caller entry kept, original order intact."""
    lib = _fake_ollama_lib(tmp_path, "cuda_v12", "cuda_v13")
    host, driver = tmp_path / "host-cuda", tmp_path / "driver"
    host.mkdir()
    driver.mkdir()
    proc = _run_gpu_check(
        tmp_path,
        OLLAMA_LIB_ROOT=str(lib),
        LD_LIBRARY_PATH=f"{host}:{driver}",
    )
    assert _ollama_ld_path(proc).split(":") == [
        str(lib),
        str(lib / "cuda_v12"),
        str(lib / "cuda_v13"),
        str(host),
        str(driver),
    ]
    _assert_no_service_started(proc)


@pytest.mark.skipif(
    not _HAS_PYTHON, reason="entrypoint.sh needs `python` to get this far"
)
def test_a_host_cuda_runtime_on_the_path_is_reported(tmp_path: Path) -> None:
    """The condition that caused the silent CPU fallback must be named in the log.

    Only reported, never removed: which of a site's LD_LIBRARY_PATH entries are safe to
    drop is not ours to decide — the prepend above already wins.
    """
    lib = _fake_ollama_lib(tmp_path, "cuda_v13")
    host = tmp_path / "alrb-cuda"
    host.mkdir()
    (host / "libcudart.so.13").touch()
    plain = tmp_path / "plain"
    plain.mkdir()
    proc = _run_gpu_check(
        tmp_path, OLLAMA_LIB_ROOT=str(lib), LD_LIBRARY_PATH=f"{host}:{plain}"
    )
    assert f"host CUDA runtime on LD_LIBRARY_PATH ({host})" in proc.stderr
    assert (
        str(plain)
        not in proc.stderr.partition("host CUDA runtime")[2].partition("\n")[0]
    )
    assert str(host) in _ollama_ld_path(proc), (
        "a host entry was dropped, not overridden"
    )
    _assert_no_service_started(proc)


@pytest.mark.skipif(
    not _HAS_PYTHON, reason="entrypoint.sh needs `python` to get this far"
)
def test_an_unset_caller_path_yields_no_empty_element(tmp_path: Path) -> None:
    """An empty element in LD_LIBRARY_PATH means "the current directory".

    So a missing lib/ollama — any non-image run — must not leave a stray `:` that puts $PWD
    at the front of Ollama's library search path.
    """
    proc = _run_gpu_check(tmp_path, OLLAMA_LIB_ROOT=str(tmp_path / "absent"))
    assert _ollama_ld_path(proc) == "", "expected an empty path, not a bare separator"
    proc = _run_gpu_check(tmp_path, OLLAMA_LIB_ROOT=str(_fake_ollama_lib(tmp_path)))
    assert "" not in _ollama_ld_path(proc).split(":")
    _assert_no_service_started(proc)


# --- BAMBOO_KEEP_WORK ----------------------------------------------------------------


def _state_file(work: Path, run_dir: Path) -> Path:
    """The state env-file `teardown` falls back to when WORK isn't already set."""
    state = work / "bamboo-batch.env"
    state.write_text(f"export BAMBOO_RUN_DIR={run_dir}\nBAMBOO_SERVICE_PIDS=\n")
    return state


@pytest.mark.parametrize("keep,survives", [("1", True), ("", False)])
def test_teardown_honours_keep_work(tmp_path: Path, keep: str, survives: bool) -> None:
    """The service logs live in the scratch dir teardown deletes on every exit path.

    Also covers the state-file key: teardown reads the scratch dir back as BAMBOO_RUN_DIR
    (exported, so a debugging shell can find the logs), not the old internal-only name.
    """
    run_dir = tmp_path / "bamboo.abc123"
    (run_dir / "ollama").mkdir(parents=True)
    (run_dir / "ollama" / "ollama.log").write_text("gpu discovery went here\n")
    _state_file(tmp_path, run_dir)
    proc = subprocess.run(
        [shutil.which("bash") or "bash", str(ENTRYPOINT), "teardown"],
        capture_output=True,
        text=True,
        timeout=60,
        env={
            "PATH": os.environ.get("PATH", ""),
            "BAMBOO_WORK": str(tmp_path),
            **({"BAMBOO_KEEP_WORK": keep} if keep else {}),
        },
    )
    assert proc.returncode == 0, proc.stderr
    assert run_dir.exists() is survives, proc.stderr
    if survives:
        assert "keeping scratch dir" in proc.stderr


# --- report_accelerator: gpu vs cpu, and BAMBOO_REQUIRE_GPU --------------------------
#
# This is the verdict a GPU job is judged on, so its four outcomes are pinned here rather
# than left to a node test. It cannot go through `gpu-check`: launch_ollama uses `setsid`,
# which macOS does not ship. So the script is *sourced* (`help` prints usage and returns,
# starting nothing) and report_accelerator called directly against a stub `curl`.


_FAKE_CURL = """#!/usr/bin/env bash
# /api/generate succeeds; /api/ps writes $FAKE_PS to the -o path.
out=""; url=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -o) out="$2"; shift 2 ;;
    http*) url="$1"; shift ;;
    -d|-H) shift 2 ;;
    *) shift ;;
  esac
done
case "$url" in
  */api/ps) printf '%s' "${FAKE_PS}" > "$out" ;;
esac
exit 0
"""

_GPU_FULL = (
    '{"models":[{"model":"qwen3.6:latest","size":19783483392,"size_vram":19783483392}]}'
)
_GPU_PARTIAL = (
    '{"models":[{"model":"qwen3.6:latest","size":19783483392,"size_vram":12884901888}]}'
)
_CPU_ONLY = '{"models":[{"model":"qwen3.6:latest","size":19783483392,"size_vram":0}]}'
_NO_ENTRY = '{"models":[]}'


def _report_accelerator(
    tmp_path: Path, api_ps: str, require_gpu: str = ""
) -> subprocess.CompletedProcess[str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    curl = bin_dir / "curl"
    curl.write_text(_FAKE_CURL)
    curl.chmod(0o755)
    work = tmp_path / "work"
    (work / "ollama").mkdir(parents=True)
    (work / "ollama" / "ollama.log").write_text("no compatible GPUs were discovered\n")
    return subprocess.run(
        [
            shutil.which("bash") or "bash",
            "-c",
            f'source "{ENTRYPOINT}" help >/dev/null\n'
            f'WORK="{work}"; BAMBOO_OLLAMA_LOG="$WORK/ollama/ollama.log"\n'
            "OLLAMA_BASE_URL=http://127.0.0.1:1; LLM_MODEL=qwen3.6\n"
            'report_accelerator; echo "FN_RC=$?"\n',
        ],
        capture_output=True,
        text=True,
        timeout=60,
        env={
            "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}",
            "FAKE_PS": api_ps,
            **({"BAMBOO_REQUIRE_GPU": require_gpu} if require_gpu else {}),
        },
    )


@pytest.mark.skipif(
    not _HAS_PYTHON, reason="report_accelerator parses /api/ps in python"
)
@pytest.mark.parametrize(
    "api_ps,expected",
    [
        (_GPU_FULL, "accelerator: gpu — qwen3.6 fully offloaded (18.4 GiB in VRAM)"),
        (_GPU_PARTIAL, "accelerator: gpu (PARTIAL) — 12.0 of 18.4 GiB in VRAM"),
        (_CPU_ONLY, "accelerator: cpu — qwen3.6 is entirely in host RAM"),
        (_NO_ENTRY, "accelerator: UNKNOWN"),
    ],
    ids=["gpu-full", "gpu-partial", "cpu-only", "no-entry"],
)
def test_report_accelerator_names_the_processor(
    tmp_path: Path, api_ps: str, expected: str
) -> None:
    """size_vram vs size is the whole verdict; every outcome must say so out loud."""
    proc = _report_accelerator(tmp_path, api_ps)
    assert expected in proc.stderr, proc.stderr
    assert "FN_RC=0" in proc.stdout, (
        "a non-required CPU/UNKNOWN result must not fail the boot"
    )


@pytest.mark.skipif(
    not _HAS_PYTHON, reason="report_accelerator parses /api/ps in python"
)
@pytest.mark.parametrize("api_ps", [_CPU_ONLY, _NO_ENTRY], ids=["cpu-only", "no-entry"])
def test_require_gpu_fails_the_boot_when_the_gpu_is_not_confirmed(
    tmp_path: Path, api_ps: str
) -> None:
    """What submit.sh sets on a GPU queue: unconfirmed must not pass for confirmed.

    A CPU fallback is silent otherwise — Ollama serves /api/tags and generates fine, just an
    order of magnitude slower — so the job's only symptom is finishing late.
    """
    proc = _report_accelerator(tmp_path, api_ps, require_gpu="1")
    assert "BAMBOO_REQUIRE_GPU=1" in proc.stderr
    assert "FN_RC=" not in proc.stdout, "die() must abort, not return"


@pytest.mark.skipif(
    not _HAS_PYTHON, reason="report_accelerator parses /api/ps in python"
)
def test_report_accelerator_reads_api_ps_from_a_file_not_a_pipe(tmp_path: Path) -> None:
    """`python - <<'PY'` already uses stdin for the script.

    Piping the JSON in as well hands json.load an exhausted stream, which reports UNKNOWN
    for every run — a GPU node included — and BAMBOO_REQUIRE_GPU then fails a healthy job.
    """
    proc = _report_accelerator(tmp_path, _GPU_FULL)
    assert "UNKNOWN" not in proc.stderr, proc.stderr
    assert (tmp_path / "work" / "ollama" / "api-ps.json").exists(), (
        "the raw /api/ps reply should be left in scratch for debugging"
    )


@pytest.mark.skipif(
    not _HAS_PYTHON, reason="report_accelerator parses /api/ps in python"
)
@pytest.mark.parametrize(
    "require,api_ps,should_die",
    [
        ("full", _GPU_FULL, False),
        ("full", _GPU_PARTIAL, True),
        ("1", _GPU_PARTIAL, False),  # back-compat: 1 has always meant "any offload"
        ("1", _GPU_FULL, False),
        ("yes", _GPU_FULL, True),  # unrecognised must not silently disable the check
    ],
    ids=[
        "full-ok",
        "full-rejects-partial",
        "1-allows-partial",
        "1-ok",
        "typo-is-an-error",
    ],
)
def test_require_gpu_levels(
    tmp_path: Path, require: str, api_ps: str, should_die: bool
) -> None:
    """`full` exists because gpu-partial is the *unstable* state.

    A partial offload means VRAM was already too tight to hold the model, so the next load
    can land at zero layers and the job quietly finishes on the CPU — which is how this came
    up. A node expected to fit the model should refuse to start when it doesn't.
    """
    proc = _report_accelerator(tmp_path, api_ps, require_gpu=require)
    if should_die:
        assert "FN_RC=" not in proc.stdout, f"expected die(), got:\n{proc.stderr}"
        assert "ERROR" in proc.stderr
    else:
        assert "FN_RC=0" in proc.stdout, proc.stderr


# --- launch_ollama: the server env ---------------------------------------------------


def test_launch_ollama_pins_keep_alive_and_the_library_path(tmp_path: Path) -> None:
    """OLLAMA_KEEP_ALIVE must reach `ollama serve`, not just the preload request.

    langchain_ollama sends `keep_alive: null` on every call, which Ollama reads as
    "unspecified" and answers with its 5-minute *server* default. A keep_alive passed only on
    the preload therefore governs nothing once bamboo starts talking: the model unloads and
    the reload re-decides GPU offload from whatever VRAM is free then. That is exactly the bug
    this asserts against — and it also gives launch_ollama its first direct test, since
    `setsid` does not exist on macOS and has to be stubbed.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "setsid").write_text('#!/usr/bin/env bash\nexec "$@"\n')
    dump = tmp_path / "ollama-env.txt"
    (bin_dir / "ollama").write_text(f'#!/usr/bin/env bash\nenv > "{dump}"\n')
    for stub in ("setsid", "ollama"):
        (bin_dir / stub).chmod(0o755)

    lib = tmp_path / "ollama-lib"
    (lib / "cuda_v13").mkdir(parents=True)
    work = tmp_path / "work"
    (work / "ollama").mkdir(parents=True)
    host_cuda = tmp_path / "host-cuda"
    host_cuda.mkdir()

    proc = subprocess.run(
        [
            shutil.which("bash") or "bash",
            "-c",
            f'source "{ENTRYPOINT}" help >/dev/null\n'
            f'WORK="{work}"; OLLAMA_PORT=12345\n'
            f'BAMBOO_OLLAMA_LOG="$WORK/ollama/ollama.log"\n'
            "launch_ollama\nwait\n",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        env={
            "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}",
            "OLLAMA_LIB_ROOT": str(lib),
            "LD_LIBRARY_PATH": str(host_cuda),
        },
    )
    assert dump.exists(), f"stub ollama never ran: {proc.stderr}"
    env = dict(
        line.split("=", 1) for line in dump.read_text().splitlines() if "=" in line
    )
    assert env.get("OLLAMA_KEEP_ALIVE") == "-1", "the server never got the keep-alive"
    assert env.get("OLLAMA_HOST") == "127.0.0.1:12345"
    assert env.get("LD_LIBRARY_PATH", "").split(":") == [
        str(lib),
        str(lib / "cuda_v13"),
        str(host_cuda),
    ]


@pytest.mark.parametrize("preset", ["", "42m"])
def test_launch_ollama_keep_alive_is_overridable(tmp_path: Path, preset: str) -> None:
    """-1 is a default, not a policy: a node that must evict can say so."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "setsid").write_text('#!/usr/bin/env bash\nexec "$@"\n')
    dump = tmp_path / "ollama-env.txt"
    (bin_dir / "ollama").write_text(f'#!/usr/bin/env bash\nenv > "{dump}"\n')
    for stub in ("setsid", "ollama"):
        (bin_dir / stub).chmod(0o755)
    work = tmp_path / "work"
    (work / "ollama").mkdir(parents=True)

    subprocess.run(
        [
            shutil.which("bash") or "bash",
            "-c",
            f'source "{ENTRYPOINT}" help >/dev/null\n'
            f'WORK="{work}"; OLLAMA_PORT=1\n'
            f'BAMBOO_OLLAMA_LOG="$WORK/ollama/ollama.log"\n'
            "launch_ollama\nwait\n",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        env={
            "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}",
            "OLLAMA_LIB_ROOT": str(tmp_path / "absent"),
            **({"OLLAMA_KEEP_ALIVE": preset} if preset else {}),
        },
    )
    env = dict(
        line.split("=", 1) for line in dump.read_text().splitlines() if "=" in line
    )
    assert env.get("OLLAMA_KEEP_ALIVE") == (preset or "-1")


# --- the durable accelerator record --------------------------------------------------


def _teardown_with_samples(
    tmp_path: Path, report_dir: Path | None, **env: str
) -> tuple[subprocess.CompletedProcess[str], Path]:
    """Run a standalone `teardown` over a scratch dir holding a recorded TSV."""
    run_dir = tmp_path / "bamboo.abc123"
    (run_dir / "ollama").mkdir(parents=True)
    tsv = run_dir / "ollama" / "accelerator.tsv"
    tsv.write_text(
        "offset_s\tstate\tsize\tsize_vram\tgpu_total_mib\tgpu_used_mib\tgpu_free_mib\n"
        "0\tgpu-partial\t100\t80\t24576\t20000\t4576\n"
        "15\tcpu\t100\t0\t24576\t500\t24076\n"
    )
    (tmp_path / "bamboo-batch.env").write_text(
        f"export BAMBOO_RUN_DIR={run_dir}\n"
        f"export BAMBOO_ACCEL_LOG={tsv}\n"
        "BAMBOO_SERVICE_PIDS=\n"
    )
    proc = subprocess.run(
        [shutil.which("bash") or "bash", str(ENTRYPOINT), "teardown"],
        capture_output=True,
        text=True,
        timeout=60,
        env={
            "PATH": os.environ.get("PATH", ""),
            "BAMBOO_WORK": str(tmp_path),
            "ACCEL_SAMPLER": str(
                Path(ENTRYPOINT).resolve().parent / "accel_sampler.py"
            ),
            **({"BAMBOO_ACCEL_REPORT_DIR": str(report_dir)} if report_dir else {}),
            **env,
        },
    )
    return proc, run_dir


@pytest.mark.skipif(
    not _HAS_PYTHON, reason="the summary is produced by accel_sampler.py"
)
def test_teardown_summarises_and_keeps_the_accelerator_record(tmp_path: Path) -> None:
    """The summary alone lives on stderr and the TSV dies with the scratch dir.

    A job that ran for hours can therefore end with no record of whether it used the GPU, so
    teardown copies both into an existing writable dir before the rm -rf.
    """
    out = tmp_path / "out"
    out.mkdir()
    proc, run_dir = _teardown_with_samples(tmp_path, out)
    assert proc.returncode == 0, proc.stderr
    assert "transitions: gpu-partial -> cpu" in proc.stderr, proc.stderr
    assert (out / "accelerator.tsv").read_text().count("\n") == 3  # header + 2 rows
    assert "transitions" in (out / "accelerator.txt").read_text()
    assert not run_dir.exists(), "the scratch dir should still be removed"


@pytest.mark.skipif(
    not _HAS_PYTHON, reason="the summary is produced by accel_sampler.py"
)
def test_teardown_says_so_when_the_record_cannot_be_kept(tmp_path: Path) -> None:
    """A read-only /out is normal for `exec` on a rootless Apptainer run.

    It must not fail teardown, and it must not be silent either — a missing artifact that
    nothing mentions reads as a clean run.
    """
    proc, _ = _teardown_with_samples(tmp_path, tmp_path / "does-not-exist")
    assert proc.returncode == 0, proc.stderr
    assert "transitions: gpu-partial -> cpu" in proc.stderr
    assert "accelerator record not kept" in proc.stderr
    assert "BAMBOO_KEEP_WORK=1" in proc.stderr
