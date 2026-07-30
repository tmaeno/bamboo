"""Tests for `deploy/batch/accel_sampler.py`.

The sampler answers two questions after the fact — *was Ollama on the GPU for this job, and did
the GPU actually do any work?* — so what is worth testing is the classification, the nvidia-smi
tick assembly, and the summary. All are pure functions over data: no GPU, no network, no Ollama.

The pieces that cannot be pure are covered narrowly: `read_api_ps` against a stub HTTP server,
and `UtilStream` against a stub `nvidia-smi` on PATH. The `--sample` loop itself is only
exercised in its `--once` form; the loop is `while True: … sleep`, killed by `teardown`, and the
interesting part is the row it writes.

The module lives under `deploy/`, not in the `bamboo` package (it is COPYed into the batch image
beside `entrypoint.sh` and must stay importable without the app's venv), so it is loaded by path.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Iterator, Optional

import pytest

_SAMPLER_PATH = (
    Path(__file__).resolve().parents[1] / "deploy" / "batch" / "accel_sampler.py"
)


def _load_sampler() -> Any:
    spec = importlib.util.spec_from_file_location("accel_sampler", _SAMPLER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["accel_sampler"] = module
    spec.loader.exec_module(module)
    return module


sampler = _load_sampler()


# --- classification -------------------------------------------------------------------


@pytest.mark.parametrize(
    "size,size_vram,expected",
    [
        (0, 0, "unloaded"),  # nothing resident: /api/ps had no entry
        (100, 0, "cpu"),  # loaded, nothing offloaded
        (100, 40, "gpu-partial"),  # the unstable state: VRAM was too tight
        (100, 100, "gpu-full"),
        (100, 120, "gpu-full"),  # >= not ==: never report partial on a rounding quirk
    ],
)
def test_classify(size: int, size_vram: int, expected: str) -> None:
    assert sampler.classify(size, size_vram) == expected


@pytest.mark.parametrize(
    "entry,model,matches",
    [
        ("qwen3.6:latest", "qwen3.6", True),  # the staged manifest omits the tag
        ("qwen3.6", "qwen3.6:latest", True),  # …or /api/ps does
        ("qwen3.6:latest", "qwen3.6:latest", True),
        ("llama3:latest", "qwen3.6", False),
        ("", "qwen3.6", False),
    ],
)
def test_model_matches_tolerates_an_implicit_latest_tag(
    entry: str, model: str, matches: bool
) -> None:
    """An exact compare would report `unloaded` for a perfectly healthy GPU run."""
    assert sampler.model_matches(entry, model) is matches


# --- reading /api/ps ------------------------------------------------------------------


@pytest.fixture
def api_ps_server() -> Iterator[Any]:
    """A stub Ollama serving whatever `.payload` is set to (or a 500 when it is None)."""
    state: dict[str, Any] = {"payload": {"models": []}}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler's API
            if state["payload"] is None:
                self.send_error(500)
                return
            body = json.dumps(state["payload"]).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: Any) -> None:
            pass  # keep pytest output clean

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    state["base_url"] = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        yield state
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_read_api_ps_reports_the_matching_entry(api_ps_server: dict[str, Any]) -> None:
    api_ps_server["payload"] = {
        "models": [
            {"model": "llama3:latest", "size": 1, "size_vram": 1},
            {"model": "qwen3.6:latest", "size": 22, "size_vram": 19},
        ]
    }
    assert sampler.read_api_ps(api_ps_server["base_url"], "qwen3.6") == (
        "gpu-partial",
        22,
        19,
    )


def test_read_api_ps_distinguishes_unloaded_from_unreachable(
    api_ps_server: dict[str, Any],
) -> None:
    """`unreachable` is not evidence about the GPU, so it must not read as `cpu`.

    Ollama is gone (or the port is closing during teardown) versus Ollama is up and has no model
    resident: the summary treats these differently, so the sampler must too.
    """
    api_ps_server["payload"] = {"models": []}
    assert sampler.read_api_ps(api_ps_server["base_url"], "qwen3.6")[0] == "unloaded"

    api_ps_server["payload"] = None  # 500
    assert sampler.read_api_ps(api_ps_server["base_url"], "qwen3.6")[0] == "unreachable"

    # Nothing listening at all.
    assert sampler.read_api_ps("http://127.0.0.1:1", "qwen3.6")[0] == "unreachable"


# --- the nvidia-smi stream ------------------------------------------------------------


@pytest.mark.parametrize(
    "line,expected",
    [
        ("0, 87, 41, 40960, 24367, 16593", (0, 87, 41, 40960, 24367, 16593)),
        ("1,0,0,40960,500,40460", (1, 0, 0, 40960, 500, 40460)),
        ("0, [N/A], [N/A], 40960, 500, 40460", None),  # unsupported device
        ("0, 87, 41", None),  # truncated
        ("some nvidia-smi warning", None),
    ],
)
def test_parse_device_row(line: str, expected: Optional[tuple[int, ...]]) -> None:
    """`[N/A]` must be dropped, never coerced to 0 — an invented 0% reads as "GPU idle"."""
    row = sampler.parse_device_row(line)
    assert (tuple(row) if row else None) == expected


def test_ticks_are_summed_across_devices_and_split_on_the_index_reset() -> None:
    """Summed, matching the site monitor's gpusmpct (">100% when multiple GPUs are active").

    An iteration boundary is where the device index stops increasing, since
    `nvidia-smi --loop-ms` writes one row per device per tick with no separator.
    """
    ticks = sampler.group_ticks(
        [
            "0, 40, 10, 20480, 1000, 19480",
            "1, 60, 20, 20480, 2000, 18480",  # tick 1: two devices
            "0, 5, 1, 20480, 100, 20380",  # index reset -> tick 2 begins
            "1, 5, 1, 20480, 100, 20380",
            "0, 90, 45, 20480, 3000, 17480",  # tick 3
            "1, 10, 5, 20480, 3000, 17480",
        ]
    )
    assert [t.sm_pct for t in ticks] == [100, 10, 100]
    assert [t.membw_pct for t in ticks] == [30, 2, 50]
    assert ticks[0].total_mib == 40960  # summed across both cards
    assert ticks[0].used_mib == 3000


def test_a_single_gpu_yields_one_tick_per_row() -> None:
    ticks = sampler.group_ticks(["0, 87, 41, 40960, 24367, 16593"] * 3)
    assert [t.sm_pct for t in ticks] == [87, 87, 87]


def test_a_partial_final_tick_is_still_flushed() -> None:
    """The stream is cut mid-iteration when teardown kills the group."""
    ticks = sampler.group_ticks(["0, 40, 10, 20480, 1000, 19480"])
    assert len(ticks) == 1 and ticks[0].sm_pct == 40


def test_aggregate_keeps_the_vram_envelope_not_a_snapshot() -> None:
    """report() takes min/max across rows, so each row must already carry its window's edges."""
    window = sampler.aggregate(
        sampler.group_ticks(
            [
                "0, 10, 5, 40960, 1000, 39960",
                "0, 90, 45, 40960, 24000, 16960",
                "0, 50, 25, 40960, 20000, 20960",
            ]
        )
    )
    assert (window.sm_min, window.sm_mean, window.sm_max) == (10, 50, 90)
    assert window.membw_max == 45
    assert window.used_mib == 24000  # peak
    assert window.free_mib == 16960  # minimum
    assert window.total_mib == 40960
    assert window.ticks == 3


def test_aggregate_of_nothing_is_unmeasured_not_zero() -> None:
    """0% and "not measured" must not be confused — that is why util_ticks is a column."""
    window = sampler.aggregate([])
    assert window.sm_max is None and window.total_mib is None
    assert window.ticks == 0


def test_util_stream_reads_a_stub_nvidia_smi(tmp_path: Path, monkeypatch: Any) -> None:
    """Covers the Popen + reader-thread + assembler path that no pure test reaches."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "nvidia-smi"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        "echo '0, 40, 10, 20480, 1000, 19480'\n"
        "echo '1, 60, 20, 20480, 2000, 18480'\n"
        "echo '0, 90, 45, 20480, 3000, 17480'\n"
        "echo '1, 10, 5, 20480, 3000, 17480'\n"
    )
    stub.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")

    stream = sampler.UtilStream(loop_ms=100)
    assert stream.start() is True
    for _ in range(50):  # the child exits immediately; wait for the reader to catch up
        time.sleep(0.02)
        ticks = stream.drain()
        if len(ticks) >= 1:
            break
    stream.stop()
    assert [t.sm_pct for t in ticks] == [100], ticks


def test_util_stream_reports_absence_rather_than_failing(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """A CPU queue has no nvidia-smi; the sampler must keep recording state either way."""
    monkeypatch.setenv("PATH", str(tmp_path))
    stream = sampler.UtilStream(loop_ms=100)
    assert stream.start() is False
    assert stream.drain() == []
    stream.stop()  # must be safe with no child


def test_sample_once_writes_a_header_and_a_row(
    api_ps_server: dict[str, Any], tmp_path: Path, monkeypatch: Any
) -> None:
    api_ps_server["payload"] = {
        "models": [{"model": "qwen3.6:latest", "size": 22, "size_vram": 22}]
    }
    monkeypatch.setenv("PATH", str(tmp_path))  # no nvidia-smi
    tsv = tmp_path / "nested" / "accelerator.tsv"  # parent must be created
    sampler.sample(tsv, api_ps_server["base_url"], "qwen3.6", 0.01, once=True)
    header, row = tsv.read_text().splitlines()
    assert header.split("\t") == list(sampler.COLUMNS)
    fields = row.split("\t")
    assert len(fields) == len(sampler.COLUMNS)
    assert fields[1:4] == ["gpu-full", "22", "22"]
    assert fields[4:11] == [""] * 7, "unmeasured GPU columns must be empty, not 0"


# --- the summary ----------------------------------------------------------------------


def _row(
    offset: int,
    state: str,
    *,
    size: int = 100,
    vram: int = 0,
    total: Optional[int] = None,
    used: Optional[int] = None,
    free: Optional[int] = None,
    sm: Optional[tuple[int, int, int]] = None,
    membw: Optional[int] = None,
    ticks: int = 0,
) -> str:
    sm_min, sm_mean, sm_max = sm if sm else (None, None, None)
    values = (
        offset, state, size, vram, total, used, free,
        sm_min, sm_mean, sm_max, membw, ticks,
    )  # fmt: skip
    return "\t".join("" if v is None else str(v) for v in values)


def _tsv(tmp_path: Path, *rows: str) -> Path:
    path = tmp_path / "accelerator.tsv"
    path.write_text("\t".join(sampler.COLUMNS) + "\n" + "".join(f"{r}\n" for r in rows))
    return path


def test_report_says_never_when_the_gpu_was_not_used(tmp_path: Path) -> None:
    tsv = _tsv(tmp_path, _row(0, "cpu"), _row(15, "cpu"), _row(30, "cpu"))
    out = sampler.report(tsv)
    assert "the GPU was NEVER used by ollama during this job" in out
    assert "transitions:" not in out  # nothing changed, so nothing to show
    assert "not recorded (no nvidia-smi" in out
    assert "sm not measured" in out


def test_report_shows_the_eviction_that_a_boot_time_verdict_hides(
    tmp_path: Path,
) -> None:
    """The case this sampler exists for: started on the GPU, ended on the CPU.

    A single boot-time measurement logs `accelerator: gpu (PARTIAL)` and nothing more, which is
    how a job came to report no GPU use at all to the site monitor while the entrypoint claimed
    otherwise.
    """
    hot = {
        "total": 24576,
        "used": 20000,
        "free": 4576,
        "sm": (3, 41, 88),
        "membw": 44,
        "ticks": 15,
    }
    cold = {
        "total": 24576,
        "used": 500,
        "free": 24076,
        "sm": (0, 0, 1),
        "membw": 0,
        "ticks": 15,
    }
    tsv = _tsv(
        tmp_path,
        _row(0, "gpu-partial", vram=80, **hot),
        _row(15, "gpu-partial", vram=80, **hot),
        _row(30, "unloaded", size=0, **cold),
        _row(45, "cpu", **cold),
        _row(60, "cpu", **cold),
    )
    out = sampler.report(tsv)
    assert "transitions: gpu-partial -> unloaded -> cpu" in out
    assert "first non-GPU sample: unloaded at +0m30s" in out
    assert "ollama was on the GPU for 40% of samples" in out
    assert "min free 4576 MiB of 24576 MiB total" in out
    assert "max used 20000 MiB" in out
    assert "sm 0-88%" in out


def test_report_says_every_sample_when_it_stayed_on_the_gpu(tmp_path: Path) -> None:
    tsv = _tsv(
        tmp_path,
        _row(
            0, "gpu-full", vram=100, total=97871, used=30000, free=67871, sm=(5, 60, 95)
        ),
        _row(
            15,
            "gpu-full",
            vram=100,
            total=97871,
            used=30000,
            free=67871,
            sm=(9, 70, 99),
        ),
    )
    out = sampler.report(tsv)
    assert "on the GPU for every sample" in out
    assert "model footprint:" in out
    assert "sm 5-99% (mean 65%)" in out


def test_report_flags_residency_without_compute(tmp_path: Path) -> None:
    """The gap this round is about: the weights are on the card, the work is not.

    `gpu-full 100%` alone reads as success. The first `gpu-check` run had exactly this shape —
    model resident, `gpusmpct: 0.0`, because it loaded the model and never generated.
    """
    tsv = _tsv(
        tmp_path,
        _row(
            0,
            "gpu-full",
            vram=100,
            total=40960,
            used=24000,
            free=16960,
            sm=(0, 0, 0),
            ticks=15,
        ),
        _row(
            15,
            "gpu-full",
            vram=100,
            total=40960,
            used=24000,
            free=16960,
            sm=(0, 0, 1),
            ticks=15,
        ),
    )
    out = sampler.report(tsv)
    assert "NO COMPUTE observed" in out
    assert "on the GPU for every sample" not in out


def test_report_does_not_flag_compute_that_did_happen(tmp_path: Path) -> None:
    tsv = _tsv(
        tmp_path,
        _row(
            0,
            "gpu-full",
            vram=100,
            total=40960,
            used=24000,
            free=16960,
            sm=(0, 40, 80),
            ticks=15,
        ),
    )
    out = sampler.report(tsv)
    assert "NO COMPUTE" not in out
    assert "on the GPU for every sample" in out


def test_report_does_not_claim_no_compute_when_sm_was_never_measured(
    tmp_path: Path,
) -> None:
    """No nvidia-smi is not evidence of idleness — it is absence of evidence."""
    tsv = _tsv(tmp_path, _row(0, "gpu-full", vram=100), _row(15, "gpu-full", vram=100))
    out = sampler.report(tsv)
    assert "NO COMPUTE" not in out
    assert "on the GPU for every sample" in out


def test_report_skips_malformed_rows_rather_than_failing(tmp_path: Path) -> None:
    """teardown kills the sampler, so a truncated final row is normal, not exceptional."""
    tsv = _tsv(
        tmp_path,
        _row(0, "gpu-full", vram=100),
        _row(15, "not-a-state"),  # unknown state
        "30\tgpu-full\tNaN\t100\t\t\t\t\t\t\t\t0",  # unparseable size
        "45\tgpu-full",  # truncated mid-write
        _row(60, "cpu"),
    )
    out = sampler.report(tsv)
    assert "2 samples" in out, out
    assert "transitions: gpu-full -> cpu" in out


def test_report_on_a_missing_or_empty_record_is_not_an_error(tmp_path: Path) -> None:
    assert "no samples recorded" in sampler.report(tmp_path / "absent.tsv")
    assert "no usable samples" in sampler.report(_tsv(tmp_path))


# --- the timeline ---------------------------------------------------------------------


def test_timeline_collapses_runs_of_one_state(tmp_path: Path) -> None:
    tsv = _tsv(
        tmp_path,
        _row(0, "gpu-full", vram=2**30, sm=(10, 50, 90)),
        _row(15, "gpu-full", vram=2**30, sm=(20, 60, 95)),
        _row(30, "cpu", sm=(0, 0, 0)),
        _row(3630, "cpu", sm=(0, 0, 1)),
    )
    lines = [ln for ln in sampler.report(tsv).splitlines() if " smp " in ln]
    assert len(lines) == 2, lines
    assert "0m00s-0m15s" in lines[0] and "gpu-full" in lines[0]
    assert "2 smp" in lines[0] and "1.0 GiB" in lines[0] and "sm 10-95%" in lines[0]
    # Past an hour the offsets switch to h/m so the column stays narrow.
    assert "0m30s-1h00m" in lines[1] and "cpu" in lines[1] and "vram —" in lines[1]


def test_timeline_is_one_line_when_nothing_changes(tmp_path: Path) -> None:
    tsv = _tsv(tmp_path, *(_row(15 * i, "gpu-full", vram=100) for i in range(10)))
    assert len([ln for ln in sampler.report(tsv).splitlines() if " smp " in ln]) == 1


# --- the record dump ------------------------------------------------------------------


def test_dump_max_emits_the_record_verbatim_and_round_trips(tmp_path: Path) -> None:
    """Verbatim TSV, not padded columns, so the dump can be fed straight back to --report."""
    rows = [
        _row(15 * i, "gpu-full", vram=100, sm=(1, 2, 3), ticks=15) for i in range(5)
    ]
    tsv = _tsv(tmp_path, *rows)
    out = sampler.report(tsv, dump_max=0)
    assert "accelerator record (5 rows)" in out
    dumped = [ln[4:] for ln in out.splitlines() if ln.startswith("  | ")]
    assert dumped[0] == "\t".join(sampler.COLUMNS)
    assert dumped[1:] == rows
    assert [r.offset_s for r in sampler.parse_rows(dumped)] == [0, 15, 30, 45, 60]


def test_dump_max_downsamples_and_says_so(tmp_path: Path) -> None:
    """The timeline above stays complete, so only resolution is lost — never the shape."""
    rows = [_row(15 * i, "gpu-full", vram=100) for i in range(100)]
    out = sampler.report(_tsv(tmp_path, *rows), dump_max=10)
    assert "downsampled, 1 in 10 of 100 rows" in out
    dumped = [ln[4:] for ln in out.splitlines() if ln.startswith("  | ")][1:]
    assert len(dumped) == 11  # 10 strided + the final row
    assert dumped[0] == rows[0] and dumped[-1] == rows[-1]


def test_report_without_dump_max_omits_the_record(tmp_path: Path) -> None:
    out = sampler.report(_tsv(tmp_path, _row(0, "gpu-full", vram=100)))
    assert "accelerator record" not in out
    assert "timeline:" in out


def test_report_mode_prints_to_stdout(tmp_path: Path, capsys: Any) -> None:
    tsv = _tsv(tmp_path, _row(0, "gpu-full", vram=100))
    assert sampler.main(["--report", "--tsv", str(tsv), "--dump-max", "0"]) == 0
    out = capsys.readouterr().out
    assert "on the GPU for every sample" in out
    assert "accelerator record" in out


def test_sample_mode_requires_a_target(tmp_path: Path) -> None:
    """A sampler started without --base-url/--model would record `unreachable` forever."""
    with pytest.raises(SystemExit):
        sampler.main(["--sample", "--tsv", str(tmp_path / "x.tsv")])
