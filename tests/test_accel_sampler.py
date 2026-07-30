"""Tests for `deploy/batch/accel_sampler.py`.

The sampler's job is to answer one question after the fact — *was Ollama on the GPU for this
job, and if not, was it ever?* — so what is worth testing is the classification and the
summary, both pure functions over data. Neither needs a GPU, a network, or Ollama.

Sampling itself is exercised only in its `--once` form against a stub HTTP server: the loop
is `while True: … sleep`, killed by `teardown`, and the interesting part is the row it writes.

The module lives under `deploy/`, not in the `bamboo` package (it is COPYed into the batch
image beside `entrypoint.sh` and must stay importable without the app's venv), so it is loaded
by path.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Iterator

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

    Ollama is gone (or the port is closing during teardown) versus Ollama is up and has no
    model resident: the summary treats these differently, so the sampler must too.
    """
    api_ps_server["payload"] = {"models": []}
    assert sampler.read_api_ps(api_ps_server["base_url"], "qwen3.6")[0] == "unloaded"

    api_ps_server["payload"] = None  # 500
    assert sampler.read_api_ps(api_ps_server["base_url"], "qwen3.6")[0] == "unreachable"

    # Nothing listening at all.
    assert sampler.read_api_ps("http://127.0.0.1:1", "qwen3.6")[0] == "unreachable"


def test_sample_once_writes_a_header_and_a_row(
    api_ps_server: dict[str, Any], tmp_path: Path
) -> None:
    api_ps_server["payload"] = {
        "models": [{"model": "qwen3.6:latest", "size": 22, "size_vram": 22}]
    }
    tsv = tmp_path / "nested" / "accelerator.tsv"  # parent must be created
    sampler.sample(tsv, api_ps_server["base_url"], "qwen3.6", 0.01, once=True)
    header, row = tsv.read_text().splitlines()
    assert header.split("\t") == list(sampler.COLUMNS)
    assert row.split("\t")[1:4] == ["gpu-full", "22", "22"]


# --- the summary ----------------------------------------------------------------------


def _tsv(tmp_path: Path, *rows: str) -> Path:
    """Build a record file: header plus `offset\tstate\tsize\tvram` rows, no VRAM columns."""
    path = tmp_path / "accelerator.tsv"
    path.write_text("\t".join(sampler.COLUMNS) + "\n" + "".join(f"{r}\n" for r in rows))
    return path


def test_report_says_never_when_the_gpu_was_not_used(tmp_path: Path) -> None:
    tsv = _tsv(
        tmp_path,
        "0\tcpu\t100\t0\t\t\t",
        "15\tcpu\t100\t0\t\t\t",
        "30\tcpu\t100\t0\t\t\t",
    )
    out = sampler.report(tsv)
    assert "the GPU was NEVER used by ollama during this job" in out
    assert "cpu" in out
    assert "transitions:" not in out  # nothing changed, so nothing to show
    assert "not recorded (no nvidia-smi" in out


def test_report_shows_the_eviction_that_a_boot_time_verdict_hides(
    tmp_path: Path,
) -> None:
    """The case this whole sampler exists for: started on the GPU, ended on the CPU.

    A single boot-time measurement logs `accelerator: gpu (PARTIAL)` and nothing more, which
    is how a job came to report no GPU use at all to the site monitor while the entrypoint
    claimed otherwise.
    """
    tsv = _tsv(
        tmp_path,
        "0\tgpu-partial\t100\t80\t24576\t20000\t4576",
        "15\tgpu-partial\t100\t80\t24576\t20000\t4576",
        "30\tunloaded\t0\t0\t24576\t500\t24076",
        "45\tcpu\t100\t0\t24576\t500\t24076",
        "60\tcpu\t100\t0\t24576\t400\t24176",
    )
    out = sampler.report(tsv)
    assert "transitions: gpu-partial -> unloaded -> cpu" in out
    assert "first non-GPU sample: unloaded at +0m30s" in out
    assert "ollama was on the GPU for 40% of samples" in out
    assert "min free 4576 MiB of 24576 MiB total" in out
    assert "max used 20000 MiB" in out


def test_report_says_every_sample_when_it_stayed_on_the_gpu(tmp_path: Path) -> None:
    tsv = _tsv(
        tmp_path,
        "0\tgpu-full\t100\t100\t97871\t30000\t67871",
        "15\tgpu-full\t100\t100\t97871\t30000\t67871",
    )
    out = sampler.report(tsv)
    assert "on the GPU for every sample" in out
    assert "model footprint:" in out


def test_report_skips_malformed_rows_rather_than_failing(tmp_path: Path) -> None:
    """teardown kills the sampler, so a truncated final row is normal, not exceptional.

    Raising here would lose the summary in exactly the runs that need it.
    """
    tsv = _tsv(
        tmp_path,
        "0\tgpu-full\t100\t100\t\t\t",
        "15\tnot-a-state\t100\t100\t\t\t",  # unknown state
        "30\tgpu-full\tNaN\t100\t\t\t",  # unparseable size
        "45\tgpu-full",  # truncated mid-write
        "60\tcpu\t100\t0\t\t\t",
    )
    out = sampler.report(tsv)
    assert "2 samples" in out, out
    assert "transitions: gpu-full -> cpu" in out


def test_report_on_a_missing_or_empty_record_is_not_an_error(tmp_path: Path) -> None:
    assert "no samples recorded" in sampler.report(tmp_path / "absent.tsv")
    assert "no usable samples" in sampler.report(_tsv(tmp_path))


def test_report_mode_prints_to_stdout(tmp_path: Path, capsys: Any) -> None:
    tsv = _tsv(tmp_path, "0\tgpu-full\t100\t100\t\t\t")
    assert sampler.main(["--report", "--tsv", str(tsv)]) == 0
    assert "on the GPU for every sample" in capsys.readouterr().out


def test_sample_mode_requires_a_target(tmp_path: Path) -> None:
    """A sampler started without --base-url/--model would record `unreachable` forever."""
    with pytest.raises(SystemExit):
        sampler.main(["--sample", "--tsv", str(tmp_path / "x.tsv")])
