#!/usr/bin/env python3
"""Record, for the whole life of a batch job, whether Ollama is generating on the GPU.

`entrypoint.sh`'s ``report_accelerator`` answers that question **once**, right after the
model is preloaded. That turned out not to be enough: a job on a GPU node logged
``accelerator: gpu (PARTIAL) - 19.1 of 22.8 GiB in VRAM`` at boot and yet the site's job
monitor reported ``ngpus: 0.0`` / ``gpufbmem: 0.0`` for its entire run. A point measurement
cannot distinguish "never used the GPU" from "started there and was evicted", and those have
completely different causes — so this samples continuously instead.

Two modes over one on-disk format (TSV), so the format has a single owner:

* ``--sample`` — poll ``{base}/api/ps`` plus ``nvidia-smi`` every ``--interval`` seconds and
  append one row per sample. Runs until killed; ``entrypoint.sh`` starts it after the boot
  verdict and ``teardown`` kills it with the services.
* ``--report`` — read a TSV and print the end-of-job summary: time in each state, the state
  *transitions* with the offset of the first non-GPU sample, and the VRAM envelope.

Deliberately stdlib-only (``urllib`` rather than httpx): this runs inside the batch image
next to the services, and a diagnostic must not be the thing that fails to import.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable, Iterator, NamedTuple, Optional

# One column layout, written by --sample and read by --report.
COLUMNS = (
    "offset_s",  # whole seconds since the sampler started
    "state",  # see classify()
    "size",  # bytes the model occupies in total
    "size_vram",  # bytes of it resident on the GPU
    "gpu_total_mib",  # summed over devices; "" when nvidia-smi is unavailable
    "gpu_used_mib",
    "gpu_free_mib",
)

# States, worst-to-best. `unreachable` means the sampler could not reach Ollama at all
# (server gone, or the port closed during teardown) and is not evidence about the GPU.
UNREACHABLE = "unreachable"
UNLOADED = "unloaded"
CPU = "cpu"
GPU_PARTIAL = "gpu-partial"
GPU_FULL = "gpu-full"
STATE_ORDER = (GPU_FULL, GPU_PARTIAL, CPU, UNLOADED, UNREACHABLE)


def model_matches(entry_name: str, model: str) -> bool:
    """Match an ``/api/ps`` entry to the configured model name.

    Same ``:latest``-tolerance as ``bamboo.llm.llm_client._ollama_model_matches`` — the
    staged manifest may say ``qwen3.6`` where ``/api/ps`` says ``qwen3.6:latest``. Inlined
    rather than imported: this script must keep working if the venv is not importable, which
    is exactly the situation where a diagnostic matters most.
    """
    if not entry_name:
        return False
    return entry_name == model or entry_name.split(":")[0] == model.split(":")[0]


def classify(size: int, size_vram: int) -> str:
    """Turn ``/api/ps`` byte counts into one of the states above.

    ``size_vram`` is what ``ollama ps`` renders as its PROCESSOR column, so this is the
    non-heuristic answer: no scraping of log wording that changes between releases.
    """
    if size <= 0:
        return UNLOADED
    if size_vram <= 0:
        return CPU
    if size_vram >= size:
        return GPU_FULL
    return GPU_PARTIAL


def read_api_ps(
    base_url: str, model: str, timeout: float = 5.0
) -> tuple[str, int, int]:
    """Return ``(state, size, size_vram)`` for ``model`` from ``{base_url}/api/ps``."""
    url = f"{base_url.rstrip('/')}/api/ps"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 — localhost
            payload = json.loads(resp.read().decode("utf-8", "replace"))
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError):
        return UNREACHABLE, 0, 0
    for entry in (payload or {}).get("models") or []:
        if model_matches(entry.get("model") or entry.get("name") or "", model):
            size = int(entry.get("size") or 0)
            vram = int(entry.get("size_vram") or 0)
            return classify(size, vram), size, vram
    return UNLOADED, 0, 0


class DeviceVram(NamedTuple):
    total_mib: Optional[int]
    used_mib: Optional[int]
    free_mib: Optional[int]

    @classmethod
    def unavailable(cls) -> DeviceVram:
        return cls(None, None, None)


def read_gpu_vram(timeout: float = 5.0) -> DeviceVram:
    """Total/used/free VRAM summed over all visible devices, via ``nvidia-smi``.

    Summed rather than per-device because the interesting question is the envelope: was the
    card simply too small for the model, or was a co-tenant holding the memory? Both show up
    as ``free`` collapsing while ``total`` stays put.
    """
    if shutil.which("nvidia-smi") is None:
        return DeviceVram.unavailable()
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return DeviceVram.unavailable()
    total = used = free = 0
    seen = False
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            t, u, f = (int(p) for p in parts)
        except ValueError:
            continue
        total, used, free, seen = total + t, used + u, free + f, True
    return DeviceVram(total, used, free) if seen else DeviceVram.unavailable()


def sample(
    tsv: Path, base_url: str, model: str, interval: float, *, once: bool = False
) -> None:
    """Append one row per interval until killed (or one row, with ``once``).

    Each row is flushed immediately: ``teardown`` kills this process, so anything still
    buffered would be lost — and the last samples before the kill are the ones describing how
    the job ended.
    """
    tsv.parent.mkdir(parents=True, exist_ok=True)
    new = not tsv.exists() or tsv.stat().st_size == 0
    started = time.monotonic()
    warned_no_smi = False
    with tsv.open("a", encoding="utf-8") as fh:
        if new:
            fh.write("\t".join(COLUMNS) + "\n")
            fh.flush()
        while True:
            state, size, vram = read_api_ps(base_url, model)
            vram_dev = read_gpu_vram()
            if vram_dev.total_mib is None and not warned_no_smi:
                # Once, not per sample: a CPU queue would otherwise fill the log with it.
                print(
                    "accel-sampler: nvidia-smi unavailable", file=sys.stderr, flush=True
                )
                warned_no_smi = True
            fh.write(
                "\t".join(
                    str(v) if v is not None else ""
                    for v in (
                        int(time.monotonic() - started),
                        state,
                        size,
                        vram,
                        vram_dev.total_mib,
                        vram_dev.used_mib,
                        vram_dev.free_mib,
                    )
                )
                + "\n"
            )
            fh.flush()
            if once:
                return
            time.sleep(interval)


class Row(NamedTuple):
    offset_s: int
    state: str
    size: int
    size_vram: int
    vram: DeviceVram


def parse_rows(lines: Iterable[str]) -> Iterator[Row]:
    """Yield well-formed rows, skipping anything else.

    Malformed rows are expected, not exceptional: the sampler is killed by ``teardown``, so
    the final line can be a partial write. A summary that raised on that would be missing
    exactly when it is wanted.
    """

    def _opt_int(value: str) -> Optional[int]:
        try:
            return int(value)
        except ValueError:
            return None

    for line in lines:
        parts = line.rstrip("\n").split("\t")
        if len(parts) != len(COLUMNS) or parts[0] == COLUMNS[0]:
            continue
        try:
            offset, state, size, vram = (
                int(parts[0]),
                parts[1],
                int(parts[2]),
                int(parts[3]),
            )
        except ValueError:
            continue
        if state not in STATE_ORDER:
            continue
        yield Row(
            offset,
            state,
            size,
            vram,
            DeviceVram(_opt_int(parts[4]), _opt_int(parts[5]), _opt_int(parts[6])),
        )


def _gib(byte_count: int) -> str:
    return f"{byte_count / 2**30:.1f} GiB"


def report(tsv: Path) -> str:
    """Render the end-of-job summary that goes into the job log."""
    if not tsv.exists():
        return f"accelerator: no samples recorded ({tsv} is missing)"
    rows = list(
        parse_rows(tsv.read_text(encoding="utf-8", errors="replace").splitlines())
    )
    if not rows:
        return f"accelerator: no usable samples in {tsv}"

    counts = dict.fromkeys(STATE_ORDER, 0)
    for row in rows:
        counts[row.state] += 1
    total = len(rows)
    span = rows[-1].offset_s - rows[0].offset_s

    lines = [
        f"accelerator over the job: {total} samples spanning {span // 60}m{span % 60}s"
    ]
    for state in STATE_ORDER:
        if counts[state]:
            lines.append(
                f"  {state:<12} {counts[state]:>5}  {100 * counts[state] // total:>3}%"
            )

    # The transitions are the point of sampling at all: "gpu-full -> unloaded -> cpu" is a
    # different bug report from "cpu" throughout.
    transitions = [rows[0].state]
    first_non_gpu: Optional[Row] = None
    for row in rows:
        if row.state != transitions[-1]:
            transitions.append(row.state)
        if first_non_gpu is None and not row.state.startswith("gpu-"):
            first_non_gpu = row
    if len(transitions) > 1:
        lines.append(f"  transitions: {' -> '.join(transitions)}")
    if first_non_gpu is not None:
        lines.append(
            f"  first non-GPU sample: {first_non_gpu.state} at +"
            f"{first_non_gpu.offset_s // 60}m{first_non_gpu.offset_s % 60}s"
        )

    on_gpu = counts[GPU_FULL] + counts[GPU_PARTIAL]
    if on_gpu == 0:
        lines.append("  verdict: the GPU was NEVER used by ollama during this job")
    elif on_gpu == total:
        lines.append("  verdict: ollama was on the GPU for every sample")
    else:
        lines.append(
            f"  verdict: ollama was on the GPU for {100 * on_gpu // total}% of samples"
        )

    # VRAM envelope: distinguishes "card too small for the model" from "someone else had it".
    frees = [r.vram.free_mib for r in rows if r.vram.free_mib is not None]
    totals = [r.vram.total_mib for r in rows if r.vram.total_mib is not None]
    if frees and totals:
        lines.append(
            f"  device VRAM: min free {min(frees)} MiB of {max(totals)} MiB total"
            f" (max used {max(r.vram.used_mib or 0 for r in rows)} MiB, all processes)"
        )
    else:
        lines.append("  device VRAM: not recorded (no nvidia-smi in the container)")

    biggest = max(rows, key=lambda r: r.size)
    if biggest.size:
        lines.append(
            f"  model footprint: {_gib(biggest.size)} total, peak {_gib(max(r.size_vram for r in rows))} in VRAM"
        )
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, add_help=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--sample", action="store_true", help="poll until killed")
    mode.add_argument("--report", action="store_true", help="summarise a recorded TSV")
    parser.add_argument("--tsv", required=True, type=Path, help="the record file")
    parser.add_argument("--base-url", default="", help="ollama base URL (--sample)")
    parser.add_argument("--model", default="", help="model name to look for (--sample)")
    parser.add_argument(
        "--interval", type=float, default=15.0, help="seconds between samples"
    )
    parser.add_argument(
        "--once", action="store_true", help="take a single sample and exit"
    )
    args = parser.parse_args(argv)

    if args.report:
        print(report(args.tsv))
        return 0

    if not args.base_url or not args.model:
        parser.error("--sample needs --base-url and --model")
    try:
        sample(args.tsv, args.base_url, args.model, args.interval, once=args.once)
    except KeyboardInterrupt:  # pragma: no cover — teardown sends the signal
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
