#!/usr/bin/env python3
"""Record, for the whole life of a batch job, whether Ollama is *using* the GPU.

`entrypoint.sh`'s ``report_accelerator`` answers that question **once**, right after the model
is preloaded, and only in terms of residency. Neither was enough:

* A job logged ``accelerator: gpu (PARTIAL)`` at boot and the site's job monitor still reported
  ``ngpus: 0.0`` for its whole run — a point measurement cannot distinguish "never used the GPU"
  from "started there and was evicted", and those have different causes.
* ``size_vram`` proves the weights are *in* VRAM, not that any token was produced there. The
  first ``gpu-check`` run was exactly that shape: the model resident, ``gpusmpct: 0.0``, because
  it loaded the model and never generated. So SM utilisation is sampled too.

Two modes over one on-disk format (TSV), so the format has a single owner:

* ``--sample`` — poll ``{base}/api/ps`` every ``--interval`` seconds, aggregate the SM stream
  (below) over that window, and append one row. Runs until killed; ``entrypoint.sh`` starts it
  after the boot verdict and ``teardown`` kills it with the services.
* ``--report`` — read a TSV and print the summary, a per-state *timeline*, and with
  ``--dump-max`` the record itself so it survives in a job log.

SM utilisation comes from a **separate, denser stream**: one long-lived
``nvidia-smi --loop-ms`` child read by a daemon thread. ``utilization.gpu`` is an instantaneous
figure over roughly the preceding second, so reading it once per 15 s poll would miss the bursts
that token generation consists of; one ``fork`` for the whole job buys ~15 ticks per row instead.

Deliberately stdlib-only (``urllib`` rather than httpx): this runs inside the batch image next to
the services, and a diagnostic must not be the thing that fails to import.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pty
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path
from typing import Any, Iterable, Iterator, NamedTuple, Optional

# One column layout, written by --sample and read by --report.
COLUMNS = (
    "offset_s",  # whole seconds since the sampler started
    "state",  # see classify()
    "size",  # bytes the model occupies in total
    "size_vram",  # bytes of it resident on the GPU
    "gpu_total_mib",  # device totals over this window; "" when not measured
    "gpu_used_mib",  # the window's PEAK used (all processes)
    "gpu_free_mib",  # the window's MINIMUM free
    "sm_min",  # SM utilisation % over the window, summed across devices
    "sm_mean",
    "sm_max",
    "membw_max",  # memory-bandwidth utilisation %, the window's peak
    "util_ticks",  # nvidia-smi ticks that landed in this window — see aggregate()
)

# States, worst-to-best. `unreachable` means the sampler could not reach Ollama at all
# (server gone, or the port closed during teardown) and is not evidence about the GPU.
UNREACHABLE = "unreachable"
UNLOADED = "unloaded"
CPU = "cpu"
GPU_PARTIAL = "gpu-partial"
GPU_FULL = "gpu-full"
STATE_ORDER = (GPU_FULL, GPU_PARTIAL, CPU, UNLOADED, UNREACHABLE)

# Below this, "the GPU held the weights but did no work" is the honest reading. An idle device
# still reports a percent or two, so 0 alone would be too strict a test; real generation drives
# this to tens of percent, so nothing in between is being papered over.
IDLE_SM_PCT = 2


def _note(message: str) -> None:
    """One-line diagnostic on stderr, prefixed so it is identifiable in a job log."""
    print(f"accel-sampler: {message}", file=sys.stderr, flush=True)


def model_matches(entry_name: str, model: str) -> bool:
    """Match an ``/api/ps`` entry to the configured model name.

    Same ``:latest``-tolerance as ``bamboo.llm.llm_client._ollama_model_matches`` — the staged
    manifest may say ``qwen3.6`` where ``/api/ps`` says ``qwen3.6:latest``. Inlined rather than
    imported: this script must keep working if the venv is not importable, which is exactly the
    situation where a diagnostic matters most.
    """
    if not entry_name:
        return False
    return entry_name == model or entry_name.split(":")[0] == model.split(":")[0]


def classify(size: int, size_vram: int) -> str:
    """Turn ``/api/ps`` byte counts into one of the states above.

    ``size_vram`` is what ``ollama ps`` renders as its PROCESSOR column, so this is the
    non-heuristic answer: no scraping of log wording that changes between releases. It speaks
    only about residency — see the SM columns for whether the GPU did any work.
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


# --------------------------------------------------------------------------------------- #
# The nvidia-smi stream
# --------------------------------------------------------------------------------------- #

UTIL_QUERY = (
    "index,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free"
)


class DeviceRow(NamedTuple):
    index: int
    sm_pct: int
    membw_pct: int
    total_mib: int
    used_mib: int
    free_mib: int


class Tick(NamedTuple):
    """One nvidia-smi iteration, summed over the devices it reported."""

    sm_pct: int
    membw_pct: int
    total_mib: int
    used_mib: int
    free_mib: int


def parse_device_row(line: str) -> Optional[DeviceRow]:
    """Parse one CSV row, or None if it is not one.

    ``[N/A]`` appears for utilisation on some devices (and for everything on a stray warning
    line), so anything unparseable is dropped rather than turned into a zero — an invented 0 %
    would read as "the GPU was idle".
    """
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 6:
        return None
    try:
        return DeviceRow(*(int(p) for p in parts))
    except ValueError:
        return None


def _sum_devices(rows: list[DeviceRow]) -> Tick:
    """Sum a tick across devices, matching the site monitor's own definition of gpusmpct:
    *"sum of the streaming multiprocessor usage … can be >100% when multiple GPUs are active"*.
    Summing rather than averaging is what makes the two numbers comparable."""
    return Tick(
        sm_pct=sum(r.sm_pct for r in rows),
        membw_pct=sum(r.membw_pct for r in rows),
        total_mib=sum(r.total_mib for r in rows),
        used_mib=sum(r.used_mib for r in rows),
        free_mib=sum(r.free_mib for r in rows),
    )


class TickAssembler:
    """Groups streamed device rows into per-iteration ticks.

    ``nvidia-smi --loop-ms`` emits one row per device per iteration with no separator, so an
    iteration boundary is where the device index stops increasing. Stateful because the stream
    is drained incrementally, a partial iteration at a time.
    """

    def __init__(self) -> None:
        self._cur: list[DeviceRow] = []
        self._prev_index: Optional[int] = None

    def feed(self, line: str) -> Optional[Tick]:
        """Absorb one line; return the tick it completed, if any."""
        row = parse_device_row(line)
        if row is None:
            return None
        done: Optional[Tick] = None
        if self._prev_index is not None and row.index <= self._prev_index and self._cur:
            done = _sum_devices(self._cur)
            self._cur = []
        self._cur.append(row)
        self._prev_index = row.index
        return done

    def flush(self) -> Optional[Tick]:
        """Close the iteration in progress — the stream ends mid-tick when we are killed."""
        if not self._cur:
            return None
        tick = _sum_devices(self._cur)
        self._cur = []
        self._prev_index = None
        return tick


def group_ticks(lines: Iterable[str]) -> list[Tick]:
    """Batch form of TickAssembler, for a finite sequence of rows."""
    assembler = TickAssembler()
    ticks = [t for t in (assembler.feed(line) for line in lines) if t is not None]
    last = assembler.flush()
    return ticks + ([last] if last else [])


class UtilWindow(NamedTuple):
    """What one row of the TSV records about the GPU, over one /api/ps interval."""

    sm_min: Optional[int]
    sm_mean: Optional[int]
    sm_max: Optional[int]
    membw_max: Optional[int]
    total_mib: Optional[int]
    used_mib: Optional[int]
    free_mib: Optional[int]
    ticks: int

    @classmethod
    def unmeasured(cls) -> UtilWindow:
        return cls(None, None, None, None, None, None, None, 0)


def aggregate(ticks: list[Tick]) -> UtilWindow:
    """Collapse a window's ticks into one row's worth of numbers.

    VRAM keeps the envelope rather than a snapshot — peak used, minimum free — so the job-level
    figures in report() stay exact however coarse the rows are. ``total`` is taken from the last
    tick since it does not move.

    No ticks yields all-None, not zeros: "not measured" and "0 %" must not be confused, which is
    the whole reason ``util_ticks`` is recorded next to them.
    """
    if not ticks:
        return UtilWindow.unmeasured()
    sm = [t.sm_pct for t in ticks]
    return UtilWindow(
        sm_min=min(sm),
        sm_mean=round(sum(sm) / len(sm)),
        sm_max=max(sm),
        membw_max=max(t.membw_pct for t in ticks),
        total_mib=ticks[-1].total_mib,
        used_mib=max(t.used_mib for t in ticks),
        free_mib=min(t.free_mib for t in ticks),
        ticks=len(ticks),
    )


MODE_STREAM = "stream"
MODE_ONESHOT = "oneshot"
MODE_NONE = "none"

# Consecutive empty windows before the stream is written off. One can be legitimate (a short
# interval, a slow first tick); two in a row means it is not delivering.
_STREAM_DRY_LIMIT = 2

# …but only once the stream has had a fair chance, in wall time. A tick is complete only when the
# *next* iteration's first row arrives, so the earliest possible tick is two loop periods in, and
# a caller polling faster than that (a small BAMBOO_ACCEL_SAMPLE_SEC) would otherwise write off a
# perfectly healthy stream on timing alone.
def _stream_grace_s(loop_ms: int) -> float:
    return 3 * loop_ms / 1000 + 1.0


def _oneshot(timeout: float = 5.0) -> list[Tick]:
    """One ``nvidia-smi`` invocation: a single tick, or none if it fails.

    Coarse — one sample per window instead of ~15 — but it cannot be defeated by buffering,
    which is why it is the fallback.
    """
    try:
        out = subprocess.run(  # noqa: S603 — fixed argv, no shell
            [
                "nvidia-smi",
                f"--query-gpu={UTIL_QUERY}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return []
    return group_ticks(out.splitlines())


class UtilSource:
    """Per-window GPU utilisation: a streamed ``nvidia-smi`` when that works, one-shot when not.

    The stream is a long-lived ``nvidia-smi --loop-ms`` child read by a daemon thread, which
    appends raw lines to a deque; assembly and aggregation happen on the main loop.
    ``deque.append``/``popleft`` are individually atomic in CPython, so draining by
    popleft-until-empty needs no lock.

    **The child is given a pty, not a pipe.** nvidia-smi's stdout goes through stdio, which
    block-buffers (8 KiB) when it is not a terminal: at ~31 bytes per row and one row per second
    the first flush lands over four minutes in, so a three-minute job recorded *zero* ticks
    while a per-poll one-shot had been working fine. A pty makes ``isatty()`` true and stdio
    line-buffers instead. (A local stub built from shell ``echo`` — one ``write(2)`` per line —
    hid this completely, which is why the fallback below exists rather than trust alone.)

    If the stream still delivers nothing for ``_STREAM_DRY_LIMIT`` consecutive windows it is
    written off and every later window uses a one-shot. Losing density beats losing the data.
    """

    def __init__(self, loop_ms: int, *, stderr_path: Optional[Path] = None) -> None:
        self._loop_ms = loop_ms
        self._stderr_path = stderr_path
        self._lines: deque[str] = deque()
        self._proc: Optional[subprocess.Popen[bytes]] = None
        self._stdout: Optional[Any] = None
        self._assembler = TickAssembler()
        self._dry = 0
        self._stream_since = 0.0
        self.mode = MODE_NONE

    def start(self) -> str:
        """Pick a mode and return it. ``none`` when there is no nvidia-smi at all."""
        if shutil.which("nvidia-smi") is None:
            self.mode = MODE_NONE
            return self.mode
        self.mode = MODE_STREAM if self._spawn_stream() else MODE_ONESHOT
        return self.mode

    def _spawn_stream(self) -> bool:
        argv = [
            "nvidia-smi",
            f"--query-gpu={UTIL_QUERY}",
            "--format=csv,noheader,nounits",
            f"--loop-ms={self._loop_ms}",
        ]
        # stderr goes to a file rather than DEVNULL: discarding it is what made the buffering
        # failure above undiagnosable from the job log.
        err = None
        try:
            if self._stderr_path is not None:
                err = self._stderr_path.open("wb")
            master, slave = pty.openpty()
        except OSError:
            if err is not None:
                err.close()
            return False
        try:
            self._proc = subprocess.Popen(  # noqa: S603 — fixed argv, no shell
                argv, stdout=slave, stderr=err or subprocess.DEVNULL, close_fds=True
            )
        except OSError:
            os.close(master)
            return False
        finally:
            os.close(slave)
            if err is not None:
                err.close()
        self._stdout = os.fdopen(master, "r", errors="replace", newline="")
        self._stream_since = time.monotonic()
        threading.Thread(target=self._read, daemon=True).start()
        return True

    def _read(self) -> None:
        if self._stdout is None:
            return
        try:
            for line in self._stdout:
                self._lines.append(line)
        except (OSError, ValueError):
            pass  # the pty closes with EIO when the child goes, incl. teardown killing the group

    def _drain_stream(self) -> list[Tick]:
        ticks: list[Tick] = []
        while True:
            try:
                line = self._lines.popleft()
            except IndexError:
                break
            tick = self._assembler.feed(line)
            if tick is not None:
                ticks.append(tick)
        return ticks

    def read_window(self) -> tuple[list[Tick], Optional[str]]:
        """Ticks for the window just ended, plus a one-off note when the mode changed."""
        if self.mode == MODE_NONE:
            return [], None
        if self.mode == MODE_ONESHOT:
            return _oneshot(), None
        ticks = self._drain_stream()
        if ticks:
            self._dry = 0
            return ticks, None
        self._dry += 1
        waited = time.monotonic() - self._stream_since
        if self._dry < _STREAM_DRY_LIMIT or waited < _stream_grace_s(self._loop_ms):
            return [], None
        self.stop()
        self.mode = MODE_ONESHOT
        return _oneshot(), (
            f"nvidia-smi --loop-ms delivered nothing in {waited:.0f}s"
            f"{self._stderr_hint()} — falling back to one sample per window"
        )

    def _stderr_hint(self) -> str:
        if self._stderr_path is None or not self._stderr_path.exists():
            return ""
        first = self._stderr_path.read_text(errors="replace").strip().splitlines()
        return f" ({first[0]})" if first else ""

    def stop(self) -> None:
        """Belt and braces: the child shares our process group, so teardown's `kill -- -PID`
        already reaps it. This covers the ordinary-exit path."""
        proc, self._proc = self._proc, None
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except (subprocess.SubprocessError, OSError):
                pass
        if self._stdout is not None:
            try:
                self._stdout.close()
            except OSError:
                pass
            self._stdout = None


# --------------------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------------------- #


def sample(
    tsv: Path,
    base_url: str,
    model: str,
    interval: float,
    *,
    util_ms: int = 1000,
    once: bool = False,
) -> None:
    """Append one row per interval until killed (or one row, with ``once``).

    Each row is flushed immediately: ``teardown`` kills this process, so anything still buffered
    would be lost — and the last samples before the kill are the ones describing how the job
    ended.
    """
    tsv.parent.mkdir(parents=True, exist_ok=True)
    new = not tsv.exists() or tsv.stat().st_size == 0
    started = time.monotonic()
    source = UtilSource(util_ms, stderr_path=tsv.with_name("nvidia-smi.err"))
    # Said once, on stderr, not per sample. It has to be visible: when this was buried in
    # ollama.log an entire run recorded no GPU data with nothing in the job log to say why.
    _note(f"GPU utilisation source: {source.start()}")
    try:
        with tsv.open("a", encoding="utf-8") as fh:
            if new:
                fh.write("\t".join(COLUMNS) + "\n")
                fh.flush()
            while True:
                # Read first so the window ends at the /api/ps read, not somewhere inside it.
                ticks, note = source.read_window()
                if note:
                    _note(note)
                util = aggregate(ticks)
                state, size, vram = read_api_ps(base_url, model)
                fh.write(
                    "\t".join(
                        "" if v is None else str(v)
                        for v in (
                            int(time.monotonic() - started),
                            state,
                            size,
                            vram,
                            util.total_mib,
                            util.used_mib,
                            util.free_mib,
                            util.sm_min,
                            util.sm_mean,
                            util.sm_max,
                            util.membw_max,
                            util.ticks,
                        )
                    )
                    + "\n"
                )
                fh.flush()
                if once:
                    return
                time.sleep(interval)
    finally:
        source.stop()


# --------------------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------------------- #


class Row(NamedTuple):
    offset_s: int
    state: str
    size: int
    size_vram: int
    total_mib: Optional[int]
    used_mib: Optional[int]
    free_mib: Optional[int]
    sm_min: Optional[int]
    sm_mean: Optional[int]
    sm_max: Optional[int]
    membw_max: Optional[int]
    util_ticks: int


def parse_rows(lines: Iterable[str]) -> Iterator[Row]:
    """Yield well-formed rows, skipping anything else.

    Malformed rows are expected, not exceptional: the sampler is killed by ``teardown``, so the
    final line can be a partial write. A summary that raised on that would be missing exactly
    when it is wanted.
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
            *(_opt_int(p) for p in parts[4:11]),
            _opt_int(parts[11]) or 0,
        )


def _gib(byte_count: int) -> str:
    return f"{byte_count / 2**30:.1f} GiB"


def _fmt_offset(seconds: int) -> str:
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


def _sm_range(rows: list[Row]) -> str:
    sm_max = [r.sm_max for r in rows if r.sm_max is not None]
    sm_min = [r.sm_min for r in rows if r.sm_min is not None]
    sm_mean = [r.sm_mean for r in rows if r.sm_mean is not None]
    if not sm_max:
        return "sm not measured"
    mean = round(sum(sm_mean) / len(sm_mean)) if sm_mean else 0
    return f"sm {min(sm_min)}-{max(sm_max)}% (mean {mean}%)"


class Segment(NamedTuple):
    rows: list[Row]

    @property
    def state(self) -> str:
        return self.rows[0].state


def segments(rows: list[Row]) -> list[Segment]:
    """Split into runs of constant state — one line each in the timeline."""
    out: list[Segment] = []
    for row in rows:
        if out and out[-1].state == row.state:
            out[-1].rows.append(row)
        else:
            out.append(Segment([row]))
    return out


def format_timeline(rows: list[Row]) -> list[str]:
    """Collapse consecutive same-state samples to one line each.

    This is the readable answer and stays short however long the job ran, which is what lets the
    raw record below it be downsampled without losing the shape.
    """
    lines = []
    for seg in segments(rows):
        span = (
            f"{_fmt_offset(seg.rows[0].offset_s)}-{_fmt_offset(seg.rows[-1].offset_s)}"
        )
        peak_vram = max(r.size_vram for r in seg.rows)
        vram = _gib(peak_vram) if peak_vram else "—"
        lines.append(
            f"  {span:<15} {seg.state:<12} {len(seg.rows):>4} smp  "
            f"vram {vram:<10} {_sm_range(seg.rows)}"
        )
    return lines


def format_record(tsv: Path, max_rows: int) -> list[str]:
    """The record itself, verbatim TSV including the header.

    Verbatim rather than padded into aligned columns so it round-trips: piped back through
    --report it reproduces the same summary, and it pastes into a spreadsheet.

    Above ``max_rows`` data rows it keeps every ``ceil(n/max)``-th plus the last one and says so.
    The timeline above is always complete, so only resolution is lost, never shape. ``0`` means
    unlimited.
    """
    raw = tsv.read_text(encoding="utf-8", errors="replace").splitlines()
    if not raw:
        return []
    header, data = raw[0], raw[1:]
    note = ""
    if max_rows and len(data) > max_rows:
        stride = math.ceil(len(data) / max_rows)
        kept = data[::stride]
        if data and kept[-1] != data[-1]:
            kept.append(data[-1])
        note = f" — downsampled, 1 in {stride} of {len(data)} rows"
        data = kept
    return [
        f"---- accelerator record ({len(data)} rows{note}) ----",
        f"  | {header}",
        *(f"  | {line}" for line in data),
        "---- end ----",
    ]


def report(tsv: Path, dump_max: Optional[int] = None) -> str:
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

    lines = [f"accelerator over the job: {total} samples spanning {_fmt_offset(span)}"]
    for state in STATE_ORDER:
        if counts[state]:
            lines.append(
                f"  {state:<12} {counts[state]:>5}  {100 * counts[state] // total:>3}%"
            )

    # The transitions are the point of sampling at all: "gpu-full -> unloaded -> cpu" is a
    # different bug report from "cpu" throughout.
    transitions = [seg.state for seg in segments(rows)]
    first_non_gpu = next((r for r in rows if not r.state.startswith("gpu-")), None)
    if len(transitions) > 1:
        lines.append(f"  transitions: {' -> '.join(transitions)}")
    if first_non_gpu is not None:
        lines.append(
            f"  first non-GPU sample: {first_non_gpu.state} at "
            f"+{_fmt_offset(first_non_gpu.offset_s)}"
        )

    on_gpu = counts[GPU_FULL] + counts[GPU_PARTIAL]
    sm_max = max((r.sm_max for r in rows if r.sm_max is not None), default=None)
    if on_gpu == 0:
        lines.append("  verdict: the GPU was NEVER used by ollama during this job")
    elif sm_max is not None and sm_max <= IDLE_SM_PCT:
        # The gap this file exists to close: residency is not compute. Reported for any
        # non-zero GPU residency, because "the weights are on the card" is exactly the finding
        # that reads as success when it isn't.
        lines.append(
            f"  verdict: resident on the GPU but NO COMPUTE observed"
            f" (sm never exceeded {sm_max}%)"
        )
    elif on_gpu == total:
        lines.append("  verdict: ollama was on the GPU for every sample")
    else:
        lines.append(
            f"  verdict: ollama was on the GPU for {100 * on_gpu // total}% of samples"
        )
    lines.append(f"  compute: {_sm_range(rows)} — device-wide, all processes")

    # VRAM envelope: distinguishes "card too small for the model" from "someone else had it".
    frees = [r.free_mib for r in rows if r.free_mib is not None]
    totals = [r.total_mib for r in rows if r.total_mib is not None]
    if frees and totals:
        lines.append(
            f"  device VRAM: min free {min(frees)} MiB of {max(totals)} MiB total"
            f" (max used {max(r.used_mib or 0 for r in rows)} MiB, all processes)"
        )
    else:
        # Deliberately no cause: this used to read "(no nvidia-smi in the container)" and said
        # so on a node that had one — the columns were empty because the sampler's stream was
        # buffered, not because the tool was missing. The reason belongs to whoever recorded it.
        lines.append(
            "  device VRAM: not recorded — no nvidia-smi samples"
            " (see the accel-sampler lines in this log)"
        )

    biggest = max(rows, key=lambda r: r.size)
    if biggest.size:
        lines.append(
            f"  model footprint: {_gib(biggest.size)} total,"
            f" peak {_gib(max(r.size_vram for r in rows))} in VRAM"
        )

    lines.append("timeline:")
    lines.extend(format_timeline(rows))
    if dump_max is not None:
        lines.extend(format_record(tsv, dump_max))
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
        "--interval", type=float, default=15.0, help="seconds between /api/ps samples"
    )
    parser.add_argument(
        "--util-ms", type=int, default=1000, help="nvidia-smi stream period, ms"
    )
    parser.add_argument(
        "--dump-max",
        type=int,
        default=None,
        help="--report: also print the record, at most N rows (0 = unlimited)",
    )
    parser.add_argument(
        "--once", action="store_true", help="take a single sample and exit"
    )
    args = parser.parse_args(argv)

    if args.report:
        print(report(args.tsv, dump_max=args.dump_max))
        return 0

    if not args.base_url or not args.model:
        parser.error("--sample needs --base-url and --model")
    try:
        sample(
            args.tsv,
            args.base_url,
            args.model,
            args.interval,
            util_ms=args.util_ms,
            once=args.once,
        )
    except KeyboardInterrupt:  # pragma: no cover — teardown sends the signal
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
