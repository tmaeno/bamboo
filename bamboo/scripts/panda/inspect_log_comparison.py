"""Diagnose how compare_job_logs sizes up on a real (failed, successful) PandaID pair.

Usage:
    python -m bamboo.scripts.panda.inspect_log_comparison \\
        --failed-panda-id 7130072572 --successful-panda-id 7130288457

    # optionally dump the full comparison dict for offline inspection:
    python -m bamboo.scripts.panda.inspect_log_comparison \\
        --failed-panda-id 7130072572 --successful-panda-id 7130288457 \\
        --output /tmp/comparison.json
"""

import asyncio
import json
from pathlib import Path

import click

from bamboo.utils.log_filters import compare_job_logs
from bamboo.utils.panda_client import get_job_log


def _line_stats(label: str, text: str) -> None:
    lines = text.splitlines()
    if not lines:
        click.echo(f"{label}: empty")
        return
    lens = sorted(len(l) for l in lines)
    n = len(lens)
    click.echo(
        f"{label}: {len(text):>10,} chars, {n:>6,} lines, "
        f"max_line={lens[-1]:>7,}, "
        f"p99={lens[max(0, int(n * 0.99) - 1)]:>7,}, "
        f"p50={lens[n // 2]:>7,}, "
        f"mean={sum(lens) // n:>7,}"
    )


def _value_size(v) -> int:
    if isinstance(v, str):
        return len(v)
    if isinstance(v, (dict, list)):
        return len(json.dumps(v, default=str))
    return len(str(v))


@click.command()
@click.option("--failed-panda-id", type=int, required=True)
@click.option("--successful-panda-id", type=int, required=True)
@click.option(
    "--output",
    type=click.Path(),
    default=None,
    help="Optional path to dump the full comparison dict as JSON.",
)
def main(failed_panda_id, successful_panda_id, output):
    """Inspect compare_job_logs output for a specific pair of PandaIDs."""
    asyncio.run(_run(failed_panda_id, successful_panda_id, output))


async def _run(failed_pid: int, successful_pid: int, output: str | None) -> None:
    click.echo(f"Fetching failed-job log (PandaID={failed_pid})...")
    failed = await get_job_log(failed_pid) or ""
    click.echo(f"Fetching successful-job log (PandaID={successful_pid})...")
    successful = await get_job_log(successful_pid) or ""

    click.echo("\n--- Raw log line stats ---")
    _line_stats("failed    ", failed)
    _line_stats("successful", successful)

    click.echo("\nRunning compare_job_logs...")
    result = compare_job_logs(failed, successful)

    click.echo("\n--- Comparison output (per-field sizes) ---")
    total = 0
    for key, value in result.items():
        size = _value_size(value)
        total += size
        if isinstance(value, str):
            lines = value.splitlines()
            n_lines = len(lines)
            max_line = max((len(l) for l in lines), default=0) if lines else 0
            click.echo(
                f"  {key:>50}: {size:>10,} chars  "
                f"({n_lines} lines, max_line={max_line:,})"
            )
        else:
            click.echo(
                f"  {key:>50}: {value!r:>10}  ({type(value).__name__})"
            )
    click.echo(f"  {'TOTAL':>50}: {total:>10,} chars")

    click.echo("\n--- Alignment context ---")
    auf = int(result.get("aligned_until_failed_line", 0) or 0)
    aus = int(result.get("aligned_until_successful_line", 0) or 0)
    fl = failed.splitlines()
    sl = successful.splitlines()
    click.echo(
        f"Failed:     aligned through line {auf:,}/{len(fl):,} "
        f"({auf / max(len(fl), 1) * 100:.1f}% of failed)"
    )
    click.echo(
        f"Successful: aligned through line {aus:,}/{len(sl):,} "
        f"({aus / max(len(sl), 1) * 100:.1f}% of successful)"
    )
    click.echo(
        f"Past alignment: {len(fl) - auf:,} unique lines in failed, "
        f"{len(sl) - aus:,} unique lines in successful"
    )

    _CTX_PAST = 20
    _CTX_BEFORE = 10
    _LINE_CAP = 200

    def _fmt_line(idx: int, text: str) -> str:
        snippet = text[:_LINE_CAP] + ("..." if len(text) > _LINE_CAP else "")
        return f"  {idx:>6}: {snippet}"

    def _excerpt_past(label: str, lines: list[str], start: int) -> None:
        end = min(len(lines), start + _CTX_PAST)
        body = "\n".join(_fmt_line(start + i, l) for i, l in enumerate(lines[start:end]))
        click.echo(f"\n[{label}] lines {start:,}..{end:,}")
        click.echo(body or "  (none)")

    def _excerpt_before(label: str, lines: list[str], end: int) -> None:
        start = max(0, end - _CTX_BEFORE)
        body = "\n".join(_fmt_line(start + i, l) for i, l in enumerate(lines[start:end]))
        click.echo(f"\n[{label}] last aligned lines {start:,}..{end:,}")
        click.echo(body or "  (none)")

    _excerpt_before("FAILED last aligned",     fl, auf)
    _excerpt_before("SUCCESSFUL last aligned", sl, aus)
    _excerpt_past("FAILED past alignment",     fl, auf)
    _excerpt_past("SUCCESSFUL past alignment", sl, aus)

    if output:
        Path(output).write_text(json.dumps(result, indent=2, default=str))
        click.echo(f"\nFull comparison dict written to {output}")


if __name__ == "__main__":
    main()
