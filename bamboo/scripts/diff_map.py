"""``bamboo diff-map`` — compare two builds of a Code Map.

The gates check the map against the source it was built from and against the
logs the deployment writes.  A moved threshold escapes both: a comparison with
the source agrees with whatever the source now says, and no condition is
echoed to a log.  So a map built from the wrong release explains a decision
with a number that has since changed, and sounds convincing doing it.

Comparing two builds is the only thing that surfaces that, which is why this
is a separate command and not a mode of ``check-map``: it needs no production
data at all, and must stay runnable when ``check-map`` cannot run.

**The other direction: one source, two versions of this code.**  The question
above is what the release changed.  A round that edits the extractor asks the
opposite -- PanDA stands still and the map moves anyway, so every difference
is something this repository did, and "the map is unchanged" stops being
available as the gate.  The two builds cannot exist at once, because they need
two checkouts of *this* repository rather than of PanDA, so one of them is
written down.

Usage::

    # Installed distribution against a checkout
    bamboo diff-map --against /path/to/panda

    # Two checkouts
    bamboo diff-map --source-root /path/to/old --against /path/to/new

    # One source, two versions of this code
    bamboo diff-map --dump before.json        # on the old code
    bamboo diff-map --baseline before.json    # on the new code
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import click

from bamboo.codemap import diff
from bamboo.codemap.factory import available_map_ids, get_code_map_plugin
from bamboo.codemap.models import MapFragment

logger = logging.getLogger(__name__)


def _build(map_id: str, source_root: Optional[Path]) -> MapFragment:
    plugin = get_code_map_plugin(map_id)
    plugin.prepare(source_root)
    return plugin.run()


def _dump(fragment: MapFragment, path: Path) -> None:
    """Write *fragment* where a later build of other code can read it back.

    The whole fragment, not a summary of it: a baseline is only worth having
    if the comparison it feeds is the same one two live builds would get, and
    a field left out here reads afterwards as a difference the round made.
    """
    path.write_text(fragment.model_dump_json())


def _load(path: Path) -> MapFragment:
    try:
        return MapFragment.model_validate_json(path.read_text())
    except (OSError, ValueError) as exc:
        raise click.ClickException(f"{path} is not a dumped map fragment: {exc}") from exc


def _report(result: diff.MapDiff, top: int) -> None:
    click.echo(f"\nold: {result.old_version}")
    click.echo(f"new: {result.new_version}")

    if result.old_version == result.new_version:
        # Two builds stamped the same way are the same release, so any
        # difference below is the extractor disagreeing with itself.
        click.echo(
            "\nnote: both builds carry the same version stamp, so differences "
            "here are not release drift."
        )

    # Changes are counted twice over: one node can differ in several fields,
    # and a reader comparing "changed" with "unchanged" needs both numbers to
    # be about nodes.
    touched = {change.node.split(" → ")[0] for change in result.changes}
    click.echo(
        f"\nnodes: unchanged {result.unchanged}, moved {len(result.moved)}, "
        f"changed {len(touched)}, added {len(result.added)}, "
        f"removed {len(result.removed)}"
    )
    if result.changes:
        click.echo(f"       {len(result.changes)} difference(s) across those {len(touched)}")

    drift = result.drift()
    if drift:
        # The headline: what no other check can see.  A strategy built on one
        # of these would explain a decision with the wrong reason.
        click.echo(f"\ncondition drift ({len(drift)}) — the gates cannot see these:")
        for change in drift[:top]:
            click.echo(f"    {change.render()}")
        if len(drift) > top:
            click.echo(f"    … {len(drift) - top} more")

    other = result.other()
    if other:
        click.echo(f"\nother changes ({len(other)}):")
        for change in other[:top]:
            click.echo(f"    {change.render()}")
        if len(other) > top:
            click.echo(f"    … {len(other) - top} more")

    for label, names in (("added", result.added), ("removed", result.removed)):
        if names:
            click.echo(f"\n{label} ({len(names)}):")
            for name in names[:top]:
                click.echo(f"  {name}")
            if len(names) > top:
                click.echo(f"  … {len(names) - top} more")

    if result.moved:
        # Reported as a count with a sample: every one of these is a node a
        # file-and-line key would have lost or duplicated, so the number is
        # the evidence for identifying nodes by signature.
        click.echo(f"\nmoved but unchanged ({len(result.moved)}):")
        for name in result.moved[:3]:
            click.echo(f"  {name}")
        if len(result.moved) > 3:
            click.echo(f"  … {len(result.moved) - 3} more")

    if result.identical:
        click.echo("\nThe two builds describe the same map.")


@click.command("diff-map")
@click.option("--map-id", default="panda", show_default=True, help="Which Code Map to compare.")
@click.option(
    "--source-root",
    default=None,
    metavar="PATH",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help=(
        "The source to build from.  Defaults to the installed distribution.  "
        "With --against it is the older side."
    ),
)
@click.option(
    "--against",
    default=None,
    metavar="PATH",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="The newer build's source.",
)
@click.option(
    "--baseline",
    default=None,
    metavar="FILE",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="A fragment written by --dump, standing in for the older build.",
)
@click.option(
    "--dump",
    "dump_to",
    default=None,
    metavar="FILE",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    help="Write this run's build there, for a later run to read with --baseline.",
)
@click.option(
    "--strict",
    is_flag=True,
    help=(
        "Exit non-zero if the builds differ.  Off by default: a difference "
        "between releases is expected, and what matters is reading it, not "
        "failing on it."
    ),
)
@click.option("--top", default=10, show_default=True, help="Rows per listing.")
@click.option("-v", "--verbose", is_flag=True, help="DEBUG logging.")
def main(
    map_id: str,
    source_root: Optional[Path],
    against: Optional[Path],
    baseline: Optional[Path],
    dump_to: Optional[Path],
    strict: bool,
    top: int,
    verbose: bool,
) -> None:
    """Compare two builds of a Code Map and report what the code changed."""
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)

    if against is not None and baseline is not None:
        # Both name the older side, so honouring both would mean picking one
        # and reporting the result as though the other had been read.
        raise click.UsageError("--baseline and --against name the same side; pass one.")
    if against is None and baseline is None and dump_to is None:
        # Without this the command would build the same source twice with the
        # same code and report no difference -- which is true, and says
        # nothing about anything.
        raise click.UsageError("pass --against, --baseline, or --dump.")

    try:
        get_code_map_plugin(map_id)
    except ValueError as exc:
        raise click.ClickException(
            f"{exc}  Available maps: {', '.join(available_map_ids())}"
        ) from exc

    try:
        # The side this run builds.  With --against that is the newer source;
        # otherwise it is this code's reading of the source it already has.
        new = _build(map_id, against if against is not None else source_root)
        old = _load(baseline) if baseline is not None else (
            _build(map_id, source_root) if against is not None else None
        )
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc

    if dump_to is not None:
        _dump(new, dump_to)
        click.echo(f"wrote {dump_to}")

    if old is None:
        return

    result = diff.compare(old, new)
    _report(result, top)

    if strict and not result.identical:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
