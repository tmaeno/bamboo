"""`bamboo serve-mattermost` — run the Mattermost bot daemon.

A long-running service that fronts bamboo for the ops team over Mattermost.
Requires the ``bamboo[mattermost]`` extra and ``MATTERMOST_URL`` /
``MATTERMOST_TOKEN`` / ``MATTERMOST_ALLOWED_CHANNELS`` configuration.
"""

from __future__ import annotations

import asyncio
import logging

import click

from bamboo.utils.logging import setup_logging

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    help=(
        "Verbose SERVER-SIDE logging: set all bamboo loggers to DEBUG (like LOG_LEVEL=DEBUG), "
        "for the whole bot. Affects the server console/log only — to surface a command's "
        "behind-the-scenes detail in its Mattermost reply, use that command's own --verbose."
    ),
)
def main(verbose: bool) -> None:
    """Run the Mattermost bot daemon (ops-facing chat frontend)."""
    setup_logging()
    if verbose:
        logging.getLogger("bamboo").setLevel(logging.DEBUG)

    # Import lazily so `bamboo --help` and other commands don't require the
    # optional Mattermost client to be installed.
    from bamboo.frontends.mattermost.serve import serve

    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        click.echo("\nMattermost bot stopped.")


if __name__ == "__main__":
    main()
