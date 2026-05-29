"""Phase 1 wedge — post an :class:`AnalysisResult` to a Mattermost channel.

A read-only step that never touches the investigation loop: it renders the
analysis as a message attachment (see :mod:`bamboo.frontends.mattermost.render`)
and posts it.  The message-building is pure and unit-tested; the network call is
isolated behind an injectable ``post`` callable so the poster can be tested
without the client or a live server.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Optional

from bamboo.config import Settings, get_settings
from bamboo.frontends.mattermost import render

logger = logging.getLogger(__name__)

# A post backend: (channel_id, message, props) -> awaitable[response dict].
PostFn = Callable[[str, str, dict[str, Any]], Awaitable[dict[str, Any]]]


async def post_analysis(
    channel_id: str,
    result: Any,
    *,
    post: Optional[PostFn] = None,
    settings: Optional[Settings] = None,
) -> dict[str, Any]:
    """Post an analysis result to *channel_id* as a Mattermost attachment.

    Args:
        channel_id: Target Mattermost channel ID.
        result:     An ``AnalysisResult`` (or compatible object).
        post:       Optional backend coroutine ``(channel_id, message, props)``;
                    defaults to building a driver from settings and posting.
                    Injected by tests.
        settings:   Optional settings override (defaults to :func:`get_settings`).

    Returns:
        The created-post response dict from the backend.
    """
    props = render.analysis_message(result)
    backend = post or _default_post(settings)
    logger.info("Posting analysis for task %s to channel %s", getattr(result, "task_id", "?"), channel_id)
    return await backend(channel_id, "", props)


def _default_post(settings: Optional[Settings]) -> PostFn:
    """Build the default post backend that talks to a real Mattermost driver."""
    from bamboo.frontends.mattermost import driver as _driver  # local: keep import lazy

    resolved = settings or get_settings()

    async def _post(channel_id: str, message: str, props: dict[str, Any]) -> dict[str, Any]:
        drv = _driver.build_async_driver(resolved)
        return await _driver.create_post(
            drv, channel_id=channel_id, message=message, props=props
        )

    return _post
