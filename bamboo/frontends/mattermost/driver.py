"""Lazy Mattermost client factory.

The third-party Mattermost client (``mattermostautodriver``) lives behind the
``bamboo[mattermost]`` extra.  Importing this module is cheap and safe without
the extra; the client is only imported when :func:`build_async_driver` is
actually called, which then raises a clear, actionable error if the extra is
missing.
"""

from __future__ import annotations

import logging
from typing import Any, Optional
from urllib.parse import urlparse

from bamboo.config import Settings, get_settings

logger = logging.getLogger(__name__)

_MISSING_MSG = (
    "The Mattermost frontend requires the optional client. Install it with:\n"
    "    pip install 'bamboo[mattermost]'"
)


def _driver_options(settings: Settings) -> dict[str, Any]:
    """Translate :class:`Settings` into mattermostautodriver options.

    Parses ``MATTERMOST_URL`` (e.g. ``https://mattermost.cern.ch`` or
    ``https://mattermost.cern.ch:8065``) into the scheme/host/port the client
    expects.  Defaults: ``https`` on port 443.
    """
    if not settings.mattermost_url:
        raise ValueError("MATTERMOST_URL is not configured.")
    if not settings.mattermost_token:
        raise ValueError("MATTERMOST_TOKEN is not configured.")

    parsed = urlparse(settings.mattermost_url)
    # Tolerate a bare host with no scheme (urlparse puts it in .path).
    host = parsed.hostname or parsed.path or settings.mattermost_url
    scheme = parsed.scheme or "https"
    port = parsed.port or (443 if scheme == "https" else 80)
    return {
        "url": host,
        "scheme": scheme,
        "port": port,
        "token": settings.mattermost_token,
        "basepath": "/api/v4",
    }


def build_async_driver(settings: Optional[Settings] = None) -> Any:
    """Construct an unconnected ``mattermostautodriver.AsyncDriver``.

    Raises:
        ImportError: If the ``bamboo[mattermost]`` extra is not installed.
        ValueError:  If ``MATTERMOST_URL`` / ``MATTERMOST_TOKEN`` are unset.
    """
    settings = settings or get_settings()
    options = _driver_options(settings)
    try:
        from mattermostautodriver import AsyncDriver  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised only without extra
        raise ImportError(_MISSING_MSG) from exc
    return AsyncDriver(options)


async def create_post(
    driver: Any,
    *,
    channel_id: str,
    message: str = "",
    root_id: str = "",
    props: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Create a post via the driver, tolerating both client API layouts.

    mattermostautodriver groups endpoints (``driver.posts.create_post``); older
    layouts expose a flat ``driver.create_post``.  We try the grouped form first.
    """
    options: dict[str, Any] = {"channel_id": channel_id, "message": message}
    if root_id:
        options["root_id"] = root_id
    if props:
        options["props"] = props

    posts = getattr(driver, "posts", None)
    if posts is not None and hasattr(posts, "create_post"):
        return await posts.create_post(options)
    return await driver.create_post(options)
