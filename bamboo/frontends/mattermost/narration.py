"""Stream engine progress narration into a Mattermost thread.

The reasoning/analysis engines report progress through
:mod:`bamboo.utils.narrator` (``say`` / ``show_block`` / ``thinking``).  In the
CLI that goes to a Rich console; in the bot we install a
:class:`MattermostNarrationSink` (per session, via the narrator's ``ContextVar``)
that turns those events into two live-updating thread posts:

* a **status** post carrying an animated spinner GIF (uploaded once) plus the
  current step — the GIF animates client-side, so we only edit the text when the
  step changes; on completion it's frozen to a static "✓ done";
* a **detail** post showing only the last *N* narration lines (Mattermost folds it
  with "Show more" when tall).

The *full* firehose is always logged to ``bamboo.narration`` (stdout/log file)
for later debugging — only a bounded view reaches chat.

Updates are coalesced by a per-session flusher task and throttled, so a long
analysis costs a handful of edits, not hundreds.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time
from collections import deque
from typing import Any, Optional

logger = logging.getLogger(__name__)
_narration_log = logging.getLogger("bamboo.narration")

_DETAIL_LINES = 5  # most-recent narration lines kept in the MM detail post
_LINE_CLIP = 200  # max chars per detail line (full text still goes to the log)
_FLUSH_INTERVAL = 1.0  # min seconds between Mattermost API edits (coalescing)


def _clip(text: str, limit: int = _LINE_CLIP) -> str:
    text = " ".join((text or "").split())  # collapse whitespace/newlines
    return text if len(text) <= limit else text[: limit - 1] + "…"


class MattermostNarrationSink:
    """Thread-safe narrator sink that mirrors progress into two MM posts.

    Sink methods may be called from worker threads (``asyncio.to_thread``); they
    only mutate guarded state + wake the flusher via ``call_soon_threadsafe``, so
    no MM I/O happens on the caller's thread.
    """

    def __init__(self, bot: Any, channel_id: str, root_id: str, loop: asyncio.AbstractEventLoop) -> None:
        self._bot = bot
        self._channel_id = channel_id
        self._root_id = root_id
        self._loop = loop
        self._lock = threading.Lock()
        self._wake = asyncio.Event()
        self._latest_step: str = "working…"
        self._detail: deque[str] = deque(maxlen=_DETAIL_LINES)
        self._status_dirty = False
        self._detail_dirty = False
        # Post ids + uploaded gif id, set lazily by the flusher; read by finalize().
        self._status_id: Optional[str] = None
        self._detail_id: Optional[str] = None
        self._gif_id: Optional[str] = None
        self._started_at: float = 0.0

    # --- NarrationSink (called from engine code, possibly off-loop) ---
    def say(self, msg: str) -> None:
        _narration_log.info("→ %s", msg)
        with self._lock:
            self._detail.append(f"→ {_clip(msg)}")
            self._detail_dirty = True
        self._signal()

    def block(self, title: str, content: str) -> None:
        _narration_log.info("[%s]\n%s", title, content)
        first = (content or "").strip().splitlines()
        head = first[0] if first else ""
        with self._lock:
            self._detail.append(f"▎{_clip(title, 80)}: {_clip(head)}")
            self._detail_dirty = True
        self._signal()

    def step(self, label: str) -> None:
        _narration_log.info("%s", label)
        with self._lock:
            self._latest_step = _clip(label, 160)
            self._status_dirty = True
        self._signal()

    def _signal(self) -> None:
        # Wake the flusher from any thread.
        try:
            self._loop.call_soon_threadsafe(self._wake.set)
        except RuntimeError:  # loop closed/closing
            pass

    # --- driven by stream_narration ---
    async def run(self, started_at: float) -> None:
        """Flusher loop: coalesce events into throttled status/detail edits."""
        self._started_at = started_at
        last_flush = 0.0
        while True:
            await self._wake.wait()
            self._wake.clear()
            # Throttle: never edit more than once per _FLUSH_INTERVAL.
            delta = time.monotonic() - last_flush
            if delta < _FLUSH_INTERVAL:
                await asyncio.sleep(_FLUSH_INTERVAL - delta)
            await self._flush_once()
            last_flush = time.monotonic()

    async def _flush_once(self) -> None:
        """Push the current status/detail snapshot to MM (create on first use)."""
        with self._lock:
            step = self._latest_step
            detail_lines = list(self._detail)
            status_dirty, self._status_dirty = self._status_dirty, False
            detail_dirty, self._detail_dirty = self._detail_dirty, False
        if status_dirty:
            if self._status_id is None:
                self._gif_id = await self._upload_spinner()
                self._status_id = await self._create(
                    f"🔎 {step}", file_ids=[self._gif_id] if self._gif_id else None
                )
            else:
                await self._patch(self._status_id, message=f"🔎 {step}")
        if detail_dirty and detail_lines:
            body = "```\n" + "\n".join(detail_lines) + "\n```"
            if self._detail_id is None:
                self._detail_id = await self._create(body)
            else:
                await self._patch(self._detail_id, message=body)

    async def finalize(self) -> None:
        """Freeze the status to a static '✓ done' and drop the spinner GIF."""
        with self._lock:
            detail_lines = list(self._detail)
        if self._status_id is None and not detail_lines:
            return  # nothing was ever streamed
        elapsed = max(0, int(time.monotonic() - (self._started_at or time.monotonic())))
        if self._status_id is not None:
            # Remove the GIF (best-effort) and show a static completion line.
            ok = await self._patch(
                self._status_id, message=f"✓ done ({elapsed}s)", file_ids=[]
            )
            if not ok and self._gif_id is not None:
                # Fallback: delete the spinner post; the result card stands alone.
                with contextlib.suppress(Exception):
                    await self._bot.delete_post(self._status_id)
        if self._detail_id is not None and detail_lines:
            body = "```\n" + "\n".join(detail_lines) + "\n```"
            await self._patch(self._detail_id, message=body)

    # --- MM I/O helpers (best-effort) ---
    async def _upload_spinner(self) -> Optional[str]:
        try:
            import importlib.resources as ir  # noqa: PLC0415

            data = (
                ir.files("bamboo.frontends.mattermost")
                .joinpath("assets/spinner.gif")
                .read_bytes()
            )
            return await self._bot.upload_file(self._channel_id, "spinner.gif", data)
        except Exception as exc:  # noqa: BLE001
            logger.warning("narration: spinner upload failed (%s); status will be text-only", exc)
            return None

    async def _create(self, message: str, *, file_ids: Optional[list[str]] = None) -> Optional[str]:
        try:
            post = await self._bot.create_post(
                self._channel_id, message, root_id=self._root_id, file_ids=file_ids
            )
            return post.get("id") if isinstance(post, dict) else None
        except Exception as exc:  # noqa: BLE001
            logger.warning("narration: create_post failed (%s)", exc)
            return None

    async def _patch(self, post_id: str, *, message: str, file_ids: Optional[list[str]] = None) -> bool:
        try:
            await self._bot.update_post(post_id, message=message, file_ids=file_ids)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("narration: update_post failed (%s)", exc)
            return False


@contextlib.asynccontextmanager
async def stream_narration(transport: Any):
    """Install a per-session narration sink that streams engine progress to MM.

    No-op for transports not backed by a bot (e.g. test fakes): narration stays
    silent, exactly as before.
    """
    from bamboo.utils.narrator import reset_narration_sink, set_narration_sink  # noqa: PLC0415

    bot = getattr(transport, "_bot", None)
    channel_id = getattr(transport, "channel_id", None)
    if bot is None or not channel_id:
        yield
        return

    loop = asyncio.get_running_loop()
    sink = MattermostNarrationSink(bot, channel_id, getattr(transport, "root_id", ""), loop)
    token = set_narration_sink(sink)
    started_at = time.monotonic()
    flusher = asyncio.ensure_future(sink.run(started_at))
    try:
        yield
    finally:
        reset_narration_sink(token)
        with contextlib.suppress(Exception):
            await sink.finalize()
        flusher.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await flusher
