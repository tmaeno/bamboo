"""Stream engine progress narration into a Mattermost thread.

The reasoning/analysis engines report progress through
:mod:`bamboo.utils.narrator` (``say`` / ``show_block`` / ``thinking``).  In the
CLI that goes to a Rich console; in the bot we install a
:class:`MattermostNarrationSink` (per session, via the narrator's ``ContextVar``)
that turns those events into a **single** live-updating thread post:

* a head line carrying the current step (``:bamboo_spinner: 🔎 <step>``) — the
  ``:bamboo_spinner:`` animated custom emoji (registered once at bot startup) spins
  client-side on MM's inline text path, so we only edit the text when the step
  changes; on completion the head is frozen to a static "✓ done". If the emoji
  isn't available the head degrades to a plain ``🔎 <step>``;
* below the head, a fenced code block of the last *N* narration lines (Mattermost
  folds it under "Show more" when tall).

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

# All narration logging (firehose + operational diagnostics) goes to one logger so
# it shows together in the bot's output (the module logger name differs and is
# easy to filter out by accident).
_narration_log = logging.getLogger("bamboo.narration")

_DETAIL_LINES = 5  # most-recent narration lines kept in the MM detail post
_LINE_CLIP = 200  # max chars per detail line (full text still goes to the log)
_FLUSH_INTERVAL = 1.0  # min seconds between Mattermost API edits (coalescing)


def _clip(text: str, limit: int = _LINE_CLIP) -> str:
    text = " ".join((text or "").split())  # collapse whitespace/newlines
    return text if len(text) <= limit else text[: limit - 1] + "…"


class MattermostNarrationSink:
    """Thread-safe narrator sink that mirrors progress into a single MM post.

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
        self._dirty = False
        # Animated spinner custom-emoji name if the bot registered one, else None
        # (renders on MM's inline text path; falls back to a static glyph).
        self._spinner: Optional[str] = getattr(bot, "spinner_emoji", None)
        # Post id, set lazily by the flusher; read by finalize().
        self._post_id: Optional[str] = None
        self._started_at: float = 0.0

    # --- NarrationSink (called from engine code, possibly off-loop) ---
    def say(self, msg: str) -> None:
        _narration_log.info("→ %s", msg)
        with self._lock:
            self._detail.append(f"→ {_clip(msg)}")
            self._dirty = True
        self._signal()

    def block(self, title: str, content: str) -> None:
        _narration_log.info("[%s]\n%s", title, content)
        first = (content or "").strip().splitlines()
        head = first[0] if first else ""
        with self._lock:
            self._detail.append(f"▎{_clip(title, 80)}: {_clip(head)}")
            self._dirty = True
        self._signal()

    def step(self, label: str) -> None:
        _narration_log.info("%s", label)
        with self._lock:
            self._latest_step = _clip(label, 160)
            self._dirty = True
        self._signal()

    def _signal(self) -> None:
        # Wake the flusher from any thread.
        try:
            self._loop.call_soon_threadsafe(self._wake.set)
        except RuntimeError:  # loop closed/closing
            pass

    # --- driven by stream_narration ---
    async def run(self, started_at: float) -> None:
        """Flusher loop: coalesce events into throttled edits to the one post."""
        self._started_at = started_at
        last_flush = 0.0
        _narration_log.info("narration: flusher started")
        while True:
            await self._wake.wait()
            self._wake.clear()
            # Throttle: never edit more than once per _FLUSH_INTERVAL.
            delta = time.monotonic() - last_flush
            if delta < _FLUSH_INTERVAL:
                await asyncio.sleep(_FLUSH_INTERVAL - delta)
            await self._flush_once()
            last_flush = time.monotonic()

    def _compose(self, step: str, detail_lines: list[str], *, done: bool = False, elapsed: int = 0) -> str:
        """Build the single post body: a head line + a folded code block of detail."""
        if done:
            head = f"✓ done ({elapsed}s)"
        elif self._spinner:
            # Animated custom emoji — renders + spins on MM's inline text path.
            head = f":{self._spinner}: 🔎 {step}"
        else:
            head = f"🔎 {step}"
        if not detail_lines:
            return head
        return head + "\n```\n" + "\n".join(detail_lines) + "\n```"

    async def _flush_once(self) -> None:
        """Push the current snapshot to MM as one post (create on first use)."""
        with self._lock:
            step = self._latest_step
            detail_lines = list(self._detail)
            dirty, self._dirty = self._dirty, False
        if not dirty:
            return
        msg = self._compose(step, detail_lines)
        if self._post_id is None:
            self._post_id = await self._create(msg)
        else:
            await self._patch(self._post_id, message=msg)

    async def finalize(self) -> None:
        """Freeze the post to a static '✓ done' (drops the spinner emoji)."""
        with self._lock:
            step = self._latest_step
            detail_lines = list(self._detail)
        if self._post_id is None and not detail_lines:
            return  # nothing was ever streamed
        elapsed = max(0, int(time.monotonic() - (self._started_at or time.monotonic())))
        msg = self._compose(step, detail_lines, done=True, elapsed=elapsed)
        if self._post_id is None:
            # Streaming produced detail but never flushed a post; create it now.
            self._post_id = await self._create(msg)
            return
        await self._patch(self._post_id, message=msg)

    # --- MM I/O helpers (best-effort) ---
    async def _create(self, message: str) -> Optional[str]:
        try:
            post = await self._bot.create_post(
                self._channel_id, message, root_id=self._root_id
            )
            pid = post.get("id") if isinstance(post, dict) else None
            _narration_log.info("narration: created post id=%s", pid)
            return pid
        except Exception as exc:  # noqa: BLE001
            _narration_log.warning("narration: create_post failed (%s)", exc)
            return None

    async def _patch(self, post_id: str, *, message: str) -> bool:
        try:
            await self._bot.update_post(post_id, message=message)
            return True
        except Exception as exc:  # noqa: BLE001
            _narration_log.warning("narration: update_post failed (%s)", exc)
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
