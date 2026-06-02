"""Stream engine progress narration into a Mattermost thread.

The reasoning/analysis engines report progress through
:mod:`bamboo.utils.narrator` (``say`` / ``show_block`` / ``thinking``).  In the
CLI that goes to a Rich console; in the bot we install a
:class:`MattermostNarrationSink` (per session, via the narrator's ``ContextVar``)
that mirrors progress into a **single live-updating post**:

* a head line carrying the current step (``:bamboo_spinner: 🔎 <step>``; the
  animated custom emoji spins client-side on MM's inline text path, frozen to
  ``✓ done`` at the end);
* below it, a **monospace code block** of the last *N* ``say()`` lines, each
  prefixed with a wall-clock timestamp — it reads like the bot's console log.

``show_block()`` content (LLM prompts, log dumps, code) is **not** sent to MM —
it's debugging detail that goes only to the ``bamboo.narration`` log.  The full
firehose (say / block / step) is always logged there for later inspection.

All Mattermost I/O happens in the per-session flusher task (coalesced/throttled);
the sink methods (which may run on ``asyncio.to_thread`` workers) only mutate
guarded state and wake the flusher.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time
from collections import deque
from datetime import datetime
from typing import Any, Optional

# All narration logging goes to one logger.  The say/block/step *firehose* is
# emitted here at INFO (this is the console log operators read); the sink's own
# post-plumbing diagnostics are at DEBUG so they don't pollute that stream.
_narration_log = logging.getLogger("bamboo.narration")

_DETAIL_ENTRIES = 5  # most-recent say-lines kept in the live post (sliding window)
_LINE_CLIP = 200  # max chars per say line (full text still goes to the log)
_FLUSH_INTERVAL = 1.0  # min seconds between Mattermost API edits (coalescing)


def _clip(text: str, limit: int = _LINE_CLIP) -> str:
    text = " ".join((text or "").split())  # collapse whitespace/newlines
    return text if len(text) <= limit else text[: limit - 1] + "…"


class MattermostNarrationSink:
    """Thread-safe narrator sink that mirrors progress into one MM post.

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
        # Live sliding window of console-style say lines (already timestamped).
        self._lines: deque[str] = deque(maxlen=_DETAIL_ENTRIES)
        self._dirty = False
        self._streamed = False
        # Animated spinner custom-emoji name if the bot registered one, else None
        # (renders on MM's inline text path; falls back to a static glyph).
        self._spinner: Optional[str] = getattr(bot, "spinner_emoji", None)
        # Live post id, set lazily by the flusher; read by finalize().
        self._post_id: Optional[str] = None
        self._started_at: float = 0.0

    # --- NarrationSink (called from engine code, possibly off-loop) ---
    def say(self, msg: str) -> None:
        _narration_log.info("→ %s", msg)
        line = f"{datetime.now().strftime('%H:%M:%S')}  → {_clip(msg)}"
        with self._lock:
            self._lines.append(line)
            self._dirty = True
            self._streamed = True
        self._signal()

    def block(self, title: str, content: str) -> None:
        # Debugging detail: log only — never sent to MM.
        _narration_log.info("[%s]\n%s", title, content)

    def step(self, label: str) -> None:
        _narration_log.info("%s", label)
        with self._lock:
            self._latest_step = _clip(label, 160)
            self._dirty = True
            self._streamed = True
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
        _narration_log.debug("narration: flusher started")
        while True:
            await self._wake.wait()
            self._wake.clear()
            # Throttle: never edit more than once per _FLUSH_INTERVAL.
            delta = time.monotonic() - last_flush
            if delta < _FLUSH_INTERVAL:
                await asyncio.sleep(_FLUSH_INTERVAL - delta)
            await self._flush_once()
            last_flush = time.monotonic()

    def _compose(self, lines: list[str], *, done: bool = False, elapsed: int = 0) -> str:
        """Build the post body: a head line + a monospace log of recent lines."""
        if done:
            head = f"✓ done ({elapsed}s)"
        elif self._spinner:
            # Animated custom emoji — renders + spins on MM's inline text path.
            head = f":{self._spinner}: 🔎 {self._latest_step}"
        else:
            head = f"🔎 {self._latest_step}"
        if not lines:
            return head
        return head + "\n```\n" + "\n".join(lines) + "\n```"

    async def _flush_once(self) -> None:
        """Push the current snapshot to MM as one post (create on first use)."""
        with self._lock:
            lines = list(self._lines)
            dirty, self._dirty = self._dirty, False
        if not dirty:
            return
        msg = self._compose(lines)
        if self._post_id is None:
            self._post_id = await self._create(msg)
        else:
            await self._patch(self._post_id, message=msg)

    async def finalize(self) -> None:
        """Freeze the post to a static '✓ done' (drops the spinner emoji)."""
        with self._lock:
            lines = list(self._lines)
            streamed = self._streamed
        if not streamed:
            return  # nothing was ever streamed
        elapsed = max(0, int(time.monotonic() - (self._started_at or time.monotonic())))
        msg = self._compose(lines, done=True, elapsed=elapsed)
        if self._post_id is None:
            self._post_id = await self._create(msg)
        else:
            await self._patch(self._post_id, message=msg)

    # --- MM I/O helpers (best-effort) ---
    async def _create(self, message: str) -> Optional[str]:
        try:
            post = await self._bot.create_post(
                self._channel_id, message, root_id=self._root_id
            )
            pid = post.get("id") if isinstance(post, dict) else None
            _narration_log.debug("narration: created post id=%s", pid)
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
