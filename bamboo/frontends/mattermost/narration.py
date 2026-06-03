"""Render engine progress narration into a Mattermost thread.

Narration is a single logging stream: :mod:`bamboo.utils.narrator`
(``say`` / ``thinking`` / ``show_block`` / ``warn`` / ``error``) emits records on
the ``bamboo.narration`` logger.  The bot attaches a :class:`MattermostLogHandler`
to that logger; the server console sees the same records, so **what MM shows is a
level-filtered subset of the console** (never divergent).

Per session, :func:`stream_narration` installs a :class:`_LivePost` (the live
post's state + a flusher task) and points a ContextVar at it; the global handler
routes each record to whichever session is active in the emitting context.  A
record's ``narration_kind`` decides its role:

* ``"step"`` → the post **head** (``:bamboo_spinner: 🔎 <step>``; the animated
  emoji spins client-side; frozen / deleted at the end);
* ``"block"`` → **ignored** (verbose detail stays in the server log only);
* otherwise → a timestamped **body line** (last *N* kept), with WARNING/ERROR
  highlighted.

The handler only ever sees records logging already admitted (≥ ``LOG_LEVEL``), and
further filters by ``NARRATION_LEVEL``.  ``emit`` is non-blocking — it just mutates
guarded state and wakes the flusher, which does all Mattermost I/O.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time
from collections import deque
from contextvars import ContextVar
from datetime import datetime
from typing import Any, Optional

# Operational diagnostics for the handler itself (post create/patch failures).
# Kept at DEBUG so they don't pollute the narration stream operators read.
_diag = logging.getLogger("bamboo.narration")

_DETAIL_ENTRIES = 10  # most-recent body lines kept in the live post (sliding window)
_LINE_CLIP = 200  # max chars per body line (full text still goes to the log)
_FLUSH_INTERVAL = 1.0  # min seconds between Mattermost API edits (coalescing)

# The active session's live post, for the emitting context. The global handler
# reads this to route a record to the right thread; None between/outside sessions.
_active_post: ContextVar[Optional["_LivePost"]] = ContextVar("mm_live_post", default=None)


def _clip(text: str, limit: int = _LINE_CLIP) -> str:
    text = " ".join((text or "").split())  # collapse whitespace/newlines
    return text if len(text) <= limit else text[: limit - 1] + "…"


class _LivePost:
    """Per-session builder of one live-updating Mattermost progress post.

    Fed records by :class:`MattermostLogHandler` (possibly from worker threads);
    ``feed`` only mutates guarded state + wakes the flusher via
    ``call_soon_threadsafe`` — no Mattermost I/O on the caller's thread.
    """

    def __init__(
        self,
        bot: Any,
        channel_id: str,
        root_id: str,
        loop: asyncio.AbstractEventLoop,
        threshold: int,
    ) -> None:
        self._bot = bot
        self._channel_id = channel_id
        self._root_id = root_id
        self._loop = loop
        self._threshold = threshold
        self._lock = threading.Lock()
        self._wake = asyncio.Event()
        self._latest_step: str = "working…"
        self._lines: deque[str] = deque(maxlen=_DETAIL_ENTRIES)
        self._dirty = False
        self._streamed = False
        self._spinner: Optional[str] = getattr(bot, "spinner_emoji", None)
        self._post_id: Optional[str] = None
        self._started_at: float = 0.0

    # --- fed by the logging handler (possibly off-loop) ---
    def feed(self, record: logging.LogRecord) -> None:
        if record.levelno < self._threshold:
            return
        kind = getattr(record, "narration_kind", "line")
        if kind == "block":
            return  # verbose detail — server log only, never chat
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001 — never let logging formatting break us
            return
        ts = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        with self._lock:
            if kind == "step":
                self._latest_step = _clip(msg, 160)
            else:
                mark = "⚠️ " if record.levelno >= logging.WARNING else "→ "
                self._lines.append(f"{ts}  {mark}{_clip(msg)}")
            self._dirty = True
            self._streamed = True
        self._signal()

    def _signal(self) -> None:
        try:
            self._loop.call_soon_threadsafe(self._wake.set)
        except RuntimeError:  # loop closed/closing
            pass

    # --- driven by stream_narration ---
    async def run(self, started_at: float) -> None:
        """Flusher loop: coalesce records into throttled edits to the one post."""
        self._started_at = started_at
        last_flush = 0.0
        while True:
            await self._wake.wait()
            self._wake.clear()
            delta = time.monotonic() - last_flush
            if delta < _FLUSH_INTERVAL:
                await asyncio.sleep(_FLUSH_INTERVAL - delta)
            await self._flush_once()
            last_flush = time.monotonic()

    def _compose(self, lines: list[str], *, done: bool = False, stopped: bool = False, elapsed: int = 0) -> str:
        """A head line + a monospace log of recent lines.

        ``stopped`` (failure) renders a static head with no spinner emoji.
        """
        if done:
            head = f"✓ done ({elapsed}s)"
        elif not stopped and self._spinner:
            head = f":{self._spinner}: 🔎 {self._latest_step}"
        else:
            head = f"🔎 {self._latest_step}"
        if not lines:
            return head
        return head + "\n```\n" + "\n".join(lines) + "\n```"

    async def _flush_once(self) -> None:
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

    async def finalize(self, *, success: bool) -> None:
        """Freeze the post when the run ends.

        On success → a terse ``✓ done (Ns)`` line with the streamed detail dropped.
        On failure → a static ``🔎 <last step>`` head (no spinner) keeping the last
        lines as a trail.
        """
        with self._lock:
            lines = list(self._lines)
            streamed = self._streamed
        if not streamed:
            return
        elapsed = max(0, int(time.monotonic() - (self._started_at or time.monotonic())))
        if success:
            msg = self._compose([], done=True, elapsed=elapsed)  # head only, detail dropped
        else:
            msg = self._compose(lines, stopped=True)
        if self._post_id is None:
            self._post_id = await self._create(msg)
        else:
            await self._patch(self._post_id, message=msg)

    # --- MM I/O helpers (best-effort) ---
    async def _create(self, message: str) -> Optional[str]:
        try:
            post = await self._bot.create_post(self._channel_id, message, root_id=self._root_id)
            pid = post.get("id") if isinstance(post, dict) else None
            _diag.debug("narration: created post id=%s", pid)
            return pid
        except Exception as exc:  # noqa: BLE001
            _diag.warning("narration: create_post failed (%s)", exc)
            return None

    async def _patch(self, post_id: str, *, message: str) -> bool:
        try:
            await self._bot.update_post(post_id, message=message)
            return True
        except Exception as exc:  # noqa: BLE001
            _diag.warning("narration: update_post failed (%s)", exc)
            return False


class MattermostLogHandler(logging.Handler):
    """Routes ``bamboo.narration`` records to the active session's live post.

    A single instance is attached to the ``bamboo.narration`` logger; it reads the
    per-session :data:`_active_post` ContextVar (set in the emitting context) so
    concurrent sessions each get their own post with no cross-talk. No-ops when no
    session is active. ``emit`` is non-blocking (delegates to ``_LivePost.feed``).
    """

    def emit(self, record: logging.LogRecord) -> None:
        post = _active_post.get()
        if post is None:
            return
        try:
            post.feed(record)
        except Exception:  # noqa: BLE001
            self.handleError(record)


# The handler is global and attached once; it self-gates on the ContextVar.
_handler: Optional[MattermostLogHandler] = None
_handler_lock = threading.Lock()


def _ensure_handler() -> None:
    global _handler
    with _handler_lock:
        if _handler is None:
            _handler = MattermostLogHandler()
            _handler.setLevel(logging.NOTSET)  # threshold applied per-post in feed()
            logging.getLogger("bamboo.narration").addHandler(_handler)


def _threshold() -> int:
    try:
        from bamboo.config import get_settings  # noqa: PLC0415

        name = (get_settings().narration_level or "INFO").upper()
        return getattr(logging, name, logging.INFO)
    except Exception:  # noqa: BLE001
        return logging.INFO


@contextlib.asynccontextmanager
async def stream_narration(transport: Any):
    """Install a per-session live post that mirrors the narration stream to MM.

    No-op for transports not backed by a bot (e.g. test fakes): narration stays on
    the logging stream only, exactly as before.
    """
    bot = getattr(transport, "_bot", None)
    channel_id = getattr(transport, "channel_id", None)
    if bot is None or not channel_id:
        yield
        return

    loop = asyncio.get_running_loop()
    threshold = _threshold()
    post = _LivePost(bot, channel_id, getattr(transport, "root_id", ""), loop, threshold)
    _ensure_handler()
    # Ensure narration records at >= threshold actually reach the handler: a record
    # is dropped before any handler if the logger's effective level is higher
    # (e.g. when LOG_LEVEL/root is above NARRATION_LEVEL). Only lower it if needed.
    nlog = logging.getLogger("bamboo.narration")
    prev_level = nlog.level
    if nlog.getEffectiveLevel() > threshold:
        nlog.setLevel(threshold)
    token = _active_post.set(post)
    started_at = time.monotonic()
    flusher = asyncio.ensure_future(post.run(started_at))
    ok = False
    try:
        yield
        ok = True
    finally:
        _active_post.reset(token)
        nlog.setLevel(prev_level)
        flusher.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await flusher
        with contextlib.suppress(Exception):
            await post.finalize(success=ok)
