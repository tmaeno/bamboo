"""Mattermost bot daemon — WebSocket event loop + thread↔session registry.

One outbound connection (REST for posting, WebSocket for ``posted`` events).
Each investigation/capture runs in its own thread: the thread root message starts
a session, replies in that thread are turns.  The bot is decoupled from bamboo's
agents via an injected ``run_session`` callback, so the routing/registry logic is
unit-testable with a fake driver.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from bamboo.frontends.mattermost.io import ThreadTransport, describe_post_failure

logger = logging.getLogger(__name__)

# Animated custom emoji used as the live-progress spinner (registered on startup;
# see ``MattermostBot.ensure_spinner_emoji``).
SPINNER_EMOJI_NAME = "bamboo_spinner"


@dataclass
class Command:
    """A parsed start command from a thread-root message."""

    kind: str  # "investigate" | "capture" | "analyze" | "login" | "logout" | "status" | "help"
    task_id: Optional[int] = None
    user_id: Optional[str] = None  # Mattermost user who issued the command
    verbose: bool = False  # `--verbose`/`-v` on the command → DEBUG narration in the live post


@dataclass
class BotStatus:
    """A snapshot of the bot's runtime health, reported by the ``status`` command."""

    functional: bool  # passed a live round-trip to Mattermost
    bot_user_id: Optional[str]
    active_sessions: int
    allowed_channels: int
    uptime_seconds: Optional[float]
    detail: str = ""  # populated with the error text when not functional


def parse_command(message: str) -> Optional[Command]:
    """Parse a start command from a message, tolerating a leading @mention.

    Recognises ``investigate [<task_id>] [--verbose|-v]``, ``capture [<task_id>]``,
    ``analyze [<task_id>]``, ``login``, ``logout``, ``status``, and ``help``.
    A ``--verbose``/``-v`` token streams behind-the-scenes DEBUG narration into the
    thread's live post. Returns ``None`` when the message is not a start command.
    """
    text = (message or "").strip()
    # Drop a leading @mention token if present.
    if text.startswith("@"):
        parts = text.split(None, 1)
        text = parts[1] if len(parts) > 1 else ""
    # Tolerate a leading slash command form ("/bamboo investigate 123").
    low = text.lower()
    for prefix in ("/bamboo ", "bamboo "):
        if low.startswith(prefix):
            text = text[len(prefix):]
            low = text.lower()
            break
    tokens = text.split()
    if not tokens:
        return None
    verb = tokens[0].lower()
    if verb in ("investigate", "capture", "analyze"):
        task_id = None
        verbose = False
        for tok in tokens[1:]:
            if tok.isdigit() and task_id is None:
                task_id = int(tok)
            elif tok.lower() in ("--verbose", "-v"):
                verbose = True
        return Command(kind=verb, task_id=task_id, verbose=verbose)
    if verb in ("login", "logout", "status", "help"):
        return Command(kind=verb)
    return None


def _is_addressed(message: str, bot_username: Optional[str]) -> bool:
    """True when *message* is clearly directed at the bot.

    Either it uses a ``/bamboo``/``bamboo `` command prefix (the same ones
    :func:`parse_command` honors) or its leading ``@token`` is the bot's own
    @mention.  Used to decide whether an *unrecognized* message should get a
    ``help`` reply or be silently ignored as ordinary channel chatter.
    """
    text = (message or "").strip()
    low = text.lower()
    if low.startswith("/bamboo") or low.startswith("bamboo ") or low == "bamboo":
        return True
    if bot_username and text.startswith("@"):
        first = text.split(None, 1)[0]
        return first.lstrip("@").lower() == bot_username.lower()
    return False


# A session runner: given a transport + command, drive the flow to completion.
RunSession = Callable[[ThreadTransport, Command], Awaitable[None]]


class _ThreadSession(ThreadTransport):
    """Per-thread transport: FIFO outbox (sequential sender) + inbound replies."""

    def __init__(self, bot: "MattermostBot", channel_id: str, root_id: str) -> None:
        self._bot = bot
        self.channel_id = channel_id
        self.root_id = root_id
        self._outbox: asyncio.Queue = asyncio.Queue()
        self._inbox: asyncio.Queue = asyncio.Queue()
        self._sender_task: Optional[asyncio.Task] = None

    # --- ThreadTransport ---
    def send(self, text: str, *, props: Optional[dict[str, Any]] = None) -> None:
        self._outbox.put_nowait((text, props))

    async def next_reply(self) -> str:
        return await self._inbox.get()

    async def thread_messages(self) -> list[str]:
        return await self._bot.get_thread_messages(self.root_id)

    async def bot_status(self) -> "BotStatus":
        return await self._bot.status_snapshot()

    async def run_in_dm(self, user_id: str, run: Any) -> Any:
        return await self._bot.run_in_dm(user_id, run)

    # --- driven by the bot ---
    def deliver(self, text: str) -> None:
        self._inbox.put_nowait(text)

    async def _sender_loop(self) -> None:
        while True:
            text, props = await self._outbox.get()
            logger.debug("outbox posting msg_len=%d props=%s", len(text or ""), bool(props))
            try:
                await self._bot.create_post(
                    self.channel_id, text, root_id=self.root_id, props=props
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "%s",
                    describe_post_failure(
                        "outbox post failed",
                        message=text, props=props, root_id=self.root_id, exc=exc,
                    ),
                )
                logger.debug("outbox post failure traceback", exc_info=True)
            finally:
                self._outbox.task_done()


class MattermostBot:
    """Routes Mattermost ``posted`` events to per-thread investigation sessions."""

    def __init__(
        self,
        driver: Any,
        *,
        allowed_channels: set[str],
        run_session: RunSession,
    ) -> None:
        self.driver = driver
        self.allowed_channels = set(allowed_channels)
        self.run_session = run_session
        self._sessions: dict[str, _ThreadSession] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        # Per-thread (per-root) locks serialising post *creation* — Mattermost
        # updates the thread's row on every reply, so simultaneous replies to one
        # root can fail with `app.post.save.app_error`. See create_post.
        self._create_locks: dict[str, asyncio.Lock] = {}
        self.bot_user_id: Optional[str] = None
        self.bot_username: Optional[str] = None
        # Name of the animated spinner custom emoji once registered, else None
        # (the narration sink uses it; falls back to a static glyph when unset).
        self.spinner_emoji: Optional[str] = None
        self._started_at: Optional[float] = None

    async def create_post(
        self,
        channel_id: str,
        message: str,
        *,
        root_id: str = "",
        props: Optional[dict[str, Any]] = None,
        file_ids: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        # Empty root_id = a new top-level post → no thread contention, no lock.
        if not root_id:
            return await self.driver.posts.create_post(
                channel_id=channel_id,
                message=message,
                root_id=root_id,
                props=props,
                file_ids=file_ids,
            )
        # Serialise creates to the SAME thread: Mattermost updates the thread's row
        # on every reply, so concurrent replies to one root can collide with
        # `app.post.save.app_error`. The three posters (outbox sender, narration
        # live post, per-turn detail message) otherwise race. Different threads
        # still create concurrently. setdefault is race-free here — asyncio is
        # single-threaded and there is no await between lookup and insert.
        lock = self._create_locks.setdefault(root_id, asyncio.Lock())
        async with lock:
            return await self.driver.posts.create_post(
                channel_id=channel_id,
                message=message,
                root_id=root_id,
                props=props,
                file_ids=file_ids,
            )

    async def open_direct_channel(self, user_id: str) -> str:
        """Return the id of the bot↔user direct-message channel (created if absent)."""
        channel = await self.driver.channels.create_direct_channel(
            [self.bot_user_id, user_id]
        )
        return channel["id"]

    async def update_post(
        self,
        post_id: str,
        *,
        message: Optional[str] = None,
        file_ids: Optional[list[str]] = None,
        props: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Patch an existing post's message, attachments, and/or props (in-place edit)."""
        return await self.driver.posts.patch_post(
            post_id, message=message, file_ids=file_ids, props=props
        )

    async def delete_post(self, post_id: str) -> Any:
        """Delete a post (used as a fallback when an attachment can't be detached)."""
        return await self.driver.posts.delete_post(post_id)

    async def upload_file(self, channel_id: str, filename: str, data: bytes) -> str:
        """Upload *data* as *filename* to *channel_id*; return the new file id.

        ``files`` is a single ``(filename, bytes, content_type)`` tuple — the driver
        wraps it as ``{"files": files}`` itself, so we must NOT nest it again.
        """
        resp = await self.driver.files.upload_file(
            files=(filename, data, "image/gif"), channel_id=channel_id
        )
        return resp["file_infos"][0]["id"]

    async def ensure_spinner_emoji(self) -> None:
        """Register the animated spinner as a custom emoji (best-effort, once).

        Custom emoji render on Mattermost's inline text path (unlike file
        attachments), so an animated-GIF custom emoji spins client-side without a
        timer.  The registration is **self-healing**: if an emoji with our name
        already exists but its stored image differs from the bundled gif (e.g. a
        stale one left over from an older build), we delete and recreate it so an
        updated asset always propagates.  On any failure (custom emoji disabled /
        no permission) we leave ``spinner_emoji`` unset and the narration sink
        degrades to a static glyph.
        """
        narration_log = logging.getLogger("bamboo.narration")
        import importlib.resources as ir  # noqa: PLC0415

        try:
            data = (
                ir.files("bamboo.frontends.mattermost")
                .joinpath("assets/spinner.gif")
                .read_bytes()
            )
        except Exception as exc:  # noqa: BLE001
            narration_log.warning("narration: spinner asset unreadable (%s); using a static glyph", exc)
            return

        # Look up an existing emoji; if its image is stale, delete it so we recreate.
        try:
            existing = await self.driver.emoji.get_emoji_by_name(SPINNER_EMOJI_NAME)
        except Exception:  # noqa: BLE001 — not found is the common, expected case
            existing = None
        if isinstance(existing, dict) and existing.get("id"):
            if await self._spinner_emoji_matches(existing["id"], data):
                self.spinner_emoji = SPINNER_EMOJI_NAME
                narration_log.debug("narration: spinner emoji present (:%s:)", SPINNER_EMOJI_NAME)
                return
            try:
                await self.driver.emoji.delete_emoji(existing["id"])
                narration_log.debug("narration: spinner emoji refreshed (stale image replaced)")
            except Exception as exc:  # noqa: BLE001 — can't delete → reuse what's there
                self.spinner_emoji = SPINNER_EMOJI_NAME
                narration_log.warning(
                    "narration: stale spinner emoji could not be replaced (%s); using it as-is", exc
                )
                return

        try:
            await self.driver.emoji.create_emoji(
                image=("spinner.gif", data, "image/gif"),
                emoji=json.dumps({"name": SPINNER_EMOJI_NAME, "creator_id": self.bot_user_id}),
            )
            self.spinner_emoji = SPINNER_EMOJI_NAME
            narration_log.debug("narration: spinner emoji created (:%s:)", SPINNER_EMOJI_NAME)
        except Exception as exc:  # noqa: BLE001
            # A concurrent create may have won the race — one recovery lookup.
            try:
                recovered = await self.driver.emoji.get_emoji_by_name(SPINNER_EMOJI_NAME)
                if isinstance(recovered, dict) and recovered.get("id"):
                    self.spinner_emoji = SPINNER_EMOJI_NAME
                    return
            except Exception:  # noqa: BLE001
                pass
            narration_log.warning(
                "narration: spinner emoji unavailable (%s); progress will use a static glyph", exc
            )

    async def _spinner_emoji_matches(self, emoji_id: str, data: bytes) -> bool:
        """True if the stored emoji image equals the bundled gif (best-effort).

        On any error fetching/reading the image we return True (reuse), so a
        transient read failure never causes us to delete a working emoji.
        """
        try:
            resp = await self.driver.emoji.get_emoji_image(emoji_id)
            stored = getattr(resp, "content", resp)
            return isinstance(stored, (bytes, bytearray)) and bytes(stored) == data
        except Exception:  # noqa: BLE001
            return True

    async def run_in_dm(self, user_id: str, run: Any) -> Any:
        """Run *run(io)* against a temporary IO bound to *user_id*'s DM channel.

        Opens (or reuses) the bot↔user DM, spins a short-lived sender so the IO's
        messages are delivered in order, awaits ``run``, then drains and tears the
        sender down. Used to drive a private exchange (e.g. auto-login) from inside
        a public-channel session.
        """
        from bamboo.frontends.mattermost.io import MattermostInteractionIO  # noqa: PLC0415

        dm_id = await self.open_direct_channel(user_id)
        sess = _ThreadSession(self, dm_id, "")
        sess._sender_task = asyncio.ensure_future(sess._sender_loop())
        try:
            return await run(MattermostInteractionIO(sess))
        finally:
            try:
                await asyncio.wait_for(sess._outbox.join(), timeout=10.0)
            except (asyncio.TimeoutError, Exception):  # noqa: BLE001
                pass
            if sess._sender_task is not None:
                sess._sender_task.cancel()

    async def get_thread_messages(self, root_id: str) -> list[str]:
        """Return human (non-bot) messages in a thread, oldest first."""
        thread = await self.driver.posts.get_post_thread(root_id)
        posts = (thread or {}).get("posts", {}) or {}
        ordered = sorted(posts.values(), key=lambda p: p.get("create_at", 0))
        out: list[str] = []
        for p in ordered:
            if p.get("user_id") and p.get("user_id") == self.bot_user_id:
                continue  # skip bamboo's own posts
            text = (p.get("message") or "").strip()
            if text:
                out.append(text)
        return out

    async def run(self) -> None:
        """Login, learn our own user id, then run the WebSocket event loop."""
        await self.driver.login()
        me = await self.driver.users.get_user("me")
        self.bot_user_id = me.get("id") if isinstance(me, dict) else None
        self.bot_username = me.get("username") if isinstance(me, dict) else None
        self._started_at = time.monotonic()
        logger.info("Mattermost bot connected as user %s", self.bot_user_id)
        await self.ensure_spinner_emoji()
        await self.driver.init_websocket(self.handle_event)

    async def status_snapshot(self) -> BotStatus:
        """Report runtime health, with a live round-trip to verify functionality.

        Re-fetching ``me`` over REST proves the bot can still reach and
        authenticate against Mattermost.  The WebSocket loop is already proven
        live by the fact that the ``status`` command itself was received.
        """
        functional = True
        detail = ""
        try:
            await self.driver.users.get_user("me")
        except Exception as exc:  # noqa: BLE001
            functional = False
            detail = str(exc) or exc.__class__.__name__
        uptime = (time.monotonic() - self._started_at) if self._started_at else None
        return BotStatus(
            functional=functional,
            bot_user_id=self.bot_user_id,
            active_sessions=len(self._sessions),
            allowed_channels=len(self.allowed_channels),
            uptime_seconds=uptime,
            detail=detail,
        )

    async def handle_event(self, event: Any) -> None:
        """Handle one WebSocket event. Tolerates str or dict payloads."""
        if isinstance(event, (str, bytes)):
            try:
                event = json.loads(event)
            except (ValueError, TypeError):
                return
        if not isinstance(event, dict) or event.get("event") != "posted":
            return
        data = event.get("data") or {}
        post_raw = data.get("post")
        try:
            post = json.loads(post_raw) if isinstance(post_raw, str) else (post_raw or {})
        except (ValueError, TypeError):
            return

        channel_id = post.get("channel_id")
        user_id = post.get("user_id")
        message = post.get("message", "")
        if not channel_id or channel_id not in self.allowed_channels:
            return  # channel allow-list (Layer 1 auth)
        if user_id and user_id == self.bot_user_id:
            return  # never react to our own posts

        # A reply within an existing session thread → feed it as the next turn.
        root_id = post.get("root_id") or post.get("id")
        sess = self._sessions.get(root_id)
        if sess is not None:
            sess.deliver(message)
            return

        # Otherwise, maybe a start command (rooted at this message).
        command = parse_command(message)
        if command is None:
            # Unrecognized: reply with help only if the bot was clearly addressed
            # (@mention or /bamboo prefix); ignore ordinary channel chatter.
            if _is_addressed(message, self.bot_username):
                command = Command(kind="help")
            else:
                return
        command.user_id = user_id  # attribute the session to the invoking user
        await self._start_session(channel_id, root_id, command)

    async def _start_session(self, channel_id: str, root_id: str, command: Command) -> None:
        sess = _ThreadSession(self, channel_id, root_id)
        # The login flow is a private exchange (auth prompt + result) that never
        # reads chat replies, so redirect it to a DM with the invoking user
        # instead of posting in the public channel.
        if command.kind == "login" and command.user_id and self.bot_user_id:
            try:
                sess.channel_id = await self.open_direct_channel(command.user_id)
                sess.root_id = ""  # DM is its own channel; don't thread under the public post
            except Exception:  # noqa: BLE001
                logger.exception("failed to open DM for login; using original channel")
        self._sessions[root_id] = sess
        sess._sender_task = asyncio.ensure_future(sess._sender_loop())
        self._tasks[root_id] = asyncio.ensure_future(self._drive(sess, command))

    async def _drive(self, sess: _ThreadSession, command: Command) -> None:
        try:
            await self.run_session(sess, command)
        except Exception:  # noqa: BLE001
            logger.exception("session for thread %s failed", sess.root_id)
            sess.send("⚠️ bamboo hit an error in this session; see server logs.")
        finally:
            await self._drain_and_cleanup(sess)

    async def _drain_and_cleanup(self, sess: _ThreadSession) -> None:
        # Let queued outbound messages flush before tearing the sender down.
        try:
            await asyncio.wait_for(sess._outbox.join(), timeout=10.0)
        except (asyncio.TimeoutError, Exception):  # noqa: BLE001
            pass
        self._sessions.pop(sess.root_id, None)
        self._tasks.pop(sess.root_id, None)
        self._create_locks.pop(sess.root_id, None)
        if sess._sender_task is not None:
            sess._sender_task.cancel()
