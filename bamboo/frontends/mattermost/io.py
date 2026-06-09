"""``MattermostInteractionIO`` — the chat frontend adapter.

Implements the :class:`~bamboo.frontends.base.InteractionIO` contract over a
single Mattermost thread.  Input is *reply-based*: ``ask``/``confirm``/``edit``
post a prompt and await the next human reply in the thread (resolved by the bot's
WebSocket handler).  This keeps the bot to one outbound connection — no inbound
HTTP callback server, which interactive buttons/dialogs would require (those are
a Phase 4 enhancement).  Output methods post Markdown messages.

The adapter talks to its thread through a :class:`ThreadTransport`.  All outbound
messages go through the transport's FIFO ``send`` so render output and prompts
stay in order (the port's render methods are synchronous, so ``send`` is too);
``next_reply`` awaits the next human message.  The transport is an interface, so
the rendering and prompt/reply logic are unit-testable with a fake (no client or
live server needed).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import Any, Optional

from bamboo.frontends.base import (
    Column,
    DetailSink,
    InteractionIO,
    ReviewOption,
    match_choice,
)

logger = logging.getLogger(__name__)


def describe_post_failure(
    context: str,
    *,
    message: str = "",
    props: Optional[dict] = None,
    root_id: str = "",
    exc: Optional[BaseException] = None,
) -> str:
    """Build a one-line diagnostic for a failed Mattermost post.

    Captures the Mattermost ``error_id``/``request_id``/``status`` (the precise
    cause), the target thread, the message length + a head/tail snippet, and the
    props / attachment-text sizes — enough to tell *which* post was rejected and
    *why* (message too long, props too large, bad ``root_id``, DB save, …) without
    dumping the whole body.
    """
    bits = [context]
    if exc is not None:
        bits.append(
            f"status={getattr(exc, 'status_code', '?')} "
            f"error_id={getattr(exc, 'error_id', None)!r} "
            f"request_id={getattr(exc, 'request_id', None)!r} "
            f"exc={type(exc).__name__}: {exc}"
        )
    bits.append(f"root_id={root_id!r}")
    msg = message or ""
    bits.append(f"msg_len={len(msg)}")
    if isinstance(props, dict):
        try:
            bits.append(f"props_json_len={len(json.dumps(props))}")
        except Exception:  # noqa: BLE001
            pass
        atts = props.get("attachments")
        if isinstance(atts, list):
            total = sum(len(a.get("text", "")) for a in atts if isinstance(a, dict))
            bits.append(f"attachments={len(atts)} attachment_text_len={total}")
    snippet = msg if len(msg) <= 400 else msg[:200] + " … " + msg[-200:]
    bits.append(f"msg_snippet={snippet!r}")
    return " | ".join(bits)

# Rich-markup → Markdown / plain. We render [bold] as **, [dim]/[italic] as _,
# and strip color tags (yellow/red/green/cyan/magenta/blue/...).
_BOLD_RE = re.compile(r"\[/?bold\]")
_ITALIC_RE = re.compile(r"\[/?(?:dim|i|italic)\]")
_ANY_TAG_RE = re.compile(r"\[/?[a-zA-Z][^\]]*\]")


def to_markdown(text: str) -> str:
    """Convert Rich console markup to Mattermost Markdown (best-effort)."""
    text = _BOLD_RE.sub("**", text or "")
    text = _ITALIC_RE.sub("_", text)
    text = _ANY_TAG_RE.sub("", text)  # strip remaining color tags
    return text


class ThreadTransport(ABC):
    """A single Mattermost thread the adapter posts to and reads replies from."""

    @abstractmethod
    def send(self, text: str, *, props: Optional[dict[str, Any]] = None) -> None:
        """Enqueue a message to the thread (FIFO, non-blocking).

        Synchronous because the port's render methods are synchronous; the
        transport is responsible for actually delivering messages in order.
        """
        ...

    @abstractmethod
    async def next_reply(self) -> str:
        """Await and return the next human reply in this thread."""
        ...

    async def thread_messages(self) -> list[str]:
        """Return prior human messages in this thread, oldest first.

        Used by capture-from-thread to assemble the discussion transcript.
        Defaults to empty for transports without history; the bot's per-thread
        transport overrides it.
        """
        return []

    async def bot_status(self) -> Any:
        """Return a snapshot of the bot's runtime health.

        Used by the ``status`` command.  Defaults to an "unknown" snapshot for
        transports detached from a running bot; the bot's per-thread transport
        overrides it to delegate to :meth:`MattermostBot.status_snapshot`.
        """
        from bamboo.frontends.mattermost.bot import BotStatus

        return BotStatus(
            functional=False,
            bot_user_id=None,
            active_sessions=0,
            allowed_channels=0,
            uptime_seconds=None,
            detail="no bot attached to this transport",
        )

    async def run_in_dm(self, user_id: str, run: Any) -> Any:
        """Run *run* with an :class:`InteractionIO` bound to *user_id*'s DM.

        Used to drive a private exchange (e.g. an auto-login prompt) from inside a
        public-channel session.  *run* is an ``async (io) -> result`` callable; the
        result is returned.  Defaults to unsupported; the bot's per-thread transport
        overrides it.
        """
        raise NotImplementedError("this transport cannot open a direct-message channel")


_DETAIL_FLUSH_INTERVAL = 1.0  # min seconds between live-message edits (coalescing)
_DETAIL_CLIP = 1500  # max chars shown for the reasoning / answer sections


def _clip_tail(text: str, limit: int) -> str:
    """Keep the last *limit* chars, prefixing an ellipsis when truncated."""
    text = (text or "").strip()
    return text if len(text) <= limit else "…" + text[-(limit - 1) :]


class _LiveMessage:
    """One live-updating, durable per-turn Mattermost message of streamed detail.

    Mirrors :class:`~bamboo.frontends.mattermost.narration._LivePost` but is
    per-turn (not per-session) and posts directly through the bot so it can be
    *edited in place* (the FIFO outbox only creates posts). Driven on the event
    loop by the engine coroutine — ``feed``/``meta`` just mutate state and wake the
    flusher, which coalesces edits to one ``update_post`` per ``_DETAIL_FLUSH_INTERVAL``.
    """

    active = True

    def __init__(
        self, bot: Any, channel_id: str, root_id: str, title: str
    ) -> None:
        self._bot = bot
        self._channel_id = channel_id
        self._root_id = root_id
        self._title = title
        self._spinner: Optional[str] = getattr(bot, "spinner_emoji", None)
        self._meta: list[str] = []
        self._reasoning: list[str] = []
        self._answer: list[str] = []
        self._dirty = False
        self._wake = asyncio.Event()
        self._post_id: Optional[str] = None

    # --- DetailSink (fed by the engine coroutine, same loop) ---
    def feed(self, text: str, *, reasoning: bool = False) -> None:
        (self._reasoning if reasoning else self._answer).append(text)
        self._dirty = True
        self._wake.set()

    def meta(self, line: str) -> None:
        self._meta.append(line)
        self._dirty = True
        self._wake.set()

    # --- rendering ---
    def _render(self, *, done: bool = False) -> tuple[str, Optional[dict]]:
        spacer = " " * 10
        if not done and self._spinner:
            head = f":{self._spinner}: **{self._title}** :{self._spinner}:{spacer}"
        else:
            head = f"🔎 **{self._title}**{spacer}"
        parts: list[str] = [f"**{to_markdown(m)}**" for m in self._meta]
        reasoning = "".join(self._reasoning).strip()
        if reasoning:
            parts.append("> " + _clip_tail(reasoning, _DETAIL_CLIP).replace("\n", "\n> "))
        answer = "".join(self._answer).strip()
        if answer:
            parts.append("```\n" + _clip_tail(answer, _DETAIL_CLIP) + "\n```")
        body = "\n\n".join(parts)
        if not body:
            return head, None
        return head, {"attachments": [{"color": "#4a90d9", "text": body}]}

    # --- flusher / MM I/O (best-effort) ---
    async def run(self) -> None:
        last = 0.0
        while True:
            await self._wake.wait()
            self._wake.clear()
            delta = time.monotonic() - last
            if delta < _DETAIL_FLUSH_INTERVAL:
                await asyncio.sleep(_DETAIL_FLUSH_INTERVAL - delta)
            await self._flush_once()
            last = time.monotonic()

    async def _flush_once(self, *, done: bool = False) -> None:
        if not self._dirty and not done:
            return
        self._dirty = False
        message, props = self._render(done=done)
        if self._post_id is None:
            if message or props:
                self._post_id = await self._create(message, props)
        else:
            await self._patch(self._post_id, message, props)

    async def finalize(self) -> None:
        """Final flush + freeze (drop the spinner). Durable — the post is never deleted."""
        if self._post_id is None and not (self._meta or self._reasoning or self._answer):
            return
        await self._flush_once(done=True)

    @property
    def post_id(self) -> Optional[str]:
        """The id of this card's Mattermost post (None until first created)."""
        return self._post_id

    async def show_review(self, message: str, props: dict) -> None:
        """Replace the card's content with a static review (caller-built message + card).

        Used to *evolve* the live "Planning…" card into the orchestration review:
        the streamed reasoning/answer is dropped and the caller's *message* (the
        reply instruction) + *props* (the code-first card) take its place. One-shot
        patch — the flusher is gone once the detail-stream context has closed.
        """
        if self._post_id is None:
            self._post_id = await self._create(message, props)
        else:
            await self._patch(self._post_id, message, props)

    async def _create(self, message: str, props: Optional[dict]) -> Optional[str]:
        logger.debug("detail message posting msg_len=%d props=%s", len(message or ""), bool(props))
        try:
            post = await self._bot.create_post(
                self._channel_id, message, root_id=self._root_id, props=props
            )
            return post.get("id") if isinstance(post, dict) else None
        except Exception as exc:  # noqa: BLE001 — best-effort; never break the turn
            logger.warning(
                "%s",
                describe_post_failure(
                    "detail message create failed",
                    message=message, props=props, root_id=self._root_id, exc=exc,
                ),
            )
            return None

    async def _patch(self, post_id: str, message: str, props: Optional[dict]) -> None:
        logger.debug("detail message patching msg_len=%d props=%s", len(message or ""), bool(props))
        try:
            await self._bot.update_post(post_id, message=message, props=props)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "%s",
                describe_post_failure(
                    "detail message patch failed",
                    message=message, props=props, root_id=self._root_id, exc=exc,
                ),
            )


class MattermostInteractionIO(InteractionIO):
    """Chat adapter bound to one :class:`ThreadTransport`."""

    def __init__(self, transport: ThreadTransport, *, verbose: bool = False) -> None:
        self.transport = transport
        self.verbose = verbose

    @property
    def supports_interaction(self) -> bool:
        """Always interactive — the bot can post a prompt and await a thread reply."""
        return True

    # ------------------------------------------------------------------
    # Input (reply-based)
    # ------------------------------------------------------------------

    async def ask(
        self,
        prompt: str,
        *,
        default: str | None = None,
        choices: list[str] | None = None,
    ) -> str:
        suffix = ""
        if choices:
            suffix += f"\n_Reply with one of: {', '.join(choices)}_"
        if default is not None:
            suffix += f"\n_(blank reply → `{default}`)_"
        self.transport.send(to_markdown(prompt) + suffix)
        while True:
            answer = (await self.transport.next_reply()).strip()
            if not answer:
                if default is not None:
                    return default
                continue
            if choices and answer not in choices:
                self.transport.send(f"_Please reply with one of: {', '.join(choices)}_")
                continue
            return answer

    async def confirm(self, prompt: str, *, default: bool | None = None) -> bool:
        self.transport.send(f"{to_markdown(prompt)} [yes/no]")
        while True:
            answer = (await self.transport.next_reply()).strip().lower()
            if not answer and default is not None:
                return default
            if answer in ("y", "yes", "true"):
                return True
            if answer in ("n", "no", "false"):
                return False
            self.transport.send("_Please reply yes or no._")

    async def edit(
        self,
        *,
        strategy_type: str,
        code: str,
        summary: str,
        triggers: list[str],
    ) -> tuple[str, str, str, list[str]]:
        self.transport.send(
            "To edit, reply with the replacement code block. "
            "Reply `ok` to keep the code as-is.\n"
            f"```python\n{code}\n```"
        )
        reply = (await self.transport.next_reply()).strip()
        if reply.lower() in ("", "ok", "keep"):
            return strategy_type, code, summary, triggers
        # Treat the reply as the new code body; strip a surrounding fenced block.
        return strategy_type, _strip_code_fence(reply), summary, triggers

    async def prompt_turn(self) -> str:
        """Await the operator's next message — post nothing.

        In chat there's no need for a ``>`` prompt each turn: the session's
        kickoff card already told the user to just type. So this only waits for the
        next thread reply.
        """
        return await self.transport.next_reply()

    # ------------------------------------------------------------------
    # Output (post Markdown, FIFO-ordered via transport.send)
    # ------------------------------------------------------------------

    def notice(self, text: str) -> None:
        self.transport.send(to_markdown(text))

    def panel(
        self,
        body: str,
        *,
        title: str | None = None,
        style: str | None = None,
        fit: bool = False,
    ) -> None:
        md = to_markdown(body)
        self.transport.send(f"**{title}**\n{md}" if title else md)

    def code(self, code: str, *, lang: str = "python") -> None:
        self.transport.send(f"```{lang}\n{code}\n```")

    def table(self, *, title: str, columns: list[Column], rows: list[list[str]]) -> None:
        self.transport.send(_markdown_table(title, columns, rows))

    def result(self, summary: str, *, title: str | None = None) -> None:
        md = to_markdown(summary)
        self.transport.send(f"**{title}**\n{md}" if title else md)

    def diff(
        self,
        rows: list[tuple[str, str, str]],
        *,
        edge_count: int,
        edges: list[tuple[str, str, str]] | None = None,
    ) -> None:
        new_count = sum(1 for _, _, a in rows if a == "new")
        merge_count = len(rows) - new_count
        summary = (
            f"**commit diff:** {new_count} new, {merge_count} will merge — "
            f"{edge_count} edge(s) to write."
        )
        # Mermaid graph: nodes coloured by action, with edges where available.
        # Renders as a diagram on Mermaid-capable instances; degrades to a
        # readable code block otherwise.
        diagram = _mermaid_graph(rows, edges or [])
        self.transport.send(f"{summary}\n{diagram}")

    @asynccontextmanager
    async def detail_stream(self, *, title: str):
        """Stream a turn's verbose detail into a live-updating, durable thread message.

        Active only when the session is verbose (``--verbose``) and a bot is
        attached; otherwise yields the inactive no-op sink so the caller skips the
        slower streaming path. The message is created on first content, edited in
        place as reasoning/answer stream (coalesced ~1/s), and frozen at the end —
        it stays in the thread (it is *not* the ephemeral session live post).
        """
        bot = getattr(self.transport, "_bot", None)
        channel_id = getattr(self.transport, "channel_id", None)
        if not self.verbose or bot is None or not channel_id:
            async with super().detail_stream(title=title) as sink:
                yield sink
            return
        msg = _LiveMessage(bot, channel_id, getattr(self.transport, "root_id", ""), title)
        flusher = asyncio.ensure_future(msg.run())
        try:
            yield msg
        finally:
            flusher.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await flusher
            with contextlib.suppress(Exception):
                await msg.finalize()

    async def review_orchestration(
        self,
        *,
        strategy_type: str,
        summary: str,
        triggers: list[str],
        code: str,
        options: list[ReviewOption],
        sink: DetailSink | None = None,
    ) -> str:
        """Show the proposal + code in one card and await a typed choice.

        When *sink* is the turn's live "Planning…" card, the card is **evolved in
        place** (its streamed reasoning replaced by the proposal); otherwise a fresh
        review message is posted. The operator replies with an option key/alias.
        """
        triggers_str = (
            "\n".join(f"  - {t}" for t in triggers) if triggers else "  - (none)"
        )
        code_shown = code if len(code) <= 3000 else code[:3000] + "\n… (truncated)"
        opts_line = " · ".join(f"`{o.key}` {o.label}" for o in options)
        # The post's main text is the reply instruction (the prominent line); the
        # card leads with the code (the thing to review), then the metadata.
        message = f"**Review** the orchestration code, then reply — {opts_line}"
        card = (
            f"```python\n{code_shown}\n```\n"
            f"**strategy:** {strategy_type}\n"
            f"**summary:** {summary or '(none)'}\n"
            f"**trigger:**\n{triggers_str}"
        )
        props = {"attachments": [{"color": "#e0a800", "text": card}]}
        if isinstance(sink, _LiveMessage) and sink.post_id:
            await sink.show_review(message, props)
        else:
            self.transport.send(message, props=props)
        keys = ", ".join(o.key for o in options)
        while True:
            answer = await self.transport.next_reply()
            choice = match_choice(answer, options)
            if choice is not None:
                return choice
            self.transport.send(f"_Please reply with one of: {keys}_")


def _strip_code_fence(text: str) -> str:
    m = re.search(r"```(?:[a-zA-Z]*)\n(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text


def _mermaid_label(text: str) -> str:
    """Sanitise a node label for a Mermaid quoted string."""
    return (text or "").replace('"', "'").replace("\n", " ").strip() or "(unnamed)"


def _mermaid_graph(
    rows: list[tuple[str, str, str]],
    edges: list[tuple[str, str, str]],
) -> str:
    """Render a ```mermaid flowchart of the commit diff (nodes + edges)."""
    lines = ["```mermaid", "graph TD"]
    idmap: dict[str, str] = {}
    for i, (ntype, name, action) in enumerate(rows):
        nid = f"n{i}"
        idmap[name] = nid
        label = _mermaid_label(f"{name} ({ntype})")
        cls = "new" if action == "new" else "merge"
        lines.append(f'  {nid}["{label}"]:::{cls}')
    for src, tgt, rtype in edges:
        s, t = idmap.get(src), idmap.get(tgt)
        if s and t:
            lines.append(f"  {s} -->|{_mermaid_label(rtype)}| {t}")
    lines.append("  classDef new fill:#2eb886,color:#ffffff;")
    lines.append("  classDef merge fill:#cccccc,color:#000000;")
    lines.append("```")
    return "\n".join(lines)


def _markdown_table(title: str, columns: list[Column], rows: list[list[str]]) -> str:
    headers = [c.header for c in columns]
    lines = [f"**{title}**", "", "| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        cells = [to_markdown(str(c)).replace("|", "\\|") for c in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
