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

import re
from abc import ABC, abstractmethod
from typing import Any, Optional

from bamboo.frontends.base import Column, InteractionIO

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


class MattermostInteractionIO(InteractionIO):
    """Chat adapter bound to one :class:`ThreadTransport`."""

    def __init__(self, transport: ThreadTransport) -> None:
        self.transport = transport

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
