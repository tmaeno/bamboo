"""Unit tests for the Phase 2 Mattermost bot + interaction adapter.

No live server and no Mattermost client: the adapter is exercised over a fake
transport, and the bot's routing/registry over a fake driver. Tests assert the
reply-based ask/confirm/edit semantics, Markdown rendering, command parsing, the
channel allow-list, own-message suppression, thread↔session routing, and session
cleanup.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from bamboo.agents.investigation_session import InvestigationOrchestrator, _Deps
from bamboo.frontends.mattermost.bot import Command, MattermostBot, parse_command
from bamboo.frontends.mattermost.io import (
    MattermostInteractionIO,
    ThreadTransport,
    to_markdown,
)
from bamboo.frontends.base import Column


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeTransport(ThreadTransport):
    def __init__(self, replies: list[str] | None = None) -> None:
        self.sent: list[str] = []
        self._replies = iter(replies or [])

    def send(self, text, *, props=None) -> None:
        self.sent.append(text)

    async def next_reply(self) -> str:
        return next(self._replies)


class _Posts:
    def __init__(self) -> None:
        self.created: list[dict] = []

    async def create_post(self, **kwargs):
        self.created.append(kwargs)
        return {"id": f"post-{len(self.created)}"}


class _Users:
    async def get_user(self, _id):
        return {"id": "bot-user"}


class _Channels:
    def __init__(self) -> None:
        self.direct: list[list[str]] = []

    async def create_direct_channel(self, options):
        self.direct.append(options)
        return {"id": "dm-chan"}


class FakeDriver:
    def __init__(self) -> None:
        self.posts = _Posts()
        self.users = _Users()
        self.channels = _Channels()
        self.logged_in = False

    async def login(self):
        self.logged_in = True

    async def init_websocket(self, handler):  # not used directly in tests
        return None


def _posted_event(channel_id, user_id, message, *, post_id="p1", root_id=""):
    post = {
        "id": post_id,
        "channel_id": channel_id,
        "user_id": user_id,
        "message": message,
        "root_id": root_id,
    }
    return {"event": "posted", "data": {"post": json.dumps(post)}}


# ---------------------------------------------------------------------------
# parse_command
# ---------------------------------------------------------------------------


def test_parse_command_investigate_with_task_id():
    cmd = parse_command("investigate 12345")
    assert cmd == Command(kind="investigate", task_id=12345)


def test_parse_command_handles_mention_and_slash_prefixes():
    assert parse_command("@bamboo investigate 7").task_id == 7
    assert parse_command("/bamboo investigate 9").task_id == 9
    assert parse_command("bamboo capture").kind == "capture"


def test_parse_command_investigate_without_task_id():
    cmd = parse_command("investigate")
    assert cmd.kind == "investigate" and cmd.task_id is None


def test_parse_command_returns_none_for_chatter():
    assert parse_command("hello team, any updates?") is None
    assert parse_command("") is None


def test_parse_command_status():
    assert parse_command("status") == Command(kind="status")
    assert parse_command("@bamboo status").kind == "status"
    assert parse_command("/bamboo status").kind == "status"


# ---------------------------------------------------------------------------
# to_markdown
# ---------------------------------------------------------------------------


def test_to_markdown_converts_bold_and_strips_colors():
    assert to_markdown("[bold]hi[/bold]") == "**hi**"
    assert to_markdown("[yellow]warn[/yellow]") == "warn"
    assert "**" in to_markdown("[bold]x[/bold] [red]y[/red]")


# ---------------------------------------------------------------------------
# MattermostInteractionIO
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ask_returns_reply_and_posts_prompt():
    t = FakeTransport(["the answer"])
    io = MattermostInteractionIO(t)
    assert await io.ask("[bold]your finding?[/bold]") == "the answer"
    assert t.sent[0].startswith("**your finding?**")


@pytest.mark.asyncio
async def test_ask_reasks_on_invalid_choice():
    t = FakeTransport(["maybe", "y"])
    io = MattermostInteractionIO(t)
    assert await io.ask("Proceed?", choices=["y", "N", "edit"]) == "y"
    # A re-ask nudge was posted after the invalid "maybe".
    assert any("Please reply with one of" in m for m in t.sent)


@pytest.mark.asyncio
async def test_ask_empty_reply_returns_default():
    t = FakeTransport([""])
    io = MattermostInteractionIO(t)
    assert await io.ask("resolution?", default="") == ""


@pytest.mark.asyncio
async def test_confirm_parses_yes_no():
    assert await MattermostInteractionIO(FakeTransport(["yes"])).confirm("ok?") is True
    assert await MattermostInteractionIO(FakeTransport(["n"])).confirm("ok?") is False


@pytest.mark.asyncio
async def test_edit_keeps_code_on_ok_and_replaces_otherwise():
    keep = await MattermostInteractionIO(FakeTransport(["ok"])).edit(
        strategy_type="s", code="orig", summary="sum", triggers=["t"]
    )
    assert keep == ("s", "orig", "sum", ["t"])

    replaced = await MattermostInteractionIO(
        FakeTransport(["```python\nnew_code()\n```"])
    ).edit(strategy_type="s", code="orig", summary="sum", triggers=["t"])
    assert replaced[1] == "new_code()"


def test_render_methods_post_markdown():
    t = FakeTransport()
    io = MattermostInteractionIO(t)
    io.notice("[dim]noted[/dim]")
    io.code("print(1)", lang="python")
    io.table(title="t", columns=[Column("a"), Column("b")], rows=[["1", "2"]])
    io.diff([("Symptom", "x", "new"), ("Cause", "y", "merge")], edge_count=3)
    blob = "\n".join(t.sent)
    assert "```python\nprint(1)\n```" in blob
    assert "| a | b |" in blob
    assert "1 new, 1 will merge — 3 edge(s)" in blob


def test_diff_renders_mermaid_graph_with_edges():
    t = FakeTransport()
    io = MattermostInteractionIO(t)
    io.diff(
        [("Symptom", "high failure", "new"), ("Cause", "OOM", "merge")],
        edge_count=1,
        edges=[("high failure", "OOM", "indicate")],
    )
    blob = "\n".join(t.sent)
    assert "```mermaid" in blob and "graph TD" in blob
    # Nodes carry type labels and action classes; the edge carries its label.
    assert "high failure (Symptom)" in blob
    assert ":::new" in blob and ":::merge" in blob
    assert "-->|indicate|" in blob


# ---------------------------------------------------------------------------
# Bot routing
# ---------------------------------------------------------------------------


def _make_bot(run_session):
    return MattermostBot(
        FakeDriver(), allowed_channels={"chan-ok"}, run_session=run_session
    )


@pytest.mark.asyncio
async def test_bot_ignores_disallowed_channel_and_own_messages():
    calls = []

    async def run_session(transport, command):
        calls.append(command)

    bot = _make_bot(run_session)
    bot.bot_user_id = "bot-user"

    await bot.handle_event(_posted_event("other-chan", "u1", "investigate 1"))
    await bot.handle_event(_posted_event("chan-ok", "bot-user", "investigate 1"))
    await bot.handle_event(_posted_event("chan-ok", "u1", "just chatting"))
    await asyncio.sleep(0)
    assert calls == []
    assert bot._sessions == {}


@pytest.mark.asyncio
async def test_bot_starts_session_and_routes_replies():
    started = asyncio.Event()
    got_reply = []

    async def run_session(transport, command):
        started.set()
        transport.send("kickoff")
        reply = await transport.next_reply()
        got_reply.append(reply)

    bot = _make_bot(run_session)
    bot.bot_user_id = "bot-user"

    # Start command (rooted at p1).
    await bot.handle_event(_posted_event("chan-ok", "u1", "investigate 42", post_id="p1"))
    await asyncio.wait_for(started.wait(), timeout=1.0)
    assert "p1" in bot._sessions

    # A reply in that thread (root_id=p1) is routed to the session.
    await bot.handle_event(
        _posted_event("chan-ok", "u1", "memory pressure", post_id="p2", root_id="p1")
    )
    await asyncio.sleep(0.05)
    assert got_reply == ["memory pressure"]

    # Session completes → cleaned up, and kickoff was actually posted via driver.
    await asyncio.sleep(0.05)
    assert bot._sessions == {}
    posted = [c["message"] for c in bot.driver.posts.created]
    assert "kickoff" in posted


@pytest.mark.asyncio
async def test_login_session_posts_to_direct_message_channel():
    async def run_session(transport, command):
        transport.send("auth prompt")

    bot = _make_bot(run_session)
    bot.bot_user_id = "bot-user"

    await bot.handle_event(_posted_event("chan-ok", "u1", "login", post_id="p1"))
    await asyncio.sleep(0.05)

    # A DM channel was opened between the bot and the invoking user...
    assert bot.driver.channels.direct == [["bot-user", "u1"]]
    # ...and the prompt was posted there, not in the public channel.
    assert len(bot.driver.posts.created) == 1
    post = bot.driver.posts.created[0]
    assert post["channel_id"] == "dm-chan"
    assert post["message"] == "auth prompt"
    assert post["root_id"] == ""  # not threaded under the public command post


# ---------------------------------------------------------------------------
# Real orchestrator over the Mattermost adapter (drop-in proof)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_real_orchestrator_runs_and_commits_over_mattermost_io():
    """The InvestigationOrchestrator's run()+finalize() loop drives unchanged
    over MattermostInteractionIO: a scripted thread (/done → cause → resolution →
    yes) reaches commit, calling the KnowledgeAccumulator storage methods."""
    graph_db = MagicMock()
    graph_db.get_node_description = AsyncMock(return_value=None)

    accumulator = MagicMock()
    accumulator.store_extracted = AsyncMock(return_value=("summary", []))

    transport = FakeTransport(["/done", "the cause", "the resolution", "yes"])
    io = MattermostInteractionIO(transport)
    deps = _Deps(
        mcp_client=MagicMock(),
        graph_db=graph_db,
        knowledge_accumulator=accumulator,
        io=io,
    )
    orch = InvestigationOrchestrator(deps=deps, session_id="mm-integ", max_turns=10)

    await orch.run()       # consumes "/done" → resolved
    await orch.finalize()  # cause → resolution → diff → confirm "yes" → commit

    assert orch.session.status == "resolved"
    accumulator.store_extracted.assert_awaited_once()
    # The cause/resolution the human typed made it into the committed graph.
    names = {n.name for n in orch.session.partial_graph.nodes}
    assert "the cause" in names
    assert "the resolution" in names


# ---------------------------------------------------------------------------
# status command
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_status_snapshot_reports_functional():
    bot = _make_bot(lambda *a: None)
    bot.bot_user_id = "bot-user"
    bot._started_at = 100.0
    bot._sessions = {"p1": object(), "p2": object()}  # type: ignore[dict-item]

    snap = await bot.status_snapshot()

    assert snap.functional is True
    assert snap.detail == ""
    assert snap.bot_user_id == "bot-user"
    assert snap.active_sessions == 2
    assert snap.allowed_channels == 1
    assert snap.uptime_seconds is not None and snap.uptime_seconds > 0


@pytest.mark.asyncio
async def test_status_snapshot_degraded_when_driver_fails():
    bot = _make_bot(lambda *a: None)
    bot.bot_user_id = "bot-user"

    async def boom(_id):
        raise RuntimeError("server unreachable")

    bot.driver.users.get_user = boom

    snap = await bot.status_snapshot()

    assert snap.functional is False
    assert "server unreachable" in snap.detail
    assert snap.uptime_seconds is None  # never started


@pytest.mark.asyncio
async def test_run_session_status_posts_summary():
    from bamboo.frontends.mattermost import serve
    from bamboo.frontends.mattermost.bot import BotStatus

    class _StatusTransport(FakeTransport):
        async def bot_status(self):
            return BotStatus(
                functional=True,
                bot_user_id="bot-user",
                active_sessions=3,
                allowed_channels=2,
                uptime_seconds=3725.0,  # 1h 2m 5s
            )

    t = _StatusTransport()
    await serve._run_session(t, Command(kind="status"))

    blob = "\n".join(t.sent)
    assert "bamboo bot status" in blob
    assert "✓ functional" in blob
    assert "Active sessions: 3" in blob
    assert "Allowed channels: 2" in blob
    assert "1h 2m" in blob
