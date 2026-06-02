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
from bamboo.frontends.mattermost.bot import (
    SPINNER_EMOJI_NAME,
    Command,
    MattermostBot,
    _is_addressed,
    parse_command,
)
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
        self.patched: list[dict] = []
        self.deleted: list[str] = []

    async def create_post(self, **kwargs):
        self.created.append(kwargs)
        return {"id": f"post-{len(self.created)}"}

    async def patch_post(self, post_id, **kwargs):
        self.patched.append({"post_id": post_id, **kwargs})
        return {"id": post_id}

    async def delete_post(self, post_id):
        self.deleted.append(post_id)


class _Users:
    async def get_user(self, _id):
        return {"id": "bot-user"}


class _Channels:
    def __init__(self) -> None:
        self.direct: list[list[str]] = []

    async def create_direct_channel(self, options):
        self.direct.append(options)
        return {"id": "dm-chan"}


class _Files:
    def __init__(self) -> None:
        self.uploaded: list[dict] = []

    async def upload_file(self, files=None, channel_id=None):
        self.uploaded.append({"files": files, "channel_id": channel_id})
        return {"file_infos": [{"id": "file-1"}]}


class _EmojiImageResp:
    def __init__(self, content: bytes) -> None:
        self.content = content


class _Emoji:
    def __init__(self, *, existing=False, stored_image=b"", create_raises=False) -> None:
        self._existing = existing  # emoji already present on the server
        self._stored_image = stored_image  # bytes the existing emoji holds
        self._create_raises = create_raises
        self.created: list[dict] = []
        self.deleted: list[str] = []

    async def get_emoji_by_name(self, name):
        if self._existing:
            return {"id": "emoji-1", "name": name}
        raise RuntimeError("not found")  # MM returns 404 for a missing emoji

    async def get_emoji_image(self, emoji_id):
        return _EmojiImageResp(self._stored_image)

    async def delete_emoji(self, emoji_id):
        self.deleted.append(emoji_id)
        self._existing = False

    async def create_emoji(self, image=None, emoji=None):
        if self._create_raises:
            raise RuntimeError("custom emoji disabled")
        self.created.append({"image": image, "emoji": emoji})
        self._existing = True
        return {"id": "emoji-1"}


class FakeDriver:
    def __init__(self, *, emoji=None) -> None:
        self.posts = _Posts()
        self.users = _Users()
        self.channels = _Channels()
        self.files = _Files()
        self.emoji = emoji or _Emoji()
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


def test_parse_command_analyze():
    assert parse_command("analyze 12345") == Command(kind="analyze", task_id=12345)
    assert parse_command("@bamboo analyze 7").task_id == 7
    assert parse_command("/bamboo analyze 9") == Command(kind="analyze", task_id=9)
    cmd = parse_command("analyze")
    assert cmd.kind == "analyze" and cmd.task_id is None


def test_parse_command_help():
    assert parse_command("help") == Command(kind="help")
    assert parse_command("@bamboo help").kind == "help"
    assert parse_command("/bamboo help") == Command(kind="help")


def test_is_addressed():
    assert _is_addressed("@bamboo foo", "bamboo") is True
    assert _is_addressed("/bamboo foo", "bamboo") is True
    assert _is_addressed("bamboo foo", "bamboo") is True
    assert _is_addressed("@bamboo", "bamboo") is True
    assert _is_addressed("hello team", "bamboo") is False
    assert _is_addressed("@someoneelse foo", "bamboo") is False


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
async def test_unknown_command_replies_help_only_when_addressed():
    started: list[str] = []

    async def run_session(transport, command):
        started.append(command.kind)

    bot = _make_bot(run_session)
    bot.bot_user_id = "bot-user"
    bot.bot_username = "bamboo"

    # Addressed but unrecognized → falls back to a help session.
    await bot.handle_event(_posted_event("chan-ok", "u1", "@bamboo bogus", post_id="p1"))
    await asyncio.sleep(0.05)
    assert started == ["help"]

    # Ordinary chatter that doesn't address the bot → ignored, no session.
    await bot.handle_event(_posted_event("chan-ok", "u1", "hello team", post_id="p2"))
    await asyncio.sleep(0.05)
    assert started == ["help"]
    assert "p2" not in bot._sessions


@pytest.mark.asyncio
async def test_update_upload_delete_post_call_driver():
    bot = _make_bot(lambda *a: None)
    bot.bot_user_id = "bot-user"

    await bot.update_post("p1", message="edited", file_ids=[])
    assert bot.driver.posts.patched[-1] == {"post_id": "p1", "message": "edited", "file_ids": []}

    fid = await bot.upload_file("chan-1", "spinner.gif", b"GIF89a...")
    assert fid == "file-1"
    up = bot.driver.files.uploaded[-1]
    assert up["channel_id"] == "chan-1"
    # `files` must be a flat (filename, bytes, content_type) tuple — NOT nested in a
    # dict (the driver wraps it as {"files": files}; double-wrapping breaks the upload).
    assert isinstance(up["files"], tuple) and up["files"][0] == "spinner.gif"
    assert up["files"][1] == b"GIF89a..."

    await bot.delete_post("p1")
    assert bot.driver.posts.deleted == ["p1"]


def _bundled_spinner_bytes() -> bytes:
    import importlib.resources as ir

    return ir.files("bamboo.frontends.mattermost").joinpath("assets/spinner.gif").read_bytes()


@pytest.mark.asyncio
async def test_ensure_spinner_emoji_creates_when_missing():
    driver = FakeDriver(emoji=_Emoji(existing=False))
    bot = MattermostBot(driver, allowed_channels={"chan-ok"}, run_session=lambda *a: None)
    bot.bot_user_id = "bot-user"

    await bot.ensure_spinner_emoji()

    assert bot.spinner_emoji == SPINNER_EMOJI_NAME
    assert len(driver.emoji.created) == 1
    created = driver.emoji.created[0]
    # multipart: image as a flat tuple, emoji as a JSON string with name + creator.
    assert isinstance(created["image"], tuple) and created["image"][0] == "spinner.gif"
    meta = json.loads(created["emoji"])
    assert meta == {"name": SPINNER_EMOJI_NAME, "creator_id": "bot-user"}


@pytest.mark.asyncio
async def test_ensure_spinner_emoji_reuses_when_image_matches():
    # Existing emoji whose stored image equals the bundled gif → reuse, no churn.
    driver = FakeDriver(emoji=_Emoji(existing=True, stored_image=_bundled_spinner_bytes()))
    bot = MattermostBot(driver, allowed_channels={"chan-ok"}, run_session=lambda *a: None)
    bot.bot_user_id = "bot-user"

    await bot.ensure_spinner_emoji()

    assert bot.spinner_emoji == SPINNER_EMOJI_NAME
    assert driver.emoji.created == []  # matching image → no create
    assert driver.emoji.deleted == []  # ...and no delete


@pytest.mark.asyncio
async def test_ensure_spinner_emoji_refreshes_when_image_stale():
    # Existing emoji with a different (stale) image → delete + recreate.
    driver = FakeDriver(emoji=_Emoji(existing=True, stored_image=b"OLD-16px-gif"))
    bot = MattermostBot(driver, allowed_channels={"chan-ok"}, run_session=lambda *a: None)
    bot.bot_user_id = "bot-user"

    await bot.ensure_spinner_emoji()

    assert bot.spinner_emoji == SPINNER_EMOJI_NAME
    assert driver.emoji.deleted == ["emoji-1"]  # stale one removed
    assert len(driver.emoji.created) == 1  # ...and recreated from the bundled gif


@pytest.mark.asyncio
async def test_ensure_spinner_emoji_degrades_on_error():
    driver = FakeDriver(emoji=_Emoji(existing=False, create_raises=True))
    bot = MattermostBot(driver, allowed_channels={"chan-ok"}, run_session=lambda *a: None)
    bot.bot_user_id = "bot-user"

    await bot.ensure_spinner_emoji()  # must not raise

    assert bot.spinner_emoji is None  # graceful degrade → narration uses a glyph


@pytest.mark.asyncio
async def test_run_in_dm_opens_dm_and_runs_callback():
    bot = _make_bot(lambda *a: None)
    bot.bot_user_id = "bot-user"

    async def run(io):
        io.notice("hello in dm")
        return "ok"

    result = await bot.run_in_dm("u1", run)

    assert result == "ok"
    # A DM channel was opened between bot and user...
    assert bot.driver.channels.direct == [["bot-user", "u1"]]
    # ...and the callback's output was posted there (not the public channel).
    posted = bot.driver.posts.created
    assert any(c.get("channel_id") == "dm-chan" and "hello in dm" in c.get("message", "") for c in posted)


@pytest.mark.asyncio
async def test_run_session_help_lists_commands():
    from bamboo.frontends.mattermost import serve

    t = FakeTransport()
    await serve._run_session(t, Command(kind="help"))

    blob = "\n".join(t.sent)
    for name in ("investigate", "capture", "analyze", "login", "status", "help"):
        assert name in blob


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
