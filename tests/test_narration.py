"""Tests for narrator→logging emission and the Mattermost narration handler."""

from __future__ import annotations

import asyncio
import logging

import pytest

from bamboo.frontends.mattermost import narration as narr
from bamboo.frontends.mattermost.narration import _LivePost, _DETAIL_ENTRIES
from bamboo.utils import narrator


# ---------------------------------------------------------------------------
# narrator → bamboo.narration logging emission
# ---------------------------------------------------------------------------


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def cap():
    lg = logging.getLogger("bamboo.narration")
    h = _Capture()
    lg.addHandler(h)
    old_level, old_prop = lg.level, lg.propagate
    lg.setLevel(logging.DEBUG)
    try:
        yield h
    finally:
        lg.removeHandler(h)
        lg.setLevel(old_level)
        lg.propagate = old_prop


def _kind(rec):
    return getattr(rec, "narration_kind", "line")


def test_say_emits_info_line(cap):
    narrator.say("hello")
    assert len(cap.records) == 1
    r = cap.records[0]
    assert r.levelno == logging.INFO and r.getMessage() == "hello" and _kind(r) == "line"


def test_say_debug_is_detail(cap):
    narrator.say("verbose detail", level=logging.DEBUG)
    assert cap.records[-1].levelno == logging.DEBUG


def test_warn_and_error_levels(cap):
    narrator.warn("careful")
    narrator.error("boom")
    assert cap.records[-2].levelno == logging.WARNING and cap.records[-2].getMessage() == "careful"
    assert cap.records[-1].levelno == logging.ERROR and cap.records[-1].getMessage() == "boom"


def test_thinking_emits_step_record(cap):
    with narrator.thinking("doing work"):
        pass
    steps = [r for r in cap.records if _kind(r) == "step"]
    assert steps and "doing work" in steps[0].getMessage()


def test_show_block_is_block_kind(cap):
    narrator.show_block("Title", "line1\nline2")
    blocks = [r for r in cap.records if _kind(r) == "block"]
    assert blocks and "Title" in blocks[0].getMessage() and "line1" in blocks[0].getMessage()


# ---------------------------------------------------------------------------
# _LivePost (fed by the handler)
# ---------------------------------------------------------------------------


class _FakeBot:
    def __init__(self, spinner_emoji="bamboo_spinner"):
        self.spinner_emoji = spinner_emoji
        self.created, self.patched, self.deleted = [], [], []
        self._n = 0

    async def create_post(self, channel_id, message, *, root_id="", props=None, file_ids=None):
        self._n += 1
        pid = f"post-{self._n}"
        self.created.append({"id": pid, "message": message})
        return {"id": pid}

    async def update_post(self, post_id, *, message=None, file_ids=None, props=None):
        self.patched.append({"id": post_id, "message": message})
        return {"id": post_id}

    async def delete_post(self, post_id):
        self.deleted.append(post_id)


def _live(bot, threshold=logging.INFO):
    return _LivePost(bot, "chan-1", "root-1", asyncio.get_event_loop(), threshold)


def _rec(msg, level=logging.INFO, kind=None):
    r = logging.LogRecord("bamboo.narration", level, __file__, 0, "%s", (msg,), None)
    if kind is not None:
        r.narration_kind = kind
    return r


@pytest.mark.asyncio
async def test_livepost_step_head_and_body_lines():
    bot = _FakeBot()
    post = _live(bot)
    post.feed(_rec("[nav] analysing", kind="step"))
    post.feed(_rec("looking at logs"))
    await post._flush_once()

    assert len(bot.created) == 1
    msg = bot.created[0]["message"]
    assert msg.startswith(":bamboo_spinner: 🔎") and "analysing" in msg
    assert "```" in msg and "→ looking at logs" in msg


@pytest.mark.asyncio
async def test_livepost_ignores_block_and_below_threshold():
    bot = _FakeBot()
    post = _live(bot, threshold=logging.INFO)
    post.feed(_rec("a prompt dump", kind="block"))   # block → never
    post.feed(_rec("debug detail", level=logging.DEBUG))  # below threshold
    await post._flush_once()
    assert bot.created == []  # nothing dirty → no post


@pytest.mark.asyncio
async def test_livepost_highlights_warning():
    bot = _FakeBot()
    post = _live(bot)
    post.feed(_rec("careful now", level=logging.WARNING))
    await post._flush_once()
    assert "⚠️ careful now" in bot.created[0]["message"]


@pytest.mark.asyncio
async def test_livepost_body_capped():
    bot = _FakeBot()
    post = _live(bot)
    for i in range(_DETAIL_ENTRIES + 10):
        post.feed(_rec(f"line {i}"))
    await post._flush_once()
    body = bot.created[-1]["message"]
    assert body.count("→ line") == _DETAIL_ENTRIES
    assert "→ line 9" not in body and f"→ line {_DETAIL_ENTRIES + 9}" in body


@pytest.mark.asyncio
async def test_livepost_falls_back_to_glyph_without_emoji():
    bot = _FakeBot(spinner_emoji=None)
    post = _live(bot)
    post.feed(_rec("[nav] analysing", kind="step"))
    await post._flush_once()
    head = bot.created[0]["message"]
    assert head.startswith("🔎 ") and ":bamboo_spinner:" not in head


@pytest.mark.asyncio
async def test_livepost_finalize_success_deletes():
    bot = _FakeBot()
    post = _live(bot)
    post.feed(_rec("[nav] analysing", kind="step"))
    await post._flush_once()
    await post.finalize(success=True)
    assert bot.deleted == ["post-1"]


@pytest.mark.asyncio
async def test_livepost_finalize_failure_keeps_static():
    bot = _FakeBot()
    post = _live(bot)
    post.feed(_rec("[nav] analysing", kind="step"))
    post.feed(_rec("looking at logs"))
    await post._flush_once()
    await post.finalize(success=False)
    last = bot.patched[-1]
    assert bot.deleted == []
    assert last["message"].startswith("🔎") and ":bamboo_spinner:" not in last["message"]
    assert "→ looking at logs" in last["message"]


@pytest.mark.asyncio
async def test_livepost_finalize_noop_when_nothing_streamed():
    bot = _FakeBot()
    post = _live(bot)
    await post.finalize(success=True)
    assert bot.created == [] and bot.patched == [] and bot.deleted == []


# ---------------------------------------------------------------------------
# stream_narration end-to-end (narrator → global handler → live post)
# ---------------------------------------------------------------------------


class _FakeTransport:
    def __init__(self, bot):
        self._bot = bot
        self.channel_id = "chan-1"
        self.root_id = "root-1"


@pytest.mark.asyncio
async def test_stream_narration_deletes_on_success():
    bot = _FakeBot()
    async with narr.stream_narration(_FakeTransport(bot)):
        narrator.say("doing work")
        await asyncio.sleep(0.1)  # let the flusher create the post
    assert bot.created, "expected a progress post"
    assert bot.deleted == [bot.created[-1]["id"]]


@pytest.mark.asyncio
async def test_stream_narration_keeps_on_failure():
    bot = _FakeBot()
    with pytest.raises(RuntimeError):
        async with narr.stream_narration(_FakeTransport(bot)):
            narrator.say("doing work")
            await asyncio.sleep(0.1)
            raise RuntimeError("boom")
    assert bot.created and bot.deleted == []
    assert bot.patched[-1]["message"].startswith("🔎")


@pytest.mark.asyncio
async def test_stream_narration_noop_without_bot():
    class _Bare:
        channel_id = None
    async with narr.stream_narration(_Bare()):
        narrator.say("nobody listening")  # no handler target → just logs
    # nothing to assert beyond "no exception"


# ---------------------------------------------------------------------------
# set_narrator routes logging through the Rich Console (CLI spinner coordination)
# ---------------------------------------------------------------------------


def test_route_logging_through_console_swaps_handlers():
    import io as _io
    import logging as _logging
    from rich.console import Console
    from bamboo.utils.narrator import _ConsoleLogHandler, _route_logging_through_console

    root = _logging.getLogger()
    saved_handlers = list(root.handlers)
    stray = _logging.StreamHandler()  # mimic setup_logging's plain stdout handler
    root.addHandler(stray)
    try:
        console = Console(file=_io.StringIO(), force_terminal=False, width=200)
        _route_logging_through_console(console)
        # The plain StreamHandler is gone; exactly one console handler is installed.
        assert stray not in root.handlers
        assert sum(isinstance(h, _ConsoleLogHandler) for h in root.handlers) == 1
        # Idempotent: a second call doesn't add another.
        _route_logging_through_console(console)
        assert sum(isinstance(h, _ConsoleLogHandler) for h in root.handlers) == 1
    finally:
        root.handlers[:] = saved_handlers


def test_console_log_handler_renders_through_console():
    import io as _io
    import logging as _logging
    from rich.console import Console
    from bamboo.utils.narrator import _ConsoleLogHandler

    buf = _io.StringIO()
    h = _ConsoleLogHandler(Console(file=buf, force_terminal=False, width=200))
    rec = _logging.LogRecord("bamboo.demo", _logging.INFO, __file__, 0, "rendered-line", None, None)
    h.emit(rec)
    assert "rendered-line" in buf.getvalue()
