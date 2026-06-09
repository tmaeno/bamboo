"""Tests for narrator→logging emission and the Mattermost narration handler."""

from __future__ import annotations

import asyncio
import logging
import re

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


def test_diag_is_not_the_narration_logger():
    """Operational post-I/O diagnostics must NOT be logged on 'bamboo.narration'.

    The MattermostLogHandler is attached to that logger, so logging there would
    feed records back into the live post — a per-patch breadcrumb would re-dirty
    the post and loop the flusher (~1 Hz update_post) in a verbose session.
    """
    assert narr._diag.name == "bamboo.frontends.mattermost.narration"

    narration_logger = logging.getLogger("bamboo.narration")
    prev_level = narration_logger.level
    narration_logger.setLevel(logging.DEBUG)
    cap_h = _Capture()
    narration_logger.addHandler(cap_h)
    try:
        narr._diag.debug("operational breadcrumb must not reach the live-post handler")
        assert cap_h.records == []  # did not propagate to bamboo.narration's handler
    finally:
        narration_logger.removeHandler(cap_h)
        narration_logger.setLevel(prev_level)


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
        self.created.append({"id": pid, "message": message, "props": props})
        return {"id": pid}

    async def update_post(self, post_id, *, message=None, file_ids=None, props=None):
        self.patched.append({"id": post_id, "message": message, "props": props})
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


def _body(post: dict) -> str:
    """The body attachment's text (where progress lines now live)."""
    atts = (post.get("props") or {}).get("attachments") or []
    return atts[0]["text"] if atts else ""


@pytest.mark.asyncio
async def test_livepost_step_head_and_body_lines():
    bot = _FakeBot()
    post = _live(bot)
    post.feed(_rec("[nav] analysing", kind="step"))
    post.feed(_rec("looking at logs"))
    await post._flush_once()

    assert len(bot.created) == 1
    # Head (message): spinner emoji + the step, with the [module] prefix stripped.
    msg = bot.created[0]["message"]
    assert msg.startswith(":bamboo_spinner:") and "analysing" in msg and "[nav]" not in msg
    assert "looking at logs" not in msg  # body is in the card, not the head
    # Body (attachment card): **HH:MM:SS** + green accent + the message; no code block.
    body = _body(bot.created[0])
    assert "looking at logs" in body  # message present (accent emoji varies per pool)
    assert "```" not in body
    assert re.search(r"\*\*\d\d:\d\d:\d\d\*\*", body)  # bold timestamp
    assert bot.created[0]["props"]["attachments"][0]["color"]  # status-colored bar


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
    assert "⚠️ careful now" in _body(bot.created[0])


@pytest.mark.asyncio
async def test_livepost_escapes_markdown_in_message():
    bot = _FakeBot()
    post = _live(bot)
    post.feed(_rec("find_similar a*b ~c~"))  # underscores/asterisks/tildes
    await post._flush_once()
    body = _body(bot.created[0])
    assert "find\\_similar" in body and "a\\*b" in body and "\\~c\\~" in body


@pytest.mark.asyncio
async def test_livepost_body_newest_first():
    bot = _FakeBot()
    post = _live(bot)
    post.feed(_rec("first"))
    post.feed(_rec("second"))
    await post._flush_once()
    body = _body(bot.created[0])
    assert body.index("second") < body.index("first")  # newest on top


@pytest.mark.asyncio
async def test_livepost_cycles_green_emoji():
    from bamboo.frontends.mattermost.narration import _GREEN_POOL

    bot = _FakeBot()
    post = _live(bot)
    for i in range(len(_GREEN_POOL)):
        post.feed(_rec(f"milestone {i}"))
    await post._flush_once()
    body = _body(bot.created[0])
    used = {e for e in _GREEN_POOL if e in body}
    assert len(used) >= 2  # not all the same green emoji


@pytest.mark.asyncio
async def test_livepost_replaces_url_with_clickable_link():
    bot = _FakeBot()
    post = _live(bot)
    post.feed(_rec("fetched 9 chars from https://h.example/x?a&b"))
    await post._flush_once()
    body = _body(bot.created[0])
    # Clickable markdown link with the raw URL as the destination.
    assert "[link](https://h.example/x?a&b)" in body
    assert "\x00" not in body  # no placeholder remnants


@pytest.mark.asyncio
async def test_livepost_body_capped():
    bot = _FakeBot()
    post = _live(bot)
    for i in range(_DETAIL_ENTRIES + 10):
        post.feed(_rec(f"line {i}"))
    await post._flush_once()
    body = _body(bot.created[-1])
    # Green accent varies per line, so count entries by the hard-break join.
    assert len(body.split("  \n")) == _DETAIL_ENTRIES
    assert "line 9" not in body and f"line {_DETAIL_ENTRIES + 9}" in body


@pytest.mark.asyncio
async def test_livepost_falls_back_to_glyph_without_emoji():
    bot = _FakeBot(spinner_emoji=None)
    post = _live(bot)
    post.feed(_rec("[nav] analysing", kind="step"))
    await post._flush_once()
    head = bot.created[0]["message"]
    assert head.startswith("🔎 ") and ":bamboo_spinner:" not in head


@pytest.mark.asyncio
async def test_livepost_finalize_success_freezes_done():
    bot = _FakeBot()
    post = _live(bot)
    post.feed(_rec("[nav] analysing", kind="step"))
    post.feed(_rec("looking at logs"))
    await post._flush_once()
    await post.finalize(success=True)
    # Frozen to a terse "done" line — body card cleared, post NOT deleted.
    last = bot.patched[-1]
    assert bot.deleted == []
    assert "done" in last["message"] and "looking at logs" not in last["message"]
    # The patch must explicitly clear attachments ([], not None) — None leaves the
    # existing card in place (the driver drops a None props).
    assert last["props"] == {"attachments": []}
    assert _body(last) == ""


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
    assert "looking at logs" in _body(last)  # last lines kept in the (red) card


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
async def test_stream_narration_freezes_done_on_success():
    bot = _FakeBot()
    async with narr.stream_narration(_FakeTransport(bot)):
        narrator.say("doing work")
        await asyncio.sleep(0.1)  # let the flusher create the post
    assert bot.created, "expected a progress post"
    assert bot.deleted == []  # kept, not deleted
    assert "done" in bot.patched[-1]["message"]
    assert bot.patched[-1]["props"] == {"attachments": []}  # body card cleared


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


def _post_text(bot) -> str:
    """All body/message text across created + patched posts."""
    chunks: list[str] = []
    for entry in (*bot.created, *bot.patched):
        chunks.append(entry.get("message") or "")
        for att in (entry.get("props") or {}).get("attachments", []):
            chunks.append(att.get("text", ""))
    return "\n".join(chunks)


@pytest.mark.asyncio
async def test_stream_narration_verbose_surfaces_debug():
    """`--verbose` lowers this session's post threshold to DEBUG."""
    bot = _FakeBot()
    async with narr.stream_narration(_FakeTransport(bot), verbose=True):
        narrator.say("behind the scenes detail", level=logging.DEBUG)
        await asyncio.sleep(0.1)
    assert "behind the scenes detail" in _post_text(bot)


@pytest.mark.asyncio
async def test_stream_narration_default_drops_debug():
    """Without `--verbose`, a DEBUG line stays out of the post (INFO threshold)."""
    bot = _FakeBot()
    async with narr.stream_narration(_FakeTransport(bot)):
        narrator.say("behind the scenes detail", level=logging.DEBUG)
        await asyncio.sleep(0.1)
    assert "behind the scenes detail" not in _post_text(bot)


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
