"""Unit tests for narrator→sink forwarding and the Mattermost narration sink."""

from __future__ import annotations

import asyncio
import logging

import pytest

import re

from bamboo.frontends.mattermost.narration import (
    MattermostNarrationSink,
    _DETAIL_ENTRIES,
)
from bamboo.utils import narrator


# ---------------------------------------------------------------------------
# narrator → sink forwarding
# ---------------------------------------------------------------------------


class _RecordingSink:
    def __init__(self):
        self.says, self.blocks, self.steps = [], [], []

    def say(self, msg):
        self.says.append(msg)

    def block(self, title, content):
        self.blocks.append((title, content))

    def step(self, label):
        self.steps.append(label)


def test_narrator_forwards_to_sink():
    sink = _RecordingSink()
    token = narrator.set_narration_sink(sink)
    try:
        narrator.say("hello")  # forwarded regardless of verbose flag
        narrator.show_block("Title", "line1\nline2")
        with narrator.thinking("doing work"):
            pass
    finally:
        narrator.reset_narration_sink(token)

    assert sink.says == ["hello"]
    assert sink.blocks == [("Title", "line1\nline2")]
    assert len(sink.steps) == 1 and "doing work" in sink.steps[0]


def test_narrator_silent_without_sink_or_console():
    # No sink, no console → all no-ops, no exceptions.
    narrator.say("nobody hears this")
    narrator.show_block("t", "c")
    with narrator.thinking("x"):
        pass


# ---------------------------------------------------------------------------
# MattermostNarrationSink
# ---------------------------------------------------------------------------


class _FakeBot:
    def __init__(self, spinner_emoji="bamboo_spinner"):
        self.spinner_emoji = spinner_emoji
        self.uploaded = []
        self.created = []
        self.patched = []
        self.deleted = []
        self._n = 0

    async def upload_file(self, channel_id, filename, data):
        self.uploaded.append((channel_id, filename, len(data)))
        return "gif-1"

    async def create_post(self, channel_id, message, *, root_id="", props=None, file_ids=None):
        self._n += 1
        pid = f"post-{self._n}"
        self.created.append(
            {"id": pid, "channel_id": channel_id, "message": message, "props": props, "file_ids": file_ids}
        )
        return {"id": pid}

    async def update_post(self, post_id, *, message=None, file_ids=None, props=None):
        self.patched.append({"id": post_id, "message": message, "props": props, "file_ids": file_ids})
        return {"id": post_id}

    async def delete_post(self, post_id):
        self.deleted.append(post_id)


def _sink(bot):
    return MattermostNarrationSink(bot, "chan-1", "root-1", asyncio.get_event_loop())


_TS = re.compile(r"\d\d:\d\d:\d\d")  # HH:MM:SS console timestamp prefix


@pytest.mark.asyncio
async def test_sink_console_log_post_then_patches(caplog):
    bot = _FakeBot()
    sink = _sink(bot)

    with caplog.at_level(logging.INFO, logger="bamboo.narration"):
        sink.step("[nav] analysing")
        sink.say("looking at logs")
        await sink._flush_once()
        # First flush: ONE live post; head carries the spinner emoji + step, the
        # say line is a timestamped entry inside a monospace code block.
        assert bot.uploaded == [] and len(bot.created) == 1
        msg = bot.created[0]["message"]
        assert msg.startswith(":bamboo_spinner: 🔎") and "analysing" in msg
        assert "```" in msg and "→ looking at logs" in msg
        body = msg.split("```")[1]
        assert _TS.search(body)  # the say line is timestamped

        # A later update edits the SAME post (no new post; both lines present).
        sink.say("found an error")
        sink.step("[nav] root cause")
        await sink._flush_once()
        assert len(bot.created) == 1
        last = bot.patched[-1]
        assert last["id"] == "post-1" and "root cause" in last["message"]
        assert "→ looking at logs" in last["message"] and "→ found an error" in last["message"]

    # Firehose stays at INFO; the post-plumbing diagnostics are NOT at INFO anymore.
    assert any("looking at logs" in r.message for r in caplog.records)
    info_msgs = [r.getMessage() for r in caplog.records if r.name == "bamboo.narration"]
    assert not any(m.startswith("narration: created post id=") for m in info_msgs)
    assert not any(m.startswith("narration: flush ") for m in info_msgs)


@pytest.mark.asyncio
async def test_sink_diagnostics_at_debug(caplog):
    bot = _FakeBot()
    sink = _sink(bot)
    with caplog.at_level(logging.DEBUG, logger="bamboo.narration"):
        sink.say("hello")
        await sink._flush_once()
    msgs = [r.getMessage() for r in caplog.records if r.name == "bamboo.narration"]
    assert any(m.startswith("narration: created post id=") for m in msgs)  # available at DEBUG


@pytest.mark.asyncio
async def test_sink_falls_back_to_glyph_without_emoji():
    bot = _FakeBot(spinner_emoji=None)
    sink = _sink(bot)
    sink.step("[nav] analysing")
    await sink._flush_once()
    head = bot.created[0]["message"]
    assert head.startswith("🔎 ") and ":bamboo_spinner:" not in head


@pytest.mark.asyncio
async def test_sink_log_window_capped_and_logged(caplog):
    bot = _FakeBot()
    sink = _sink(bot)
    with caplog.at_level(logging.INFO, logger="bamboo.narration"):
        for i in range(_DETAIL_ENTRIES + 10):
            sink.say(f"line {i}")
        await sink._flush_once()

    body = bot.created[-1]["message"]
    # Only the last _DETAIL_ENTRIES survive in the live post...
    assert "→ line 9" not in body  # evicted
    assert f"→ line {_DETAIL_ENTRIES + 9}" in body
    assert body.count("→ line") == _DETAIL_ENTRIES
    # ...but the full firehose reached the log.
    assert sum("line" in r.message for r in caplog.records) >= _DETAIL_ENTRIES + 10


@pytest.mark.asyncio
async def test_sink_block_is_log_only_not_sent_to_mm(caplog):
    bot = _FakeBot()
    sink = _sink(bot)
    with caplog.at_level(logging.INFO, logger="bamboo.narration"):
        sink.block("LLM prompt: root cause", "PROMPT-" + "x" * 5000)  # big block
        await sink._flush_once()

    # Nothing posted to MM for a block (only say/step drive the live post).
    assert bot.created == [] and bot.patched == []
    # ...but the full block content reached the log.
    assert any("PROMPT-" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_sink_finalize_freezes_post_to_done():
    bot = _FakeBot()
    sink = _sink(bot)
    sink.step("[nav] analysing")
    await sink._flush_once()
    await sink.finalize()

    last = bot.patched[-1]
    assert last["id"] == "post-1"
    # Frozen to a static completion line; the spinner emoji is gone from the head.
    assert last["message"].startswith("✓ done")
    assert ":bamboo_spinner:" not in last["message"]


@pytest.mark.asyncio
async def test_sink_finalize_noop_when_nothing_streamed():
    bot = _FakeBot()
    sink = _sink(bot)
    await sink.finalize()
    assert bot.created == [] and bot.patched == [] and bot.uploaded == []
