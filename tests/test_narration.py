"""Unit tests for narrator→sink forwarding and the Mattermost narration sink."""

from __future__ import annotations

import asyncio
import logging

import pytest

from bamboo.frontends.mattermost.narration import (
    MattermostNarrationSink,
    _DETAIL_LINES,
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
    def __init__(self):
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
        self.created.append({"id": pid, "channel_id": channel_id, "message": message, "file_ids": file_ids})
        return {"id": pid}

    async def update_post(self, post_id, *, message=None, file_ids=None):
        self.patched.append({"id": post_id, "message": message, "file_ids": file_ids})
        return {"id": post_id}

    async def delete_post(self, post_id):
        self.deleted.append(post_id)


def _sink(bot):
    return MattermostNarrationSink(bot, "chan-1", "root-1", asyncio.get_event_loop())


@pytest.mark.asyncio
async def test_sink_creates_status_with_gif_then_patches_on_step(caplog):
    bot = _FakeBot()
    sink = _sink(bot)

    with caplog.at_level(logging.INFO, logger="bamboo.narration"):
        sink.step("[nav] analysing")
        await sink._flush_once()
        # First status flush: gif uploaded + status post created with it.
        assert bot.uploaded == [("chan-1", "spinner.gif", 704)] or bot.uploaded[0][1] == "spinner.gif"
        status = bot.created[0]
        assert status["file_ids"] == ["gif-1"]
        assert "analysing" in status["message"] and status["message"].startswith("🔎")

        # A later step edits the same status post (no new post, gif stays).
        sink.step("[nav] root cause")
        await sink._flush_once()
        assert bot.patched and "root cause" in bot.patched[-1]["message"]
        assert len([c for c in bot.created if c["id"] == "post-1"]) == 1

    # Every event was logged for dev debugging.
    assert any("analysing" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_sink_detail_capped_and_logged(caplog):
    bot = _FakeBot()
    sink = _sink(bot)
    with caplog.at_level(logging.INFO, logger="bamboo.narration"):
        for i in range(_DETAIL_LINES + 10):
            sink.say(f"line {i}")
        await sink._flush_once()

    detail = bot.created[-1]["message"]
    # Only the last _DETAIL_LINES survive in the MM post...
    assert "line 9" not in detail  # evicted
    assert f"line {_DETAIL_LINES + 9}" in detail
    assert detail.count("→ line") == _DETAIL_LINES
    # ...but the full firehose reached the log.
    assert sum("line" in r.message for r in caplog.records) >= _DETAIL_LINES + 10


@pytest.mark.asyncio
async def test_sink_finalize_freezes_status_and_drops_gif():
    bot = _FakeBot()
    sink = _sink(bot)
    sink.step("[nav] analysing")
    await sink._flush_once()
    await sink.finalize()

    last = bot.patched[-1]
    assert last["id"] == "post-1"
    assert last["message"].startswith("✓ done")
    assert last["file_ids"] == []  # gif removed


@pytest.mark.asyncio
async def test_sink_finalize_noop_when_nothing_streamed():
    bot = _FakeBot()
    sink = _sink(bot)
    await sink.finalize()
    assert bot.created == [] and bot.patched == [] and bot.uploaded == []
