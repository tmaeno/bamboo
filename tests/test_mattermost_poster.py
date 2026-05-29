"""Unit tests for the Phase 1 Mattermost wedge.

Covers the pure attachment builder, the poster (with an injected post backend so
no client/server is needed), and the driver-options URL parsing.  The
third-party Mattermost client is never imported here.
"""

from __future__ import annotations

import pytest

from bamboo.config import Settings
from bamboo.frontends.mattermost import render
from bamboo.frontends.mattermost.driver import _driver_options
from bamboo.frontends.mattermost.poster import post_analysis
from bamboo.models.knowledge_entity import AnalysisResult


def _result(**overrides) -> AnalysisResult:
    base = dict(
        task_id="12345",
        root_cause="scout jobs OOMed under memory pressure",
        confidence=0.82,
        resolution="resubmit to a high-memory queue",
        explanation="The scout phase failed; all failed jobs hit the memory limit.",
    )
    base.update(overrides)
    return AnalysisResult(**base)


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------


def test_confidence_color_bands():
    assert render.confidence_color(0.9) == render._COLOR_HIGH
    assert render.confidence_color(0.7) == render._COLOR_HIGH
    assert render.confidence_color(0.5) == render._COLOR_MEDIUM
    assert render.confidence_color(0.4) == render._COLOR_MEDIUM
    assert render.confidence_color(0.1) == render._COLOR_LOW


def test_analysis_attachment_core_fields():
    att = render.analysis_attachment(_result())
    assert att["color"] == render._COLOR_HIGH
    assert "12345" in att["title"]
    titles = {f["title"]: f["value"] for f in att["fields"]}
    assert "memory pressure" in titles["Root cause"]
    assert titles["Confidence"] == "82%"
    assert titles["Task"] == "12345"
    assert "high-memory queue" in titles["Suggested resolution"]
    # Explanation goes in the secondary text, not a field.
    assert "scout phase failed" in att["text"]
    assert att["footer"] == "bamboo"


def test_analysis_attachment_includes_novel_symptoms_and_gaps():
    att = render.analysis_attachment(
        _result(
            unmatched_symptoms=["brand new error string"],
            capability_gaps=[
                {"investigation": "check pilot logs", "suggested_tool_capability": "fetch_pilot_log"}
            ],
        )
    )
    titles = {f["title"]: f["value"] for f in att["fields"]}
    assert "brand new error string" in titles["Novel symptoms (no KB match)"]
    assert "fetch_pilot_log" in titles["Capability gaps"]


def test_analysis_attachment_omits_optional_sections_when_empty():
    att = render.analysis_attachment(_result())
    field_titles = {f["title"] for f in att["fields"]}
    assert "Novel symptoms (no KB match)" not in field_titles
    assert "Capability gaps" not in field_titles


def test_analysis_message_wraps_attachment():
    msg = render.analysis_message(_result())
    assert "attachments" in msg
    assert len(msg["attachments"]) == 1


# ---------------------------------------------------------------------------
# poster
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_post_analysis_uses_injected_backend():
    captured = {}

    async def fake_post(channel_id, message, props):
        captured["channel_id"] = channel_id
        captured["message"] = message
        captured["props"] = props
        return {"id": "post-1"}

    resp = await post_analysis("chan-9", _result(), post=fake_post)

    assert resp == {"id": "post-1"}
    assert captured["channel_id"] == "chan-9"
    assert captured["message"] == ""
    assert "attachments" in captured["props"]
    assert captured["props"]["attachments"][0]["title"].endswith("12345")


# ---------------------------------------------------------------------------
# driver options
# ---------------------------------------------------------------------------


def test_driver_options_parses_https_url():
    opts = _driver_options(
        Settings(mattermost_url="https://mattermost.cern.ch", mattermost_token="tok")
    )
    assert opts == {
        "url": "mattermost.cern.ch",
        "scheme": "https",
        "port": 443,
        "token": "tok",
        "basepath": "/api/v4",
    }


def test_driver_options_honours_explicit_port():
    opts = _driver_options(
        Settings(mattermost_url="https://mm.example.com:8065", mattermost_token="tok")
    )
    assert opts["port"] == 8065
    assert opts["url"] == "mm.example.com"


def test_driver_options_requires_url_and_token():
    with pytest.raises(ValueError):
        _driver_options(Settings(mattermost_url="", mattermost_token="tok"))
    with pytest.raises(ValueError):
        _driver_options(Settings(mattermost_url="https://mm", mattermost_token=""))
