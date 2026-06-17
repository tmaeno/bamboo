"""PandaMcpClient brokerage tool: result-shaping moved into the tool.

``_fetch_brokerage_context`` now parses each fetched brokerage log into a
structured ``summary`` (keyed by URL) instead of returning raw logs and leaving
the parse to a downstream consumer. These tests pin the summary derivation and
the ``sites_of_interest`` construction.
"""

from __future__ import annotations

import pytest

from bamboo.mcp.panda_mcp_client import (
    PandaMcpClient,
    _brokerage_sites_of_interest,
    _parse_brokerage_summaries,
)

# Minimal but parseable brokerage log: the "initial N candidates" body line the
# parser keys on, one skip line for a site of interest, plus a summary funnel
# table (header + initial/final counts + two stages).
_MINI_BROKERAGE_LOG = """\
2026-01-01 00:00:01.001000 : initial 100 candidates
2026-01-01 00:00:01.051000 :   skip site=SITE_A/SCORE  consider SITE_A unsuitable for the user due to long queue criteria=-badsite
2026-01-01 00:00:01.063000 : ===== Job brokerage summary =====
2026-01-01 00:00:01.065000 : the number of initial candidates: 100
2026-01-01 00:00:01.066000 :  100 ->  40 candidates,  60% cut : input data check
2026-01-01 00:00:01.069000 :    5 ->   0 candidates, 100% cut : final check
2026-01-01 00:00:01.070000 : the number of final candidates: 0
"""


# ---------------------------------------------------------------------------
# sites_of_interest derivation
# ---------------------------------------------------------------------------


def test_sites_of_interest_combines_site_and_included_string():
    sites = _brokerage_sites_of_interest(
        {"site": "SITE_A", "includedSite": "SITE_B, SITE_C"}
    )
    assert sites == ["SITE_A", "SITE_B", "SITE_C"]


def test_sites_of_interest_dedups_and_accepts_list():
    sites = _brokerage_sites_of_interest(
        {"site": "SITE_A", "includedSite": ["SITE_A", "SITE_B"]}
    )
    assert sites == ["SITE_A", "SITE_B"]


def test_sites_of_interest_handles_missing_fields():
    assert _brokerage_sites_of_interest({}) == []
    assert _brokerage_sites_of_interest(None) == []


# ---------------------------------------------------------------------------
# per-URL summary parsing
# ---------------------------------------------------------------------------


def test_parse_brokerage_summaries_keys_by_url_and_annotates_sites():
    summaries = _parse_brokerage_summaries(
        {"http://logs/x": _MINI_BROKERAGE_LOG},
        {"site": "SITE_A/SCORE"},
    )
    assert "http://logs/x" in summaries
    summary = summaries["http://logs/x"]
    assert summary["initial_candidates"] == 100
    assert summary["final_candidates"] == 0
    assert summary["terminal_filter"] == "final check"
    assert summary["sites_of_interest"] == ["SITE_A/SCORE"]
    assert any(f["site"] == "SITE_A/SCORE" for f in summary["sites_of_interest_fates"])


def test_parse_brokerage_summaries_skips_logs_without_summary_section():
    summaries = _parse_brokerage_summaries(
        {"http://logs/x": "just some unrelated log text"}, {}
    )
    assert summaries == {}


# ---------------------------------------------------------------------------
# the tool handler wires fetch → parse → summary
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_brokerage_context_includes_summary(monkeypatch):
    client = PandaMcpClient()

    monkeypatch.setattr(
        "bamboo.mcp.panda_mcp_client.extract_log_urls",
        lambda _err: ["http://logs/x"],
    )

    async def _fake_fetch(url):  # noqa: ANN001
        return _MINI_BROKERAGE_LOG

    monkeypatch.setattr(
        "bamboo.mcp.panda_mcp_client.async_fetch_log_content", _fake_fetch
    )

    async def _fake_doc(self):  # noqa: ANN001
        return ("BROKERAGE DOC", "advanced/brokerage.rst")

    monkeypatch.setattr(PandaMcpClient, "_fetch_brokerage_doc", _fake_doc)

    result = await client._fetch_brokerage_context(
        {"errorDialog": "<a href='http://logs/x'>log</a>", "site": "SITE_A/SCORE"}
    )

    assert set(result) == {"logs", "brokerage_doc", "doc_path", "summary"}
    assert result["logs"] == {"http://logs/x": _MINI_BROKERAGE_LOG}
    assert result["brokerage_doc"] == "BROKERAGE DOC"
    assert "http://logs/x" in result["summary"]
    assert result["summary"]["http://logs/x"]["sites_of_interest"] == ["SITE_A/SCORE"]
