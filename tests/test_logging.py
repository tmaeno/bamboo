"""Tests for logging configuration — the Neo4j notification one-liner filter.

The Neo4j driver logs each server notification as the full GqlStatusObject repr
plus the entire query (a NotificationPrinter arg). ``_Neo4jNotificationFilter``
rewrites that record into a concise, still-informative one line. These tests use
fakes shaped like the driver's NotificationPrinter (no live Neo4j).
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

from bamboo.utils.logging import _Neo4jNotificationFilter


def _record(arg) -> logging.LogRecord:
    # Mirrors the driver's call: warning("Received notification ...: %s", printer)
    return logging.LogRecord(
        name="neo4j.notifications",
        level=logging.WARNING,
        pathname=__file__,
        lineno=1,
        msg="Received notification from DBMS server: %s",
        args=(arg,),
        exc_info=None,
    )


def _fake_printer():
    pos = SimpleNamespace(line=2, column=35)
    note = SimpleNamespace(
        gql_status="01N52",
        classification=SimpleNamespace(value="UNRECOGNIZED"),
        status_description="property key `auto_run` does not exist",
        position=pos,
    )
    query = "MATCH (p:Procedure)\nRETURN coalesce(p.auto_run, false) AS auto_run"
    return SimpleNamespace(notification=note, query=query)


def test_filter_condenses_notification_to_one_line():
    rec = _record(_fake_printer())
    assert _Neo4jNotificationFilter().filter(rec) is True

    msg = rec.getMessage()
    assert "01N52" in msg
    assert "UNRECOGNIZED" in msg
    assert "auto_run" in msg
    assert "line 2" in msg and "col 35" in msg
    # the offending query line is included…
    assert "coalesce(p.auto_run, false)" in msg
    # …and the verbose dump is gone
    assert "GqlStatusObject" not in msg
    assert "\n" not in msg  # genuinely one line
    assert rec.args == ()


def test_filter_passes_through_unexpected_records():
    # A record whose arg isn't a NotificationPrinter must be left untouched.
    rec = _record("some other %s")
    rec2 = logging.LogRecord(
        "neo4j.notifications", logging.WARNING, __file__, 1, "plain message", (), None
    )
    assert _Neo4jNotificationFilter().filter(rec) is True
    assert _Neo4jNotificationFilter().filter(rec2) is True
    assert rec2.getMessage() == "plain message"
