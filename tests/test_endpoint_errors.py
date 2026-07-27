"""Unit tests for :mod:`bamboo.utils.errors` — the service-agnostic diagnostics.

Same defect as the LLM case, one layer out: a dead Qdrant, an unreachable Neo4j and
a blocked api.github.com all raised the identical ``All connection attempts failed``
with nothing naming the service or URL, all at WARNING as if they were minor.

``bamboo.llm.errors`` builds its provider/model variant on these primitives, so the
connection detection and the severity rule are tested once, here.
"""

from __future__ import annotations

import json
import logging

import httpx
import pytest

from bamboo.utils import errors

_QDRANT = "http://127.0.0.1:6333"


def _anyio_style_error() -> OSError:
    """The exact exception anyio raises from ``connect_tcp`` — a plain OSError."""
    return OSError("All connection attempts failed")


def _wrapped(depth: int) -> Exception:
    """An anyio OSError re-wrapped ``depth`` times, as the client libraries do."""
    exc: BaseException = _anyio_style_error()
    for i in range(depth):
        try:
            raise (httpx.ConnectError("connect failed") if i == 0 else RuntimeError("sdk wrapper")) from exc
        except Exception as raised:  # noqa: BLE001
            exc = raised
    return exc  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# is_connection_error
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("depth", [0, 1, 2, 3])
def test_connection_error_detected_through_the_cause_chain(depth):
    """Client libraries re-wrap transport errors, so the detector must walk __cause__."""
    exc = _anyio_style_error() if depth == 0 else _wrapped(depth)
    assert errors.is_connection_error(exc) is True


def test_httpx_connect_error_detected_although_it_is_not_an_oserror():
    """httpx.ConnectError derives from httpx.HTTPError, not OSError — both families
    have to be listed or one of the two real-world shapes slips through."""
    assert not isinstance(httpx.ConnectError("x"), OSError)
    assert errors.is_connection_error(httpx.ConnectError("x")) is True


@pytest.mark.parametrize(
    "exc",
    [
        ValueError("Expected 5 items, got 3"),
        json.JSONDecodeError("Expecting value", "not json", 0),
        KeyError("nl_query"),
    ],
)
def test_non_transport_errors_are_not_connection_errors(exc):
    assert errors.is_connection_error(exc) is False


def test_detector_tolerates_a_self_referential_chain():
    """A cyclic __context__ must not hang the diagnostic path."""
    a = RuntimeError("a")
    b = RuntimeError("b")
    a.__context__ = b
    b.__context__ = a
    assert errors.is_connection_error(a) is False


# --------------------------------------------------------------------------- #
# describe_endpoint_failure
# --------------------------------------------------------------------------- #


def test_message_names_the_service_url_and_hint():
    msg = errors.describe_endpoint_failure(
        "PandaDocNavigator: failed to read index meta",
        _wrapped(2),
        service="qdrant",
        endpoint=_QDRANT,
        fallback="treating the doc index as absent",
    )
    assert "PandaDocNavigator: failed to read index meta" in msg
    assert f"qdrant={_QDRANT}" in msg
    assert "docker compose up -d" in msg
    assert "QDRANT_URL" in msg
    assert "treating the doc index as absent" in msg
    assert "\n" not in msg  # one line, so it survives grep


@pytest.mark.parametrize(
    "service,expected",
    [
        ("qdrant", "QDRANT_URL"),
        ("neo4j", "NEO4J_URI"),
        ("github", "HTTPS_PROXY"),
    ],
)
def test_hint_is_per_service(service, expected):
    msg = errors.describe_endpoint_failure(
        "x", _anyio_style_error(), service=service, endpoint="http://h:1"
    )
    assert expected in msg


def test_unknown_service_gets_the_generic_hint():
    msg = errors.describe_endpoint_failure(
        "x", _anyio_style_error(), service="something-new", endpoint="http://h:1"
    )
    assert "something-new=http://h:1" in msg
    assert "HTTPS_PROXY" in msg


def test_multiline_exception_text_is_collapsed_to_one_line():
    """neo4j's ServiceUnavailable embeds a newline in str(exc); left alone it splits
    the record across log lines and you can no longer grep for the whole thing."""
    exc = OSError("Couldn't connect to 127.0.0.1:3\nFailed to establish connection")
    msg = errors.describe_endpoint_failure(
        "Failed to connect to Neo4j", exc, service="neo4j", endpoint="bolt://127.0.0.1:3"
    )
    assert "\n" not in msg
    assert "Couldn't connect to 127.0.0.1:3 Failed to establish connection" in msg
    assert "NEO4J_URI" in msg


def test_explicit_hint_overrides_the_default():
    msg = errors.describe_endpoint_failure(
        "x", _anyio_style_error(), service="qdrant", endpoint=_QDRANT, hint="do the thing"
    )
    assert "hint: do the thing" in msg
    assert "docker compose" not in msg


def test_no_hint_for_a_non_transport_error():
    """A 404 or a bad payload reached the service — 'where do I point this' is not
    the question, so the advice would be noise."""
    msg = errors.describe_endpoint_failure(
        "x", ValueError("bad payload"), service="qdrant", endpoint=_QDRANT
    )
    assert "hint:" not in msg
    assert f"qdrant={_QDRANT}" in msg  # the endpoint is still worth knowing


# --------------------------------------------------------------------------- #
# severity
# --------------------------------------------------------------------------- #


def test_connection_failure_logs_at_error(caplog):
    logger = logging.getLogger("bamboo.test.endpoint_errors")
    with caplog.at_level(logging.DEBUG, logger=logger.name):
        errors.log_endpoint_failure(
            logger, "ctx", _wrapped(2), service="qdrant", endpoint=_QDRANT
        )
    assert [r.levelno for r in caplog.records] == [logging.ERROR]
    assert _QDRANT in caplog.records[0].getMessage()


def test_other_failures_stay_at_warning(caplog):
    logger = logging.getLogger("bamboo.test.endpoint_errors")
    with caplog.at_level(logging.DEBUG, logger=logger.name):
        errors.log_endpoint_failure(
            logger, "ctx", ValueError("bad payload"), service="qdrant", endpoint=_QDRANT
        )
    assert [r.levelno for r in caplog.records] == [logging.WARNING]


def test_exc_info_captures_the_passed_exception_not_the_ambient_one(caplog):
    """Called outside any ``except`` block, so a bare ``exc_info=True`` would record
    ``(None, None, None)``. The traceback must come from the exception we passed."""
    logger = logging.getLogger("bamboo.test.endpoint_errors")
    exc = ValueError("x")
    with caplog.at_level(logging.DEBUG, logger=logger.name):
        errors.log_diagnostic(logger, "msg", exc, exc_info=True)
    assert caplog.records[0].exc_info[1] is exc


def test_exc_info_defaults_off(caplog):
    logger = logging.getLogger("bamboo.test.endpoint_errors")
    with caplog.at_level(logging.DEBUG, logger=logger.name):
        errors.log_diagnostic(logger, "msg", ValueError("x"))
    assert caplog.records[0].exc_info is None
