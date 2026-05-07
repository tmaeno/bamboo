"""Tests for bamboo.utils.panda_client — PanDA data-fetching helper."""

from unittest.mock import MagicMock, patch

import pytest

from bamboo.utils.panda_client import fetch_task_data

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SAMPLE_TASK = {
    "jediTaskID": 12345,
    "status": "failed",
    "taskName": "user.test.task",
    "userName": "atlasprod",
    "ramCount": 2000,
    "walltime": 7200,
}


def _patched_http_client(status=0, response=None):
    """Return a (context_manager, mock_instance) pair that stubs HttpClient.

    ``_call_api`` does ``from pandaserver.api.v1.http_client import HttpClient``
    at call time, so we inject a fake module into sys.modules the same way the
    original tests injected a fake pandaclient module.

    ``status``   — 0 for a completed HTTP exchange, 255 for connection error.
    ``response`` — the second element of the (status, response) tuple returned
                   by HttpClient.get / HttpClient.post.
    """
    mock_instance = MagicMock()
    mock_instance.get.return_value = (status, response)
    mock_instance.post.return_value = (status, response)

    mock_class = MagicMock(return_value=mock_instance)

    mock_module = MagicMock()
    mock_module.HttpClient = mock_class
    mock_module.api_url_ssl = "https://mock-panda.example.com/api/v1"

    return (
        patch.dict(
            "sys.modules",
            {
                "pandaserver.api.v1.http_client": mock_module,
            },
        ),
        mock_instance,
    )


def _success_response(data):
    return {"success": True, "data": data, "message": ""}


def _error_response(message="task not found"):
    return {"success": False, "data": None, "message": message}


# ---------------------------------------------------------------------------
# fetch_task_data — happy path
# ---------------------------------------------------------------------------


class TestFetchTaskDataSuccess:
    @pytest.mark.asyncio
    async def test_returns_dict_on_success(self):
        ctx, _ = _patched_http_client(status=0, response=_success_response(_SAMPLE_TASK))
        with ctx:
            result = await fetch_task_data(12345)
        assert result == _SAMPLE_TASK

    @pytest.mark.asyncio
    async def test_accepts_string_task_id(self):
        ctx, mock_instance = _patched_http_client(
            status=0, response=_success_response(_SAMPLE_TASK)
        )
        with ctx:
            result = await fetch_task_data("12345")
        assert result["jediTaskID"] == 12345
        # The underlying call must have received an int, not a string.
        _, call_kwargs = mock_instance.get.call_args
        args, _ = mock_instance.get.call_args
        assert args[1].get("task_id") == 12345

    @pytest.mark.asyncio
    async def test_all_fields_preserved(self):
        large_task = {k: f"v{i}" for i, k in enumerate(["a", "b", "c", "d"])}
        ctx, _ = _patched_http_client(
            status=0, response=_success_response(large_task)
        )
        with ctx:
            result = await fetch_task_data(9999)
        assert result == large_task


# ---------------------------------------------------------------------------
# fetch_task_data — error cases
# ---------------------------------------------------------------------------


class TestFetchTaskDataErrors:
    @pytest.mark.asyncio
    async def test_raises_runtime_error_on_connection_failure(self):
        ctx, _ = _patched_http_client(status=255, response="connection refused")
        with ctx:
            with pytest.raises(RuntimeError, match="task_id=99999"):
                await fetch_task_data(99999)

    @pytest.mark.asyncio
    async def test_raises_runtime_error_on_api_error(self):
        """Status 0 but success=False must raise RuntimeError."""
        ctx, _ = _patched_http_client(
            status=0, response=_error_response("task not found")
        )
        with ctx:
            with pytest.raises(RuntimeError, match="task not found"):
                await fetch_task_data(12345)

    @pytest.mark.asyncio
    async def test_raises_value_error_on_non_numeric_task_id(self):
        """A non-numeric string task_id must raise ValueError before any network call."""
        ctx, mock_instance = _patched_http_client(
            status=0, response=_success_response(_SAMPLE_TASK)
        )
        with ctx:
            with pytest.raises(ValueError):
                await fetch_task_data("not-a-number")
        mock_instance.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_error_message_includes_task_id(self):
        """The RuntimeError message must include the offending task_id."""
        ctx, _ = _patched_http_client(status=255, response="err")
        with ctx:
            with pytest.raises(RuntimeError, match="99999"):
                await fetch_task_data(99999)

    @pytest.mark.asyncio
    async def test_error_message_mentions_panda_api_url(self):
        """The RuntimeError message should hint at PANDA_API_URL_SSL."""
        ctx, _ = _patched_http_client(status=255, response="err")
        with ctx:
            with pytest.raises(RuntimeError, match="PANDA_API_URL_SSL"):
                await fetch_task_data(12345)
