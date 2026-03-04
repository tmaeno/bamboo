"""Tests for bamboo.data.panda_client — PanDA data-fetching helper."""

import sys
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


def _patched_pandaclient(status=0, data=None):
    """Return a (context_manager, mock_client) pair.

    ``fetch_task_data`` does ``from pandaclient import Client`` at call time, so
    we mock ``pandaclient`` in ``sys.modules`` so that the import resolves to our
    stub before any real network call is made.
    """
    mock_client = MagicMock()
    mock_client.get_task_details_json.return_value = (status, data)

    mock_pandaclient = MagicMock()
    mock_pandaclient.Client = mock_client

    return patch.dict("sys.modules", {"pandaclient": mock_pandaclient}), mock_client


# ---------------------------------------------------------------------------
# fetch_task_data — happy path
# ---------------------------------------------------------------------------


class TestFetchTaskDataSuccess:
    @pytest.mark.asyncio
    async def test_returns_dict_on_success(self):
        ctx, _ = _patched_pandaclient(status=0, data=_SAMPLE_TASK)
        with ctx:
            result = await fetch_task_data(12345)
        assert result == _SAMPLE_TASK

    @pytest.mark.asyncio
    async def test_accepts_string_task_id(self):
        ctx, mock_client = _patched_pandaclient(status=0, data=_SAMPLE_TASK)
        with ctx:
            result = await fetch_task_data("12345")
        assert result["jediTaskID"] == 12345
        # The underlying call must have received an int, not a string.
        mock_client.get_task_details_json.assert_called_once_with(12345)

    @pytest.mark.asyncio
    async def test_all_fields_preserved(self):
        large_task = {k: f"v{i}" for i, k in enumerate(["a", "b", "c", "d"])}
        ctx, _ = _patched_pandaclient(status=0, data=large_task)
        with ctx:
            result = await fetch_task_data(9999)
        assert result == large_task


# ---------------------------------------------------------------------------
# fetch_task_data — error cases
# ---------------------------------------------------------------------------


class TestFetchTaskDataErrors:
    @pytest.mark.asyncio
    async def test_raises_runtime_error_on_nonzero_status(self):
        ctx, _ = _patched_pandaclient(status=255, data=None)
        with ctx:
            with pytest.raises(RuntimeError, match="status=255"):
                await fetch_task_data(99999)

    @pytest.mark.asyncio
    async def test_raises_runtime_error_when_data_is_none(self):
        """Status 0 but data=None must still raise (server returned empty body)."""
        ctx, _ = _patched_pandaclient(status=0, data=None)
        with ctx:
            with pytest.raises(RuntimeError, match="status=0"):
                await fetch_task_data(12345)

    @pytest.mark.asyncio
    async def test_raises_runtime_error_when_data_is_list(self):
        """If the server sends a list instead of a dict, raise RuntimeError."""
        ctx, _ = _patched_pandaclient(status=0, data=[{"key": "val"}])
        with ctx:
            with pytest.raises(RuntimeError, match="Unexpected response type"):
                await fetch_task_data(12345)

    @pytest.mark.asyncio
    async def test_raises_runtime_error_when_data_is_string(self):
        ctx, _ = _patched_pandaclient(status=0, data="error message from server")
        with ctx:
            with pytest.raises(RuntimeError, match="Unexpected response type"):
                await fetch_task_data(12345)

    @pytest.mark.asyncio
    async def test_raises_value_error_on_non_numeric_task_id(self):
        """A non-numeric string task_id must raise ValueError before any network call."""
        ctx, mock_client = _patched_pandaclient(status=0, data=_SAMPLE_TASK)
        with ctx:
            with pytest.raises(ValueError):
                await fetch_task_data("not-a-number")
        # The network call must never have been made.
        mock_client.get_task_details_json.assert_not_called()

    @pytest.mark.asyncio
    async def test_raises_import_error_when_pandaclient_missing(self):
        """ImportError is raised with a helpful message when panda-client-light is absent."""
        saved = {k: v for k, v in sys.modules.items() if "pandaclient" in k}
        for key in list(saved):
            del sys.modules[key]

        import builtins
        real_import = builtins.__import__

        def _block_pandaclient(name, *args, **kwargs):
            if name == "pandaclient" or name.startswith("pandaclient."):
                raise ImportError(f"No module named '{name}'")
            return real_import(name, *args, **kwargs)

        try:
            with patch("builtins.__import__", side_effect=_block_pandaclient):
                with pytest.raises(ImportError, match="panda-client-light"):
                    await fetch_task_data(12345)
        finally:
            sys.modules.update(saved)

    @pytest.mark.asyncio
    async def test_error_message_includes_task_id(self):
        """The RuntimeError message must include the offending task_id."""
        ctx, _ = _patched_pandaclient(status=255, data=None)
        with ctx:
            with pytest.raises(RuntimeError, match="99999"):
                await fetch_task_data(99999)

    @pytest.mark.asyncio
    async def test_error_message_mentions_panda_url(self):
        """The RuntimeError message should hint at PANDA_URL env vars."""
        ctx, _ = _patched_pandaclient(status=255, data=None)
        with ctx:
            with pytest.raises(RuntimeError, match="PANDA_URL"):
                await fetch_task_data(12345)

