"""Thin async wrapper around ``pandaclient.Client.get_task_details_json``.

This module provides :func:`fetch_task_data` — a single async helper that
retrieves full task details from a live PanDA server and returns them as a
plain Python ``dict`` ready for use as ``task_data`` in the Bamboo pipeline.

PanDA server configuration
--------------------------
The underlying ``panda-client-light`` library reads the server URL from the
environment:

* ``PANDA_URL``      – plain HTTP base URL  (default: ``http://pandaserver.cern.ch:25080``)
* ``PANDA_URL_SSL``  – HTTPS base URL       (default: ``https://pandaserver.cern.ch``)

Set these variables (or place them in your ``.env`` file) to point at a
different PanDA instance (e.g. a development server).

The function runs the blocking ``Client.get_task_details_json`` call in a
thread-pool executor so it can be awaited from async code without
blocking the event loop.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Status code returned by panda-client-light on success.
_PANDA_OK = 0


async def fetch_task_data(task_id: int | str, verbose: bool = False) -> dict[str, Any]:
    """Fetch full task details from PanDA and return them as a ``dict``.

    The function calls :func:`pandaclient.Client.get_task_details_json` in a
    thread-pool executor so it does not block the event loop.

    Args:
        task_id: The PanDA ``jediTaskID`` to look up (int or numeric string).
        verbose: If ``True``, the underlying ``panda-client-light`` library
            will print every curl command it constructs and the raw server
            response to stdout, which is useful for diagnosing network or
            authentication issues.

    Returns:
        A ``dict`` containing the task details as returned by the PanDA
        server, ready to be passed as ``task_data`` to
        :class:`~bamboo.agents.knowledge_accumulator.KnowledgeAccumulator`
        or any extraction strategy.

    Raises:
        ImportError:  If ``panda-client-light`` is not installed.
        RuntimeError: If the PanDA server returns a non-zero status code or
                      ``None`` data.
        ValueError:   If *task_id* cannot be converted to ``int``.
    """
    try:
        from pandaclient import Client  # noqa: PLC0415  (conditional import)
    except ImportError as exc:
        raise ImportError(
            "panda-client-light is required to fetch task data from PanDA. "
            "Install it with: pip install panda-client-light"
        ) from exc

    task_id_int = int(task_id)

    if verbose:
        logging.getLogger("bamboo").setLevel(logging.DEBUG)

    logger.info("Fetching task details from PanDA for task_id=%s", task_id_int)

    loop = asyncio.get_event_loop()
    status, data = await loop.run_in_executor(
        None,
        lambda: Client.get_task_details_json(task_id_int, verbose=verbose),
    )

    if status != _PANDA_OK or data is None:
        raise RuntimeError(
            f"PanDA returned status={status} for task_id={task_id_int}. "
            "Check that the task ID is valid and the PanDA server is reachable "
            "(PANDA_URL / PANDA_URL_SSL environment variables)."
        )

    if not isinstance(data, dict):
        raise RuntimeError(
            f"Unexpected response type from PanDA: {type(data).__name__!r}. "
            f"Expected dict, got: {data!r}"
        )

    logger.info(
        "Successfully fetched task details for task_id=%s (%d fields)",
        task_id_int,
        len(data),
    )
    return data
