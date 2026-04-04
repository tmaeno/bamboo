"""Abstract base classes for MCP-style tool clients.

An :class:`McpClient` exposes a catalogue of named :class:`McpTool` descriptors
to an LLM (via :meth:`McpClient.list_tools`) and executes specific tool calls
(via :meth:`McpClient.execute`).  Concrete subclasses implement the actual
data-fetching logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class McpTool:
    """Descriptor for a single callable tool exposed by an :class:`McpClient`.

    Attributes:
        name:              Unique tool identifier used in LLM output and
                           dispatch (must match what :meth:`McpClient.execute`
                           accepts as ``tool_name``).
        description:       Plain-English description the LLM reads to decide
                           whether to call this tool.  Should describe **when**
                           to use it (which problem it solves), not just what
                           it does.
        parameters_schema: JSON Schema ``object`` describing the tool's
                           keyword arguments.  Must match the ``**kwargs``
                           accepted by the concrete ``execute`` implementation.
        metadata:          Optional extra fields (e.g. cost hints, rate
                           limits) — not shown to the LLM by default.
    """

    name: str
    description: str
    parameters_schema: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


class McpClient(ABC):
    """Abstract base for MCP-style tool clients.

    Subclasses register tools in ``__init__`` and implement :meth:`execute`
    to dispatch calls to the appropriate handler.

    In-process clients (e.g. :class:`~bamboo.mcp.PandaMcpClient`) do not need
    a network connection, so :meth:`connect` and :meth:`close` are no-ops by
    default.  Clients that wrap an external server (e.g.
    :class:`~bamboo.mcp.ExternalMcpClient`) override them to manage the
    connection lifecycle.
    """

    @property
    def name(self) -> str:
        """Human-readable client identifier used in log messages and UI."""
        return type(self).__name__

    async def connect(self) -> None:
        """Establish connection and discover remote tools.

        Called by :class:`~bamboo.agents.ExtraSourceExplorer` before the first
        :meth:`list_tools` call.  No-op for in-process clients.
        """

    async def close(self) -> None:
        """Release the connection acquired by :meth:`connect`.

        Called by :class:`~bamboo.agents.ExtraSourceExplorer` after all tool
        calls have completed.  No-op for in-process clients.
        """

    def task_data_tools(self) -> frozenset[str]:
        """Return names of tools that expect ``task_data`` to be injected.

        The explorer calls this to determine which tool calls should have the
        full ``task_data`` dict automatically added to their kwargs before
        execution — so the LLM never has to specify a ``task_id`` argument.

        Override in subclasses that have task-data-aware tools.  The default
        returns an empty set (safe for clients with no such tools).
        """
        return frozenset()

    @abstractmethod
    def list_tools(self) -> list[McpTool]:
        """Return the full catalogue of tools this client exposes."""
        ...

    @abstractmethod
    async def execute(self, tool_name: str, **kwargs: Any) -> Any:
        """Execute one tool call.

        Args:
            tool_name: Name of the tool as returned by :meth:`list_tools`.
            **kwargs:  Arguments matching the tool's ``parameters_schema``.

        Returns:
            Tool-specific result.  Shape documented per tool in the concrete
            subclass.

        Raises:
            ValueError:   If *tool_name* is not registered.
            RuntimeError: If the underlying data fetch fails fatally.
        """
        ...
