"""External MCP server clients.

:class:`ExternalMcpClient`
    Wraps one external MCP server via StreamableHTTP transport.  The ``mcp``
    package is imported lazily — users who only use
    :class:`~bamboo.mcp.PandaMcpClient` do not need it installed.

:class:`StdioMcpClient`
    Wraps one external MCP server via stdio transport.  Spawns the server as a
    subprocess and communicates over its stdin/stdout.  Also uses the ``mcp``
    package lazily.

:class:`CompositeMcpClient`
    Aggregates any number of :class:`~bamboo.mcp.base.McpClient` instances
    into a single client.  Dispatches :meth:`execute` calls to whichever
    sub-client registered the requested tool.  The first client to register a
    tool name wins on name clashes (PanDA tools always come first in practice).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from contextlib import AsyncExitStack
from typing import Any

from bamboo.mcp.base import McpClient, McpTool

logger = logging.getLogger(__name__)


def _filter_tools(
    tools: list[McpTool],
    include_patterns: list[str],
    exclude_patterns: list[str],
) -> list[McpTool]:
    """Apply whitelist then blacklist regex filters to *tools*.

    Args:
        tools:            Full tool list returned by the server.
        include_patterns: ``re.search`` patterns — keep only matching tools.
                          Empty list means keep all.
        exclude_patterns: ``re.search`` patterns — drop matching tools.
                          Applied after ``include_patterns``.

    Returns:
        Filtered list (original order preserved).
    """
    result = tools
    if include_patterns:
        result = [t for t in result if any(re.search(p, t.name) for p in include_patterns)]
    if exclude_patterns:
        result = [t for t in result if not any(re.search(p, t.name) for p in exclude_patterns)]
    if len(result) != len(tools):
        hidden = {t.name for t in tools} - {t.name for t in result}
        logger.debug("_filter_tools: hidden %d tool(s): %s", len(hidden), sorted(hidden))
    return result


class ExternalMcpClient(McpClient):
    """MCP client that connects to an external server via StreamableHTTP.

    The connection lifecycle is managed explicitly:
    - :meth:`connect` opens the HTTP session, initialises the MCP protocol,
      and discovers available tools.
    - :meth:`close` tears down the session.
    - :meth:`execute` requires an active connection (call :meth:`connect` first).

    Fail-open behaviour:
    - If ``mcp`` is not installed, :meth:`connect` logs a helpful error and
      returns without raising — this client contributes zero tools.
    - If the server is unreachable, :meth:`connect` logs the error and returns
      — this client contributes zero tools.
    - :meth:`execute` re-raises on failure so that ``asyncio.gather`` in the
      explorer can handle it as an exception.

    Args:
        name:    Human-readable label for log messages.
        url:     StreamableHTTP endpoint, e.g. ``http://host:8080/mcp``.
        headers: Extra HTTP headers (e.g. ``{"Authorization": "Bearer …"}``).
    """

    def __init__(
        self,
        name: str,
        url: str,
        headers: dict[str, str] | None = None,
        include_tools: list[str] | None = None,
        exclude_tools: list[str] | None = None,
    ) -> None:
        self._name = name
        self._url = url
        self._headers = headers or {}
        self._include_tools = include_tools or []
        self._exclude_tools = exclude_tools or []
        self._tools: list[McpTool] = []
        self._session: Any = None  # mcp.ClientSession, typed as Any for lazy import
        self._stack: AsyncExitStack | None = None

    @property
    def name(self) -> str:
        return self._name

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Open the StreamableHTTP connection and discover tools.

        Silently returns (after logging) if the ``mcp`` package is missing or
        if the server cannot be reached.
        """
        try:
            from mcp import ClientSession  # noqa: PLC0415
            from mcp.client.streamable_http import streamablehttp_client  # noqa: PLC0415
        except ImportError:
            logger.warning(
                "ExternalMcpClient(%s): 'mcp' package not installed — "
                "install with: pip install 'bamboo'",
                self._name,
            )
            return

        try:
            self._stack = AsyncExitStack()
            read, write, _ = await self._stack.enter_async_context(
                streamablehttp_client(self._url, headers=self._headers)
            )
            self._session = await self._stack.enter_async_context(
                ClientSession(read, write)
            )
            await self._session.initialize()
            result = await self._session.list_tools()
            self._tools = _filter_tools(
                [self._convert_tool(t) for t in result.tools],
                self._include_tools,
                self._exclude_tools,
            )
            logger.info(
                "ExternalMcpClient(%s): connected to %s — %d tool(s): %s",
                self._name,
                self._url,
                len(self._tools),
                [t.name for t in self._tools],
            )
        except Exception as exc:
            logger.warning(
                "ExternalMcpClient(%s): failed to connect to %s: %s — "
                "this server will contribute no tools",
                self._name,
                self._url,
                exc,
            )
            if self._stack is not None:
                await self._stack.aclose()
                self._stack = None
            self._session = None
            self._tools = []

    async def close(self) -> None:
        """Close the StreamableHTTP session."""
        if self._stack is not None:
            try:
                await self._stack.aclose()
            except Exception as exc:
                logger.debug(
                    "ExternalMcpClient(%s): error during close: %s", self._name, exc
                )
            finally:
                self._stack = None
                self._session = None

    # ------------------------------------------------------------------
    # McpClient interface
    # ------------------------------------------------------------------

    def list_tools(self) -> list[McpTool]:
        return list(self._tools)

    async def execute(self, tool_name: str, **kwargs: Any) -> Any:
        if self._session is None:
            raise RuntimeError(
                f"ExternalMcpClient({self._name}): not connected — call connect() first"
            )
        result = await self._session.call_tool(tool_name, kwargs)
        if result.isError:
            raise RuntimeError(
                f"ExternalMcpClient({self._name}): tool {tool_name!r} returned error: "
                f"{result.content}"
            )
        return self._extract_content(result.content)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_tool(t: Any) -> McpTool:
        """Convert an ``mcp`` SDK tool descriptor to a :class:`McpTool`."""
        schema = t.inputSchema
        if not isinstance(schema, dict):
            schema = {"type": "object", "properties": {}}
        return McpTool(
            name=t.name,
            description=t.description or "",
            parameters_schema=schema,
        )

    @staticmethod
    def _extract_content(content: list[Any]) -> Any:
        """Convert an MCP content list to a plain Python value.

        - Empty list → ``None``
        - Single text item → try JSON-parse, fall back to raw string
        - Multiple items → list of text strings
        """
        if not content:
            return None
        texts = [c.text if hasattr(c, "text") else str(c) for c in content]
        if len(texts) == 1:
            try:
                return json.loads(texts[0])
            except (json.JSONDecodeError, TypeError):
                return texts[0]
        return texts


class StdioMcpClient(McpClient):
    """MCP client that spawns a server subprocess and communicates via stdio.

    The server process is started on :meth:`connect` and terminated on
    :meth:`close`.  Fail-open behaviour is identical to
    :class:`ExternalMcpClient`: missing ``mcp`` package or subprocess failure
    logs a warning and contributes zero tools rather than raising.

    Args:
        name:    Human-readable label for log messages.
        command: Executable to spawn, e.g. ``"python3"``.
        args:    Arguments, e.g. ``["-m", "bamboo.server"]``.
        env:     Extra environment variables merged on top of the current
                 process environment.  ``None`` means inherit as-is.
    """

    def __init__(
        self,
        name: str,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        include_tools: list[str] | None = None,
        exclude_tools: list[str] | None = None,
    ) -> None:
        self._name = name
        self._command = command
        self._args = args or []
        self._env = env  # None → inherit; non-empty → merged in connect()
        self._include_tools = include_tools or []
        self._exclude_tools = exclude_tools or []
        self._tools: list[McpTool] = []
        self._session: Any = None
        self._stack: AsyncExitStack | None = None

    @property
    def name(self) -> str:
        return self._name

    async def connect(self) -> None:
        """Spawn the subprocess and initialise the MCP session."""
        try:
            from mcp import ClientSession  # noqa: PLC0415
            from mcp.client.stdio import StdioServerParameters, stdio_client  # noqa: PLC0415
        except ImportError:
            logger.warning(
                "StdioMcpClient(%s): 'mcp' package not installed — "
                "install with: pip install 'bamboo'",
                self._name,
            )
            return

        import os  # noqa: PLC0415

        merged_env: dict[str, str] | None = None
        if self._env:
            merged_env = {**os.environ, **self._env}

        try:
            params = StdioServerParameters(
                command=self._command,
                args=self._args,
                env=merged_env,
            )
            self._stack = AsyncExitStack()
            read, write = await self._stack.enter_async_context(
                stdio_client(params)
            )
            self._session = await self._stack.enter_async_context(
                ClientSession(read, write)
            )
            await self._session.initialize()
            result = await self._session.list_tools()
            self._tools = _filter_tools(
                [ExternalMcpClient._convert_tool(t) for t in result.tools],
                self._include_tools,
                self._exclude_tools,
            )
            logger.info(
                "StdioMcpClient(%s): spawned %s %s — %d tool(s): %s",
                self._name,
                self._command,
                " ".join(self._args),
                len(self._tools),
                [t.name for t in self._tools],
            )
        except Exception as exc:
            logger.warning(
                "StdioMcpClient(%s): failed to start %s: %s — "
                "this server will contribute no tools",
                self._name,
                self._command,
                exc,
            )
            if self._stack is not None:
                await self._stack.aclose()
                self._stack = None
            self._session = None
            self._tools = []

    async def close(self) -> None:
        """Terminate the subprocess and close the session."""
        if self._stack is not None:
            try:
                await self._stack.aclose()
            except Exception as exc:
                logger.debug(
                    "StdioMcpClient(%s): error during close: %s", self._name, exc
                )
            finally:
                self._stack = None
                self._session = None

    def list_tools(self) -> list[McpTool]:
        return list(self._tools)

    async def execute(self, tool_name: str, **kwargs: Any) -> Any:
        if self._session is None:
            raise RuntimeError(
                f"StdioMcpClient({self._name}): not connected — call connect() first"
            )
        result = await self._session.call_tool(tool_name, kwargs)
        if result.isError:
            raise RuntimeError(
                f"StdioMcpClient({self._name}): tool {tool_name!r} returned error: "
                f"{result.content}"
            )
        return ExternalMcpClient._extract_content(result.content)


class CompositeMcpClient(McpClient):
    """Aggregates multiple :class:`~bamboo.mcp.base.McpClient` instances.

    Tool names are deduplicated: the first client that registers a given name
    owns it for :meth:`execute` dispatch.  In practice :class:`PandaMcpClient`
    is always first, so its tool names take precedence over external servers.

    :meth:`connect` and :meth:`close` fan out to all sub-clients concurrently,
    using ``return_exceptions=True`` so one failure never blocks the others.

    Args:
        clients: Ordered list of clients.  First entry's tool names win on
                 clashes.
    """

    def __init__(self, clients: list[McpClient]) -> None:
        self._clients = clients
        self._tool_owner: dict[str, McpClient] = {}

    async def connect(self) -> None:
        await asyncio.gather(
            *(c.connect() for c in self._clients), return_exceptions=True
        )
        # Build dispatch map after all clients have populated their tool lists.
        self._tool_owner = {}
        for client in self._clients:
            for tool in client.list_tools():
                self._tool_owner.setdefault(tool.name, client)

    async def close(self) -> None:
        await asyncio.gather(
            *(c.close() for c in self._clients), return_exceptions=True
        )

    def list_tools(self) -> list[McpTool]:
        seen: set[str] = set()
        tools: list[McpTool] = []
        for client in self._clients:
            for t in client.list_tools():
                if t.name not in seen:
                    seen.add(t.name)
                    tools.append(t)
        return tools

    async def execute(self, tool_name: str, **kwargs: Any) -> Any:
        owner = self._tool_owner.get(tool_name)
        if owner is None:
            raise ValueError(f"CompositeMcpClient: unknown tool {tool_name!r}")
        return await owner.execute(tool_name, **kwargs)
