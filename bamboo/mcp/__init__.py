"""Lightweight MCP (Model Context Protocol) sub-package for Bamboo.

Provides a minimal abstraction layer for tool-calling agents.
:class:`McpClient` subclasses register named tools and handle their execution;
agents call :meth:`McpClient.list_tools` to discover what is available and
:meth:`McpClient.execute` to invoke a specific tool by name.
"""

from .base import McpClient, McpTool
from .external_mcp_client import CompositeMcpClient, ExternalMcpClient
from .factory import build_mcp_client
from .panda_mcp_client import PandaMcpClient

__all__ = [
    "McpTool",
    "McpClient",
    "PandaMcpClient",
    "ExternalMcpClient",
    "CompositeMcpClient",
    "build_mcp_client",
]
