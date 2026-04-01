"""Factory for building the MCP client used by :class:`ExtraSourceExplorer`.

:func:`build_mcp_client` is the single entry point.  It returns a bare
:class:`~bamboo.mcp.PandaMcpClient` when no external servers are configured
(identical behaviour to before), and a
:class:`~bamboo.mcp.external_mcp_client.CompositeMcpClient` wrapping
PanDA + any enabled external servers otherwise.
"""

from __future__ import annotations

import logging

from bamboo.mcp.base import McpClient
from bamboo.mcp.panda_mcp_client import PandaMcpClient

logger = logging.getLogger(__name__)


def build_mcp_client(settings: object) -> McpClient:
    """Build the MCP client for :class:`~bamboo.agents.ExtraSourceExplorer`.

    Args:
        settings: :class:`~bamboo.config.Settings` instance.

    Returns:
        :class:`~bamboo.mcp.PandaMcpClient` when ``settings.mcp_servers_config``
        is empty.  :class:`~bamboo.mcp.external_mcp_client.CompositeMcpClient`
        combining PanDA tools with all *enabled* external server tools otherwise.
    """
    from bamboo.mcp.external_mcp_client import (  # noqa: PLC0415
        CompositeMcpClient,
        ExternalMcpClient,
        StdioMcpClient,
    )
    from bamboo.mcp.server_config import load_server_configs  # noqa: PLC0415

    clients: list[McpClient] = [PandaMcpClient()]

    config_path: str = getattr(settings, "mcp_servers_config", "") or ""
    if config_path:
        for cfg in load_server_configs(config_path):
            if not cfg.enabled:
                continue
            if cfg.command:
                clients.append(
                    StdioMcpClient(
                        cfg.name,
                        cfg.command,
                        cfg.args,
                        cfg.env or None,
                    )
                )
                logger.info(
                    "build_mcp_client: registered stdio MCP server %r (%s %s)",
                    cfg.name,
                    cfg.command,
                    " ".join(cfg.args),
                )
            else:
                clients.append(
                    ExternalMcpClient(cfg.name, cfg.url, cfg.headers)
                )
                logger.info(
                    "build_mcp_client: registered external MCP server %r at %s",
                    cfg.name,
                    cfg.url,
                )

    if len(clients) == 1:
        return clients[0]

    logger.info(
        "build_mcp_client: composite client with %d server(s): PandaMcpClient + %s",
        len(clients),
        [c._name for c in clients[1:]],  # type: ignore[attr-defined]
    )
    return CompositeMcpClient(clients)
