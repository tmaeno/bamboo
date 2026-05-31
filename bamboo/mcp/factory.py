"""Factory for building the MCP client used by :class:`ContextEnricher`.

:func:`build_mcp_client` is the single entry point.  It returns a
:class:`~bamboo.mcp.external_mcp_client.CompositeMcpClient` combining the
strategy's built-in clients with any configured external servers.
"""

from __future__ import annotations

import logging

from bamboo.mcp.base import McpClient

logger = logging.getLogger(__name__)


def build_mcp_client(settings: object, strategy=None, *, io=None) -> McpClient:
    """Build the MCP client for :class:`~bamboo.agents.ContextEnricher`.

    Args:
        settings: :class:`~bamboo.config.Settings` instance.
        strategy: Optional :class:`~bamboo.agents.extractors.base.ExtractionStrategy`.
                  When ``None`` the active strategy is resolved via
                  :func:`~bamboo.agents.extractors.get_extraction_strategy`.
                  Built-in clients are sourced from
                  :meth:`~bamboo.agents.extractors.base.ExtractionStrategy.builtin_mcp_clients`.
        io:       Optional :class:`~bamboo.frontends.base.InteractionIO` for the
                  interactive client to gather human input through (terminal or
                  chat). When ``None`` the interactive client falls back to stdin.

    Returns:
        :class:`~bamboo.mcp.external_mcp_client.CompositeMcpClient` combining
        strategy built-in tools, the interactive client, and any enabled
        external server tools.
    """
    from bamboo.agents.extractors import get_extraction_strategy  # noqa: PLC0415
    from bamboo.mcp.external_mcp_client import (  # noqa: PLC0415
        CompositeMcpClient,
        ExternalMcpClient,
        StdioMcpClient,
    )
    from bamboo.mcp.interactive_mcp_client import InteractiveMcpClient  # noqa: PLC0415
    from bamboo.mcp.server_config import load_server_configs  # noqa: PLC0415

    if strategy is None:
        strategy = get_extraction_strategy()

    clients: list[McpClient] = [*strategy.builtin_mcp_clients(), InteractiveMcpClient(io)]

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
                        cfg.include_tools or None,
                        cfg.exclude_tools or None,
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
                    ExternalMcpClient(
                        cfg.name,
                        cfg.url,
                        cfg.headers,
                        cfg.include_tools or None,
                        cfg.exclude_tools or None,
                    )
                )
                logger.info(
                    "build_mcp_client: registered external MCP server %r at %s",
                    cfg.name,
                    cfg.url,
                )

    if len(clients) == 2 and not config_path:
        # No external servers — return the bare composite (PanDA + interactive).
        pass

    logger.info(
        "build_mcp_client: composite client with %d client(s): %s",
        len(clients),
        [c.name for c in clients],
    )
    return CompositeMcpClient(clients)
