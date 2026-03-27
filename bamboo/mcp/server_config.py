"""Declarative configuration for external MCP servers.

External servers are described in a JSON file whose path is set via the
``MCP_SERVERS_CONFIG`` environment variable (or ``.env`` field).  Header
values may reference environment variables using ``${VAR_NAME}`` notation —
they are expanded at load time so secrets stay out of the config file.

Example ``mcp_servers.json``::

    {
      "servers": [
        {
          "name": "my_atlas",
          "url": "http://localhost:8080/mcp",
          "headers": {"Authorization": "Bearer ${ATLAS_TOKEN}"},
          "enabled": true
        }
      ]
    }
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

from pydantic import BaseModel, field_validator

logger = logging.getLogger(__name__)

_ENV_VAR_RE = re.compile(r"\$\{([^}]+)\}")


def _expand_env_vars(value: str) -> str:
    """Replace ``${VAR}`` placeholders with their ``os.environ`` values.

    Unknown variables are left as-is so the user gets a visible hint rather
    than a silent empty string.
    """
    def _replace(match: re.Match) -> str:
        var = match.group(1)
        return os.environ.get(var, match.group(0))

    return _ENV_VAR_RE.sub(_replace, value)


class McpServerConfig(BaseModel):
    """Configuration for a single external MCP server.

    Attributes:
        name:    Human-readable label used in logs and tool-name disambiguation.
        url:     StreamableHTTP endpoint URL, e.g. ``http://host:8080/mcp``.
        headers: HTTP headers sent with every request.  Values support
                 ``${ENV_VAR}`` expansion.
        enabled: Set to ``false`` to skip this server without removing the
                 entry from the file.
    """

    name: str
    url: str
    headers: dict[str, str] = {}
    enabled: bool = True

    @field_validator("headers", mode="before")
    @classmethod
    def _expand_header_values(cls, v: dict) -> dict:
        if not isinstance(v, dict):
            return v
        return {k: _expand_env_vars(str(val)) for k, val in v.items()}


def load_server_configs(path: str) -> list[McpServerConfig]:
    """Load and validate external MCP server entries from *path*.

    Returns an empty list (never raises) when the file is missing, empty,
    or malformed — the pipeline degrades gracefully to built-in tools only.
    """
    p = Path(path)
    if not p.exists():
        logger.warning("MCP server config not found: %s — skipping external servers", path)
        return []
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        entries = raw.get("servers", [])
        configs = [McpServerConfig.model_validate(e) for e in entries]
        logger.info(
            "Loaded %d external MCP server config(s) from %s", len(configs), path
        )
        return configs
    except Exception as exc:
        logger.warning(
            "Failed to load MCP server config from %s: %s — skipping external servers",
            path,
            exc,
        )
        return []
