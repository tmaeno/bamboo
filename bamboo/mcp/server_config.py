"""Declarative configuration for external MCP servers.

External servers are described in a JSON file whose path is set via the
``MCP_SERVERS_CONFIG`` environment variable (or ``.env`` field).  Header
values may reference environment variables using ``${VAR_NAME}`` notation —
they are expanded at load time so secrets stay out of the config file.

Each entry must specify **exactly one** of ``url`` (HTTP) or ``command`` (stdio):

HTTP example::

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

stdio example (bamboo-mcp spawned as a subprocess)::

    {
      "servers": [
        {
          "name": "bamboo_mcp",
          "command": "python3",
          "args": ["-m", "bamboo.server"],
          "env": {"PYTHONPATH": "/path/to/bamboo-mcp/core"},
          "include_tools": ["panda_.*", "atlas\\..*"],
          "exclude_tools": ["bamboo_llm_answer"],
          "enabled": true
        }
      ]
    }

Tool filtering rules:

* ``include_tools`` — whitelist: only tools whose name matches **any** pattern are
  exposed.  Empty list means *allow all*.
* ``exclude_tools`` — blacklist: tools whose name matches **any** pattern are hidden,
  applied **after** the whitelist.  Empty list means *exclude nothing*.
* Patterns are Python ``re.search`` expressions (partial match, case-sensitive).
  Compile errors are caught at config-load time and reported as warnings.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

from pydantic import BaseModel, field_validator, model_validator

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

    Exactly one of ``url`` (HTTP transport) or ``command`` (stdio transport)
    must be set.

    Attributes:
        name:    Human-readable label used in logs and tool-name disambiguation.
        url:     StreamableHTTP endpoint URL, e.g. ``http://host:8080/mcp``.
                 Mutually exclusive with ``command``.
        headers: HTTP headers sent with every request.  Values support
                 ``${ENV_VAR}`` expansion.  Only used when ``url`` is set.
        command: Executable to spawn for stdio transport, e.g. ``"python3"``.
                 Mutually exclusive with ``url``.
        args:    Command-line arguments passed to ``command``,
                 e.g. ``["-m", "bamboo.server"]``.
        env:           Extra environment variables for the subprocess.  Merged on top
                       of the current process environment.  Values support
                       ``${ENV_VAR}`` expansion.
        include_tools: Whitelist of ``re.search`` patterns.  When non-empty, only
                       tools whose name matches at least one pattern are exposed.
        exclude_tools: Blacklist of ``re.search`` patterns.  Tools whose name
                       matches at least one pattern are hidden (applied after
                       ``include_tools``).
        enabled:       Set to ``false`` to skip this server without removing the
                       entry from the file.
    """

    name: str
    url: str = ""
    headers: dict[str, str] = {}
    command: str = ""
    args: list[str] = []
    env: dict[str, str] = {}
    include_tools: list[str] = []
    exclude_tools: list[str] = []
    enabled: bool = True

    @field_validator("headers", mode="before")
    @classmethod
    def _expand_header_values(cls, v: dict) -> dict:
        if not isinstance(v, dict):
            return v
        return {k: _expand_env_vars(str(val)) for k, val in v.items()}

    @field_validator("env", mode="before")
    @classmethod
    def _expand_env_values(cls, v: dict) -> dict:
        if not isinstance(v, dict):
            return v
        return {k: _expand_env_vars(str(val)) for k, val in v.items()}

    @field_validator("include_tools", "exclude_tools", mode="before")
    @classmethod
    def _validate_patterns(cls, patterns: list) -> list:
        """Compile each pattern to catch syntax errors at config-load time."""
        if not isinstance(patterns, list):
            return patterns
        valid = []
        for p in patterns:
            try:
                re.compile(p)
                valid.append(p)
            except re.error as exc:
                logger.warning(
                    "McpServerConfig: invalid regex pattern %r — skipped (%s)", p, exc
                )
        return valid

    def model_post_init(self, __context: object) -> None:
        if not self.url and not self.command:
            raise ValueError(
                f"McpServerConfig({self.name!r}): must specify either 'url' (HTTP) "
                "or 'command' (stdio)"
            )
        if self.url and self.command:
            raise ValueError(
                f"McpServerConfig({self.name!r}): 'url' and 'command' are mutually "
                "exclusive — use one transport only"
            )


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
