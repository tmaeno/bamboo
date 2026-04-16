"""Interactive MCP client: tools that require human operator input.

:class:`InteractiveMcpClient` exposes tools marked ``requires_interaction=True``
that are only offered to the LLM when running in an interactive terminal.
The :class:`~bamboo.agents.extra_source_explorer.ExtraSourceExplorer` filters
these out automatically when ``sys.stdout.isatty()`` is ``False``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from bamboo.mcp.base import McpClient, McpTool
from bamboo.utils.narrator import say

logger = logging.getLogger(__name__)


class InteractiveMcpClient(McpClient):
    """MCP client providing tools that prompt the human operator for input.

    Currently exposes a single tool: ``request_human_input``.
    """

    def __init__(self) -> None:
        self._tools: list[McpTool] = [
            McpTool(
                name="request_human_input",
                description=(
                    "Ask the human operator to provide information that cannot be "
                    "obtained from automated sources.  Use this when all other tools "
                    "have been considered and the gap can only be filled by the person "
                    "who handled the incident — for example, investigation steps that "
                    "were taken but not written in the email."
                ),
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "The specific question to display to the operator",
                        }
                    },
                    "required": ["prompt"],
                },
                requires_interaction=True,
            ),
        ]

    @property
    def name(self) -> str:
        return "InteractiveMcpClient"

    def list_tools(self) -> list[McpTool]:
        return list(self._tools)

    async def execute(self, tool_name: str, **kwargs: Any) -> Any:
        if tool_name == "request_human_input":
            return await self._request_human_input(**kwargs)
        raise ValueError(f"InteractiveMcpClient: unknown tool {tool_name!r}")

    async def _request_human_input(self, prompt: str) -> str:
        """Display ``prompt`` to the operator and return their typed response."""
        say("─" * 60)
        say(f"[Human input requested]\n{prompt}")
        say("─" * 60)
        loop = asyncio.get_event_loop()
        try:
            text = await loop.run_in_executor(None, input, "> ")
            return text.strip()
        except EOFError:
            logger.warning("InteractiveMcpClient: EOF on stdin — returning empty string")
            return ""
