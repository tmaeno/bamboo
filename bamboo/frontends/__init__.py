"""Frontends — presentation adapters for bamboo's interactive engines.

Each frontend implements the :class:`~bamboo.frontends.base.InteractionIO`
contract.  The terminal frontend (:class:`~bamboo.frontends.cli.CliInteractionIO`)
is the default; ``bamboo.frontends.mattermost`` adds a chat frontend.  See
:mod:`bamboo.frontends.base` for the design rationale.
"""

from __future__ import annotations

from bamboo.frontends.base import Column, InteractionIO
from bamboo.frontends.cli import CliInteractionIO
from bamboo.frontends.factory import get_frontend, list_frontends, register_frontend

__all__ = [
    "Column",
    "InteractionIO",
    "CliInteractionIO",
    "get_frontend",
    "list_frontends",
    "register_frontend",
]
