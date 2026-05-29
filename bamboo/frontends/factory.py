"""Frontend factory and registry.

Frontends are registered by name at import time via :func:`register_frontend`.
The default frontend is the Rich terminal (``"cli"``).  Concrete frontends
mirror the ``base.py`` + implementations + ``factory.py`` shape used by
:mod:`bamboo.mcp` and :mod:`bamboo.database`.

Built-in frontends registered automatically:

========  ============================================
Name      Class
========  ============================================
``cli``   :class:`~bamboo.frontends.cli.CliInteractionIO`
========  ============================================

The Mattermost frontend is *not* argless (it needs a live bot connection and a
per-session thread binding), so it is constructed directly by the daemon rather
than via this factory; the factory is for argless, process-wide frontends.
"""

from __future__ import annotations

import logging
from typing import Callable

from bamboo.frontends.base import InteractionIO

logger = logging.getLogger(__name__)

# Registry: lower-cased name → zero-arg factory returning an InteractionIO.
_frontends: dict[str, Callable[[], InteractionIO]] = {}


def register_frontend(name: str, factory: Callable[[], InteractionIO]) -> None:
    """Register an argless frontend factory under *name*."""
    _frontends[name.lower()] = factory
    logger.debug("Registered frontend: %s", name)


def get_frontend(name: str = "cli") -> InteractionIO:
    """Return a fresh :class:`InteractionIO` for the named frontend.

    Args:
        name: Lower-cased registry key (default ``"cli"``).

    Raises:
        ValueError: If *name* is not registered.
    """
    key = name.lower()
    if key not in _frontends:
        raise ValueError(
            f"Frontend {key!r} not registered. Available: {sorted(_frontends.keys())}"
        )
    return _frontends[key]()


def list_frontends() -> list[str]:
    """Return the names of all registered argless frontends."""
    return sorted(_frontends.keys())


def _register_builtin_frontends() -> None:
    """Register built-in frontends.  Called once at import time."""
    from bamboo.frontends.cli import CliInteractionIO

    register_frontend("cli", CliInteractionIO)


_register_builtin_frontends()
