"""Logging configuration for the Bamboo application.

Call :func:`setup_logging` once at application startup (e.g. in the CLI
entry point or ``main()``).  The log level is read from the
``LOG_LEVEL`` configuration key (default ``"INFO"``).

Third-party libraries that are excessively verbose at INFO level
(``httpx``, ``httpcore``, ``neo4j``) are clamped to WARNING.
"""

import logging
import sys

from bamboo.config import get_settings


def setup_logging() -> None:
    """Configure root logging from application settings.

    Sets the root logger level and attaches a :class:`logging.StreamHandler`
    that writes to ``stdout``.  Safe to call multiple times — ``basicConfig``
    is a no-op if handlers are already configured.
    """
    settings = get_settings()

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Suppress chatty third-party loggers
    for noisy in ("httpx", "httpcore", "neo4j"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
