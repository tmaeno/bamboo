"""Logging configuration for the Bamboo application.

Call :func:`setup_logging` once at application startup (e.g. in the CLI
entry point or ``main()``).  The log level is read from the
``LOG_LEVEL`` configuration key (default ``"INFO"``).

Third-party libraries that are excessively verbose at INFO level are clamped:
- ``httpx``, ``httpcore``, ``neo4j`` → WARNING
- ``transformers``, ``sentence_transformers`` → ERROR  (suppresses the
  harmless "LOAD REPORT / embeddings.position_ids UNEXPECTED" noise emitted
  by transformers 5.x on every model load)
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

    # Suppress chatty third-party loggers.
    for noisy in ("httpx", "httpcore", "neo4j"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # transformers installs its own StreamHandler(stderr) on the
    # "transformers" logger at import time with propagate=False, so
    # simply setting the level on the logger is not enough — the handler
    # itself must also be silenced.  sentence-transformers 5.x / transformers
    # 5.x emit a "LOAD REPORT" warning (harmless embeddings.position_ids
    # artefact) on every model load; suppress it at ERROR level.
    try:
        import transformers as _transformers  # noqa: F401 — triggers handler registration
    except ImportError:
        pass
    for name in ("sentence_transformers", "transformers"):
        lg = logging.getLogger(name)
        lg.setLevel(logging.ERROR)
        for handler in lg.handlers:
            handler.setLevel(logging.ERROR)


