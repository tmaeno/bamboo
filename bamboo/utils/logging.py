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


class _Neo4jNotificationFilter(logging.Filter):
    """Condense ``neo4j.notifications`` records into a one-line summary.

    The Neo4j driver logs each server notification as the full ``GqlStatusObject``
    repr **plus the entire query** (via a ``NotificationPrinter`` arg). We rewrite
    the record in place to::

        Neo4j notice <gql_status>/<classification>: <description> (line L, col C) | <offending query line>

    so notifications stay **visible** — a genuinely mistyped property name still
    surfaces (now readable) — without the multi-line dump. The driver's arg
    exposes ``.notification`` (``gql_status``/``status_description``/``position``/
    ``classification``) and ``.query``; if a record doesn't match that shape we
    leave it untouched. Never raises (a formatting failure must not break logging).
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            arg = record.args[0] if record.args else None
            note = getattr(arg, "notification", None)
            if note is None:
                return True  # not the expected shape — leave as-is
            pos = getattr(note, "position", None)
            where = f" (line {pos.line}, col {pos.column})" if pos else ""
            cls = getattr(note.classification, "value", note.classification)
            query = getattr(arg, "query", None) or ""
            lines = query.splitlines()
            snippet = ""
            if pos and 0 < getattr(pos, "line", 0) <= len(lines):
                snippet = f" | {lines[pos.line - 1].strip()[:120]}"
            record.msg = (
                f"Neo4j notice {note.gql_status}/{cls}: "
                f"{note.status_description}{where}{snippet}"
            )
            record.args = ()
        except Exception:  # noqa: BLE001 — never break logging; fall back to default
            return True
        return True


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

    # Keep Neo4j server notifications visible (so a mistyped property surfaces),
    # but render them as a readable one-line summary instead of the driver's full
    # GqlStatusObject + query dump. Attach once.
    _notif_log = logging.getLogger("neo4j.notifications")
    if not any(isinstance(f, _Neo4jNotificationFilter) for f in _notif_log.filters):
        _notif_log.addFilter(_Neo4jNotificationFilter())

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
