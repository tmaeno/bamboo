"""Sanitization utilities: strip or pseudonymise sensitive fields.

Two complementary operations are provided:

:func:`pseudonymise`
    Replaces a sensitive raw value with a **stable, opaque short token**
    (e.g. ``"user-a3f2c1"``) derived via HMAC-SHA256.  The same raw value
    always maps to the same token, so graph relationships between incidents
    that share the same user/group are preserved — without ever storing or
    forwarding the real identity.  Use this when writing to the graph/vector
    databases.

:func:`sanitize_for_llm`
    Replaces sensitive values with the literal string ``"<redacted>"`` before
    any dict is serialised into an LLM prompt.  Pseudonyms are *also* replaced
    here — even an opaque token should not leave the system via an external API.
    Use this immediately before calling the LLM.

Usage::

    from bamboo.utils.sanitize import pseudonymise, sanitize_for_llm

    # In the extractor — store pseudonym, not real name:
    safe_value = pseudonymise("userName", raw_value)   # → "user-a3f2c1"

    # Before sending to LLM — redact everything sensitive:
    safe_data = sanitize_for_llm(task_data)            # → {"userName": "<redacted>", ...}

The salt for pseudonymisation is read from the ``PSEUDONYM_SALT`` environment
variable (via :func:`~bamboo.config.get_settings`).  Set a random secret in
production; the built-in default is only suitable for development.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Built-in sensitive keys — fields that directly identify a person or org.
# ---------------------------------------------------------------------------
SENSITIVE_TASK_KEYS: frozenset[str] = frozenset(
    {
        # Identity
        "userName",
        "prodUserName",
        "prodUserID",
    }
)

_REDACTED = "<redacted>"
# Fallback salt used when no PSEUDONYM_SALT is configured.  Not secret — set
# PSEUDONYM_SALT in .env for production deployments.
_DEFAULT_SALT = "bamboo-default-pseudonym-salt-change-me"


def _get_salt() -> str:
    """Return the configured pseudonym salt, falling back to the built-in default."""
    try:
        from bamboo.config import get_settings  # noqa: PLC0415

        salt = get_settings().pseudonym_salt
        if salt:
            return salt
    except Exception:
        pass
    return _DEFAULT_SALT


def pseudonymise(field: str, value: str) -> str:
    """Return a stable, opaque short token for *value* of *field*.

    The token is ``"<field_prefix>-<6-hex-chars>"``, e.g. ``"user-a3f2c1"``.
    The same *(field, value)* pair always produces the same token for a given
    salt, so graph relationships across incidents that share the same user or
    group are preserved.

    Args:
        field: The field name (used as a readable prefix in the token).
        value: The raw sensitive value to pseudonymise.

    Returns:
        A short opaque string suitable for use as a graph node name.
    """
    salt = _get_salt()
    # hmac.new(key, msg, digestmod) — standard library
    digest = hmac.new(
        salt.encode(),
        f"{field}:{value}".encode(),
        hashlib.sha256,
    ).hexdigest()[:8]
    # Derive a short readable prefix from the field name by stripping common
    # suffixes (Name, ID) and lowercasing, e.g.:
    #   "userName"    → "user"
    #   "prodUserName"→ "produser"
    #   "prodUserID"  → "produser"
    prefix = re.sub(r"(?i)(Name|ID)$", "", field).lower()[:8]
    return f"{prefix}-{digest}"


def pseudonymise_dict(
    data: dict[str, Any] | None,
    extra_keys: set[str] | None = None,
) -> dict[str, Any]:
    """Return a copy of *data* with sensitive values replaced by stable pseudonyms.

    Suitable for writing to databases — graph patterns (same user → same node)
    are preserved without storing the real identity.

    Args:
        data:       The dict to process (typically raw ``task_data``).
        extra_keys: Additional field names to pseudonymise beyond the defaults.

    Returns:
        A shallow copy with sensitive values pseudonymised.
    """
    if not data:
        return data  # type: ignore[return-value]

    effective_keys = _effective_sensitive_keys(extra_keys)
    result: dict[str, Any] = {}
    pseudonymised_fields: list[str] = []
    for key, value in data.items():
        if key in effective_keys and value is not None:
            result[key] = pseudonymise(key, str(value))
            pseudonymised_fields.append(key)
        else:
            result[key] = value

    if pseudonymised_fields:
        logger.debug(
            "pseudonymise_dict: pseudonymised %d field(s): %s",
            len(pseudonymised_fields),
            pseudonymised_fields,
        )
    return result


def sanitize_for_llm(
    data: dict[str, Any] | None,
    extra_keys: set[str] | None = None,
) -> dict[str, Any]:
    """Return a copy of *data* with sensitive fields replaced by ``"<redacted>"``.

    Both raw values *and* pseudonyms are redacted — nothing identifying should
    leave the system via an external LLM API.

    Args:
        data:       The dict to sanitize (typically raw ``task_data``).
                    ``None`` is accepted and returned as-is.
        extra_keys: Optional set of additional field names to redact on top of
                    the defaults.

    Returns:
        A shallow copy of *data* with sensitive values replaced.  The original
        dict is never mutated.
    """
    if not data:
        return data  # type: ignore[return-value]

    effective_keys = _effective_sensitive_keys(extra_keys)
    sanitized: dict[str, Any] = {}
    redacted_fields: list[str] = []
    for key, value in data.items():
        if key in effective_keys:
            sanitized[key] = _REDACTED
            redacted_fields.append(key)
        else:
            sanitized[key] = value

    if redacted_fields:
        logger.debug(
            "sanitize_for_llm: redacted %d field(s): %s",
            len(redacted_fields),
            redacted_fields,
        )
    return sanitized


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _effective_sensitive_keys(extra_keys: set[str] | None) -> set[str]:
    """Build the full effective redaction key set from defaults + config + caller."""
    effective: set[str] = set(SENSITIVE_TASK_KEYS)
    if extra_keys:
        effective.update(extra_keys)
    try:
        from bamboo.config import get_settings  # noqa: PLC0415

        cfg_keys = get_settings().sensitive_task_keys
        if cfg_keys:
            effective.update(k.strip() for k in cfg_keys.split(",") if k.strip())
    except Exception:
        pass
    return effective
