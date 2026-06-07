"""Pytest bootstrap — fail loudly if the test run is using the wrong interpreter.

Most of the suite imports cleanly under almost any Python that has the heavy deps
(langchain, pydantic, …), so a run launched with a *system* interpreter that is
missing a declared dependency can pass the majority of tests while silently being
in the wrong environment. That once masked a real problem (a run under the system
`python3`, which lacks ``textual``, "passed" 329/330 and made a single failure look
pre-existing).

This guard converts that silent, misleading state into an immediate, explanatory
abort: if a representative import-time dependency is not importable, the whole
session stops with a message pointing at the project virtualenv. It is keyed on
*dependency presence* (not a hardcoded path), so it stays correct for any
contributor's venv.
"""

from __future__ import annotations

import importlib.util
import sys

import pytest

# Representative import-time dependencies that a fully-provisioned project env has
# but a bare/system interpreter typically does not. ``textual`` is the canonical
# example: it is a declared core dependency, imported at module load by
# ``bamboo.scripts.review_drafts`` (so the CLI import path needs it), yet absent on
# a stock framework/system Python — exactly the gap that caused the silent failure.
_REQUIRED_IMPORTS = ("textual",)


def pytest_configure(config: pytest.Config) -> None:
    missing = [m for m in _REQUIRED_IMPORTS if importlib.util.find_spec(m) is None]
    if missing:
        raise pytest.UsageError(
            "Wrong Python interpreter for this project: missing declared "
            f"dependenc{'y' if len(missing) == 1 else 'ies'} {', '.join(missing)} "
            f"under {sys.executable!r}.\n"
            "You are almost certainly not in the project virtualenv. Run the tests "
            "through the project venv's interpreter (e.g. activate it, or use its "
            "`bin/python -m pytest`) — do not fall back to the system `python3`. "
            "See CLAUDE.md / CLAUDE.local.md."
        )
