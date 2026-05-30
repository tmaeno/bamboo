"""Wire and run the Mattermost bot daemon.

Builds the shared Mattermost driver, parses the channel allow-list, and supplies
the ``run_session`` callback that constructs a per-session
:class:`~bamboo.agents.investigation_session.InvestigationOrchestrator` bound to a
:class:`~bamboo.frontends.mattermost.io.MattermostInteractionIO`.  Collaborator
objects are built per session (mirroring ``bamboo investigate``'s wiring) so
sessions don't share mutable client state.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

from bamboo.config import Settings, get_settings
from bamboo.frontends.mattermost.bot import Command, MattermostBot
from bamboo.frontends.mattermost.driver import build_async_driver
from bamboo.frontends.mattermost.io import MattermostInteractionIO, ThreadTransport

logger = logging.getLogger(__name__)


def _parse_allowed_channels(settings: Settings) -> set[str]:
    return {c.strip() for c in (settings.mattermost_allowed_channels or "").split(",") if c.strip()}


def _build_deps(io: MattermostInteractionIO):
    """Build a per-session ``_Deps`` (mirrors ``bamboo.scripts.investigate._run``)."""
    from bamboo.agents.context_enricher import ContextEnricher
    from bamboo.agents.extractors.panda_knowledge_extractor import (
        ErrorCategoryClassifier,
        PandaKnowledgeExtractor,
    )
    from bamboo.agents.investigation_session import _Deps
    from bamboo.agents.knowledge_accumulator import KnowledgeAccumulator
    from bamboo.agents.reasoning_navigator import ReasoningNavigator
    from bamboo.database.graph_database_client import GraphDatabaseClient
    from bamboo.database.vector_database_client import VectorDatabaseClient
    from bamboo.mcp.panda_mcp_client import PandaMcpClient

    mcp_client = PandaMcpClient()
    graph_db = GraphDatabaseClient()
    vector_db = VectorDatabaseClient()
    extractor = PandaKnowledgeExtractor()
    error_classifier = ErrorCategoryClassifier()
    explorer = ContextEnricher(mcp_client=mcp_client)
    accumulator = KnowledgeAccumulator(
        extractor=extractor,
        graph_db=graph_db,
        vector_db=vector_db,
        reviewer=None,
        explorer=explorer,
    )
    reasoning_navigator = ReasoningNavigator(
        graph_db=graph_db,
        vector_db=vector_db,
        extractor=extractor,
        explorer=explorer,
    )
    return _Deps(
        mcp_client=mcp_client,
        graph_db=graph_db,
        vector_db=vector_db,
        extractor=extractor,
        reasoning_navigator=reasoning_navigator,
        knowledge_accumulator=accumulator,
        error_classifier=error_classifier,
        io=io,
    )


def _auth_vo() -> str:
    return os.environ.get("PANDA_AUTH_VO") or os.environ.get("OIDC_AUTH_VO") or ""


async def resolve_user_credentials(user_id: Optional[str], settings: Settings):
    """Resolve per-user PanDA OIDC credentials for the invoking Mattermost user.

    Returns :class:`~bamboo.utils.panda_client.PandaCredentials` to act as that
    user (when they have a valid stored token from ``login``), else ``None`` so
    the caller falls back to the bot's service identity. The channel allow-list
    (Layer 1) gates *who* can invoke the bot; this is Layer 2 (*whose* identity).
    """
    if not user_id:
        return None
    from bamboo.frontends.mattermost import oidc  # local: lazy pandaclient import

    try:
        return await oidc.valid_credentials(user_id, _auth_vo(), settings)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not resolve per-user credentials for %s: %s", user_id, exc)
        return None


async def _run_login(io: MattermostInteractionIO, user_id: Optional[str], settings: Settings) -> None:
    from bamboo.frontends.mattermost import oidc  # local: lazy pandaclient import

    if not user_id:
        io.notice("Could not determine your Mattermost user id; cannot log in.")
        return
    try:
        login = await asyncio.to_thread(oidc.begin_device_login, user_id, settings)
    except Exception as exc:  # noqa: BLE001
        io.notice(f"Could not start login: {exc}")
        return
    code_hint = f" (code: `{login.user_code}`)" if login.user_code else ""
    io.notice(
        f"To log in as yourself, open {login.verification_uri_complete} and sign "
        f"in with CERN IAM{code_hint}. Waiting for you to finish…"
    )
    try:
        claims = await oidc.poll_for_token(user_id, login, settings)
    except Exception as exc:  # noqa: BLE001
        io.notice(f"Login failed: {exc}")
        return
    who = claims.get("email") or claims.get("sub") or "you"
    io.notice(f"✓ Logged in as **{who}**. Your PanDA actions now run under your identity.")


def _format_uptime(seconds: Optional[float]) -> str:
    """Render an uptime in seconds as a compact ``"2h 5m"`` string."""
    if seconds is None:
        return "unknown"
    total = int(seconds)
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if not parts:  # under a minute
        parts.append(f"{secs}s")
    return " ".join(parts)


async def _run_session(transport: ThreadTransport, command: Command) -> None:
    """Drive one session (investigate/capture/login/logout/status) over a thread."""
    from bamboo.agents.investigation_session import InvestigationOrchestrator
    from bamboo.frontends.mattermost import oidc
    from bamboo.frontends.mattermost.capture import run_capture
    from bamboo.utils.panda_client import panda_credentials

    settings = get_settings()
    io = MattermostInteractionIO(transport)

    if command.kind == "login":
        await _run_login(io, command.user_id, settings)
        return
    if command.kind == "logout":
        await asyncio.to_thread(oidc.logout, command.user_id or "", settings)
        io.notice("✓ Logged out; the bot will use its service identity for you.")
        return
    if command.kind == "status":
        s = await transport.bot_status()
        health = "✓ functional" if s.functional else f"⚠️ degraded — {s.detail}"
        io.notice(
            "**bamboo bot status**\n"
            f"- Health: {health}\n"
            f"- Active sessions: {s.active_sessions}\n"
            f"- Allowed channels: {s.allowed_channels}\n"
            f"- Uptime: {_format_uptime(s.uptime_seconds)}"
        )
        return

    deps = _build_deps(io)
    creds = await resolve_user_credentials(command.user_id, settings)
    if creds is None and settings.mattermost_require_user_login:
        io.notice("Per-user login is required. Run `login` first to act as yourself.")
        return
    # Bind per-user PanDA identity for every API call in this session (no-op when
    # creds is None). ContextVars propagate across asyncio.to_thread.
    with panda_credentials(creds):
        if command.kind == "investigate":
            orch = InvestigationOrchestrator(deps=deps)
            await orch.start(task_id=command.task_id)
            await orch.run()
            await orch.finalize()
        elif command.kind == "capture":
            messages = await transport.thread_messages()
            transcript = "\n".join(messages)
            await run_capture(
                io,
                transcript=transcript,
                task_id=command.task_id,
                accumulator=deps.knowledge_accumulator,
                graph_db=deps.graph_db,
                mcp_client=deps.mcp_client,
            )
        else:  # pragma: no cover - parse_command only emits the above
            transport.send(f"Unknown command: {command.kind}")


async def serve(settings: Optional[Settings] = None) -> None:
    """Build and run the Mattermost bot until the process is stopped."""
    settings = settings or get_settings()
    allowed = _parse_allowed_channels(settings)
    if not allowed:
        logger.warning(
            "MATTERMOST_ALLOWED_CHANNELS is empty — the bot will ignore every "
            "channel. Set it to the channel IDs bamboo should act in."
        )
    driver = build_async_driver(settings)
    bot = MattermostBot(driver, allowed_channels=allowed, run_session=_run_session)
    logger.info("Starting Mattermost bot (allowed channels: %s)", sorted(allowed))
    await bot.run()
