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

_HELP = (
    "**bamboo commands** — a leading `@bamboo` or `/bamboo` is optional.\n"
    "- `investigate <taskID>` — live, turn-by-turn co-investigation of a task.\n"
    "- `capture [<taskID>]` — turn this thread's discussion into curated knowledge.\n"
    "- `analyze <taskID>` — one-shot root-cause analysis; posts a result card.\n"
    "- `login` / `logout` — sign in as yourself via IAM / revert to the service identity.\n"
    "- `status` — check the bot is alive and functional.\n"
    "- `help` — show this list."
)


def _parse_allowed_channels(settings: Settings) -> set[str]:
    return {c.strip() for c in (settings.mattermost_allowed_channels or "").split(",") if c.strip()}


def _build_deps(io: MattermostInteractionIO):
    """Build a per-session ``_Deps`` via the shared factory (IO is the only diff)."""
    from bamboo.agents.deps import build_deps

    return build_deps(io=io)


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


async def _run_login(io: MattermostInteractionIO, user_id: Optional[str], settings: Settings) -> bool:
    """Drive the device-flow login over *io*. Returns True iff the user signed in."""
    from bamboo.frontends.mattermost import oidc  # local: lazy pandaclient import

    if not user_id:
        io.notice("Could not determine your Mattermost user id; cannot log in.")
        return False
    try:
        login = await asyncio.to_thread(oidc.begin_device_login, user_id, settings)
    except Exception as exc:  # noqa: BLE001
        io.notice(f"Could not start login: {exc}")
        return False
    from bamboo.frontends.mattermost import render  # local: Mattermost-only render

    # The login prompt is delivered entirely as the attachment card (clickable
    # title link + fallback carry the URL/code); the message body is left empty.
    io.transport.send(
        "",
        props=render.login_message(login.verification_uri_complete, login.user_code),
    )
    try:
        claims = await oidc.poll_for_token(user_id, login, settings)
    except Exception as exc:  # noqa: BLE001
        io.notice(f"Login failed: {exc}")
        return False
    who = claims.get("email") or claims.get("sub") or "you"
    io.notice(f"✓ Logged in as **{who}**. Your PanDA actions now run under your identity.")
    return True


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
    from bamboo.frontends.mattermost.analyze import run_analyze
    from bamboo.frontends.mattermost.capture import run_capture
    from bamboo.frontends.mattermost.narration import stream_narration
    from bamboo.utils.panda_client import panda_credentials

    settings = get_settings()
    io = MattermostInteractionIO(transport)

    if command.kind == "help":
        io.notice(_HELP)
        return
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
        # Auto-start the device-flow login (privately, in the user's DM), wait for
        # them to sign in, then continue the original operation under their identity.
        if not command.user_id:
            io.notice("Could not determine your Mattermost user id; cannot log in.")
            return
        io.notice("You're not logged in — I've sent a login link to your DM; I'll continue once you sign in.")
        try:
            ok = await transport.run_in_dm(
                command.user_id,
                lambda dm_io: _run_login(dm_io, command.user_id, settings),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("auto-login failed for user %s: %s", command.user_id, exc)
            ok = False
        if not ok:
            io.notice("Login didn't complete — please run the command again.")
            return
        creds = await resolve_user_credentials(command.user_id, settings)
        if creds is None:
            io.notice("Still no valid credentials after login — please try again.")
            return
    # Bind per-user PanDA identity for every API call in this session (no-op when
    # creds is None). ContextVars propagate across asyncio.to_thread.
    with panda_credentials(creds):
        # Stream the engine's progress narration into this thread (status spinner +
        # foldable last-N detail; full firehose goes to the `bamboo.narration` log).
        async with stream_narration(transport):
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
            elif command.kind == "analyze":
                await run_analyze(io, task_id=command.task_id, deps=deps)
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
