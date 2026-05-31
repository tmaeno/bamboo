"""Per-user PanDA OIDC login for the Mattermost bot.

Drives the OAuth2 **device flow** so an ops user can authenticate *as
themselves* through the bot; their PanDA actions then run under their identity
via :func:`bamboo.utils.panda_client.panda_credentials`.

This runs entirely in the bamboo bot process — not inside Mattermost. The bot
talks to IAM directly (device-code request + token polling + refresh); the user
authenticates in their own browser; Mattermost only relays the verification
URL/code. Neither the bot nor Mattermost ever sees the user's IAM credentials,
only the issued token.

Heavy reuse of :class:`pandaclient.openidc_utils.OpenIdConnect_Utils` (config
discovery, device-code request, refresh, expiry check). The one method we cannot
reuse is ``get_id_token`` — it blocks on an interactive ``[y/n]`` prompt — so we
replicate just its polling loop. The per-user token store is simply a per-user
``token_dir`` (``OpenIdConnect_Utils`` writes/reads ``<token_dir>/.token``).
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlencode, urlparse

from bamboo.config import Settings, get_settings
from bamboo.utils.panda_client import PandaCredentials

logger = logging.getLogger(__name__)

_DEVICE_GRANT = "urn:ietf:params:oauth:grant-type:device_code"

_MISSING_PANDA_MSG = (
    "Per-user OIDC login requires the PanDA client. Install it with:\n"
    "    pip install 'bamboo[panda]'"
)


def _decode_id_token(enc: str) -> dict:
    """Decode a JWT id-token's payload (no signature verification needed here).

    Mirrors ``pandaclient.openidc_utils.decode_id_token`` but kept local so the
    hot path doesn't import the PanDA client.
    """
    payload = enc.split(".")[1]
    payload += "=" * (-len(payload) % 4)
    return json.loads(base64.urlsafe_b64decode(payload.encode()))


@dataclass
class DeviceLogin:
    """Everything needed to show the user a login prompt and poll for the token."""

    verification_uri_complete: str
    user_code: str
    device_code: str
    interval: int
    expires_in: int
    token_endpoint: str
    client_id: str
    client_secret: str


def _require_vo(settings: Settings) -> str:
    vo = os.environ.get("PANDA_AUTH_VO") or os.environ.get("OIDC_AUTH_VO")
    if not vo:
        raise ValueError("PANDA_AUTH_VO is not set; cannot resolve the OIDC auth config.")
    return vo


def auth_config_url(settings: Optional[Settings] = None) -> str:
    """Build the PanDA auth-config URL: ``{scheme}://{host}[:{port}]/auth/{VO}_auth_config.json``.

    Host/port come from ``PANDA_API_URL_SSL`` (path stripped); VO from
    ``PANDA_AUTH_VO``. Mirrors ``pandaclient.Client._Curl.get_oidc``.
    """
    settings = settings or get_settings()
    vo = _require_vo(settings)
    base = os.environ.get("PANDA_API_URL_SSL")
    if not base:
        raise ValueError("PANDA_API_URL_SSL is not set; cannot resolve the OIDC auth config.")
    parsed = urlparse(base)
    host = parsed.hostname or parsed.path
    scheme = parsed.scheme or "https"
    netloc = f"{host}:{parsed.port}" if parsed.port else host
    return f"{scheme}://{netloc}/auth/{vo}_auth_config.json"


def _token_root(settings: Settings) -> Path:
    root = settings.mattermost_token_dir or str(Path.home() / ".bamboo" / "mattermost_tokens")
    return Path(os.path.expanduser(root))


def _sanitize(user_id: str) -> str:
    """Make a Mattermost user id safe to use as a directory name."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", user_id or "unknown")


def _utils(user_id: str, settings: Settings):
    """Build an ``OpenIdConnect_Utils`` scoped to this user's token directory."""
    try:
        from pandaclient import openidc_utils  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised only without extra
        raise ImportError(_MISSING_PANDA_MSG) from exc
    user_dir = _token_root(settings) / _sanitize(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(user_dir, 0o700)
    except OSError:  # pragma: no cover - best effort on exotic filesystems
        pass
    return openidc_utils.OpenIdConnect_Utils(
        auth_config_url(settings), token_dir=str(user_dir), log_stream=logger
    )


def _load_endpoints(util: Any) -> tuple[dict, dict]:
    """Fetch (auth_config, endpoint_config) via the util's cached fetch_page."""
    ok, auth_config = util.fetch_page(util.auth_config_url)
    if not ok:
        raise RuntimeError(f"Failed to fetch PanDA auth config: {auth_config}")
    ok, endpoint_config = util.fetch_page(auth_config["oidc_config_url"])
    if not ok:
        raise RuntimeError(f"Failed to fetch OIDC endpoint config: {endpoint_config}")
    return auth_config, endpoint_config


def begin_device_login(user_id: str, settings: Optional[Settings] = None) -> DeviceLogin:
    """Request a device code from IAM and return what's needed to prompt + poll."""
    settings = settings or get_settings()
    util = _utils(user_id, settings)
    auth_config, endpoint_config = _load_endpoints(util)
    ok, dev = util.get_device_code(
        endpoint_config["device_authorization_endpoint"],
        auth_config["client_id"],
        auth_config["audience"],
        auth_config.get("jwt_profile"),
    )
    if not ok:
        raise RuntimeError(f"Failed to get device code: {dev}")
    return DeviceLogin(
        verification_uri_complete=dev["verification_uri_complete"],
        user_code=dev.get("user_code", ""),
        device_code=dev["device_code"],
        interval=int(dev.get("interval", 5)),
        expires_in=int(dev.get("expires_in", 600)),
        token_endpoint=endpoint_config["token_endpoint"],
        client_id=auth_config["client_id"],
        client_secret=auth_config.get("client_secret", ""),
    )


async def poll_for_token(
    user_id: str, login: DeviceLogin, settings: Optional[Settings] = None
) -> dict:
    """Poll the token endpoint until the user authenticates (device flow).

    Replicates ``OpenIdConnect_Utils.get_id_token``'s polling loop without its
    interactive ``[y/n]`` prompt, using ``httpx`` (async). On success the full
    token response is written to the user's ``.token`` (so ``check_token`` /
    ``refresh_token`` work later) and the decoded id-token claims are returned.
    """
    import httpx  # noqa: PLC0415

    settings = settings or get_settings()
    data = {
        "client_id": login.client_id,
        "grant_type": _DEVICE_GRANT,
        "device_code": login.device_code,
    }
    if login.client_secret:
        data["client_secret"] = login.client_secret
    body = urlencode(data)
    headers = {"content-type": "application/x-www-form-urlencoded"}

    deadline = login.expires_in
    waited = 0
    async with httpx.AsyncClient(timeout=30.0) as client:
        while waited < deadline:
            resp = await client.post(login.token_endpoint, content=body, headers=headers)
            if resp.status_code == 200:
                token = resp.json()
                _write_token(user_id, token, settings)
                return _decode_id_token(token["id_token"])
            # Pending → wait the polling interval and retry; anything else is fatal.
            err = ""
            try:
                err = resp.json().get("error", "")
            except Exception:  # noqa: BLE001
                pass
            if err != "authorization_pending":
                raise RuntimeError(f"Device-flow token request failed ({resp.status_code}): {resp.text}")
            await asyncio.sleep(login.interval + 1)
            waited += login.interval + 1
    raise TimeoutError("Login timed out before the device code was authorized.")


def _write_token(user_id: str, token: dict, settings: Settings) -> None:
    util = _utils(user_id, settings)
    path = util.get_token_path()
    with open(path, "w") as f:
        json.dump(token, f)
    try:
        os.chmod(path, 0o600)
    except OSError:  # pragma: no cover
        pass


async def valid_credentials(
    user_id: str, auth_vo: str, settings: Optional[Settings] = None
) -> Optional[PandaCredentials]:
    """Return per-user :class:`PandaCredentials`, refreshing if near expiry.

    ``None`` when the user has no stored token (caller falls back to the service
    identity or asks them to ``login``). The blocking check/refresh runs in a
    worker thread.
    """
    settings = settings or get_settings()
    id_token = await asyncio.to_thread(_resolve_id_token, user_id, settings)
    if not id_token:
        return None
    return PandaCredentials(id_token=id_token, auth_vo=auth_vo)


def _resolve_id_token(user_id: str, settings: Settings) -> Optional[str]:
    util = _utils(user_id, settings)
    valid, token_or_refresh, _dec = util.check_token()
    if valid:
        return token_or_refresh  # a still-valid id_token
    if not token_or_refresh:
        return None  # no token stored at all
    # Near expiry but a refresh token is available — refresh it.
    try:
        _auth_config, endpoint_config = _load_endpoints(util)
        ok, new_id = util.refresh_token(
            endpoint_config["token_endpoint"],
            _auth_config["client_id"],
            _auth_config.get("client_secret", ""),
            token_or_refresh,
        )
        return new_id if ok else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("OIDC refresh failed for user %s: %s", user_id, exc)
        return None


def logout(user_id: str, settings: Optional[Settings] = None) -> None:
    """Remove the user's stored token + page caches."""
    settings = settings or get_settings()
    try:
        _utils(user_id, settings).cleanup()
    except Exception as exc:  # noqa: BLE001
        logger.warning("logout cleanup failed for user %s: %s", user_id, exc)
