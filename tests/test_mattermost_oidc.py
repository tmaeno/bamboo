"""Unit tests for per-user PanDA OIDC login (Mattermost bot).

No live IAM / PanDA client: ``_utils`` is monkeypatched with a fake
``OpenIdConnect_Utils``, and ``poll_for_token`` is driven against a fake
``httpx.AsyncClient``. Verifies URL derivation, the device-login bootstrap,
polling (pending → success, and timeout), credential resolution (valid /
refresh / none), and command routing.
"""

from __future__ import annotations

import base64
import json

import httpx
import pytest

from bamboo.config import Settings
from bamboo.frontends.mattermost import oidc
from bamboo.frontends.mattermost.bot import parse_command
from bamboo.utils.panda_client import PandaCredentials


def _jwt(payload: dict) -> str:
    body = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    return f"header.{body}.sig"


class FakeUtil:
    """Stand-in for pandaclient OpenIdConnect_Utils."""

    def __init__(self, *, check=(True, "valid-id", {}), token_path="/tmp/bamboo-test.token"):
        self.auth_config_url = "https://panda/auth/atlas_auth_config.json"
        self._check = check
        self._token_path = token_path
        self.cleaned = False

    def fetch_page(self, url):
        if url == self.auth_config_url:
            return True, {
                "client_id": "cid",
                "client_secret": "sec",
                "audience": "aud",
                "jwt_profile": None,
                "oidc_config_url": "https://idp/.well-known",
            }
        return True, {
            "device_authorization_endpoint": "https://idp/device",
            "token_endpoint": "https://idp/token",
        }

    def get_device_code(self, dev_ep, cid, aud, jp):
        return True, {
            "verification_uri_complete": "https://idp/device?code=ABC",
            "user_code": "ABC-123",
            "device_code": "dev-code",
            "interval": 0,
            "expires_in": 5,
        }

    def check_token(self):
        return self._check

    def refresh_token(self, token_ep, cid, sec, rt):
        return True, "refreshed-id"

    def get_token_path(self):
        return self._token_path

    def cleanup(self):
        self.cleaned = True


@pytest.fixture
def patch_utils(monkeypatch):
    holder = {}

    def make(check=(True, "valid-id", {})):
        util = FakeUtil(check=check)
        holder["util"] = util
        monkeypatch.setattr(oidc, "_utils", lambda user_id, settings: util)
        return util

    return make


# ---------------------------------------------------------------------------
# auth_config_url
# ---------------------------------------------------------------------------


def test_auth_config_url_strips_path_and_keeps_port(monkeypatch):
    monkeypatch.setenv("PANDA_API_URL_SSL", "https://pandaserver.cern.ch:25443/api/v1")
    monkeypatch.setenv("PANDA_AUTH_VO", "atlas")
    assert (
        oidc.auth_config_url(Settings())
        == "https://pandaserver.cern.ch:25443/auth/atlas_auth_config.json"
    )


def test_auth_config_url_errors_without_vo(monkeypatch):
    monkeypatch.setenv("PANDA_API_URL_SSL", "https://pandaserver.cern.ch/api/v1")
    monkeypatch.delenv("PANDA_AUTH_VO", raising=False)
    monkeypatch.delenv("OIDC_AUTH_VO", raising=False)
    with pytest.raises(ValueError):
        oidc.auth_config_url(Settings())


# ---------------------------------------------------------------------------
# begin_device_login
# ---------------------------------------------------------------------------


def test_begin_device_login_returns_prompt_and_poll_params(patch_utils):
    patch_utils()
    login = oidc.begin_device_login("u1", Settings())
    assert login.verification_uri_complete.startswith("https://idp/device")
    assert login.user_code == "ABC-123"
    assert login.device_code == "dev-code"
    assert login.token_endpoint == "https://idp/token"
    assert login.client_id == "cid"


# ---------------------------------------------------------------------------
# poll_for_token
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class _FakeAsyncClient:
    """Returns the queued responses in order from .post()."""

    queue: list = []

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, content=None, headers=None):
        return _FakeAsyncClient.queue.pop(0)


@pytest.mark.asyncio
async def test_poll_for_token_pending_then_success(monkeypatch, patch_utils):
    patch_utils()
    monkeypatch.setattr(oidc, "_write_token", lambda *a, **k: None)
    _FakeAsyncClient.queue = [
        _FakeResponse(400, {"error": "authorization_pending"}),
        _FakeResponse(200, {"id_token": _jwt({"email": "ops@cern.ch"}), "refresh_token": "r"}),
    ]
    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)

    login = oidc.DeviceLogin(
        verification_uri_complete="x", user_code="c", device_code="dc",
        interval=0, expires_in=5, token_endpoint="https://idp/token",
        client_id="cid", client_secret="sec",
    )
    claims = await oidc.poll_for_token("u1", login, Settings())
    assert claims["email"] == "ops@cern.ch"


@pytest.mark.asyncio
async def test_poll_for_token_times_out(monkeypatch):
    class _AlwaysPending(_FakeAsyncClient):
        async def post(self, url, content=None, headers=None):
            return _FakeResponse(400, {"error": "authorization_pending"})

    monkeypatch.setattr(httpx, "AsyncClient", _AlwaysPending)
    login = oidc.DeviceLogin(
        verification_uri_complete="x", user_code="c", device_code="dc",
        interval=0, expires_in=0, token_endpoint="https://idp/token",
        client_id="cid", client_secret="",
    )
    with pytest.raises(TimeoutError):
        await oidc.poll_for_token("u1", login, Settings())


# ---------------------------------------------------------------------------
# valid_credentials
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_valid_credentials_returns_token_when_valid(patch_utils):
    patch_utils(check=(True, "valid-id", {}))
    creds = await oidc.valid_credentials("u1", "atlas", Settings())
    assert creds == PandaCredentials(id_token="valid-id", auth_vo="atlas")


@pytest.mark.asyncio
async def test_valid_credentials_refreshes_when_near_expiry(patch_utils):
    patch_utils(check=(False, "refresh-tok", {}))  # not valid, but refresh token present
    creds = await oidc.valid_credentials("u1", "atlas", Settings())
    assert creds.id_token == "refreshed-id"


@pytest.mark.asyncio
async def test_valid_credentials_none_when_no_token(patch_utils):
    patch_utils(check=(False, None, None))
    assert await oidc.valid_credentials("u1", "atlas", Settings()) is None


# ---------------------------------------------------------------------------
# command routing
# ---------------------------------------------------------------------------


def test_parse_command_login_logout():
    assert parse_command("login").kind == "login"
    assert parse_command("@bamboo logout").kind == "logout"
    assert parse_command("/bamboo login").kind == "login"


# ---------------------------------------------------------------------------
# serve._run_session login flow
# ---------------------------------------------------------------------------


class _CaptureTransport:
    def __init__(self):
        self.sent = []
        self.props = []

    def send(self, text, *, props=None):
        self.sent.append(text)
        self.props.append(props)

    async def next_reply(self):  # pragma: no cover - not used in login flow
        return ""

    async def thread_messages(self):  # pragma: no cover
        return []


@pytest.mark.asyncio
async def test_serve_login_flow_posts_url_then_success(monkeypatch):
    from bamboo.frontends.mattermost import serve
    from bamboo.frontends.mattermost.bot import Command

    login = oidc.DeviceLogin(
        verification_uri_complete="https://idp/device?code=ABC",
        user_code="ABC-123", device_code="dc", interval=0, expires_in=5,
        token_endpoint="https://idp/token", client_id="cid", client_secret="sec",
    )
    monkeypatch.setattr(oidc, "begin_device_login", lambda uid, s: login)

    async def fake_poll(uid, dev, s):
        return {"email": "ops@cern.ch"}

    monkeypatch.setattr(oidc, "poll_for_token", fake_poll)

    t = _CaptureTransport()
    await serve._run_session(t, Command(kind="login", user_id="u1"))

    blob = "\n".join(t.sent)
    assert "Logged in as" in blob and "ops@cern.ch" in blob

    # The device-code prompt is delivered as the login link-card attachment; the
    # URL (title_link) and code live in the card, not the message body.
    attachments = [
        p["attachments"][0] for p in t.props if isinstance(p, dict) and "attachments" in p
    ]
    assert len(attachments) == 1
    att = attachments[0]
    assert att["title_link"] == "https://idp/device?code=ABC"
    assert "https://idp/device?code=ABC" in att["fallback"]
    assert "ABC-123" in att["fallback"]
