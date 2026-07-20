"""Tests for `bamboo stage-model` — model/out resolution + pull-path selection.

Pure-function precedence is asserted directly; the command's ollama-vs-docker branch is
exercised with `shutil.which` / `subprocess.run` mocked (no real pull), so it runs in CI.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from click.testing import CliRunner

from bamboo.scripts import stage_model
from bamboo.scripts.stage_model import DEFAULT_MODEL, _resolve_model, _resolve_out

# --- model precedence: --model > LLM_MODEL (Ollama) > default -------------------------

def test_resolve_model_explicit_wins(monkeypatch):
    monkeypatch.setattr(
        "bamboo.config.get_settings",
        lambda: SimpleNamespace(llm_provider="ollama", llm_model="from-config"),
    )
    assert _resolve_model("explicit") == "explicit"


def test_resolve_model_from_config_when_ollama(monkeypatch):
    monkeypatch.setattr(
        "bamboo.config.get_settings",
        lambda: SimpleNamespace(llm_provider="ollama", llm_model="qwen3.6:1b"),
    )
    assert _resolve_model(None) == "qwen3.6:1b"


def test_resolve_model_default_when_not_ollama(monkeypatch):
    monkeypatch.setattr(
        "bamboo.config.get_settings",
        lambda: SimpleNamespace(llm_provider="openai", llm_model="gpt-4o"),
    )
    assert _resolve_model(None) == DEFAULT_MODEL


def test_resolve_model_default_when_config_unavailable(monkeypatch):
    def _boom():
        raise RuntimeError("no config")

    monkeypatch.setattr("bamboo.config.get_settings", _boom)
    assert _resolve_model(None) == DEFAULT_MODEL


# --- out-dir precedence: --out > $MODELS_OUT > ${SHARED:-/shared}/bamboo/ollama --------

def test_resolve_out_explicit_wins(monkeypatch):
    monkeypatch.setenv("MODELS_OUT", "/env/out")
    monkeypatch.setenv("SHARED", "/env/shared")
    assert _resolve_out("/explicit") == "/explicit"


def test_resolve_out_models_out_env(monkeypatch):
    monkeypatch.setenv("MODELS_OUT", "/env/out")
    monkeypatch.setenv("SHARED", "/env/shared")
    assert _resolve_out(None) == "/env/out"


def test_resolve_out_shared_env(monkeypatch):
    monkeypatch.delenv("MODELS_OUT", raising=False)
    monkeypatch.setenv("SHARED", "/env/shared")
    assert _resolve_out(None) == "/env/shared/bamboo/ollama"


def test_resolve_out_default(monkeypatch):
    monkeypatch.delenv("MODELS_OUT", raising=False)
    monkeypatch.delenv("SHARED", raising=False)
    assert _resolve_out(None) == "/shared/bamboo/ollama"


# --- pull-path selection --------------------------------------------------------------

class _FakeServer:
    """Stand-in for the transient `ollama serve` Popen handle."""

    def __init__(self):
        self.terminated = False

    def poll(self):
        return None  # stays "alive" through the readiness loop

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return 0

    def kill(self):  # pragma: no cover - only on wait timeout
        self.terminated = True


def test_pull_starts_transient_server_and_pulls_into_out(tmp_path, monkeypatch):
    """The local path must run its OWN `ollama serve` with OLLAMA_MODELS=<out>.

    Regression guard: pulling against a pre-existing daemon ignores the client's
    OLLAMA_MODELS and lands the model in the daemon's default store.
    """
    out = str(tmp_path / "models")
    monkeypatch.setattr(
        stage_model.shutil,
        "which",
        lambda name: "/usr/bin/ollama" if name == "ollama" else None,
    )
    monkeypatch.setattr(stage_model, "_free_port", lambda: 12345)
    server = _FakeServer()
    monkeypatch.setattr(stage_model.subprocess, "Popen", lambda *a, **k: server)
    run = MagicMock(return_value=SimpleNamespace(returncode=0))
    monkeypatch.setattr(stage_model.subprocess, "run", run)

    result = CliRunner().invoke(stage_model.main, ["--model", "qwen3.6:1b", "--out", out])
    assert result.exit_code == 0, result.output

    pull_calls = [c for c in run.call_args_list if c.args[0][:2] == ["ollama", "pull"]]
    assert len(pull_calls) == 1
    args, kwargs = pull_calls[0]
    assert args[0] == ["ollama", "pull", "qwen3.6:1b"]
    # The pull runs against OUR transient server, writing into <out>.
    assert kwargs["env"]["OLLAMA_MODELS"] == out
    assert kwargs["env"]["OLLAMA_HOST"] == "127.0.0.1:12345"
    # And the server is torn down.
    assert server.terminated
    # A manifest records the staged tag so entrypoint.sh can derive LLM_MODEL.
    manifest = json.loads((tmp_path / "models" / "bamboo-model.json").read_text())
    assert manifest == {"llm_model": "qwen3.6:1b"}


def test_pull_errors_when_no_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(stage_model.shutil, "which", lambda name: None)
    run = MagicMock()
    monkeypatch.setattr(stage_model.subprocess, "run", run)

    result = CliRunner().invoke(
        stage_model.main, ["--model", "m", "--out", str(tmp_path / "models")]
    )
    assert result.exit_code != 0
    assert "neither 'ollama' nor 'docker'" in result.output
    run.assert_not_called()
