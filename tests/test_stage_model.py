"""Tests for `bamboo stage-model` — model/out resolution + pull-path selection.

Pure-function precedence is asserted directly; the command's ollama-vs-docker branch is
exercised with `shutil.which` / `subprocess.run` mocked (no real pull), so it runs in CI.
"""

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
        lambda: SimpleNamespace(llm_provider="ollama", llm_model="llama3.2:1b"),
    )
    assert _resolve_model(None) == "llama3.2:1b"


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

def test_pull_uses_local_ollama_when_present(tmp_path, monkeypatch):
    monkeypatch.setattr(
        stage_model.shutil, "which", lambda name: "/usr/bin/ollama" if name == "ollama" else None
    )
    run = MagicMock()
    monkeypatch.setattr(stage_model.subprocess, "run", run)

    result = CliRunner().invoke(
        stage_model.main, ["--model", "llama3.2:1b", "--out", str(tmp_path / "models")]
    )
    assert result.exit_code == 0, result.output
    run.assert_called_once()
    args, kwargs = run.call_args
    assert args[0] == ["ollama", "pull", "llama3.2:1b"]
    assert kwargs["env"]["OLLAMA_MODELS"] == str(tmp_path / "models")


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
