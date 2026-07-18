"""Tests for `bamboo stage-embeddings` — model/reranker/out resolution + warm/manifest.

Pure-function precedence is asserted directly; the warm step is exercised with
`subprocess.run` mocked (no real download), so it runs in CI.
"""

import json
import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock

from click.testing import CliRunner

from bamboo.scripts import stage_embeddings
from bamboo.scripts.stage_embeddings import (
    DEFAULT_MODEL,
    _resolve_model,
    _resolve_out,
    _resolve_reranker,
)

# --- model precedence: --model > embedding_model (local) > default -------------------

def test_resolve_model_explicit_wins(monkeypatch):
    monkeypatch.setattr(
        "bamboo.config.get_settings",
        lambda: SimpleNamespace(embeddings_provider="local", embedding_model="from-config"),
    )
    assert _resolve_model("explicit") == "explicit"


def test_resolve_model_from_config_when_local(monkeypatch):
    monkeypatch.setattr(
        "bamboo.config.get_settings",
        lambda: SimpleNamespace(embeddings_provider="local", embedding_model="all-mpnet-base-v2"),
    )
    assert _resolve_model(None) == "all-mpnet-base-v2"


def test_resolve_model_default_when_not_local(monkeypatch):
    # Under the default openai provider the configured model is an OpenAI name, not a valid
    # sentence-transformers repo — fall back to the local default.
    monkeypatch.setattr(
        "bamboo.config.get_settings",
        lambda: SimpleNamespace(embeddings_provider="openai", embedding_model="text-embedding-3-small"),
    )
    assert _resolve_model(None) == DEFAULT_MODEL


def test_resolve_model_default_when_config_unavailable(monkeypatch):
    def _boom():
        raise RuntimeError("no config")

    monkeypatch.setattr("bamboo.config.get_settings", _boom)
    assert _resolve_model(None) == DEFAULT_MODEL


# --- reranker precedence: --reranker > reranker_model (non-empty) > None --------------

def test_resolve_reranker_explicit_wins(monkeypatch):
    monkeypatch.setattr(
        "bamboo.config.get_settings",
        lambda: SimpleNamespace(reranker_model="from-config"),
    )
    assert _resolve_reranker("explicit") == "explicit"


def test_resolve_reranker_from_config_when_set(monkeypatch):
    monkeypatch.setattr(
        "bamboo.config.get_settings",
        lambda: SimpleNamespace(reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2"),
    )
    assert _resolve_reranker(None) == "cross-encoder/ms-marco-MiniLM-L-6-v2"


def test_resolve_reranker_none_when_config_empty(monkeypatch):
    monkeypatch.setattr(
        "bamboo.config.get_settings",
        lambda: SimpleNamespace(reranker_model=""),
    )
    assert _resolve_reranker(None) is None


# --- out-dir precedence: --out > $EMBEDDINGS_OUT > ${SHARED:-/shared}/bamboo/embeddings

def test_resolve_out_explicit_wins(monkeypatch):
    monkeypatch.setenv("EMBEDDINGS_OUT", "/env/out")
    monkeypatch.setenv("SHARED", "/env/shared")
    assert _resolve_out("/explicit") == "/explicit"


def test_resolve_out_embeddings_out_env(monkeypatch):
    monkeypatch.setenv("EMBEDDINGS_OUT", "/env/out")
    monkeypatch.setenv("SHARED", "/env/shared")
    assert _resolve_out(None) == "/env/out"


def test_resolve_out_shared_env(monkeypatch):
    monkeypatch.delenv("EMBEDDINGS_OUT", raising=False)
    monkeypatch.setenv("SHARED", "/env/shared")
    assert _resolve_out(None) == "/env/shared/bamboo/embeddings"


def test_resolve_out_default(monkeypatch):
    monkeypatch.delenv("EMBEDDINGS_OUT", raising=False)
    monkeypatch.delenv("SHARED", raising=False)
    assert _resolve_out(None) == "/shared/bamboo/embeddings"


# --- warm + manifest -----------------------------------------------------------------

def test_warm_downloads_embedding_and_writes_manifest(tmp_path, monkeypatch):
    """The embedding model is warmed with SentenceTransformer into HF_HOME=<out>, offline off."""
    out = str(tmp_path / "emb")
    run = MagicMock(return_value=SimpleNamespace(returncode=0))
    monkeypatch.setattr(stage_embeddings.subprocess, "run", run)

    result = CliRunner().invoke(
        stage_embeddings.main, ["--model", "all-MiniLM-L6-v2", "--out", out]
    )
    assert result.exit_code == 0, result.output

    assert run.call_count == 1
    cmd, kwargs = run.call_args_list[0].args[0], run.call_args_list[0].kwargs
    assert cmd[0] == stage_embeddings.sys.executable
    assert "SentenceTransformer" in cmd[2]
    assert cmd[-1] == "all-MiniLM-L6-v2"
    assert kwargs["env"]["HF_HOME"] == out
    assert kwargs["env"]["HF_HUB_OFFLINE"] == "0"
    assert kwargs["env"]["TRANSFORMERS_OFFLINE"] == "0"

    manifest = json.loads((tmp_path / "emb" / "bamboo-embeddings.json").read_text())
    assert manifest == {"embedding_model": "all-MiniLM-L6-v2", "reranker_model": ""}


def test_warm_stages_reranker_too(tmp_path, monkeypatch):
    """--reranker warms a CrossEncoder into the same dir and records it in the manifest."""
    out = str(tmp_path / "emb")
    run = MagicMock(return_value=SimpleNamespace(returncode=0))
    monkeypatch.setattr(stage_embeddings.subprocess, "run", run)

    result = CliRunner().invoke(
        stage_embeddings.main,
        ["--model", "m", "--reranker", "cross-encoder/r", "--out", out],
    )
    assert result.exit_code == 0, result.output

    assert run.call_count == 2
    scripts = [c.args[0][2] for c in run.call_args_list]
    assert any("SentenceTransformer" in s for s in scripts)
    assert any("CrossEncoder" in s for s in scripts)
    rer_call = next(c for c in run.call_args_list if "CrossEncoder" in c.args[0][2])
    assert rer_call.args[0][-1] == "cross-encoder/r"

    manifest = json.loads((tmp_path / "emb" / "bamboo-embeddings.json").read_text())
    assert manifest == {"embedding_model": "m", "reranker_model": "cross-encoder/r"}


def test_warm_failure_reports_local_extra(tmp_path, monkeypatch):
    """A download failure surfaces the `bamboo[local]` hint and writes no manifest."""
    def _boom(*a, **k):
        raise subprocess.CalledProcessError(1, a[0] if a else "python")

    monkeypatch.setattr(stage_embeddings.subprocess, "run", _boom)

    result = CliRunner().invoke(
        stage_embeddings.main, ["--model", "m", "--out", str(tmp_path / "emb")]
    )
    assert result.exit_code != 0
    assert "bamboo[local]" in result.output
    assert not (tmp_path / "emb" / "bamboo-embeddings.json").exists()
