#!/usr/bin/env python
"""Pull the Ollama LLM model into shared storage for the batch pipeline.

`bamboo stage-model` is the CLI counterpart to `deploy/batch/stage-model.sh` — it ships
with `pip install bamboo` (no repo checkout needed) and resolves the model from your
bamboo config so you never have to repeat it. The batch container mounts the output dir
read-only at ``/models`` (``OLLAMA_MODELS``).

Uses a local ``ollama`` binary when present, otherwise a throwaway ``ollama/ollama``
Docker container that writes into the mounted output dir.

See ``website/src/content/docs/guides/batch.md``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time

import click

from bamboo.utils.logging import setup_logging

DEFAULT_MODEL = "llama3.2:3b"


def _resolve_model(explicit: str | None) -> str:
    """Model precedence: ``--model`` > ``LLM_MODEL`` (Ollama only) > ``llama3.2:3b``.

    The config is consulted only for an Ollama setup, since the batch path is
    Ollama-only; an OpenAI/Anthropic ``llm_model`` is not a valid ``ollama pull`` tag.
    """
    if explicit:
        return explicit
    try:
        from bamboo.config import get_settings

        settings = get_settings()
        if settings.llm_provider == "ollama" and settings.llm_model:
            return settings.llm_model
    except Exception:  # noqa: BLE001 — config is best-effort; fall back to the default
        pass
    return DEFAULT_MODEL


def _resolve_out(explicit: str | None) -> str:
    """Output-dir precedence: ``--out`` > ``$MODELS_OUT`` > ``${SHARED:-/shared}/bamboo/ollama``."""
    if explicit:
        return explicit
    if os.environ.get("MODELS_OUT"):
        return os.environ["MODELS_OUT"]
    shared = os.environ.get("SHARED", "/shared")
    return os.path.join(shared, "bamboo", "ollama")


def _pull_with_local_ollama(model: str, out: str) -> None:
    """Pull with a local ``ollama`` binary, writing into *out* (OLLAMA_MODELS)."""
    env = {**os.environ, "OLLAMA_MODELS": out}
    subprocess.run(["ollama", "pull", model], env=env, check=True)


def _pull_with_docker(model: str, out: str) -> None:
    """Fallback: run a throwaway ``ollama/ollama`` container and pull into *out*."""
    cname = f"bamboo-stage-ollama-{os.getpid()}"
    subprocess.run(
        [
            "docker", "run", "-d", "--name", cname,
            "-v", f"{out}:/root/.ollama/models", "ollama/ollama",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    try:
        for _ in range(60):
            ready = subprocess.run(
                ["docker", "exec", cname, "ollama", "list"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if ready.returncode == 0:
                break
            time.sleep(2)
        subprocess.run(["docker", "exec", cname, "ollama", "pull", model], check=True)
    finally:
        subprocess.run(
            ["docker", "rm", "-f", cname],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


@click.command("stage-model")
@click.option(
    "--model",
    default=None,
    help="Ollama model tag. Default: LLM_MODEL from config when LLM_PROVIDER=ollama, "
    "else llama3.2:3b.",
)
@click.option(
    "--out",
    "out_dir",
    default=None,
    type=click.Path(file_okay=False),
    help="Models output dir. Default: $MODELS_OUT or ${SHARED:-/shared}/bamboo/ollama.",
)
def main(model, out_dir):
    """Pull the Ollama model into shared storage (mounted read-only at /models).

    \b
      bamboo stage-model                      # model from your bamboo config
      bamboo stage-model --model llama3.2:3b  # explicit
      SHARED=/shared bamboo stage-model       # out = /shared/bamboo/ollama
    """
    setup_logging()
    model = _resolve_model(model)
    out = _resolve_out(out_dir)
    os.makedirs(out, exist_ok=True)

    click.echo(f"[stage-model] pulling '{model}' into {out}")
    try:
        if shutil.which("ollama"):
            _pull_with_local_ollama(model, out)
        elif shutil.which("docker"):
            _pull_with_docker(model, out)
        else:
            raise click.ClickException(
                "neither 'ollama' nor 'docker' found on PATH — install one to stage the model."
            )
    except subprocess.CalledProcessError as exc:
        raise click.ClickException(
            f"model pull failed (exit {exc.returncode})"
        ) from exc

    click.echo(
        f"[stage-model] ✓ '{model}' staged. Set LLM_MODEL={model} at submit time."
    )


if __name__ == "__main__":
    main()
