#!/usr/bin/env python
"""Pull the Ollama LLM model into shared storage for the batch pipeline.

`bamboo stage-model` ships with `pip install bamboo` (no repo checkout needed) and resolves
the model from your bamboo config so you never have to repeat it. The batch container mounts
the output dir read-only at ``/models`` (``OLLAMA_MODELS``).

Runs a *dedicated transient* Ollama server pointed at the output dir (a local ``ollama``
binary when present, else a throwaway ``ollama/ollama`` Docker container), so the model
lands in that dir regardless of any Ollama daemon already running on the host.

A tiny ``bamboo-model.json`` manifest is written alongside the model recording the pulled
tag, so ``deploy/batch/run-analyze.sh`` can derive ``LLM_MODEL`` from the staged files —
nothing needs to be repeated at submit time. See ``website/src/content/docs/guides/batch.md``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time

import click

from bamboo.utils.logging import setup_logging

DEFAULT_MODEL = "qwen3.6"
MANIFEST_NAME = "bamboo-model.json"


def _resolve_model(explicit: str | None) -> str:
    """Model precedence: ``--model`` > ``LLM_MODEL`` (Ollama only) > ``qwen3.6``.

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


def _free_port() -> int:
    """Return an ephemeral free TCP port on localhost."""
    import socket

    sock = socket.socket()
    try:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]
    finally:
        sock.close()


def _pull_with_local_ollama(model: str, out: str) -> None:
    """Pull with a *dedicated transient* ``ollama serve`` writing into *out*.

    ``ollama pull`` is a client that delegates to a server, and ``OLLAMA_MODELS`` is read
    by the **server**, not the client. Pulling against an already-running daemon (e.g. the
    Ollama.app) therefore ignores it and lands the model in that daemon's default store.
    So we start our own server on a free port with ``OLLAMA_MODELS=<out>`` and pull against
    it, then shut it down — guaranteeing the files land in *out*.
    """
    port = _free_port()
    env = {**os.environ, "OLLAMA_MODELS": out, "OLLAMA_HOST": f"127.0.0.1:{port}"}
    server = subprocess.Popen(
        ["ollama", "serve"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(60):
            if server.poll() is not None:
                raise click.ClickException(
                    "transient 'ollama serve' exited before becoming ready"
                )
            ready = subprocess.run(
                ["ollama", "list"],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if ready.returncode == 0:
                break
            time.sleep(1)
        else:
            raise click.ClickException(
                "transient 'ollama serve' did not become ready in time"
            )
        subprocess.run(["ollama", "pull", model], env=env, check=True)
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()


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
    "else qwen3.6.",
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
      bamboo stage-model --model qwen3.6  # explicit
      SHARED=/shared bamboo stage-model       # out = /shared/bamboo/ollama
    """
    setup_logging()
    model = _resolve_model(model)
    # Absolutise: the transient `ollama serve` runs with its own cwd, and Docker's -v
    # treats a non-absolute source as a named volume rather than a host path.
    out = os.path.abspath(_resolve_out(out_dir))
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

    # Manifest last: its presence implies the model landed. run-analyze.sh reads llm_model
    # from it to derive LLM_MODEL, so the model choice travels with the staged files.
    manifest = os.path.join(out, MANIFEST_NAME)
    with open(manifest, "w", encoding="utf-8") as fh:
        json.dump({"llm_model": model}, fh, indent=2)
        fh.write("\n")

    click.echo(f"[stage-model] ✓ '{model}' staged to {out}")


if __name__ == "__main__":
    main()
