#!/usr/bin/env python
"""Stage the local embedding model (and optional reranker) into shared storage.

`bamboo stage-embeddings` is the embeddings counterpart to `bamboo stage-model`: it
downloads the sentence-transformers model your KB was populated with into a directory the
batch container mounts read-only at ``/embeddings`` (``HF_HOME``). It ships with
``pip install bamboo`` (the `[local]` extra provides ``sentence-transformers``), so no repo
checkout is needed, and it resolves the model from your bamboo config so you never repeat it.

The model files are warmed into the output dir by a *dedicated child process* run with
``HF_HOME=<out>`` — ``HF_HOME`` is read at import time by ``huggingface_hub``, so a clean
subprocess is the only reliable way to land the cache in that dir (this also mirrors the
Dockerfile's old bake command byte-for-byte, so the on-disk layout matches).

If ``RERANKER_MODEL`` is set in your config, its cross-encoder is staged alongside the
embedding model (the reranker is opt-in and configured, not a CLI flag). A tiny
``bamboo-embeddings.json`` manifest is written alongside the cache recording what was staged,
so ``deploy/batch/run-analyze.sh`` can derive ``RERANKER_MODEL`` from it (the embedding model
itself is derived from the KB snapshot's ``metadata.json`` — the KB is its source of truth).
See ``website/src/content/docs/guides/batch.md``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import click

from bamboo.utils.logging import setup_logging

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MANIFEST_NAME = "bamboo-embeddings.json"


def _resolve_model(explicit: str | None) -> str:
    """Model precedence: ``--model`` > config ``embedding_model`` (local only) > default.

    The config is consulted only for a *local* embeddings setup: under the default
    ``embeddings_provider=openai`` the configured ``embedding_model`` is an OpenAI name
    (e.g. ``text-embedding-3-small``), which is not a valid sentence-transformers repo.
    """
    if explicit:
        return explicit
    try:
        from bamboo.config import get_settings

        settings = get_settings()
        if settings.embeddings_provider == "local" and settings.embedding_model:
            return settings.embedding_model
    except Exception:  # noqa: BLE001 — config is best-effort; fall back to the default
        pass
    return DEFAULT_MODEL


def _resolve_reranker() -> str | None:
    """Reranker from config ``reranker_model`` (i.e. ``RERANKER_MODEL``), else ``None``.

    Reranking is opt-in and configured — there is no ``--reranker`` flag and no default
    cross-encoder. ``None`` means "don't stage a reranker" (and the batch run leaves reranking
    off). The reranker is not KB-bound; whatever you configure here is what a batch run uses
    (``run-analyze.sh`` derives ``RERANKER_MODEL`` from the manifest this writes).
    """
    try:
        from bamboo.config import get_settings

        settings = get_settings()
        if settings.reranker_model:
            return settings.reranker_model
    except Exception:  # noqa: BLE001 — config is best-effort; no reranker by default
        pass
    return None


def _resolve_out(explicit: str | None) -> str:
    """Output-dir precedence: ``--out`` > ``$EMBEDDINGS_OUT`` > ``${SHARED:-/shared}/bamboo/embeddings``."""
    if explicit:
        return explicit
    if os.environ.get("EMBEDDINGS_OUT"):
        return os.environ["EMBEDDINGS_OUT"]
    shared = os.environ.get("SHARED", "/shared")
    return os.path.join(shared, "bamboo", "embeddings")


def _warm(model: str, out: str, loader: str) -> None:
    """Warm the HF cache in *out* by loading *model* with *loader* in a child process.

    *loader* is the ``sentence_transformers`` class the runtime uses for this artifact —
    ``SentenceTransformer`` for embeddings, ``CrossEncoder`` for the reranker — so the
    downloaded cache layout matches exactly what ``get_embeddings()`` / ``get_reranker()``
    will resolve at runtime. ``HF_HOME`` must be set in the child's env (it is read at
    import time), with the offline flags off so the download can hit the network.
    """
    subprocess.run(
        [
            sys.executable,
            "-c",
            f"import sys; from sentence_transformers import {loader}; {loader}(sys.argv[1])",
            model,
        ],
        env={
            **os.environ,
            "HF_HOME": out,
            "HF_HUB_OFFLINE": "0",
            "TRANSFORMERS_OFFLINE": "0",
        },
        check=True,
    )


@click.command("stage-embeddings")
@click.option(
    "--model",
    default=None,
    help="Sentence-transformers model. Default: EMBEDDING_MODEL from config when "
    "EMBEDDINGS_PROVIDER=local, else sentence-transformers/all-MiniLM-L6-v2.",
)
@click.option(
    "--out",
    "out_dir",
    default=None,
    type=click.Path(file_okay=False),
    help="Embeddings output dir. Default: $EMBEDDINGS_OUT or ${SHARED:-/shared}/bamboo/embeddings.",
)
def main(model, out_dir):
    """Stage the local embedding model into shared storage (mounted read-only at /embeddings).

    Set RERANKER_MODEL in your config to also stage that cross-encoder alongside the embedding
    model (reranking is opt-in and configured — there is no --reranker flag).

    \b
      bamboo stage-embeddings                             # model from your bamboo config
      bamboo stage-embeddings --model all-mpnet-base-v2   # explicit
      SHARED=/shared bamboo stage-embeddings              # out = /shared/bamboo/embeddings
    """
    setup_logging()
    model = _resolve_model(model)
    reranker = _resolve_reranker()
    # Absolutise: the child process inherits its own cwd; a relative HF_HOME would land the
    # cache under that cwd rather than the intended shared path.
    out = os.path.abspath(_resolve_out(out_dir))
    os.makedirs(out, exist_ok=True)

    click.echo(f"[stage-embeddings] warming '{model}' into {out}")
    try:
        _warm(model, out, "SentenceTransformer")
        if reranker:
            click.echo(f"[stage-embeddings] warming reranker '{reranker}' into {out}")
            _warm(reranker, out, "CrossEncoder")
    except FileNotFoundError as exc:
        raise click.ClickException(
            "could not run the Python interpreter to warm the model — is this a valid "
            "environment?"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise click.ClickException(
            f"model download failed (exit {exc.returncode}). Ensure sentence-transformers is "
            "installed:  pip install 'bamboo[local]'"
        ) from exc

    # Manifest last: its presence implies the cache is populated. run-analyze.sh reads
    # reranker_model from it; embedding_model is recorded for a best-effort cross-check
    # against the KB snapshot's metadata.json.
    manifest = os.path.join(out, MANIFEST_NAME)
    with open(manifest, "w", encoding="utf-8") as fh:
        json.dump({"embedding_model": model, "reranker_model": reranker or ""}, fh, indent=2)
        fh.write("\n")

    staged = f"'{model}'" + (f" + reranker '{reranker}'" if reranker else "")
    click.echo(f"[stage-embeddings] ✓ {staged} staged to {out}")


if __name__ == "__main__":
    main()
