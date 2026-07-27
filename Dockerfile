# syntax=docker/dockerfile:1
#
# Two-target build (see docs/BATCH.md and the plan):
#   --target bamboo                → Image 1: the bamboo app, talks to EXTERNAL services
#                                    (the standalone Docker artifact).
#   --target bamboo-batch-analyze  → Image 2: FROM bamboo, bundles Neo4j + Qdrant + Ollama
#                                    so a single container is self-sufficient on an
#                                    air-gapped batch node. Convert to .sif with Apptainer.
#
ARG PYTHON_VERSION=3.12
ARG NEO4J_VERSION=2026.06.0
# Pin explicitly (not a floating `v1`): a Qdrant snapshot is not portable across
# minor versions, so the image's Qdrant must match the version that built the KB
# snapshot (`bamboo dump-kb`). Bump this and rebuild the KB together.
ARG QDRANT_VERSION=v1.18.3
# Pin explicitly (not a floating `latest`): an Ollama "install" is a set of files
# whose layout changes across releases — 0.32.x moved inference out of the Go
# binary into a separate llama-server + lib/ollama runtime. A floating version
# silently changes what this image has to ship. The install URL below embeds it.
ARG OLLAMA_VERSION=0.32.4

# --------------------------------------------------------------------------- #
# Binary source stages — we COPY official binaries out of these (no hand-built
# installs). Pulling the whole image as a source stage is cheap; only the COPYed
# paths land in the final image.
#
# COPY-from-image is only safe when the copied path is the *whole* program. That
# holds for the two below; it did NOT hold for Ollama, whose runtime is split
# across several files — so Ollama is installed from its release tarball instead.
# --------------------------------------------------------------------------- #
# Neo4j: /var/lib/neo4j in this image *is* the extracted neo4j-community-<ver>-unix
# tarball, so copying it out is the official install tree, not a subset of one.
FROM neo4j:${NEO4J_VERSION} AS neo4j-src
# Qdrant: genuinely a single self-contained binary. Use the unprivileged variant
# (built to run as a non-root user) — the natural fit for rootless model.
# (No ollama-src stage: Ollama is installed from its release tarball below, because
#  it is *not* a single file — see the install block in image 2.)
FROM qdrant/qdrant:${QDRANT_VERSION}-unprivileged AS qdrant-src

# =========================================================================== #
# Image 1 — bamboo (the app; standalone Docker artifact)
# =========================================================================== #
FROM python:${PYTHON_VERSION}-slim-bookworm AS bamboo

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# curl/ca-certificates: runtime readiness probes + build-time downloads.
# git: some panda extras install from VCS.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Pin CPU-only torch BEFORE installing bamboo so sentence-transformers doesn't
# drag in multi-GB CUDA wheels (Ollama does its own GPU; embeddings run on CPU).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy the project and install with the offline-model (+ panda) extras.
COPY . /app
RUN git clone https://github.com/PanDAWMS/panda-client.git && cd panda-client && cp packages/light/pyproject.toml . && pip install --no-cache-dir .
RUN pip install --no-cache-dir ".[local,panda]"

# Image 1 is pure app: configured entirely by env (NEO4J_URI/QDRANT_URL/LLM_*/…).
ENTRYPOINT ["bamboo"]
CMD ["--help"]

# =========================================================================== #
# Image 2 — bamboo-batch-analyze (FROM bamboo; bundles the service stack)
# =========================================================================== #
FROM bamboo AS bamboo-batch-analyze

# Re-declare (no default) to inherit the values from the global ARGs above.
ARG NEO4J_VERSION
ARG QDRANT_VERSION
ARG OLLAMA_VERSION

# Surface the baked service versions at runtime so entrypoint.sh can guard the KB
# snapshot against a version-incompatible Neo4j dump / Qdrant snapshot (see docs/BATCH.md).
ENV NEO4J_VERSION=${NEO4J_VERSION} \
    QDRANT_VERSION=${QDRANT_VERSION}

# --- JRE + Neo4j ---
# The JRE is taken from the neo4j image rather than eclipse-temurin/apt so it is
# byte-for-byte the runtime upstream ships and tests this Neo4j against.
ENV JAVA_HOME=/opt/java/openjdk
COPY --from=neo4j-src /opt/java/openjdk /opt/java/openjdk
ENV NEO4J_HOME=/opt/neo4j
COPY --from=neo4j-src /var/lib/neo4j /opt/neo4j
ENV PATH="${NEO4J_HOME}/bin:${JAVA_HOME}/bin:${PATH}"

# APOC must be present at build time because entrypoint.sh bypasses the official
# neo4j entrypoint (which would otherwise fetch plugins at runtime — impossible
# air-gapped). The APOC core jar is released in lockstep with NEO4J_VERSION.
RUN curl -fsSL -o "${NEO4J_HOME}/plugins/apoc-${NEO4J_VERSION}-core.jar" \
      "https://github.com/neo4j/apoc/releases/download/${NEO4J_VERSION}/apoc-${NEO4J_VERSION}-core.jar"

# --- System libraries for the copied/installed service binaries ---
# libunwind8: the Qdrant binary is dynamically linked against libunwind (used for
#   backtraces); the slim base lacks it, so the copied binary dies at startup with
#   "libunwind-ptrace.so.0: cannot open shared object file". libunwind8 provides both
#   libunwind-ptrace.so.0 and the arch variant (libunwind-<arch>.so.8), on amd64 and
#   arm64 alike.
# Nothing is listed here for Ollama on purpose: its tarball vendors what it needs
# (including its own libgomp.so.1) beside the binaries, and only libstdc++6/libgcc_s
# — already in the base — are taken from the system. Don't add packages here on a
# hunch; the linkage assertion further down names anything that is actually missing.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libunwind8 \
    && rm -rf /var/lib/apt/lists/*

# --- Qdrant ---
# Only the binary: Qdrant is self-contained, and the stack is configured entirely
# through QDRANT__* env vars in entrypoint.sh (port, storage path). The official
# image's config/{config.yaml,production.yaml} are deliberately NOT copied — Qdrant
# resolves them relative to its cwd, which is /app here rather than the image's
# /qdrant, so they would need an explicit --config-path and would become a second
# source of truth beside the env overrides. Compiled-in defaults + env is the
# intended configuration for this container.
COPY --from=qdrant-src /qdrant/qdrant /usr/local/bin/qdrant

# --- Ollama ---
# Installed from the official release tarball, i.e. what upstream's install.sh does
# (`curl … | zstd -d | tar -x -C /usr/local`), NOT by copying a binary out of the
# ollama/ollama image. Ollama is not one file: since 0.32.x inference runs in a
# separate llama-server binary plus libggml*.so under lib/ollama/. Extracting the
# archive gets that whole set by construction and puts it at /usr/local/lib/ollama,
# the <exedir>/../lib/ollama path the runtime already searches — no LD_LIBRARY_PATH.
# The CUDA runtime under lib/ollama/cuda_v*/ is kept on purpose: submit.sh supports
# USE_GPU=1 (apptainer --nv), where the host supplies the driver and this supplies
# the runtime. It is most of the download size.
# .tar.zst is the only format published for current releases (the old .tgz URL 404s),
# hence the build-only zstd.
ARG TARGETARCH
RUN apt-get update \
    && apt-get install -y --no-install-recommends zstd \
    && curl -fsSL "https://github.com/ollama/ollama/releases/download/v${OLLAMA_VERSION}/ollama-linux-${TARGETARCH}.tar.zst" \
       | zstd -d | tar -x -C /usr/local \
    && apt-get purge -y --auto-remove zstd \
    && rm -rf /var/lib/apt/lists/*

# --- Assert the service binaries are complete and linkable ---
# Nothing else catches an incomplete install until a job runs: Ollama answers
# GET /api/tags (entrypoint.sh's readiness probe) with no inference runtime present,
# so setup reports "all services ready" while generation is impossible. This turns
# that into a build failure — both for a moved/renamed runtime and for any missing
# system library, which it names instead of leaving you to guess.
#
# The ggml CPU backends (libggml-cpu-*.so) are dlopen'd, not linked, so they are
# checked individually rather than via llama-server. cuda_v*/ is deliberately NOT
# checked: those legitimately report libcuda.so.1 "not found" at build time — the
# NVIDIA driver comes from the host under `apptainer --nv` (submit.sh USE_GPU=1).
RUN set -eu; \
    test -x /usr/local/lib/ollama/llama-server || { \
      echo "FATAL: no llama-server under /usr/local/lib/ollama — Ollama's layout changed" >&2; \
      ls -R /usr/local/lib/ollama >&2 || true; exit 1; }; \
    for b in /usr/local/bin/ollama /usr/local/bin/qdrant \
             /usr/local/lib/ollama/llama-server /usr/local/lib/ollama/*.so*; do \
      if ldd "$b" 2>/dev/null | grep "not found"; then \
        echo "FATAL: $b has unresolved shared libraries (listed above)" >&2; exit 1; \
      fi; \
    done; \
    echo "ok: ollama + qdrant present and fully linked ($(ls /usr/local/lib/ollama/*.so* | wc -l) shared objects checked)"

# --- Local embeddings resolve from a read-only HF cache mounted at /embeddings ---
# The embedding model is NOT baked: it is staged onto shared storage with
# `bamboo stage-embeddings` and mounted at /embeddings (HF_HOME), like the Ollama model at
# /models. EMBEDDING_MODEL/EMBEDDING_DIMENSION are derived at runtime from the KB snapshot's
# metadata.json (see deploy/batch/entrypoint.sh) so query embeddings always match the KB.
ENV HF_HOME=/embeddings \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    EMBEDDINGS_PROVIDER=local \
    LLM_PROVIDER=ollama

# --- Entry script (orchestrates the localhost stack per job) ---
# Dispatches subcommands (setup/batch/teardown/shell); no subcommand = full run
# (setup → batch → teardown), the backward-compatible default. For interactive
# debugging: `docker run -it … bamboo-batch-analyze shell`.
COPY deploy/batch/entrypoint.sh /opt/bamboo/entrypoint.sh
RUN chmod +x /opt/bamboo/entrypoint.sh

ENTRYPOINT ["/opt/bamboo/entrypoint.sh"]
CMD []
