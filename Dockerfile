# syntax=docker/dockerfile:1
#
# Two-target build (see docs/BATCH.md and the plan):
#   --target bamboo                → Image 1: the bamboo app, talks to EXTERNAL services
#                                    (the standalone Docker artifact).
#   --target bamboo-batch-analyze  → Image 2: FROM bamboo, bundles Neo4j + Qdrant + Ollama
#                                    so a single container is self-sufficient on an
#                                    air-gapped batch node. Convert to .sif with Apptainer.
#
# ⚠ SCAFFOLD — UNVERIFIED. Several paths below (JRE/Neo4j/Qdrant/Ollama binary
#   locations, APOC URL) are best-effort and MUST be confirmed by the Phase 0 spike.
#   Grep for "VERIFY:" for the spots to check on the cluster.

ARG PYTHON_VERSION=3.12
ARG NEO4J_VERSION=2026.06.0
# Pin explicitly (not a floating `v1`): a Qdrant snapshot is not portable across
# minor versions, so the image's Qdrant must match the version that built the KB
# snapshot (`bamboo dump-kb`). Bump this and rebuild the KB together.
ARG QDRANT_VERSION=v1.18.3
ARG OLLAMA_VERSION=latest

# --------------------------------------------------------------------------- #
# Binary source stages — we COPY official binaries out of these (no hand-built
# installs). Pulling the whole image as a source stage is cheap; only the COPYed
# paths land in the final image.
# --------------------------------------------------------------------------- #
FROM neo4j:${NEO4J_VERSION} AS neo4j-src
# Use the unprivileged Qdrant variant (built to run as a non-root user) — the
# natural fit for rootless model.
FROM qdrant/qdrant:${QDRANT_VERSION}-unprivileged AS qdrant-src
FROM ollama/ollama:${OLLAMA_VERSION} AS ollama-src

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

# Surface the baked service versions at runtime so run-analyze.sh can guard the KB
# snapshot against a version-incompatible Neo4j dump / Qdrant snapshot (see docs/BATCH.md).
ENV NEO4J_VERSION=${NEO4J_VERSION} \
    QDRANT_VERSION=${QDRANT_VERSION}

# --- JRE + Neo4j (VERIFY: paths in the neo4j:5.x image; it is eclipse-temurin based) ---
ENV JAVA_HOME=/opt/java/openjdk
COPY --from=neo4j-src /opt/java/openjdk /opt/java/openjdk
ENV NEO4J_HOME=/opt/neo4j
COPY --from=neo4j-src /var/lib/neo4j /opt/neo4j
ENV PATH="${NEO4J_HOME}/bin:${JAVA_HOME}/bin:${PATH}"

# APOC must be present at build time because run-analyze.sh bypasses the official
# neo4j entrypoint (which would otherwise fetch plugins at runtime — impossible
# air-gapped). VERIFY: the APOC core jar version matches NEO4J_VERSION.
RUN curl -fsSL -o "${NEO4J_HOME}/plugins/apoc-${NEO4J_VERSION}-core.jar" \
      "https://github.com/neo4j/apoc/releases/download/${NEO4J_VERSION}/apoc-${NEO4J_VERSION}-core.jar"

# --- Qdrant + Ollama binaries (VERIFY: source paths) ---
# The Qdrant binary is dynamically linked against libunwind (used for backtraces);
# the slim base lacks it, so the copied binary dies at startup with
# "libunwind-ptrace.so.0: cannot open shared object file". libunwind8 provides both
# libunwind-ptrace.so.0 and the arch variant (libunwind-<arch>.so.8), on amd64 and
# arm64 alike. Ollama is a static Go binary and needs nothing extra.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libunwind8 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=qdrant-src /qdrant/qdrant /usr/local/bin/qdrant
COPY --from=ollama-src /usr/bin/ollama /usr/local/bin/ollama

# --- Local embeddings resolve from a read-only HF cache mounted at /embeddings ---
# The embedding model is NOT baked: it is staged onto shared storage with
# `bamboo stage-embeddings` and mounted at /embeddings (HF_HOME), like the Ollama model at
# /models. EMBEDDING_MODEL/EMBEDDING_DIMENSION are derived at runtime from the KB snapshot's
# metadata.json (see deploy/batch/run-analyze.sh) so query embeddings always match the KB.
ENV HF_HOME=/embeddings \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    EMBEDDINGS_PROVIDER=local \
    LLM_PROVIDER=ollama

# --- Entry script (orchestrates the localhost stack per job) ---
COPY deploy/batch/run-analyze.sh /opt/bamboo/run-analyze.sh
RUN chmod +x /opt/bamboo/run-analyze.sh

ENTRYPOINT ["/opt/bamboo/run-analyze.sh"]
CMD []
