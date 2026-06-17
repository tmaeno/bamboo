#!/usr/bin/env bash
# build-kb-snapshot.sh — produce the read-only KB snapshot consumed by the batch
# container, on a NETWORKED host (this is the only step that needs egress).
#
# Stands up a transient Neo4j + Qdrant, lets you populate them with `bamboo
# populate` / `bamboo batch-populate`, then exports:
#   <db>.dump                 Neo4j offline dump (neo4j-admin database dump)
#   qdrant_storage.tar.zst    full Qdrant storage dir
#   metadata.json             embedding model + dimension (for the run-time guard)
# and stages them to $KB_OUT (default $SHARED/bamboo/kb).
#
# ⚠ SCAFFOLD — UNVERIFIED. This subsumes the dev-only docker-compose.yml. Adapt the
#   populate step to your actual inputs. Requires Docker.
set -euo pipefail

NEO4J_VERSION="${NEO4J_VERSION:-5.16.0}"
QDRANT_VERSION="${QDRANT_VERSION:-v1.13.6}"
NEO4J_DATABASE="${NEO4J_DATABASE:-graph_db}"
NEO4J_PASSWORD="${NEO4J_PASSWORD:-password}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"
EMBEDDING_DIMENSION="${EMBEDDING_DIMENSION:-384}"
KB_OUT="${KB_OUT:-${SHARED:-/shared}/bamboo/kb}"

WORK="$(mktemp -d)"
NEO4J_DATA="${WORK}/neo4j-data"
QDRANT_STORAGE="${WORK}/qdrant-storage"
mkdir -p "${NEO4J_DATA}" "${QDRANT_STORAGE}" "${KB_OUT}"

cname_neo4j="bamboo-kb-neo4j-$$"
cname_qdrant="bamboo-kb-qdrant-$$"
cleanup() { docker rm -f "${cname_neo4j}" "${cname_qdrant}" >/dev/null 2>&1 || true; }
trap cleanup EXIT

echo "[build-kb] starting transient Neo4j + Qdrant…"
docker run -d --name "${cname_neo4j}" \
  -p 7687:7687 -p 7474:7474 \
  -e NEO4J_AUTH="${NEO4J_DATABASE}/${NEO4J_PASSWORD}" \
  -e NEO4J_PLUGINS='["apoc"]' \
  -e NEO4J_dbms_security_procedures_unrestricted='apoc.*' \
  -v "${NEO4J_DATA}:/data" \
  "neo4j:${NEO4J_VERSION}" >/dev/null
docker run -d --name "${cname_qdrant}" \
  -p 6333:6333 \
  -v "${QDRANT_STORAGE}:/qdrant/storage" \
  "qdrant/qdrant:${QDRANT_VERSION}" >/dev/null

echo "[build-kb] waiting for services…"
for _ in $(seq 60); do curl -fsS http://localhost:6333/readyz >/dev/null 2>&1 && break; sleep 2; done
sleep 10  # give Neo4j/APOC time to finish first-boot

# ---------------------------------------------------------------------------
# POPULATE — replace this with your real population commands. Both connect via
# the env below (bamboo reads NEO4J_URI/QDRANT_URL and the embedding settings).
# ---------------------------------------------------------------------------
export NEO4J_URI="bolt://localhost:7687"
export QDRANT_URL="http://localhost:6333"
export NEO4J_DATABASE NEO4J_PASSWORD
export NEO4J_USERNAME="${NEO4J_DATABASE}"
export EMBEDDINGS_PROVIDER=local EMBEDDING_MODEL EMBEDDING_DIMENSION

echo "[build-kb] >>> populate the KB now (edit this script), e.g.:"
echo "    bamboo batch-populate --drafts drafts/ --yes"
echo "    bamboo populate --task-data some_task.json"
# bamboo batch-populate --drafts drafts/ --yes    # <-- uncomment / adapt

# ---------------------------------------------------------------------------
# EXPORT
# ---------------------------------------------------------------------------
echo "[build-kb] dumping Neo4j (offline)…"
docker stop "${cname_neo4j}" >/dev/null
# Offline dump via a one-off admin container over the same data volume.
docker run --rm -v "${NEO4J_DATA}:/data" "neo4j:${NEO4J_VERSION}" \
  neo4j-admin database dump "${NEO4J_DATABASE}" --to-path=/data >/dev/null
cp "${NEO4J_DATA}/${NEO4J_DATABASE}.dump" "${KB_OUT}/"

echo "[build-kb] archiving Qdrant storage…"
if command -v zstd >/dev/null 2>&1; then
  tar --use-compress-program=zstd -cf "${KB_OUT}/qdrant_storage.tar.zst" -C "${QDRANT_STORAGE}" .
else
  tar -czf "${KB_OUT}/qdrant_storage.tar.gz" -C "${QDRANT_STORAGE}" .
fi

echo "[build-kb] writing metadata.json…"
cat >"${KB_OUT}/metadata.json" <<EOF
{
  "embedding_model": "${EMBEDDING_MODEL}",
  "embedding_dimension": ${EMBEDDING_DIMENSION},
  "neo4j_version": "${NEO4J_VERSION}",
  "neo4j_database": "${NEO4J_DATABASE}",
  "qdrant_version": "${QDRANT_VERSION}"
}
EOF

rm -rf "${WORK}"
echo "[build-kb] ✓ snapshot staged to ${KB_OUT}"
ls -la "${KB_OUT}"
