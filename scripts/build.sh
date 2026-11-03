#!/usr/bin/env bash
# Build the shared sandbox base (once) and the agent image.
#
#   scripts/build.sh [kit-path] [image-tag]
#
# kit-path : a clone of https://github.com/Agenthon-2026/track1-coding-public (default: ../track1-coding-public)
# image-tag: local tag for the agent image (default: agenthon-t1-agent:dev)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KIT="${1:-$HERE/../track1-coding-public}"
TAG="${2:-agenthon-t1-agent:dev}"

if [ ! -f "$KIT/docker/sandbox.Dockerfile" ]; then
  echo "kit not found at $KIT — clone it first:" >&2
  echo "  git clone https://github.com/Agenthon-2026/track1-coding-public.git $KIT" >&2
  exit 2
fi

if ! docker image inspect finance-bench-sandbox:latest >/dev/null 2>&1 || [ "${REBUILD_BASE:-0}" = "1" ]; then
  echo "== building finance-bench-sandbox:latest (shared base; compiles TA-Lib, takes a few minutes)"
  docker build --platform linux/amd64 -t finance-bench-sandbox:latest -f "$KIT/docker/sandbox.Dockerfile" "$KIT"
else
  echo "== finance-bench-sandbox:latest already present (REBUILD_BASE=1 to rebuild)"
fi

echo "== building $TAG"
# --provenance/--sbom=false keeps the pushed artifact a plain single-arch manifest (no attestation index).
docker build --platform linux/amd64 --provenance=false --sbom=false -t "$TAG" "$HERE" \
  || docker build --platform linux/amd64 -t "$TAG" "$HERE"

echo "== verb smoke test (must print the agent banner and exit 0)"
TMP="$(mktemp -d)"; mkdir -p "$TMP/unit" "$TMP/out"; chmod 777 "$TMP/out"
printf '# Task: smoke\nWrite /app/output/results.json with {"ok": true}.\n' > "$TMP/unit/instruction.md"
docker run --rm --network=none -v "$TMP/unit":/input:ro -v "$TMP/out":/app/output -v "$TMP/out":/output "$TAG" solve --task-dir /input --out /app/output
echo "== ok: $TAG"
