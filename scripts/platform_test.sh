#!/usr/bin/env bash
# Run one unit with the platform's own container settings (DEVELOPMENT-RUNTIME.md):
# non-root uid 65534, read-only root FS, 64 MiB noexec /tmp, pid/fd limits, no network.
#
#   scripts/platform_test.sh <unit-id> [kit-path] [image-tag]
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIT_ID="${1:?unit id}"
KIT="${2:-$HERE/../track1-coding-public}"
TAG="${3:-agenthon-t1-agent:dev}"
UNIT="$(cd "$KIT/units/$UNIT_ID" && pwd)" || exit 2
OUT="$(mktemp -d)"; chmod 777 "$OUT"

CPUS=$(python3 -c "import tomllib;print(int(tomllib.load(open('$UNIT/card.toml','rb'))['environment'].get('cpus',16)))" 2>/dev/null || echo 4)
MEM=$(python3 -c "import tomllib;print(tomllib.load(open('$UNIT/card.toml','rb'))['environment'].get('memory','128G'))" 2>/dev/null || echo 8g)
# Docker Desktop may not have 128 GiB; cap to what the host offers (the platform enforces the card).
if [ -n "${PLATFORM_TEST_MEM:-}" ]; then MEM="$PLATFORM_TEST_MEM"; fi
if [ -n "${PLATFORM_TEST_CPUS:-}" ]; then CPUS="$PLATFORM_TEST_CPUS"; fi

if [ -n "${MODEL_ENDPOINT:-}" ]; then
  NET=(--add-host=host.docker.internal:host-gateway -e MODEL_ENDPOINT -e MODEL_NAME -e MODEL_TOKEN -e QFBENCH_NETWORK=restricted)
else
  NET=(--network=none -e QFBENCH_NETWORK=none)
fi

set -x
docker run --rm \
  --read-only --user 65534:65534 --cap-drop=ALL --security-opt no-new-privileges \
  --tmpfs /tmp:rw,noexec,nosuid,nodev,size=64m \
  --pids-limit 256 --ulimit nofile=1024:1024 --ulimit nproc=256:256 \
  --cpus "$CPUS" --memory "$MEM" --memory-swap "$MEM" \
  "${NET[@]}" \
  -v "$UNIT":/input:ro -v "$OUT":/output -v "$OUT":/app/output \
  -e QFBENCH_SEED=0 \
  "$TAG" solve --task-dir /input --out /app/output
RC=$?
set +x
echo "exit code: $RC (must be 0)"
ls -la "$OUT"
