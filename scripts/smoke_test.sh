#!/usr/bin/env bash
# Run the agent image on one public unit, then run that unit's own checker (README step 6).
#
#   scripts/smoke_test.sh <unit-id> [kit-path] [image-tag]
#
# Environment (optional, to exercise a House-like model locally):
#   MODEL_ENDPOINT=http://host.docker.internal:8000  MODEL_NAME=<model>  MODEL_TOKEN=<any>
# Without MODEL_ENDPOINT the agent runs with --network=none and gets no model (expect reward 0;
# this still proves the verb, mounts and exit code).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIT_ID="${1:?unit id, e.g. t1-EXAMPLE-bs-greeks-pde}"
KIT="${2:-$HERE/../track1-coding-public}"
TAG="${3:-agenthon-t1-agent:dev}"
UNIT="$(cd "$KIT/units/$UNIT_ID" && pwd)" || { echo "no such unit"; exit 2; }
OUT="$(mktemp -d)"; chmod 777 "$OUT"

if [ -n "${MODEL_ENDPOINT:-}" ]; then
  NET=(--add-host=host.docker.internal:host-gateway -e MODEL_ENDPOINT -e MODEL_NAME -e MODEL_TOKEN -e QFBENCH_NETWORK=restricted)
else
  NET=(--network=none -e QFBENCH_NETWORK=none)
fi

echo "== agent on $UNIT_ID -> $OUT"
docker run --rm "${NET[@]}" \
  -e QFBENCH_SEED=0 ${AGENT_DOCKER_ENV:-} \
  -v "$UNIT":/input:ro -v "$OUT":/app/output -v "$OUT":/output \
  "$TAG" solve --task-dir /input --out /app/output
echo "agent exit code: $?"
ls -la "$OUT"

echo "== checker"
# Mount the unit's build-time data locations too, because several checkers open them directly.
EXTRA=()
[ -d "$UNIT/environment/data" ] && EXTRA+=(-v "$UNIT/environment/data":/app/data:ro)
[ -d "$UNIT/checks/reference_data" ] && EXTRA+=(-v "$UNIT/checks/reference_data":/tests/reference_data:ro)
if [ -d "$UNIT/environment/data" ]; then
  for f in "$UNIT"/environment/data/*; do
    [ -e "$f" ] || continue
    EXTRA+=(-v "$f":"/app/$(basename "$f")":ro)
  done
fi
docker run --rm --network=none \
  -e OUTPUT_DIR=/app/output -e PYTHONDONTWRITEBYTECODE=1 \
  -v "$UNIT":/input:ro -v "$OUT":/app/output -v "$OUT":/output "${EXTRA[@]}" \
  finance-bench-sandbox:latest bash /input/checks/test.sh
python3 - "$OUT/reward.json" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1])); print("reward:", r)
sys.exit(0 if r.get("reward") == 1 else 1)
EOF
