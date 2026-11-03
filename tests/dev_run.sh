#!/usr/bin/env bash
# Sandbox-only developer loop (needs root, no Docker): run the agent on one public unit against the mock
# House server, then run that unit's own checker with the paths it hardcodes shimmed via symlinks.
#
#   tests/dev_run.sh <unit-id> [mock-mode] [kit-path]
#
set -uo pipefail
UNIT_ID="${1:?unit id}"
MODE="${2:-ok}"
KIT="${3:-/home/claude/track1-coding-public}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIT="$KIT/units/$UNIT_ID"
[ -d "$UNIT" ] || { echo "no such unit: $UNIT"; exit 2; }
OUT="$(mktemp -d /tmp/agent-out-XXXX)"
PORT="${MOCK_PORT:-8765}"

MOCK_MODE="$MODE" MOCK_PORT="$PORT" MOCK_SOLUTIONS_DIR="$HERE/tests/solutions" \
  ${PYTHON:-python} "$HERE/tests/mock_model_server.py" 2>"$OUT/.mock.log" &
MOCK_PID=$!
sleep 0.7

MODEL_ENDPOINT="http://127.0.0.1:$PORT" MODEL_NAME="mock-house" MODEL_TOKEN="mock-token" \
  AGENT_WORKDIR="/tmp/agent_work" env ${AGENT_ENV:-} \
  ${PYTHON:-python} "$HERE/agent/agent.py" solve --task-dir "$UNIT" --out "$OUT"
RC=$?
kill "$MOCK_PID" 2>/dev/null
echo "agent exit code: $RC"
echo "output files:"; ls -la "$OUT" | grep -v '^total' | grep -v '\.mock\.log'

# ---- checker with path shims (root only) ----
rm -rf /app/output /output /input /tests/reference_data 2>/dev/null
mkdir -p /app /tests /logs/verifier
ln -sfn "$OUT" /app/output
ln -sfn "$OUT" /output
ln -sfn "$UNIT" /input
if [ -d "$UNIT/checks/reference_data" ]; then ln -sfn "$UNIT/checks/reference_data" /tests/reference_data; fi
DATA="$UNIT/environment/data"
if [ -d "$DATA" ]; then
  rm -rf /app/data; ln -sfn "$DATA" /app/data
  for f in "$DATA"/*; do b="$(basename "$f")"; [ -e "/app/$b" ] || ln -sfn "$f" "/app/$b"; done
fi
cd "$UNIT/checks" && OUTPUT_DIR="$OUT" PYTHONDONTWRITEBYTECODE=1 ${PYTHON:-python} -m pytest test_outputs.py -q --tb=short -p no:cacheprovider 2>&1 | tail -15
echo "OUT=$OUT"
