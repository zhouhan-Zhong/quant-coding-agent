#!/usr/bin/env bash
# Sandbox-only (root, no Docker): run a hand-written solution for one public unit, then that unit's checker.
#
#   tests/solve_unit.sh <unit-id> <solution.py> [kit-path]
#
# The solution reads DATA_DIR / OUTPUT_DIR from the environment (defaults /app/data, /app/output), which
# is also the convention the agent sets for generated scripts.  The checker's hardcoded paths are shimmed
# with symlinks (/app/output, /output, /input, /app/data, /app/<file>, /tests/reference_data).
set -uo pipefail
# Serialise runs: the path shims below are global, so concurrent runs would clobber each other.
if [ -z "${SOLVE_UNIT_LOCKED:-}" ]; then
  exec env SOLVE_UNIT_LOCKED=1 flock -w 1800 /tmp/solve_unit.lock bash "$0" "$@"
fi
UNIT_ID="${1:?unit id}"; SOL="$(cd "$(dirname "${2:?solution.py}")" && pwd)/$(basename "$2")"; KIT="${3:-/home/claude/track1-coding-public}"
UNIT="$KIT/units/$UNIT_ID"; [ -d "$UNIT" ] || { echo "no such unit: $UNIT"; exit 2; }
PY="${PYTHON:-/opt/venv313/bin/python}"
OUT="$(mktemp -d /tmp/sol-out-XXXX)"
DATA="$UNIT/environment/data"
[ -d "$DATA" ] || DATA="$UNIT/environment"

# shims (the checker and some instructions use these absolute paths)
rm -rf /app/output /output /input /tests/reference_data /app/data 2>/dev/null
mkdir -p /app /tests /logs/verifier
find /app -maxdepth 1 -type l -delete 2>/dev/null   # stale /app/<file> shims from a previous unit
ln -sfn "$OUT" /app/output; ln -sfn "$OUT" /output; ln -sfn "$UNIT" /input
[ -d "$UNIT/checks/reference_data" ] && ln -sfn "$UNIT/checks/reference_data" /tests/reference_data
mkdir -p /app/data && cp -as "$DATA"/. /app/data/ 2>/dev/null     # symlink tree (so renamed aliases can be added)
for f in "$DATA"/*; do b="$(basename "$f")"; [ -e "/app/$b" ] || ln -sfn "$f" "/app/$b"; done
# Dockerfile COPY targets (possibly renamed files, e.g. stock_chars.pqt -> /app/data/stock_data.parquet)
"$PY" - "$UNIT" <<'PYEOF'
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])) if False else "/home/claude/agenthon-t1-agent/agent"))
from pathlib import Path
import agent
for alias, real in agent.parse_dockerfile_aliases(Path(sys.argv[1])).items():
    p = Path(alias)
    if alias.startswith("/app/") and not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(real, p)
PYEOF

echo "== running $SOL (DATA_DIR=$DATA OUTPUT_DIR=$OUT)"
START=$(date +%s)
( cd "$OUT" && DATA_DIR="$DATA" OUTPUT_DIR="$OUT" MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 timeout ${SOLVE_TIMEOUT:-600} "$PY" "$SOL" )
RC=$?; echo "== solution exit=$RC in $(( $(date +%s) - START ))s"; ls -la "$OUT" | grep -v '^total'
echo "== checker"
cd "$UNIT/checks" && OUTPUT_DIR="$OUT" PYTHONDONTWRITEBYTECODE=1 "$PY" -m pytest test_outputs.py -q --tb=short -p no:cacheprovider -x 2>&1 | tail -${TAIL:-25}
echo "OUT=$OUT"
