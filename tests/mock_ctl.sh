#!/usr/bin/env bash
# tests/mock_ctl.sh start <mode> [port] | stop
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIDF=/tmp/mock_model_server.pid
case "${1:-}" in
  start)
    [ -f "$PIDF" ] && kill "$(cat "$PIDF")" 2>/dev/null; sleep 0.3
    MOCK_MODE="${2:-ok}" MOCK_HOST=0.0.0.0 MOCK_PORT="${3:-8765}" MOCK_SOLUTIONS_DIR="$HERE/tests/solutions" \
      nohup ${PYTHON:-python} "$HERE/tests/mock_model_server.py" >/tmp/mock.log 2>&1 &
    echo $! > "$PIDF"; sleep 0.8; echo "mock started pid $(cat $PIDF) mode=${2:-ok}" ;;
  stop) [ -f "$PIDF" ] && kill "$(cat "$PIDF")" 2>/dev/null; rm -f "$PIDF"; echo "mock stopped" ;;
esac
