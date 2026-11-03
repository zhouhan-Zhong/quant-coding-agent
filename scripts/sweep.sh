#!/usr/bin/env bash
# Run the organizers' conformance sweep (exit 0 + expected filename on every public unit, offline).
#
#   scripts/sweep.sh [image-tag] [kit-path]
#
# Without a model the agent writes only a placeholder results.json, so units whose checker expects
# another filename show up as "wrote none of ..." -- that is expected offline.  What must NOT appear
# is a CRASH line.  The sweep script itself is fetched from the hub repo if it is not present.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAG="${1:-agenthon-t1-agent:dev}"
KIT="${2:-$HERE/../track1-coding-public}"
HUB="$HERE/../Agenthon2026-public"
if [ ! -f "$HUB/starter-packs/track1/conformance.sh" ]; then
  git clone --depth 1 https://github.com/Agenthon-2026/Agenthon2026-public.git "$HUB"
fi
bash "$HUB/starter-packs/track1/conformance.sh" "$TAG" "$KIT"
