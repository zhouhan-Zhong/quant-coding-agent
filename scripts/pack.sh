#!/usr/bin/env bash
# Seal submission.json with your team id and write submission.zip (submission.json + team-claim.json).
#
#   scripts/pack.sh <team-number>            # asks for the Team Key on a hidden prompt
#   scripts/pack.sh <team-number> --team-key-file ~/team-key.txt   # or read it from a mode-600 file
#
# Needs Python >= 3.13 with the toolkit:
#   pip install "qfbench2-common @ git+https://github.com/Agenthon-2026/Agenthon2026-public.git@v2.5.1#subdirectory=common"
# The Team Key never goes into the zip and is never printed.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEAM="${1:?team number (from the Agenthon website)}"
shift || true
[ -f "$HERE/submission.json" ] || { echo "submission.json not found; run scripts/push_image.sh first" >&2; exit 2; }
command -v qfbench2 >/dev/null || { echo "qfbench2 not found: install the toolkit (see header)" >&2; exit 2; }
rm -f "$HERE/submission.zip"
qfbench2 submission pack --descriptor "$HERE/submission.json" --team-number "$TEAM" --out "$HERE/submission.zip" "$@"
echo
echo "contents of submission.zip:"; unzip -l "$HERE/submission.zip"
echo
echo "Upload $HERE/submission.zip on the Track 1 CodaBench page: My Submissions -> Development (1 per day)."
