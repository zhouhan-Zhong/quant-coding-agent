#!/usr/bin/env bash
# One-time setup inside WSL2 (Ubuntu) on Windows, after Docker Desktop is installed with WSL integration.
#
#   bash scripts/setup_wsl.sh
#
# Installs: git/curl/unzip, uv + Python 3.13, a venv with the qfbench2 toolkit, and clones the public kit
# next to this folder (../track1-coding-public).  Re-running is harmless.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "== apt packages"
sudo apt-get update -qq && sudo apt-get install -y -qq git curl unzip ca-certificates >/dev/null

echo "== docker (Docker Desktop -> Settings -> Resources -> WSL integration must be ON for this distro)"
if ! docker version >/dev/null 2>&1; then
  echo "docker is not reachable from WSL. Start Docker Desktop on Windows and enable WSL integration, then re-run." >&2
  exit 2
fi

echo "== uv + Python 3.13"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv python install 3.13 >/dev/null
VENV="$HOME/.venvs/agenthon"
[ -d "$VENV" ] || uv venv --python 3.13 "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
uv pip install -q "qfbench2-common @ git+https://github.com/Agenthon-2026/Agenthon2026-public.git@v2.5.1#subdirectory=common"
python -c "from qfbench2_common.contracts import CONTRACT_SET; print('toolkit ok, contract set', CONTRACT_SET)"

echo "== public kit"
KIT="$HERE/../track1-coding-public"
if [ -d "$KIT/.git" ]; then git -C "$KIT" pull --ff-only || true; else git clone https://github.com/Agenthon-2026/track1-coding-public.git "$KIT"; fi
export PYTHONPATH="$KIT"

cat <<EOF

Setup done.  Every new WSL shell needs:
  source $VENV/bin/activate
  export PATH="\$HOME/.local/bin:\$PATH"

Next steps:
  scripts/build.sh                                  # builds finance-bench-sandbox:latest + agenthon-t1-agent:dev
  scripts/smoke_test.sh t1-EXAMPLE-bs-greeks-pde     # offline run + checker (expect exit 0; reward 0 without a model)
  scripts/platform_test.sh t1-EXAMPLE-bs-greeks-pde  # platform-like container settings
  bash $KIT/../Agenthon2026-public/starter-packs/track1/conformance.sh agenthon-t1-agent:dev $KIT   # optional sweep
EOF
