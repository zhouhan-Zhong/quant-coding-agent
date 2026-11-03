#!/usr/bin/env bash
# Serve the House model locally with vLLM's OpenAI-compatible API (for eval/local_eval.sh).
#
#   bash eval/serve_vllm.sh                       # all defaults
#   GPUS=2 MAXLEN=65536 bash eval/serve_vllm.sh   # tensor parallel over 2 GPUs
#   GPUS=1 MAXLEN=32768 GPU_UTIL=0.95 EXTRA="--enforce-eager" bash eval/serve_vllm.sh   # squeeze onto 1x H200
#
# The competition serves the FP8 snapshot "rl-030326-fp8" of NVIDIA Nemotron 3 Super (120B total, 12B active);
# the public FP8 checkpoint below is its counterpart (Agenthon2026-public docs/HOUSE-MODEL.md).  The model card
# lists 2x H100-80GB (or 1x B200) as the minimum for FP8; one H200 (141 GB) is tight, so prefer GPUS=2.
# Thinking is on by default in the chat template; the agent sends chat_template_kwargs.enable_thinking=false.
# No --reasoning-parser: any reasoning stays inline in `content`, the less convenient case the agent handles.
set -euo pipefail
MODEL="${MODEL:-nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8}"
NGPU="$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)"
GPUS="${GPUS:-$(( NGPU >= 2 ? 2 : 1 ))}"
MAXLEN="${MAXLEN:-65536}"
exec vllm serve "$MODEL" \
  --served-model-name "${SERVED_NAME:-house}" \
  --tensor-parallel-size "$GPUS" \
  --max-model-len "$MAXLEN" \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization "${GPU_UTIL:-0.92}" \
  --max-num-seqs "${MAX_SEQS:-8}" \
  --trust-remote-code \
  --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}" \
  ${EXTRA:-}
