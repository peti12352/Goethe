#!/usr/bin/env bash
# DeepSeek-V4.1-Flash *ternary overlay* (not NVFP4). Needs full base + Engram host RAM.
# Lab measurement: ternary is not a throughput win vs FP4 experts (+~3% decode, worse TTFT).
set -euo pipefail
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0a}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT}/ternary-kernels${PYTHONPATH:+:$PYTHONPATH}"

echo "WARNING: DeepSeek base is large; Engram wants tens–hundreds of GB host RAM." >&2

python - <<'PY'
import ternary_runtime
m = ternary_runtime.fetch("meshapplied/DeepSeek-V4.1-Flash-Ternary-Latest")
ternary_runtime.bootstrap()
print("profile", m.profile)
print("pack", m.pack_dir)
print("base", m.base_dir)
PY

: "${TERNARY_MODEL_DIR:?fetch did not set TERNARY_MODEL_DIR}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30001}"
TP="${TP:-1}"

exec python -m sglang.launch_server \
  --model-path "$TERNARY_MODEL_DIR" \
  --host "$HOST" --port "$PORT" \
  --tp-size "$TP" \
  --trust-remote-code \
  --served-model-name deepseek-v41-flash-ternary \
  "$@"
