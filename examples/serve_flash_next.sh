#!/usr/bin/env bash
# Fetch Flash-Next ternary pack + Qwen base, patch SGLang, serve. EP=TP=1 by default.
set -euo pipefail
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0a}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT}/ternary-kernels${PYTHONPATH:+:$PYTHONPATH}"

python - <<'PY'
import ternary_runtime
m = ternary_runtime.fetch("meshapplied/Qwen3.8-Flash-Next-Ternary-Latest")
ternary_runtime.bootstrap()
print("profile", m.profile)
print("pack", m.pack_dir)
print("base", m.base_dir)
PY

: "${TERNARY_MODEL_DIR:?fetch did not set TERNARY_MODEL_DIR}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
TP="${TP:-1}"

exec python -m sglang.launch_server \
  --model-path "$TERNARY_MODEL_DIR" \
  --host "$HOST" --port "$PORT" \
  --tp-size "$TP" \
  --trust-remote-code \
  --served-model-name qwen38-flash-next \
  "$@"
