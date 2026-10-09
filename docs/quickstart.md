# Quickstart

## 1. Install

```bash
git clone https://github.com/meshapplied/Goethe.git
cd Goethe
pip install -e ./ternary-kernels
```

## 2. Fetch a ternary model

```python
import ternary_runtime

m = ternary_runtime.fetch("meshapplied/Qwen3.8-Flash-Next-Ternary-Latest")
# or: meshapplied/DeepSeek-V4.1-Flash-Ternary-Latest
print(m.profile, m.pack_dir, m.base_dir)
ternary_runtime.bootstrap()
```

`fetch` sets:

| Env | Meaning |
|-----|---------|
| `TERNARY_PACK_DIR` | Local expert pack |
| `TERNARY_MODEL_DIR` | Local base checkpoint |
| `TERNARY_PROFILE` | `flash_next` or `deepseek_v41` |

## 3. Serve with SGLang

Keep EP=TP (Hadamard blocks must not split across ranks). Default Flash-Next script uses `--tp-size 1`.

```bash
./examples/serve_flash_next.sh
```

Equivalent manual launch (`--model-path` = `m.base_dir`):

```bash
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0a}"
python -m sglang.launch_server \
  --model-path "$TERNARY_MODEL_DIR" \
  --tp-size 1 \
  --host 127.0.0.1 --port 30000 \
  --trust-remote-code
```

DeepSeek: [`serve_deepseek.sh`](../examples/serve_deepseek.sh). Needs the full base and large host RAM for Engram. Do not expect a throughput win vs FP4 experts.

## 4. Local packs (offline)

Place `ternary_config.json` in your pack directory, then:

```python
m = ternary_runtime.resolve_local("/data/my-pack", base_dir="/data/base-ckpt")
ternary_runtime.bootstrap()
```
