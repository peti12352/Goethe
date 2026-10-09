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

m = ternary_runtime.fetch("meshapplied/qwen3.8-flash-next-ternary-latest")
# or: meshapplied/DeepSeek-V4.1-Flash-NVFP4
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

Launch SGLang with `--model-path` = `m.base_dir`, tensor parallel = expert parallel as required by the Hadamard block sizes (see [pack.md](pack.md)). Keep the process environment from step 2 so the runtime can load the pack.

```bash
python -m sglang.launch_server \
  --model-path "$TERNARY_MODEL_DIR" \
  --tp-size 1 \
  --host 127.0.0.1 --port 30000 \
  --trust-remote-code
```

## 4. Local packs (offline)

Place `ternary_config.json` in your pack directory, then:

```python
m = ternary_runtime.resolve_local("/data/my-pack", base_dir="/data/base-ckpt")
ternary_runtime.bootstrap()
```
