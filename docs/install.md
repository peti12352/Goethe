# Install

## Requirements

- Linux x86_64, NVIDIA GPU with a recent CUDA toolkit
- Python ≥ 3.10
- PyTorch with CUDA matching your driver
- Optional: [SGLang](https://github.com/sgl-project/sglang) for the patched MoE serve path (see [`examples/SGLANG_PIN`](../examples/SGLANG_PIN))

## Framework

```bash
git clone https://github.com/meshapplied/goethe.git
cd goethe
pip install -e ./ternary-kernels
```

Optional extras:

```bash
pip install -e "./ternary-kernels[dev]"   # pytest
```

First kernel use JIT-compiles CUDA extensions into:

- `$TERNARY_BUILD_DIR` if set, else
- `$XDG_CACHE_HOME/ternary-flash/build`, else
- `~/.cache/ternary-flash/build`

Set `TORCH_CUDA_ARCH_LIST` for your GPU if the default (`12.0a`) is wrong for your machine.

## Models

```python
import ternary_runtime

m = ternary_runtime.fetch("meshapplied/Qwen3.8-Flash-Next-Ternary-Latest")
# or: meshapplied/DeepSeek-V4.1-Flash-Ternary-Latest
```

This downloads:

1. The ternary pack from the meshapplied Hugging Face repo
2. The **base** model from `base_model` in `ternary_config.json` (original publisher)

| Pack | Hugging Face |
|------|----------------|
| Qwen3.8-Flash-Next ternary | [meshapplied/Qwen3.8-Flash-Next-Ternary-Latest](https://huggingface.co/meshapplied/Qwen3.8-Flash-Next-Ternary-Latest) |
| DeepSeek-V4.1-Flash ternary overlay | [meshapplied/DeepSeek-V4.1-Flash-Ternary-Latest](https://huggingface.co/meshapplied/DeepSeek-V4.1-Flash-Ternary-Latest) |

`meshapplied/DeepSeek-V4.1-Flash-NVFP4` is a **legacy alias** (Hub 307 + `fetch()` rewrite). It is not NVFP4 weights.

Cache root: `$TERNARY_HF_CACHE` or `~/.cache/ternary-flash/hub`.

## SGLang

Install SGLang in the same environment (or a sibling venv), then:

```python
import ternary_runtime
ternary_runtime.fetch("meshapplied/Qwen3.8-Flash-Next-Ternary-Latest")
ternary_runtime.bootstrap()
# launch_server with --model-path = resolved base_dir
```

Or run [`examples/serve_flash_next.sh`](../examples/serve_flash_next.sh).
