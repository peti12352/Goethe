# Install

## Requirements

- Linux x86_64, NVIDIA GPU with a recent CUDA toolkit
- Python ≥ 3.10
- PyTorch with CUDA matching your driver
- Optional: [SGLang](https://github.com/sgl-project/sglang) for the patched MoE serve path

## Framework

```bash
git clone https://github.com/meshapplied/Goethe.git
cd Goethe
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

Set `TORCH_CUDA_ARCH_LIST` for your GPU if the default is wrong for your machine.

## Models

```python
import ternary_runtime

m = ternary_runtime.fetch("meshapplied/qwen3.8-flash-next-ternary-latest")
# or: meshapplied/DeepSeek-V4.1-Flash-NVFP4
```

This downloads:

1. The ternary pack from the Goethe/meshapplied Hugging Face repo
2. The **base** model from `base_model` in `ternary_config.json` (original publisher)

| Pack | Hugging Face |
|------|----------------|
| Qwen3.8-Flash-Next ternary | [meshapplied/qwen3.8-flash-next-ternary-latest](https://huggingface.co/meshapplied/qwen3.8-flash-next-ternary-latest) |
| DeepSeek-V4.1-Flash ternary | [meshapplied/DeepSeek-V4.1-Flash-NVFP4](https://huggingface.co/meshapplied/DeepSeek-V4.1-Flash-NVFP4) |

Cache root: `$TERNARY_HF_CACHE` or `~/.cache/ternary-flash/hub`.

## SGLang

Install SGLang in the same environment (or a sibling venv), then:

```python
import ternary_runtime
ternary_runtime.fetch("meshapplied/qwen3.8-flash-next-ternary-latest")
ternary_runtime.bootstrap()
# launch_server with --model-path = resolved base_dir
```
