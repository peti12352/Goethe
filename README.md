# Goethe

Apache-2.0 ternary MoE runtime for SGLang. Packs store Hadamard-rotated ternary (`{-1,0,+1}`) routed experts; attention, router, shared experts, and MTP stay in the upstream base checkpoint. `ternary_runtime.fetch()` downloads a pack and its base automatically.

## Supported models

| Pack (Hugging Face) | Base | Profile |
|---------------------|------|---------|
| [`meshapplied/qwen3.8-flash-next-ternary-latest`](https://huggingface.co/meshapplied/qwen3.8-flash-next-ternary-latest) | [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) | `flash_next` |
| [`meshapplied/DeepSeek-V4.1-Flash-NVFP4`](https://huggingface.co/meshapplied/DeepSeek-V4.1-Flash-NVFP4) | [`deepseek-ai/DeepSeek-V4.1-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | `deepseek_v41` |

The DeepSeek Hub repo is a **ternary expert overlay** (not an NVFP4 weight dump).

## Install

Not on PyPI. Clone and install:

```bash
git clone https://github.com/meshapplied/Goethe.git
cd Goethe
pip install -e ./ternary-kernels
python -c "import ternary_runtime; print(ternary_runtime.__version__)"
```

Requires Linux, CUDA-capable NVIDIA GPU, Python ≥ 3.10, and a matching PyTorch build. Optional: [SGLang](https://github.com/sgl-project/sglang) for the patched MoE serve path.

JIT CUDA kernels cache under `~/.cache/ternary-flash/build` (or `$TERNARY_BUILD_DIR`).

## Quick start

```python
import ternary_runtime

m = ternary_runtime.fetch("meshapplied/qwen3.8-flash-next-ternary-latest")
# or: meshapplied/DeepSeek-V4.1-Flash-NVFP4
ternary_runtime.bootstrap()
print(m.profile, m.pack_dir, m.base_dir)
```

Serve with SGLang (`--model-path` = `$TERNARY_MODEL_DIR`, set EP=TP as needed for the pack):

```bash
python -m sglang.launch_server \
  --model-path "$TERNARY_MODEL_DIR" \
  --host 127.0.0.1 --port 8003 \
  --tp-size 1 \
  --trust-remote-code \
  --served-model-name qwen38-flash-next
```

Example TOML templates: [`examples/`](examples/).

## Documentation

- [Install](docs/install.md)
- [Quickstart](docs/quickstart.md)
- [Architecture](docs/architecture.md)
- [Pack format](docs/pack.md)
- [Changelog](CHANGELOG.md)

## Layout

| Path | Role |
|------|------|
| [`ternary-kernels/ternary_runtime/`](ternary-kernels/ternary_runtime/) | Loader, Hub fetch, SGLang patches, CUDA kernels |
| [`docs/`](docs/) | Install, pack contract, architecture |
| [`examples/`](examples/) | Placeholder serve configs |

## License

Apache-2.0 for this repository. Base model licenses on Hugging Face still apply when you download and run with a base checkpoint.
