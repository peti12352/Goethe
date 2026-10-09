# Goethe

Apache-2.0 ternary MoE runtime for SGLang. Packs store Hadamard-rotated ternary (`{-1,0,+1}`) routed experts; attention, router, shared experts, and MTP stay in the upstream base checkpoint. `ternary_runtime.fetch()` downloads a pack and its base automatically.

## Supported models

| Pack (Hugging Face) | Base | Profile |
|---------------------|------|---------|
| [`meshapplied/Qwen3.8-Flash-Next-Ternary-Latest`](https://huggingface.co/meshapplied/Qwen3.8-Flash-Next-Ternary-Latest) | [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) | `flash_next` |
| [`meshapplied/DeepSeek-V4.1-Flash-Ternary-Latest`](https://huggingface.co/meshapplied/DeepSeek-V4.1-Flash-Ternary-Latest) | [`deepseek-ai/DeepSeek-V4.1-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | `deepseek_v41` |

The DeepSeek Hub repo is a **ternary expert overlay**, not an NVFP4 weight dump. The old id `meshapplied/DeepSeek-V4.1-Flash-NVFP4` still 307s on Hub; `fetch()` rewrites it.

DeepSeek still needs the **full base checkpoint** plus host RAM for Engram. Ternary experts are a bytes/quality trade, not a 3×GPU throughput SKU.

## Install

Not on PyPI. Clone and install:

```bash
git clone https://github.com/meshapplied/Goethe.git
cd Goethe
pip install -e ./ternary-kernels
python -c "import ternary_runtime; print(ternary_runtime.__version__)"
```

Requires Linux, CUDA-capable NVIDIA GPU (kernels default `TORCH_CUDA_ARCH_LIST=12.0a`), Python ≥ 3.10, and a matching PyTorch build. Optional: [SGLang](https://github.com/sgl-project/sglang) for the patched MoE serve path (pin in [`examples/SGLANG_PIN`](examples/SGLANG_PIN)).

JIT CUDA kernels cache under `~/.cache/ternary-flash/build` (or `$TERNARY_BUILD_DIR`).

## Quick start

```python
import ternary_runtime

m = ternary_runtime.fetch("meshapplied/Qwen3.8-Flash-Next-Ternary-Latest")
# or: meshapplied/DeepSeek-V4.1-Flash-Ternary-Latest
ternary_runtime.bootstrap()
print(m.profile, m.pack_dir, m.base_dir)
```

Serve (EP=TP so Hadamard blocks stay on one rank):

```bash
./examples/serve_flash_next.sh
# DeepSeek (large base + Engram host RAM): ./examples/serve_deepseek.sh
```

## Documentation

- [Install](docs/install.md)
- [Quickstart](docs/quickstart.md)
- [Architecture](docs/architecture.md)
- [Pack format](docs/pack.md)
- [Hub card notes](docs/HUB_CARDS.md)
- [Changelog](CHANGELOG.md)

## Layout

| Path | Role |
|------|------|
| [`ternary-kernels/ternary_runtime/`](ternary-kernels/ternary_runtime/) | Loader, Hub fetch, SGLang patches, CUDA kernels |
| [`docs/`](docs/) | Install, pack contract, architecture |
| [`examples/`](examples/) | Serve scripts (Flash-Next, DeepSeek) |

## License

Apache-2.0 for this repository. Base model licenses on Hugging Face still apply when you download and run with a base checkpoint.
