# Changelog

## 0.1.1

- Canonical Hub ids: Flash-Next and DeepSeek **Ternary-Latest**. `fetch()` rewrites the legacy NVFP4 id.
- Public profiles are Flash-Next and DeepSeek only. Bonsai is behind `TERNARY_EXPERIMENTAL=1`.
- Serve scripts: `examples/serve_flash_next.sh`, `examples/serve_deepseek.sh`.
- CPU pytest CI (hub/profile/config). CUDA tests skip without a GPU.
- Docs no longer claim unmeasured tok/s or that DeepSeek ternary is NVFP4.

## 0.1.0

- Public Goethe runtime: `ternary_runtime` under Apache-2.0.
- Plug-and-play Hub helper: `ternary_runtime.fetch()` + `ternary_config.json`.
- Supported packs: Qwen3.8-Flash-Next ternary, DeepSeek-V4.1-Flash ternary overlay.
- CUDA build cache under `~/.cache/ternary-flash` (or `TERNARY_BUILD_DIR`).
- Docs: install, quickstart, architecture, pack format.
