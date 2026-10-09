# Architecture

```text
┌─────────────────────┐     ┌──────────────────────────┐
│ Base HF checkpoint  │     │ Ternary pack (*-latest)  │
│ attention, router,  │     │ layer-XX.safetensors     │
│ shared, MTP, tok    │     │ ternary_config.json      │
└──────────┬──────────┘     └────────────┬─────────────┘
           │                             │
           │    ternary_runtime.fetch    │
           └──────────────┬──────────────┘
                          ▼
                 TERNARY_MODEL_DIR
                 TERNARY_PACK_DIR
                 TERNARY_PROFILE
                          │
                          ▼
              bootstrap() → SGLang patches
                          │
                          ▼
                 MoE forward uses ternary
                 CUDA/Triton kernels
```

## Components

1. **Pack**: per-layer safetensors, format `ternary-g128-hadamard-gptq` (published on Hugging Face).
2. **Runtime** (`ternary_runtime`): load pack, rotate activations, decode GEMV / prefill GMM, patch SGLang experts.
3. **Hub helper**: `fetch()` reads `ternary_config.json`, downloads pack + `base_model`.

## Why not a single fused HF checkpoint?

Base weights are hundreds of GB and already published by upstream. Shipping the ~30 GB expert overlay plus a config that names the base model keeps installs smaller and license attribution clear, while `fetch()` still feels like one model id.
