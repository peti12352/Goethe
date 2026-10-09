# Expert pack format

Format id: `ternary-g128-hadamard-gptq`, group `128`.

Routed experts only. MTP / attention / router / shared experts stay in the **base** checkpoint.

## Stored weight

Ternary `{-1, 0, +1}` times one FP16 scale per 128 input weights, in a block Walsh-Hadamard basis. `R = H_b S / sqrt(b)`. Block length is `b = min(1024, largest power of two dividing K)`. `S` is a fixed `+/-1` vector from `signs_for(K, seed)` with seed 0. The stored matrix is `W R^T`; the runtime applies the same transform to activations or refuses the file.

Pack encoding: `code = value + 1`, 16 codes per int32, earlier K element in the low bits.

## DeepSeek-V4.1-Flash (`profile=deepseek_v41`)

| | |
|--|--|
| Layout | `w123` (`w1`, `w2`, `w3`) |
| Layers | 0..39 |
| K | 5120 (w1/w3), 2304 (w2) |
| Signs | `hadamard.signs.5120`, `hadamard.signs.2304` |
| Fallback | FP4 from base (`experts_fp4` in manifest) |

Files: `layer-LL.safetensors`, `manifest-layer-LL.json`, plus `eval.json` / NLL sidecars.

Keys: `layers.L.ffn.experts.E.{w1,w2,w3}.{weight,scale}`.

## Qwen3.8-Flash-Next (`profile=flash_next`)

| | |
|--|--|
| Layout | `gate_up_down` (fused `gate_up_proj` + `down_proj`) |
| Layers | 0..47 |
| K | 2560 (gate_up), 640 (down) |
| Signs | `hadamard.signs.2560`, `hadamard.signs.640` |
| Fallback | bf16 from base (`experts_bf16` in manifest) |

Keys: `layers.L.mlp.experts.E.{gate_up_proj,down_proj}.{weight,scale}`.

## Runtime contract (both)

For a ternary expert with route weight `c`:

1. `x` is bf16 FFN input after the official norm (no FP8 act quant on the ternary path).
2. Rotated GEMVs in fp32 with `R`, results rounded to bf16.
3. Official expert nonlinearity (SwiGLU; DeepSeek applies `swiglu_limit` clamps).
4. Down-projection with `R` on the intermediate as required by layout.
5. MoE sum in fp32, then cast to bf16.

Expert intermediates must stay on one rank under TP (use EP=TP) so Hadamard blocks are not split incorrectly.

## Hub packaging

Public packs include `ternary_config.json`:

```json
{
  "format": "ternary-g128-hadamard-gptq",
  "profile": "flash_next",
  "base_model": "Qwen/Qwen3.8-Flash-Next",
  "base_revision": "...",
  "pack_layout": "gate_up_down",
  "n_layers": 48
}
```

`ternary_runtime.fetch(repo_id)` loads this config, downloads the pack, then downloads `base_model`.

## Loader checks

Refuse a layer file unless metadata `format` / `group` match. Every routed expert appears in exactly one of ternary vs fallback lists in the layer manifest.
