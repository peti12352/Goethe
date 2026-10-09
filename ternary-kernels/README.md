# ternary-kernels

Installable Goethe runtime (`pip install -e .` → import `ternary_runtime`).

| Profile | Kind | Specialized K | Pack layout | Hugging Face pack |
|---------|------|---------------|-------------|-------------------|
| `flash_next` | MoE | 2560, 640 | `gate_up_down` | [qwen3.8-flash-next-ternary-latest](https://huggingface.co/meshapplied/qwen3.8-flash-next-ternary-latest) |
| `deepseek_v41` | MoE | 5120, 2304 | `w123` | [DeepSeek-V4.1-Flash-NVFP4](https://huggingface.co/meshapplied/DeepSeek-V4.1-Flash-NVFP4) (ternary overlay) |

```python
import ternary_runtime
m = ternary_runtime.fetch("meshapplied/qwen3.8-flash-next-ternary-latest")
ternary_runtime.bootstrap()
```

See repository root [`docs/`](../docs/) for install and pack format.
