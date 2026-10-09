# ternary-kernels

Installable goethe runtime (`pip install -e .` -> import `ternary_runtime`).

| Profile | Kind | Specialized K | Pack layout | Hugging Face pack |
|---------|------|---------------|-------------|-------------------|
| `flash_next` | MoE | 2560, 640 | `gate_up_down` | [Qwen3.8-Flash-Next-Ternary-Latest](https://huggingface.co/meshapplied/Qwen3.8-Flash-Next-Ternary-Latest) |
| `deepseek_v41` | MoE | 5120, 2304 | `w123` | [DeepSeek-V4.1-Flash-Ternary-Latest](https://huggingface.co/meshapplied/DeepSeek-V4.1-Flash-Ternary-Latest) (ternary overlay, not NVFP4) |

```python
import ternary_runtime
m = ternary_runtime.fetch("meshapplied/Qwen3.8-Flash-Next-Ternary-Latest")
ternary_runtime.bootstrap()
```

See repository root [`docs/`](../docs/) for install and pack format.
