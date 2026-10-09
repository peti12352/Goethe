"""Ternary routed-expert runtime (Qwen3.8-Flash-Next and DeepSeek-V4.1-Flash MoE)."""

__version__ = "0.1.1"

from ternary_runtime.hub import ResolvedModel, fetch, resolve_local

__all__ = [
    "__version__",
    "bootstrap",
    "fetch",
    "resolve_local",
    "ResolvedModel",
]


def bootstrap() -> None:
    """Apply SGLang monkeypatches when sglang is importable."""
    from ternary_runtime.patch import apply_patches

    apply_patches()
