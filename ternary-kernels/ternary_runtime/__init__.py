"""Ternary routed-expert runtime (DeepSeek-V4.1 MoE, Flash-Next MoE, Bonsai)."""

__version__ = "0.1.0"

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
