"""Model-aware ternary runtime profiles.

Two production targets today:
  - deepseek_v41: MoE w1/w2/w3 packs (SGLang EP)
  - bonsai:       dense TernaryLinear (ckpt embeds codes+scales; vLLM/SSD path)

Flash-Next (qwen4_exp MoE gate_up/down) is registered for detection; MoE path
wired, decode Ks specialized — full serve bring-up deferred.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import FrozenSet, Optional, Tuple

from safetensors import safe_open


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str  # "moe" | "dense"
    layout: str  # "w123" | "gate_up_down" | "dense_linear"
    num_layers: int
    num_experts: int  # 0 for dense
    hidden: int
    intermediate: int
    top_k: int
    block_hidden: int
    block_inter: int
    pack_stem: str
    ckpt_prefix: str
    fallback: str  # "fp4" | "bf16" | "none"
    manifest_fallback_key: str
    routed_scaling_factor: float
    swiglu_limit: Optional[float]
    fuse_routed_scaling_in_topk: bool
    specialized_k: Tuple[int, ...]
    features: FrozenSet[str]

    @property
    def signs_hidden_key(self) -> str:
        return f"hadamard.signs.{self.hidden}"

    @property
    def signs_inter_key(self) -> str:
        return f"hadamard.signs.{self.intermediate}"

    def pack_expert_stem(self, layer: int, expert: int) -> str:
        return self.pack_stem.format(layer=layer, expert=expert)

    def is_ternary_layer(self, layer_id: int) -> bool:
        return 0 <= layer_id < self.num_layers

    def has(self, feature: str) -> bool:
        return feature in self.features


# Feature flags (gate patch / experts / loader branches).
F_MOE_PATCH = "moe_patch"
F_ENGRAM = "engram_readahead"
F_FP4_FALLBACK = "fp4_fallback"
F_BF16_FALLBACK = "bf16_fallback"
F_MTP_SKIP = "mtp_skip_ternary"
F_SWIGLU_CLAMP = "swiglu_clamp"
F_FUSED_DECODE = "fused_moe_decode"
F_HC_MIX = "hc_mix_fast"

_MOE_DS = frozenset(
    {F_MOE_PATCH, F_ENGRAM, F_FP4_FALLBACK, F_MTP_SKIP, F_SWIGLU_CLAMP}
)
_MOE_QWEN = frozenset(
    {F_MOE_PATCH, F_BF16_FALLBACK, F_MTP_SKIP, F_FUSED_DECODE, F_HC_MIX}
)
_DENSE = frozenset()


DEEPSEEK_V41 = ModelSpec(
    name="deepseek_v41",
    kind="moe",
    layout="w123",
    num_layers=40,
    num_experts=384,
    hidden=5120,
    intermediate=2304,
    top_k=6,
    block_hidden=1024,
    block_inter=256,
    pack_stem="layers.{layer}.ffn.experts.{expert}",
    ckpt_prefix="",
    fallback="fp4",
    manifest_fallback_key="experts_fp4",
    routed_scaling_factor=1.5,
    swiglu_limit=10.0,
    fuse_routed_scaling_in_topk=True,
    specialized_k=(5120, 2304),
    features=_MOE_DS,
)

BONSAI = ModelSpec(
    name="bonsai",
    kind="dense",
    layout="dense_linear",
    num_layers=64,
    num_experts=0,
    hidden=5120,
    intermediate=17408,
    top_k=0,
    block_hidden=1024,
    block_inter=1024,
    pack_stem="",
    ckpt_prefix="model.language_model.",
    fallback="none",
    manifest_fallback_key="",
    routed_scaling_factor=1.0,
    swiglu_limit=None,
    fuse_routed_scaling_in_topk=False,
    # down_proj K=17408, gate/up/attn K=5120 (same pack format as MoE GEMV).
    specialized_k=(5120, 17408),
    features=_DENSE,
)

FLASH_NEXT = ModelSpec(
    name="flash_next",
    kind="moe",
    layout="gate_up_down",
    num_layers=48,
    num_experts=512,
    hidden=2560,
    intermediate=640,
    top_k=10,
    block_hidden=512,
    block_inter=128,
    pack_stem="layers.{layer}.mlp.experts.{expert}",
    ckpt_prefix="model.language_model.",
    fallback="bf16",
    manifest_fallback_key="experts_bf16",
    routed_scaling_factor=1.0,
    swiglu_limit=None,
    fuse_routed_scaling_in_topk=False,
    specialized_k=(2560, 640),
    features=_MOE_QWEN,
)

# Back-compat alias used by older env / docs.
QWEN4_EXP = FLASH_NEXT

_BY_NAME = {
    "deepseek_v41": DEEPSEEK_V41,
    "deepseek": DEEPSEEK_V41,
    "ds": DEEPSEEK_V41,
    "bonsai": BONSAI,
    "bonsai-27b": BONSAI,
    "ternary-bonsai": BONSAI,
    "flash_next": FLASH_NEXT,
    "flash-next": FLASH_NEXT,
    "qwen4_exp": FLASH_NEXT,
    "qwen38": FLASH_NEXT,
    "qwen": FLASH_NEXT,
}

# Union of every registered specialization (CUDA launch table).
ALL_SPECIALIZED_K: Tuple[int, ...] = tuple(
    sorted({k for p in (DEEPSEEK_V41, BONSAI, FLASH_NEXT) for k in p.specialized_k})
)

# Legacy name kept for imports.
PackProfile = ModelSpec


def _count_layers(pack_dir: Path) -> int:
    n = 0
    while (pack_dir / f"layer-{n:02d}.safetensors").is_file():
        n += 1
    return n


def _text_config(cfg: dict) -> dict:
    for key in ("text_config", "language_config", "llm_config"):
        nested = cfg.get(key)
        if isinstance(nested, dict) and nested:
            return nested
    return cfg


def detect_from_hf(model_dir: str | Path) -> ModelSpec:
    """Map HF config.json → ModelSpec."""
    path = Path(model_dir) / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"no config.json under {model_dir}")
    cfg = json.loads(path.read_text())
    text = _text_config(cfg)
    mt = str(cfg.get("model_type") or text.get("model_type") or "").lower()
    arch = " ".join(cfg.get("architectures") or []).lower()

    if mt.startswith("deepseek") or "deepseek" in arch:
        return DEEPSEEK_V41
    if mt in ("qwen3_5", "qwen3_5_text") or "qwen3_5" in arch or "bonsai" in str(model_dir).lower():
        layers = int(text.get("num_hidden_layers") or BONSAI.num_layers)
        hidden = int(text.get("hidden_size") or BONSAI.hidden)
        inter = int(text.get("intermediate_size") or BONSAI.intermediate)
        if layers != BONSAI.num_layers or hidden != BONSAI.hidden or inter != BONSAI.intermediate:
            return replace(
                BONSAI, num_layers=layers, hidden=hidden, intermediate=inter
            )
        return BONSAI
    if mt.startswith("qwen4_exp") or "qwen4exp" in arch.replace("-", ""):
        layers = int(text.get("num_hidden_layers") or FLASH_NEXT.num_layers)
        experts = int(
            text.get("num_experts")
            or text.get("n_routed_experts")
            or FLASH_NEXT.num_experts
        )
        if layers != FLASH_NEXT.num_layers or experts != FLASH_NEXT.num_experts:
            return replace(FLASH_NEXT, num_layers=layers, num_experts=experts)
        return FLASH_NEXT
    raise ValueError(f"unsupported HF model_type={mt!r} under {model_dir}")


def detect_from_pack(pack_dir: str | Path) -> ModelSpec:
    """Infer MoE profile from pack keys / Hadamard signs."""
    pack_dir = Path(pack_dir)
    path = pack_dir / "layer-00.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"no layer-00.safetensors under {pack_dir}")
    with safe_open(str(path), framework="pt") as f:
        keys = set(f.keys())
    if "hadamard.signs.5120" in keys and any(".ffn.experts." in k for k in keys):
        return DEEPSEEK_V41
    if "hadamard.signs.2560" in keys and any(".mlp.experts." in k for k in keys):
        n = _count_layers(pack_dir)
        if n and n != FLASH_NEXT.num_layers:
            return replace(FLASH_NEXT, num_layers=n)
        return FLASH_NEXT
    raise ValueError(f"cannot detect ternary pack profile under {pack_dir}")


def detect_profile(pack_dir: str | Path) -> ModelSpec:
    """Pack-dir detection with TERNARY_PROFILE override (MoE only)."""
    forced = _forced_name()
    if forced is not None:
        return forced
    return detect_from_pack(pack_dir)


def _forced_name() -> Optional[ModelSpec]:
    raw = os.environ.get("TERNARY_PROFILE", "").strip().lower()
    if not raw:
        return None
    if raw not in _BY_NAME:
        raise ValueError(
            f"unknown TERNARY_PROFILE={raw!r}; expect one of {sorted(set(_BY_NAME))}"
        )
    return _BY_NAME[raw]


def resolve_model(
    *,
    pack_dir: Optional[str] = None,
    model_dir: Optional[str] = None,
) -> ModelSpec:
    """Resolve active model: env override → pack → HF config."""
    forced = _forced_name()
    if forced is not None:
        return forced
    pack = (pack_dir or os.environ.get("TERNARY_PACK_DIR", "")).strip()
    if pack:
        return detect_from_pack(pack)
    ckpt = (
        model_dir
        or os.environ.get("TERNARY_MODEL_DIR", "")
        or os.environ.get("MODEL_PATH", "")
    ).strip()
    if ckpt:
        return detect_from_hf(ckpt)
    raise RuntimeError(
        "cannot resolve ternary model: set TERNARY_PROFILE, TERNARY_PACK_DIR, "
        "or TERNARY_MODEL_DIR / MODEL_PATH"
    )


@lru_cache(maxsize=4)
def profile_for_pack_dir(pack_dir: str) -> ModelSpec:
    return detect_profile(pack_dir)


def active_profile() -> ModelSpec:
    return resolve_model()
