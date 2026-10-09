"""Load ternary expert packs and local fallbacks for one EP rank."""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch
from safetensors import safe_open

from ternary_runtime.profile import PackProfile, active_profile, profile_for_pack_dir

FORMAT = "ternary-g128-hadamard-gptq"
GROUP = "128"


@dataclass
class FallbackExpert:
    global_id: int
    w13: torch.Tensor
    w2: torch.Tensor


@dataclass
class LayerWeights:
    layer_id: int
    signs_hidden: torch.Tensor
    signs_inter: torch.Tensor
    w13_codes: torch.Tensor
    w13_scales: torch.Tensor
    w2_codes: torch.Tensor
    w2_scales: torch.Tensor
    ternary_mask: torch.Tensor
    zero_experts: List[int] = field(default_factory=list)
    fallbacks: List[FallbackExpert] = field(default_factory=list)

    # Back-compat aliases used by DeepSeek call sites / older code.
    @property
    def signs5120(self) -> torch.Tensor:
        return self.signs_hidden

    @property
    def signs2304(self) -> torch.Tensor:
        return self.signs_inter


def _layer_pack_path(pack_dir: Path, layer_id: int) -> Path:
    return pack_dir / f"layer-{layer_id:02d}.safetensors"


def _layer_manifest_path(pack_dir: Path, layer_id: int) -> Path:
    return pack_dir / f"manifest-layer-{layer_id:02d}.json"


def _validate_pack_meta(meta: dict) -> None:
    if meta.get("format") != FORMAT or meta.get("group") != GROUP:
        raise ValueError(
            f"refusing pack: format={meta.get('format')!r} group={meta.get('group')!r}"
        )


def _load_weight_map(ckpt_dir: Path) -> Dict[str, str]:
    index = ckpt_dir / "model.safetensors.index.json"
    if index.is_file():
        return json.loads(index.read_text())["weight_map"]
    single = ckpt_dir / "model.safetensors"
    if single.is_file():
        with safe_open(str(single), framework="pt", device="cpu") as f:
            return {k: single.name for k in f.keys()}
    raise FileNotFoundError(f"no safetensors index under {ckpt_dir}")


def _is_zero_codes(codes: torch.Tensor) -> bool:
    """True when packed codes unpack to an all-zero matrix (Wx is identically 0)."""
    # codes int32 words: trit 0 is packed as code value 1 → word bits pattern can be nonzero.
    # All-zero reconstructed weight ⇔ every 2-bit field is 01, or we unpack. Cheap check:
    # if every word equals the "all trit-0" pattern 0x55555555, scales irrelevant.
    if codes.numel() == 0:
        return True
    return bool(torch.equal(codes, torch.full_like(codes, 0x55555555)))


def _copy_expert_w123(
    pack: safe_open,
    stem: str,
    local_slot: int,
    inter: int,
    w13_codes: torch.Tensor,
    w13_scales: torch.Tensor,
    w2_codes: torch.Tensor,
    w2_scales: torch.Tensor,
) -> bool:
    """Copy DS w1/w3/w2. Returns False if the expert is the zero matrix."""
    w1 = pack.get_tensor(f"{stem}.w1.weight")
    w3 = pack.get_tensor(f"{stem}.w3.weight")
    w2 = pack.get_tensor(f"{stem}.w2.weight")
    if _is_zero_codes(w1) and _is_zero_codes(w3) and _is_zero_codes(w2):
        return False
    w13_codes[local_slot, :inter].copy_(w1)
    w13_codes[local_slot, inter:].copy_(w3)
    w13_scales[local_slot, :inter].copy_(pack.get_tensor(f"{stem}.w1.scale"))
    w13_scales[local_slot, inter:].copy_(pack.get_tensor(f"{stem}.w3.scale"))
    w2_codes[local_slot].copy_(w2)
    w2_scales[local_slot].copy_(pack.get_tensor(f"{stem}.w2.scale"))
    return True


def _copy_expert_gate_up_down(
    pack: safe_open,
    stem: str,
    local_slot: int,
    w13_codes: torch.Tensor,
    w13_scales: torch.Tensor,
    w2_codes: torch.Tensor,
    w2_scales: torch.Tensor,
) -> bool:
    """Copy Qwen gate_up / down. Returns False if the expert is the zero matrix."""
    gu_w = pack.get_tensor(f"{stem}.gate_up_proj.weight")
    dn_w = pack.get_tensor(f"{stem}.down_proj.weight")
    if _is_zero_codes(gu_w) and _is_zero_codes(dn_w):
        return False
    w13_codes[local_slot].copy_(gu_w)
    w13_scales[local_slot].copy_(pack.get_tensor(f"{stem}.gate_up_proj.scale"))
    w2_codes[local_slot].copy_(dn_w)
    w2_scales[local_slot].copy_(pack.get_tensor(f"{stem}.down_proj.scale"))
    return True


def _load_fp4_expert(
    ckpt_dir: Path,
    weight_map: Dict[str, str],
    layer_id: int,
    expert_id: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    from ternary_runtime.fp4 import dequant_fp4

    stem = f"layers.{layer_id}.ffn.experts.{expert_id}"
    w1 = _ckpt_fp4(ckpt_dir, weight_map, f"{stem}.w1", device, dequant_fp4)
    w3 = _ckpt_fp4(ckpt_dir, weight_map, f"{stem}.w3", device, dequant_fp4)
    w2 = _ckpt_fp4(ckpt_dir, weight_map, f"{stem}.w2", device, dequant_fp4)
    return torch.cat([w1, w3], dim=0), w2


def _ckpt_fp4(ckpt_dir, weight_map, key, device, dequant_fp4) -> torch.Tensor:
    wkey = key if key.endswith(".weight") else f"{key}.weight"
    skey = wkey.replace(".weight", ".scale")
    shard = ckpt_dir / weight_map[wkey]
    with safe_open(str(shard), framework="pt", device="cpu") as f:
        packed = f.get_tensor(wkey).to(device)
        scale = f.get_tensor(skey).to(device)
    return dequant_fp4(packed, scale).to(torch.bfloat16)


def _load_bf16_expert(
    ckpt_dir: Path,
    weight_map: Dict[str, str],
    layer_id: int,
    expert_id: int,
    device: torch.device,
    profile: PackProfile,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Slice one expert from fused Qwen gate_up / down tensors."""
    gu_key = f"{profile.ckpt_prefix}layers.{layer_id}.mlp.experts.gate_up_proj"
    dn_key = f"{profile.ckpt_prefix}layers.{layer_id}.mlp.experts.down_proj"
    gu_shard = ckpt_dir / weight_map[gu_key]
    dn_shard = ckpt_dir / weight_map[dn_key]
    with safe_open(str(gu_shard), framework="pt", device="cpu") as f:
        gu = f.get_slice(gu_key)[expert_id].to(device=device, dtype=torch.bfloat16)
    with safe_open(str(dn_shard), framework="pt", device="cpu") as f:
        dn = f.get_slice(dn_key)[expert_id].to(device=device, dtype=torch.bfloat16)
    return gu.contiguous(), dn.contiguous()


def load_layer(
    pack_dir: Path,
    ckpt_dir: Path,
    weight_map: Dict[str, str],
    layer_id: int,
    ep_rank: int,
    ep_size: int,
    device: torch.device,
    profile: Optional[PackProfile] = None,
) -> LayerWeights:
    """Load one layer's local experts onto ``device``."""
    profile = profile or profile_for_pack_dir(str(pack_dir))
    if profile.num_experts % ep_size != 0:
        raise ValueError(
            f"num_experts {profile.num_experts} not divisible by ep_size {ep_size}"
        )
    n_local = profile.num_experts // ep_size
    expert_lo = ep_rank * n_local
    expert_hi = expert_lo + n_local
    inter = profile.intermediate
    hidden = profile.hidden
    w13_rows = inter * 2

    manifest = json.loads(_layer_manifest_path(pack_dir, layer_id).read_text())
    ternary_set: Set[int] = set(manifest["experts_ternary"])
    fallback_set: Set[int] = set(manifest.get(profile.manifest_fallback_key, []))

    w13_codes = torch.zeros(
        (n_local, w13_rows, hidden // 16), dtype=torch.int32, device=device
    )
    w13_scales = torch.zeros(
        (n_local, w13_rows, hidden // 128), dtype=torch.float16, device=device
    )
    w2_codes = torch.zeros(
        (n_local, hidden, inter // 16), dtype=torch.int32, device=device
    )
    w2_scales = torch.zeros(
        (n_local, hidden, inter // 128), dtype=torch.float16, device=device
    )
    mask = torch.zeros(n_local, dtype=torch.bool)
    zero_experts: List[int] = []

    pack_path = _layer_pack_path(pack_dir, layer_id)
    with safe_open(str(pack_path), framework="pt", device="cpu") as pack:
        _validate_pack_meta(pack.metadata())
        signs_h = pack.get_tensor(profile.signs_hidden_key).to(device, torch.float32)
        signs_i = pack.get_tensor(profile.signs_inter_key).to(device, torch.float32)
        for eid in range(expert_lo, expert_hi):
            slot = eid - expert_lo
            if eid not in ternary_set:
                if eid not in fallback_set:
                    raise ValueError(
                        f"layer {layer_id} expert {eid} missing from manifest lists"
                    )
                continue
            stem = profile.pack_expert_stem(layer_id, eid)
            if profile.layout == "w123":
                ok = _copy_expert_w123(
                    pack, stem, slot, inter, w13_codes, w13_scales, w2_codes, w2_scales
                )
            else:
                ok = _copy_expert_gate_up_down(
                    pack, stem, slot, w13_codes, w13_scales, w2_codes, w2_scales
                )
            if ok:
                mask[slot] = True
            else:
                # Ternary W is the zero matrix → MoE contribution is 0; skip compute.
                zero_experts.append(eid)
    ternary_mask = mask.to(device)

    fb_gpu: List[FallbackExpert] = []
    for eid in sorted(fallback_set):
        if expert_lo <= eid < expert_hi:
            if profile.fallback == "fp4":
                w13, w2 = _load_fp4_expert(ckpt_dir, weight_map, layer_id, eid, device)
            else:
                w13, w2 = _load_bf16_expert(
                    ckpt_dir, weight_map, layer_id, eid, device, profile
                )
            fb_gpu.append(FallbackExpert(eid, w13, w2))
    torch.cuda.synchronize(device)

    return LayerWeights(
        layer_id=layer_id,
        signs_hidden=signs_h,
        signs_inter=signs_i,
        w13_codes=w13_codes,
        w13_scales=w13_scales,
        w2_codes=w2_codes,
        w2_scales=w2_scales,
        ternary_mask=ternary_mask,
        zero_experts=zero_experts,
        fallbacks=fb_gpu,
    )


def load_all_layers(
    pack_dir: str | Path,
    ckpt_dir: str | Path,
    ep_rank: int,
    ep_size: int,
    device: torch.device,
    *,
    workers: Optional[int] = None,
    progress: bool = True,
    profile: Optional[PackProfile] = None,
) -> Tuple[List[LayerWeights], float, int]:
    """Load every ternary layer in parallel; return (layers, seconds, bytes)."""
    pack_dir = Path(pack_dir)
    ckpt_dir = Path(ckpt_dir)
    profile = profile or profile_for_pack_dir(str(pack_dir))
    workers = workers or int(os.environ.get("TERNARY_LOAD_WORKERS", "16"))
    weight_map = _load_weight_map(ckpt_dir)
    n_layers = profile.num_layers

    t0 = time.perf_counter()
    last_report = t0
    layers: List[Optional[LayerWeights]] = [None] * n_layers
    done = 0
    bytes_est = 0
    zero_total = 0

    def _one(layer_id: int) -> LayerWeights:
        return load_layer(
            pack_dir, ckpt_dir, weight_map, layer_id, ep_rank, ep_size, device, profile
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_one, lid): lid for lid in range(n_layers)}
        for fut in as_completed(futs):
            lid = futs[fut]
            lw = fut.result()
            layers[lid] = lw
            done += 1
            zero_total += len(lw.zero_experts)
            bytes_est += (
                lw.w13_codes.numel() * 4
                + lw.w13_scales.numel() * 2
                + lw.w2_codes.numel() * 4
                + lw.w2_scales.numel() * 2
            )
            for fb in lw.fallbacks:
                bytes_est += fb.w13.numel() * 2 + fb.w2.numel() * 2
            now = time.perf_counter()
            if progress and (done % 8 == 0 or now - last_report >= 3.0 or done == n_layers):
                print(
                    f"ternary[{profile.name}]: rank {ep_rank} loaded {done}/{n_layers} layers, "
                    f"{bytes_est / 2**30:.2f} GiB, {now - t0:.1f} s",
                    flush=True,
                )
                last_report = now

    if zero_total and progress:
        print(
            f"ternary[{profile.name}]: rank {ep_rank} skipped {zero_total} zero-weight "
            f"experts (Wx identically 0)",
            flush=True,
        )
    elapsed = time.perf_counter() - t0
    return [layers[i] for i in range(n_layers)], elapsed, bytes_est
