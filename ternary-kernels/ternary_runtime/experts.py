"""Ternary routed experts (replaces FusedMoE for packed ternary layers)."""

from __future__ import annotations

import json
import os
import time
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.topk import (
    TopKOutput,
    TopKOutputChecker,
)
from sglang.srt.runtime_context import get_parallel

from ternary_runtime.loader import LayerWeights, load_all_layers
from ternary_runtime.profile import PackProfile, active_profile

_kernels = None
_kernels_failed = False

FP8_MAX = 448.0
FP8_BLOCK = 32
_FB_SPARSE_MAX_T = int(os.environ.get("TERNARY_FB_SPARSE_MAX_T", "16"))


def _get_kernels():
    """Lazy import of CUDA kernels (optional until built)."""
    global _kernels, _kernels_failed
    if _kernels_failed:
        raise RuntimeError("ternary kernels unavailable")
    if _kernels is None:
        try:
            from ternary_runtime import kernels as k

            _kernels = k
        except ImportError as exc:
            _kernels_failed = True
            raise RuntimeError("ternary_runtime.kernels not built") from exc
    return _kernels


def rotate(x: torch.Tensor, signs: torch.Tensor, block: int) -> torch.Tensor:
    """Apply block Walsh-Hadamard rotation R to rows of x."""
    return _get_kernels().rotate(x, signs, block)


def ternary_grouped(
    xr: torch.Tensor,
    tok: torch.Tensor,
    lid: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Grouped ternary GEMV over local expert slots."""
    return _get_kernels().ternary_grouped(
        xr, tok, lid, codes, scales, out_dtype=out_dtype
    )


def act_quant_emulate(x: torch.Tensor) -> torch.Tensor:
    """FP8 activation per 32 values with UE8M0 power-of-two scale (DeepSeek fallbacks)."""
    if x.dtype != torch.bfloat16:
        x = x.to(torch.bfloat16)
    flat = x.reshape(-1, FP8_BLOCK).float()
    amax = flat.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    scale = torch.pow(2.0, torch.ceil(torch.log2(amax / FP8_MAX)))
    q = (flat / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return (q.float() * scale).reshape_as(x).to(torch.bfloat16)


class TernaryExperts(nn.Module):
    """Routed experts from ternary packs; FP4/bf16 fallbacks from the base checkpoint."""

    supports_deferred_finalize = False
    quant_method = None
    num_fused_shared_experts = 0

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        layer_id: int,
        top_k: Optional[int] = None,
        num_fused_shared_experts: int = 0,
        routed_scaling_factor: Optional[float] = None,
        swiglu_limit: Optional[float] = None,
        **kwargs,
    ):
        super().__init__()
        del kwargs
        if num_fused_shared_experts:
            raise ValueError("TernaryExperts does not fuse shared experts")
        self.profile: PackProfile = active_profile()
        self.should_fuse_routed_scaling_factor_in_topk = (
            self.profile.fuse_routed_scaling_in_topk
        )
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.top_k = top_k or self.profile.top_k
        self.routed_scaling_factor = (
            routed_scaling_factor
            if routed_scaling_factor is not None
            else self.profile.routed_scaling_factor
        )
        if swiglu_limit is not None:
            self.swiglu_limit = float(swiglu_limit)
        elif self.profile.has("swiglu_clamp") and self.profile.swiglu_limit is not None:
            self.swiglu_limit = float(self.profile.swiglu_limit)
        else:
            self.swiglu_limit = None

        if self.profile.kind != "moe":
            raise ValueError(
                f"TernaryExperts is MoE-only; got kind={self.profile.kind} "
                f"model={self.profile.name}"
            )
        if num_experts != self.profile.num_experts:
            raise ValueError(
                f"num_experts={num_experts} != profile {self.profile.name} "
                f"num_experts={self.profile.num_experts}"
            )
        if hidden_size != self.profile.hidden:
            raise ValueError(
                f"hidden_size={hidden_size} != profile {self.profile.name} "
                f"hidden={self.profile.hidden}"
            )
        if intermediate_size != self.profile.intermediate:
            raise ValueError(
                f"intermediate_size={intermediate_size} != profile "
                f"{self.profile.name} intermediate={self.profile.intermediate}"
            )

        parallel = get_parallel()
        self.moe_ep_size = parallel.moe_ep_size
        self.moe_ep_rank = parallel.moe_ep_rank
        self.moe_tp_size = parallel.moe_tp_size
        self.moe_tp_rank = parallel.moe_tp_rank
        self._num_global_routed = num_experts
        self._num_local_routed = num_experts // self.moe_ep_size
        self.num_local_experts = self._num_local_routed
        self.num_experts = num_experts
        self.reduce_results = False
        self._fallback_kind = self.profile.fallback

        self.moe_runner_config = MoeRunnerConfig(
            num_experts=num_experts,
            num_local_experts=self.num_local_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size // max(1, self.moe_tp_size),
            layer_id=layer_id,
            top_k=self.top_k,
            num_fused_shared_experts=0,
            params_dtype=torch.bfloat16,
            inplace=True,
            routed_scaling_factor=self.routed_scaling_factor,
            swiglu_limit=self.swiglu_limit if self.swiglu_limit is not None else 0.0,
            layer=self,
        )

        self.register_buffer(
            "local_map",
            torch.full((num_experts,), -1, dtype=torch.int32),
            persistent=False,
        )

        self._weights: Optional[LayerWeights] = None
        self._fallback_entries: List[Tuple[int, int]] = []
        self._fused = False

    def load(
        self,
        pack_dir: str,
        ckpt_dir: str,
        device: Optional[torch.device] = None,
    ) -> None:
        """Load this layer's pack slice onto ``device``."""
        from pathlib import Path

        from ternary_runtime.loader import _load_weight_map, load_layer

        device = device or torch.device("cuda", torch.cuda.current_device())
        ckpt = Path(ckpt_dir)
        weight_map = _load_weight_map(ckpt)
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        lw = load_layer(
            Path(pack_dir),
            ckpt,
            weight_map,
            self.layer_id,
            self.moe_ep_rank,
            self.moe_ep_size,
            device,
            self.profile,
        )
        t1.record()
        t1.synchronize()
        self._apply_layer_weights(lw)
        nbytes = (
            lw.w13_codes.numel() * 4
            + lw.w13_scales.numel() * 2
            + lw.w2_codes.numel() * 4
            + lw.w2_scales.numel() * 2
        )
        for fb in lw.fallbacks:
            nbytes += fb.w13.numel() * 2 + fb.w2.numel() * 2
        print(
            f"ternary[{self.profile.name}]: layer {self.layer_id} rank {self.moe_ep_rank} "
            f"{nbytes / 2**30:.3f} GiB {t0.elapsed_time(t1) / 1000:.1f} s",
            flush=True,
        )

    @classmethod
    def load_all(
        cls,
        modules: List["TernaryExperts"],
        pack_dir: str,
        ckpt_dir: str,
        device: Optional[torch.device] = None,
    ) -> None:
        """One parallel load pass shared across all TernaryExperts modules."""
        if not modules:
            return
        device = device or torch.device("cuda", torch.cuda.current_device())
        ep_rank = modules[0].moe_ep_rank
        ep_size = modules[0].moe_ep_size
        profile = modules[0].profile
        layers, elapsed, nbytes = load_all_layers(
            pack_dir,
            ckpt_dir,
            ep_rank,
            ep_size,
            device,
            profile=profile,
        )
        by_id = {m.layer_id: m for m in modules}
        for lw in layers:
            by_id[lw.layer_id]._apply_layer_weights(lw)
        print(
            f"ternary[{profile.name}]: rank {ep_rank} loaded {profile.num_layers} layers, "
            f"{nbytes / 2**30:.1f} GiB, {elapsed:.1f} s",
            flush=True,
        )
        if (
            profile.has("fused_moe_decode")
            and os.environ.get("TERNARY_PREFILL", "gmm") != "dequant"
            and os.environ.get("TERNARY_PREFILL_WARMUP", "1") != "0"
        ):
            from ternary_runtime.prefill import warmup_ternary_gmm

            t0 = time.perf_counter()
            # gate_up: act K=2560 → N=1280; down: act K=640 → N=2560.
            warmup_ternary_gmm(device, k=2560, n_list=(1280,))
            warmup_ternary_gmm(device, k=640, n_list=(2560,))
            print(
                f"ternary[{profile.name}]: prefill GMM warmup {time.perf_counter() - t0:.2f}s",
                flush=True,
            )

    def _apply_layer_weights(self, lw: LayerWeights) -> None:
        self._weights = lw
        expert_lo = self.moe_ep_rank * self._num_local_routed
        local_map = torch.full((self.num_experts,), -1, dtype=torch.int32)
        for slot in range(self._num_local_routed):
            if lw.ternary_mask[slot]:
                local_map[expert_lo + slot] = slot
        self.local_map = local_map.to(device=lw.w13_codes.device)
        emap = local_map.clone()
        for i, fb in enumerate(lw.fallbacks):
            emap[fb.global_id] = -(i + 2)
        self.register_buffer("_emap", emap.to(device=lw.w13_codes.device), persistent=False)
        self._fused = (
            self.profile.has("fused_moe_decode")
            and self._fallback_kind == "bf16"
            and os.environ.get("TERNARY_FUSED_DECODE", "1") != "0"
        )
        if lw.fallbacks:
            w13 = torch.stack([fb.w13 for fb in lw.fallbacks], dim=0)
            w2 = torch.stack([fb.w2 for fb in lw.fallbacks], dim=0)
            self.register_buffer("_fallback_w13", w13, persistent=False)
            self.register_buffer("_fallback_w2", w2, persistent=False)
            self._fallback_entries = [(fb.global_id, i) for i, fb in enumerate(lw.fallbacks)]
            self.register_buffer(
                "_fallback_ids",
                torch.tensor(
                    [fb.global_id for fb in lw.fallbacks],
                    dtype=torch.int64,
                    device=w13.device,
                ),
                persistent=False,
            )
        else:
            self._fallback_entries = []
            empty = torch.empty(0, dtype=torch.bfloat16, device=lw.w13_codes.device)
            self.register_buffer("_fallback_w13", empty, persistent=False)
            self.register_buffer("_fallback_w2", empty, persistent=False)

    def _standard_topk(
        self, topk_output: TopKOutput
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if TopKOutputChecker.format_is_standard(topk_output):
            return topk_output.topk_weights, topk_output.topk_ids
        if TopKOutputChecker.format_is_bypassed(topk_output):
            std = topk_output.to_standard(layer_id=self.layer_id)
            return std.topk_weights, std.topk_ids
        raise NotImplementedError(
            f"TernaryExperts needs standard topk, got {topk_output.format}"
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
        pre_quant_input: Optional[Tuple] = None,
    ) -> torch.Tensor:
        del pre_quant_input
        if self._weights is None:
            raise RuntimeError(
                f"TernaryExperts layer {self.layer_id} weights not loaded"
            )
        return self._forward_impl(hidden_states, topk_output)

    def forward_impl(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
        pre_quant_input: Optional[Tuple] = None,
    ) -> torch.Tensor:
        return self.forward(hidden_states, topk_output, pre_quant_input)

    def _forward_impl(
        self, hidden_states: torch.Tensor, topk_output: TopKOutput
    ) -> torch.Tensor:
        w = self._weights
        assert w is not None
        topk_weights, topk_ids = self._standard_topk(topk_output)
        t = hidden_states.shape[0]
        if t == 0:
            return hidden_states

        x = hidden_states.to(torch.bfloat16)
        limit = self.swiglu_limit
        k = self.top_k
        if self._fused and (
            torch.cuda.is_current_stream_capturing()
            or t * k <= _get_kernels()._p_max()
        ):
            clamp = limit if (limit is not None and self.profile.has("swiglu_clamp")) else 0.0
            return _get_kernels().moe_decode(
                x.contiguous(),
                topk_ids.contiguous(),
                topk_weights.float().contiguous(),
                self._emap,
                w.signs_hidden,
                w.signs_inter,
                w.w13_codes,
                w.w13_scales,
                w.w2_codes,
                w.w2_scales,
                self._fallback_w13,
                self._fallback_w2,
                clamp,
            )
        inter = self.intermediate_size
        hidden = self.hidden_size
        b_h = self.profile.block_hidden
        b_i = self.profile.block_inter

        xr = rotate(x, w.signs_hidden, b_h)
        tok = torch.arange(t, device=x.device, dtype=torch.int64).repeat_interleave(k)
        ids = topk_ids.reshape(-1).to(torch.int64)
        weights = topk_weights.reshape(-1).to(x.dtype)
        lid = self.local_map[ids]

        gu = ternary_grouped(
            xr, tok, lid, w.w13_codes, w.w13_scales, out_dtype=torch.bfloat16
        )
        gate, up = gu.split(inter, dim=1)
        gate_f = gate.float()
        up_f = up.float()
        if limit is not None and self.profile.has("swiglu_clamp"):
            up_f = up_f.clamp(min=-limit, max=limit)
            gate_f = gate_f.clamp(max=limit)
        h = (weights.unsqueeze(1) * (F.silu(gate_f) * up_f)).to(torch.bfloat16)

        p = t * k
        hr = rotate(h, w.signs_inter, b_i)
        y = ternary_grouped(
            hr,
            torch.arange(p, device=x.device, dtype=torch.int64),
            lid,
            w.w2_codes,
            w.w2_scales,
            out_dtype=torch.bfloat16,
        )
        out = y.float().view(t, k, hidden).sum(dim=1)

        if self._fallback_entries and hasattr(self, "_fallback_w13"):
            if self.profile.has("fp4_fallback") and self._fallback_kind == "fp4":
                sparse = torch.cuda.is_current_stream_capturing() or t <= _FB_SPARSE_MAX_T
                if sparse and not os.environ.get("FP4_LIVE_LOG", "").strip():
                    return self._add_fp4_fallbacks_sparse(
                        x, topk_ids, topk_weights, out
                    ).to(torch.bfloat16)
                out = self._add_fp4_fallbacks(
                    x,
                    topk_ids,
                    topk_weights,
                    out,
                    self._fallback_entries,
                    self._fallback_w13,
                    self._fallback_w2,
                )
            elif self.profile.has("bf16_fallback"):
                out = self._add_bf16_fallbacks(
                    x, topk_ids, topk_weights, out, self._fallback_w13, self._fallback_w2
                )

        return out.to(torch.bfloat16)

    def _add_bf16_fallbacks(
        self,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        out: torch.Tensor,
        fb_w13: torch.Tensor,
        fb_w2: torch.Tensor,
    ) -> torch.Tensor:
        """Plain bf16 leftovers (Qwen); graph-safe (no host sync on coef)."""
        hit = topk_ids.to(torch.int64).unsqueeze(-1) == self._fallback_ids
        coef = (
            (topk_weights.unsqueeze(-1) * hit.to(topk_weights.dtype))
            .sum(dim=1)
            .to(x.dtype)
        )
        # coef: [T, F]. Always run all F experts; unrouted slots get coef=0 → zero contrib.
        result = out
        inter = self.intermediate_size
        for f in range(fb_w13.shape[0]):
            c = coef[:, f : f + 1]
            gu = F.linear(x, fb_w13[f]).float()
            gate, up = gu.split(inter, dim=1)
            h = (c * (F.silu(gate) * up)).to(torch.bfloat16)
            result = result + F.linear(h, fb_w2[f]).float()
        return result

    def _add_fp4_fallbacks_sparse(
        self,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        out: torch.Tensor,
    ) -> torch.Tensor:
        """All fallbacks of the layer in two batched GEMVs; unrouted experts skip weight reads."""
        hit = topk_ids.to(torch.int64).unsqueeze(-1) == self._fallback_ids
        coef = (
            (topk_weights.unsqueeze(-1) * hit.to(topk_weights.dtype))
            .sum(dim=1)
            .float()
            .contiguous()
        )
        k = _get_kernels()
        gu = k.fb_gemv(k.act_quant(x.contiguous()), self._fallback_w13, coef)
        h_q = k.fb_swiglu_quant(gu, coef, self.swiglu_limit or 10.0)
        y = k.fb_gemv(h_q, self._fallback_w2, coef)
        return out + y.float().sum(dim=0)

    def _add_fp4_fallbacks(
        self,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        out: torch.Tensor,
        fb_entries: List[Tuple[int, int]],
        fb_w13: torch.Tensor,
        fb_w2: torch.Tensor,
    ) -> torch.Tensor:
        limit = self.swiglu_limit if self.swiglu_limit is not None else 10.0
        result = out
        x_q = act_quant_emulate(x)
        live_log = os.environ.get("FP4_LIVE_LOG", "").strip()
        live_bits = []
        inter = self.intermediate_size
        for eid, idx in fb_entries:
            coef = (topk_weights * (topk_ids == eid).to(topk_weights.dtype)).sum(
                dim=1, keepdim=True
            )
            if live_log:
                live_bits.append((coef != 0).any())
            gu = F.linear(x_q, fb_w13[idx]).float()
            gate, up = gu.split(inter, dim=1)
            up = up.clamp(min=-limit, max=limit)
            gate = gate.clamp(max=limit)
            h = (coef * (F.silu(gate) * up)).to(torch.bfloat16)
            h_q = act_quant_emulate(h)
            result = result + F.linear(h_q, fb_w2[idx]).float()
        if live_log and live_bits:
            n_live = int(torch.stack(live_bits).sum().item())
            rec = {
                "layer": int(self.layer_id),
                "rank": int(self.moe_ep_rank),
                "n_fb": len(fb_entries),
                "n_live": n_live,
                "T": int(x.shape[0]),
            }
            with open(live_log, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
        return result
