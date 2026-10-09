"""Ternary expert kernels: block Hadamard rotation and grouped ternary GEMM.

CUDA specializations track ``profile.ALL_SPECIALIZED_K``:
  deepseek_v41 → 5120, 2304
  flash_next   → 2560, 640
  (Bonsai 17408 remains in the .cu file; not a public profile)
"""

import os
from pathlib import Path

import torch

from ternary_runtime.profile import ALL_SPECIALIZED_K

_EXT = None
_FLASH_EXT = None
_DECODE_MAX_P = int(os.environ.get("TERNARY_P_MAX", "512"))
_PREFILL_EXPERT_CHUNK = 8
_PREFILL_MODE = os.environ.get("TERNARY_PREFILL", "gmm")
def _default_build_root() -> Path:
    """Writable cache for JIT CUDA extensions (override with TERNARY_BUILD_DIR)."""
    override = os.environ.get("TERNARY_BUILD_DIR")
    if override:
        return Path(override)
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg) / "ternary-flash" / "build"
    return Path.home() / ".cache" / "ternary-flash" / "build"


_BUILD_DIR = str(_default_build_root() / "ternary_moe")
_FLASH_BUILD_DIR = str(_default_build_root() / "flash_decode")


def _p_max():
    """Largest pair count routed to the decode GEMV (eager); above it, prefill GEMM."""
    return _DECODE_MAX_P


class _PrefillWorkspace:
    """Reused fp16 weight buffer for chunked expert dequant."""

    def __init__(self):
        self.weight = None
        self.k = 0
        self.n = 0
        self.device = None

    def weight_view(self, chunk_e, k, n, device):
        """Return a [chunk_e, K, N] fp16 view, growing storage only when needed."""
        if (
            self.weight is None
            or self.k != k
            or self.n != n
            or self.device != device
            or self.weight.size(0) < chunk_e
        ):
            cap = max(chunk_e, _PREFILL_EXPERT_CHUNK)
            self.weight = torch.empty(cap, k, n, device=device, dtype=torch.float16)
            self.k = k
            self.n = n
            self.device = device
        return self.weight[:chunk_e]


_PREFILL_WS = _PrefillWorkspace()


def _extension():
    """Load the cached CUDA extension."""
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load

        os.makedirs(_BUILD_DIR, exist_ok=True)
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")
        src = Path(__file__).resolve().parent / "csrc" / "ternary_moe.cu"
        _EXT = load(
            name="ternary_moe",
            sources=[str(src)],
            extra_cuda_cflags=["-O3", "-std=c++17"],
            build_directory=_BUILD_DIR,
            with_cuda=True,
            verbose=False,
        )
        # Sanity: every registered model K is listed for ops engineers.
        _ = ALL_SPECIALIZED_K
    return _EXT


def _flash_extension():
    """Load the cached fused-decode CUDA extension (MoE decode, HC mix)."""
    global _FLASH_EXT
    if _FLASH_EXT is None:
        from torch.utils.cpp_extension import load

        os.makedirs(_FLASH_BUILD_DIR, exist_ok=True)
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")
        src = Path(__file__).resolve().parent / "csrc" / "flash_decode.cu"
        _FLASH_EXT = load(
            name="flash_decode",
            sources=[str(src)],
            extra_cuda_cflags=["-O3", "-std=c++17"],
            build_directory=_FLASH_BUILD_DIR,
            with_cuda=True,
            verbose=False,
        )
    return _FLASH_EXT


def moe_decode(x, ids, topw, emap, signs_h, signs_i, w13c, w13s, w2c, w2s, fb13, fb2, limit=0.0):
    """Fused routed-expert decode: bf16 [T, K] = sum_k w * W2[e](R silu(W13 R x)); two launches.

    emap[e] is the local ternary slot (>= 0), -1 when expert e is not computed on this rank,
    or -(f + 2) for bf16 fallback f. Unrouted experts are never read. limit <= 0 disables clamp.
    """
    return _flash_extension().moe_decode(
        x, ids, topw, emap, signs_h, signs_i, w13c, w13s, w2c, w2s, fb13, fb2, float(limit)
    )


def hc_mix(x, w_down, w_up, hc, hs):
    """Hyper-connection low-rank mix for M <= 16 rows (split-K down, fused up/sigmoid/mean)."""
    return _flash_extension().hc_mix(x, w_down, w_up, int(hc), int(hs))


def rotate(x, signs, block):
    """R x per row, fp32. x is [M, K] bf16 or fp32; signs is [block] fp32."""
    return _extension().rotate(x, signs, int(block))


def fb_gemv(x, w, coef):
    """bf16 [F, T, N] = x @ w[f].T per expert, fp32 accumulate; zeros where coef[:, f] == 0.

    x is [T, K] (shared) or [F, T, K] bf16, w is [F, N, K] bf16, coef is [T, F] fp32.
    Graph-safe: unrouted experts are skipped on device without reading their weights.
    """
    return _extension().fb_gemv(x, w, coef)


def act_quant(x):
    """Fused experts.act_quant_emulate for contiguous bf16 x."""
    return _extension().act_quant(x)


def fb_swiglu_quant(gu, coef, limit):
    """bf16 [F, T, I] = act_quant(bf16(coef[t, f] * silu(min(gate, L)) * clamp(up, -L, L)))."""
    return _extension().fb_swiglu_quant(gu, coef, float(limit))


def _decode(xr, tok, lid, codes, scales, out_dtype, out=None):
    """Memory-bound GEMV. One output row per pair, fp32 accumulate.

    Exact fp32 activations by default; TERNARY_DECODE=int8 selects the lossy
    int8 + dp4a path.
    """
    p = tok.shape[0]
    n = codes.shape[1]
    if out is None:
        out = torch.empty(p, n, device=xr.device, dtype=out_dtype)
    elif out.shape != (p, n):
        raise ValueError(f"decode out shape {tuple(out.shape)} != ({p}, {n})")
    if p == 0:
        return out
    _extension().grouped_decode(xr, tok, lid, codes, scales, out)
    return out


def _prefill(xr, tok, lid, codes, scales, out_dtype):
    """Sort pairs by expert; fp16 tensor-core grouped GEMM over chunks of experts.

    Dequantized fp16 weights exist for at most _PREFILL_EXPERT_CHUNK experts at a
    time, and experts without tokens are skipped.
    """
    e, n, _words = codes.shape
    k = xr.shape[1]
    p = tok.shape[0]
    device = xr.device
    key = torch.where(lid < 0, torch.full_like(lid, e), lid)
    order = torch.argsort(key)
    counts = torch.bincount(key, minlength=e + 1)[:e]
    starts = torch.cumsum(counts, 0) - counts
    ends = torch.cumsum(counts, 0)
    out = torch.zeros(p, n, device=device, dtype=out_dtype)
    if p == 0:
        return out
    ext = _extension()
    counts_cpu = counts.cpu()
    starts_cpu = starts.cpu()
    ends_cpu = ends.cpu()
    e0 = 0
    while e0 < e:
        e1 = min(e0 + _PREFILL_EXPERT_CHUNK, e)
        r0 = int(starts_cpu[e0].item())
        r1 = int(ends_cpu[e1 - 1].item())
        if r1 > r0:
            rows = order[r0:r1]
            act = xr.index_select(0, tok.index_select(0, rows).to(torch.int64)).to(
                torch.float16
            )
            chunk_e = e1 - e0
            weight = _PREFILL_WS.weight_view(chunk_e, k, n, device)
            ext.dequant_kn_into(codes[e0:e1], scales[e0:e1], weight)
            local_offs = (ends[e0:e1] - r0).to(torch.int32)
            prod = torch.nn.functional.grouped_mm(act, weight, offs=local_offs)
            out.index_copy_(0, rows, prod.to(out_dtype))
        e0 = e1
    return out


def ternary_grouped(
    xr,
    tok,
    lid,
    codes,
    scales,
    out_dtype=torch.bfloat16,
    out=None,
):
    """Row p is W'[lid[p]] @ xr[tok[p]], or zeros when lid[p] < 0.

    xr is [M, K] fp32. tok is [P] int32 or int64. lid is [P] int32.
    codes is [E, N, K/16] int32. scales is [E, N, K/128] fp16.
    Decode (P <= TERNARY_P_MAX, default 512, and any P while a CUDA graph is
    capturing) is graph-safe; larger eager batches use the prefill GEMM.
    """
    if out_dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise TypeError("out_dtype must be bf16, fp16, or fp32")
    p = tok.shape[0]
    if torch.cuda.is_current_stream_capturing() or p <= _DECODE_MAX_P:
        return _decode(xr, tok, lid, codes, scales, out_dtype, out=out)
    if _PREFILL_MODE == "dequant":
        return _prefill(xr, tok, lid, codes, scales, out_dtype)
    from ternary_runtime.prefill import ternary_gmm

    return ternary_gmm(xr, tok, lid, codes, scales, out_dtype)


def ternary_grouped_ref(xr, tok, lid, codes, scales, out_dtype=torch.bfloat16):
    """Dequantize codes times scales to fp32 and matmul. Testing only."""
    e, n, words = codes.shape
    k = words * 16
    shifts = (torch.arange(16, device=codes.device, dtype=torch.int64) * 2).view(1, 1, 1, 16)
    bits = codes.to(torch.int64) & 0xFFFFFFFF
    trits = ((bits.unsqueeze(-1) >> shifts) & 3).to(torch.float32) - 1.0
    weight = trits.reshape(e, n, k) * scales.to(torch.float32).repeat_interleave(128, dim=-1)
    out = torch.zeros(tok.shape[0], n, device=xr.device, dtype=torch.float32)
    tok_l = tok.to(torch.int64)
    for slot in range(e):
        sel = torch.nonzero(lid == slot, as_tuple=False).flatten()
        if sel.numel() == 0:
            continue
        gathered = xr.index_select(0, tok_l.index_select(0, sel))
        out.index_copy_(0, sel, gathered @ weight[slot].transpose(0, 1))
    return out.to(out_dtype)
