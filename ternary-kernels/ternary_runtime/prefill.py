"""Ternary grouped GEMM for prefill: trits decoded in registers, fp16 tensor cores."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_BM = 64
_BN = 128
_WARMED: set[tuple] = set()


@triton.jit
def _ternary_gmm_kernel(
    a_ptr,
    codes_ptr,
    scales_ptr,
    out_ptr,
    dst_ptr,
    tile_e_ptr,
    tile_r0_ptr,
    end_ptr,
    N,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    """out[dst[r], n] = sum_g (a[r, g] . trits[e, n, g]) * scale[e, n, g] over 128-wide groups g."""
    pid_t = tl.program_id(0)
    pid_n = tl.program_id(1)
    e = tl.load(tile_e_ptr + pid_t).to(tl.int64)
    r0 = tl.load(tile_r0_ptr + pid_t)
    r1 = tl.load(end_ptr + e)
    words: tl.constexpr = K // 16
    groups: tl.constexpr = K // 128
    rows = r0 + tl.arange(0, BM)
    mrow = rows < r1
    n = pid_n * BN + tl.arange(0, BN)
    mn = n < N
    offs_k = tl.arange(0, 128)
    offs_w = tl.arange(0, 8)
    shifts = tl.arange(0, 16) * 2
    a_base = a_ptr + rows.to(tl.int64)[:, None] * K + offs_k[None, :]
    c_base = codes_ptr + (e * N + n.to(tl.int64))[:, None] * words + offs_w[None, :]
    s_base = scales_ptr + (e * N + n.to(tl.int64)) * groups
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for g in range(groups):
        a = tl.load(a_base + g * 128, mask=mrow[:, None], other=0.0)
        c = tl.load(c_base + g * 8, mask=mn[:, None], other=0x55555555)
        tr = ((c[:, :, None] >> shifts[None, None, :]) & 3) - 1
        w = tl.reshape(tr, (BN, 128)).to(tl.float16)
        d = tl.dot(a, tl.trans(w))
        s = tl.load(s_base + g, mask=mn, other=0.0).to(tl.float32)
        acc += d * s[None, :]
    dst = tl.load(dst_ptr + rows, mask=mrow, other=0).to(tl.int64)
    tl.store(
        out_ptr + dst[:, None] * N + n[None, :],
        acc.to(out_ptr.dtype.element_ty),
        mask=mrow[:, None] & mn[None, :],
    )


def ternary_gmm(xr, tok, lid, codes, scales, out_dtype=torch.bfloat16):
    """Row p is W'[lid[p]] @ xr[tok[p]] (zeros where lid[p] < 0); one host sync for the tile plan."""
    e, n, words = codes.shape
    k = words * 16
    p = tok.shape[0]
    device = xr.device
    out = torch.zeros(p, n, device=device, dtype=out_dtype)
    if p == 0:
        return out
    key = torch.where(lid < 0, torch.full_like(lid, e), lid)
    order = torch.argsort(key)
    counts = torch.bincount(key, minlength=e + 1)
    ends = torch.cumsum(counts[:e], 0)
    counts_cpu = counts[:e].cpu()
    valid = int(counts_cpu.sum())
    if valid == 0:
        return out
    rows = order[:valid]
    act = xr.index_select(0, tok.index_select(0, rows).to(torch.int64)).to(torch.float16)
    ntile = (counts_cpu + _BM - 1) // _BM
    tile_e = torch.repeat_interleave(torch.arange(e, dtype=torch.int64), ntile)
    starts_cpu = torch.cumsum(counts_cpu, 0) - counts_cpu
    first = torch.cumsum(ntile, 0) - ntile
    local = torch.arange(int(ntile.sum()), dtype=torch.int64) - first[tile_e]
    tile_r0 = (starts_cpu[tile_e] + local * _BM).to(torch.int32)
    tile_e = tile_e.to(torch.int32).to(device, non_blocking=True)
    tile_r0 = tile_r0.to(device, non_blocking=True)
    grid = (tile_e.numel(), triton.cdiv(n, _BN))
    _ternary_gmm_kernel[grid](
        act,
        codes,
        scales,
        out,
        rows,
        tile_e,
        tile_r0,
        ends.to(torch.int32),
        n,
        K=k,
        BM=_BM,
        BN=_BN,
        num_warps=8,
        num_stages=3,
    )
    return out


def warmup_ternary_gmm(device=None, k=2560, n_list=(1280, 2560), e=8, p=512):
    """JIT-compile `_ternary_gmm_kernel` for flash_next gate_up/down shapes (idempotent)."""
    device = device or torch.device("cuda", torch.cuda.current_device())
    key = (str(device), int(k), tuple(int(n) for n in n_list))
    if key in _WARMED:
        return
    words = k // 16
    groups = k // 128
    xr = torch.zeros(max(p // 4, 1), k, device=device, dtype=torch.float16)
    tok = torch.zeros(p, device=device, dtype=torch.int64)
    lid = torch.zeros(p, device=device, dtype=torch.int32)
    for n in n_list:
        codes = torch.zeros(e, n, words, device=device, dtype=torch.int32)
        scales = torch.zeros(e, n, groups, device=device, dtype=torch.float16)
        ternary_gmm(xr, tok, lid, codes, scales, out_dtype=torch.bfloat16)
    torch.cuda.synchronize(device)
    _WARMED.add(key)
