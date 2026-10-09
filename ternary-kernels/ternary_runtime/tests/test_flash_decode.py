"""Fused decode MoE, HC mix and prefill GEMM vs references (flash_next shapes, one GPU)."""

import pytest
import torch
import torch.nn.functional as F

from ternary_runtime import kernels as k

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

E, K, I, TOPK = 12, 2560, 640, 10
BH, BI = 512, 128


def _codes(e, n, kk, dev):
    """Random trit codes {0,1,2} packed 16 per int32."""
    trits = torch.randint(0, 3, (e, n, kk // 16, 16), device=dev, dtype=torch.int64)
    shifts = torch.arange(16, device=dev, dtype=torch.int64) * 2
    words = (trits << shifts).sum(-1)
    return torch.where(words >= 2**31, words - 2**32, words).to(torch.int32)


def _layer(dev, n_fb=3):
    g = torch.Generator(device=dev).manual_seed(0)
    torch.manual_seed(0)
    w13c = _codes(E, 2 * I, K, dev)
    w13s = (torch.rand(E, 2 * I, K // 128, device=dev, generator=g) * 0.02).half()
    w2c = _codes(E, K, I, dev)
    w2s = (torch.rand(E, K, I // 128, device=dev, generator=g) * 0.02).half()
    sh = (torch.randint(0, 2, (BH,), device=dev) * 2 - 1).float()
    si = (torch.randint(0, 2, (BI,), device=dev) * 2 - 1).float()
    fb13 = (torch.randn(n_fb, 2 * I, K, device=dev) * 0.02).bfloat16()
    fb2 = (torch.randn(n_fb, K, I, device=dev) * 0.02).bfloat16()
    n_global = 64
    emap = torch.full((n_global,), -1, dtype=torch.int32)
    for s in range(E):
        emap[s] = s
    fb_ids = [40, 41, 42][:n_fb]
    for i, gid in enumerate(fb_ids):
        emap[gid] = -(i + 2)
    return dict(w13c=w13c, w13s=w13s, w2c=w2c, w2s=w2s, sh=sh, si=si, fb13=fb13, fb2=fb2, emap=emap.to(dev))


def _reference(x, ids, w, L):
    """fp32 reference: rotate + dequantized ternary matmuls + dense bf16 fallbacks."""
    t = x.shape[0]
    xr = k.rotate(x, L["sh"], BH)
    out = torch.zeros(t, K, device=x.device, dtype=torch.float32)
    emap = L["emap"].cpu()
    for ti in range(t):
        for j in range(ids.shape[1]):
            c = int(emap[int(ids[ti, j])])
            if c == -1:
                continue
            wt = float(w[ti, j])
            if c >= 0:
                one = torch.zeros(1, dtype=torch.int32, device=x.device)
                lid = torch.tensor([c], dtype=torch.int32, device=x.device)
                gu = k.ternary_grouped_ref(xr[ti : ti + 1], one, lid, L["w13c"], L["w13s"], out_dtype=torch.float32)
                h = wt * F.silu(gu[:, :I]) * gu[:, I:]
                hr = k.rotate(h.contiguous(), L["si"], BI)
                y = k.ternary_grouped_ref(hr, one, lid, L["w2c"], L["w2s"], out_dtype=torch.float32)
            else:
                f = -c - 2
                gu = x[ti : ti + 1].float() @ L["fb13"][f].float().T
                h = wt * F.silu(gu[:, :I]) * gu[:, I:]
                y = h @ L["fb2"][f].float().T
            out[ti] += y[0]
    return out


@pytest.mark.parametrize("t", [1, 3, 8])
@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
def test_moe_decode_matches_reference(t, id_dtype):
    dev = torch.device("cuda")
    L = _layer(dev)
    x = (torch.randn(t, K, device=dev)).bfloat16()
    pool = torch.tensor([0, 1, 2, 3, 5, 7, 9, 11, 20, 30, 40, 41, 42, 50, 60], device=dev)
    ids = torch.stack([pool[torch.randperm(pool.numel(), device=dev)[:TOPK]] for _ in range(t)]).to(id_dtype)
    w = torch.rand(t, TOPK, device=dev)
    w = w / w.sum(-1, keepdim=True)
    got = k.moe_decode(x, ids, w, L["emap"], L["sh"], L["si"], L["w13c"], L["w13s"], L["w2c"], L["w2s"], L["fb13"], L["fb2"], 0.0)
    ref = _reference(x, ids, w, L)
    err = (got.float() - ref).abs().max().item()
    scale = ref.abs().max().item()
    assert err <= 1e-2 * scale + 1e-3, (err, scale)


def test_moe_decode_no_fallbacks():
    dev = torch.device("cuda")
    L = _layer(dev, n_fb=0)
    empty = torch.empty(0, dtype=torch.bfloat16, device=dev)
    L["fb13"], L["fb2"] = empty, empty
    x = torch.randn(2, K, device=dev).bfloat16()
    ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 11, 20, 21, 22, 23, 24, 25, 26, 27]], device=dev, dtype=torch.int32)
    w = torch.full((2, TOPK), 0.1, device=dev)
    got = k.moe_decode(x, ids, w, L["emap"], L["sh"], L["si"], L["w13c"], L["w13s"], L["w2c"], L["w2s"], empty, empty, 0.0)
    ref = _reference(x, ids, w, L)
    assert (got.float() - ref).abs().max().item() <= 1e-2 * ref.abs().max().item() + 1e-3


@pytest.mark.parametrize("m", [1, 2, 5, 16])
def test_hc_mix_matches_reference(m):
    dev = torch.device("cuda")
    hc, hs, r = 4, 2560, 320
    x = torch.randn(m, hc * hs, device=dev).bfloat16()
    wd = (torch.randn(r, hc * hs, device=dev) * 0.01).bfloat16()
    wu = (torch.randn(hc * hs, r, device=dev) * 0.05).bfloat16()
    t = F.silu((x.float() @ wd.float().T) / hc).bfloat16().float()
    gate = torch.sigmoid(t @ wu.float().T).unflatten(-1, (hc, hs))
    ref = (gate * x.float().unflatten(-1, (hc, hs))).mean(dim=-2)
    got = k.hc_mix(x, wd, wu, hc, hs)
    assert (got.float() - ref).abs().max().item() < 2e-2


@pytest.mark.parametrize("p", [600, 3000])
def test_prefill_gmm_matches_reference(p):
    from ternary_runtime.prefill import ternary_gmm

    dev = torch.device("cuda")
    L = _layer(dev, n_fb=0)
    m = p // 4
    xr = torch.randn(m, K, device=dev)
    tok = torch.randint(0, m, (p,), device=dev)
    lid = torch.randint(-1, E, (p,), device=dev, dtype=torch.int32)
    got = ternary_gmm(xr, tok, lid, L["w13c"], L["w13s"], out_dtype=torch.float32)
    ref = k.ternary_grouped_ref(xr, tok, lid, L["w13c"], L["w13s"], out_dtype=torch.float32)
    assert (got - ref).abs().max().item() <= 2e-3 * ref.abs().max().item() + 1e-3
    assert torch.all(got[lid < 0] == 0)
