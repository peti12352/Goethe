"""Correctness tests for grouped ternary decode GEMV."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ternary_runtime.kernels import _p_max, rotate, ternary_grouped, ternary_grouped_ref


@pytest.fixture(autouse=True, params=["fp32", "int8"])
def decode_mode(request, monkeypatch):
    monkeypatch.setenv("TERNARY_DECODE", request.param)
    return request.param


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    diff = (a.float() - b.float()).reshape(-1)
    ref = b.float().reshape(-1)
    return (diff.norm() / ref.norm().clamp(min=1e-8)).item()


def _random_codes(e: int, n: int, k: int, device: torch.device) -> torch.Tensor:
    words = k // 16
    trits = torch.randint(0, 3, (e, n, words, 16), device=device, dtype=torch.int64)
    shifts = (torch.arange(16, device=device, dtype=torch.int64) * 2).view(1, 1, 1, 16)
    return (trits << shifts).sum(dim=-1).to(torch.int32).contiguous()


def _random_scales(e: int, n: int, k: int, device: torch.device) -> torch.Tensor:
    return torch.rand(e, n, k // 128, device=device, dtype=torch.float16) * 0.1 + 0.01


@pytest.mark.parametrize("k,n", [(2304, 128), (5120, 64)])
@pytest.mark.parametrize("tok_dtype", [torch.int32, torch.int64])
def test_ternary_grouped_gaussian(k, n, tok_dtype):
    device = torch.device("cuda")
    e = 4
    m = 32
    torch.manual_seed(k + n)
    xr = torch.randn(m, k, device=device, dtype=torch.float32)
    codes = _random_codes(e, n, k, device)
    scales = _random_scales(e, n, k, device)
    p = 24
    tok = torch.randint(0, m, (p,), device=device, dtype=tok_dtype)
    lid = torch.randint(0, e, (p,), device=device, dtype=torch.int32)
    got = ternary_grouped(xr, tok, lid, codes, scales, out_dtype=torch.bfloat16)
    ref = ternary_grouped_ref(xr, tok, lid, codes, scales, out_dtype=torch.bfloat16)
    assert _rel_l2(got, ref) < 2e-2


def _hadamard(b: int, device: torch.device) -> torch.Tensor:
    h = torch.ones(1, 1, device=device, dtype=torch.float64)
    while h.shape[0] < b:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h


@pytest.mark.parametrize("k,b,m", [(5120, 1024, 8), (2304, 256, 48), (5120, 1024, 777)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rotate_matches_hadamard(k, b, m, dtype, decode_mode):
    if decode_mode != "fp32":
        pytest.skip("rotate is mode independent")
    device = torch.device("cuda")
    torch.manual_seed(k + m)
    x = torch.randn(m, k, device=device).to(dtype)
    signs = torch.randint(0, 2, (b,), device=device).float() * 2 - 1
    got = rotate(x, signs, b)
    xd = x.double().view(m, k // b, b) * signs.double()
    ref = (xd @ _hadamard(b, device) / b**0.5).view(m, k)
    assert _rel_l2(got, ref) < 1e-6


def _quant_emulate(xr: torch.Tensor) -> torch.Tensor:
    m, k = xr.shape
    g = xr.view(m, k // 128, 128)
    scale = g.abs().amax(dim=-1, keepdim=True) / 127.0
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    v = (g / safe).clamp(-127.0, 127.0)
    q = torch.sign(v) * torch.floor(v.abs() + 0.5)
    return (q * scale).view(m, k)


@pytest.mark.parametrize("k,n", [(2304, 128), (5120, 64)])
def test_int8_matches_quant_emulation(k, n, decode_mode):
    """int8 path equals the fp32 reference on q*sx activations (layout + xsum check)."""
    if decode_mode != "int8":
        pytest.skip("int8 only")
    device = torch.device("cuda")
    e, m, p = 4, 24, 40
    torch.manual_seed(7 + k)
    xr = torch.randn(m, k, device=device, dtype=torch.float32)
    xr[3] = 0.0
    codes = _random_codes(e, n, k, device)
    scales = _random_scales(e, n, k, device)
    tok = torch.randint(0, m, (p,), device=device, dtype=torch.int32)
    lid = torch.randint(0, e, (p,), device=device, dtype=torch.int32)
    got = ternary_grouped(xr, tok, lid, codes, scales, out_dtype=torch.float32)
    ref = ternary_grouped_ref(_quant_emulate(xr), tok, lid, codes, scales, out_dtype=torch.float32)
    assert _rel_l2(got, ref) < 1e-5


@pytest.mark.parametrize("k,n,p", [(5120, 64, 768), (5120, 32, 2048)])
def test_ternary_grouped_prefill(k, n, p):
    device = torch.device("cuda")
    e, m = 16, 128
    torch.manual_seed(k + n + p)
    xr = torch.randn(m, k, device=device, dtype=torch.float32)
    codes = _random_codes(e, n, k, device)
    scales = _random_scales(e, n, k, device)
    tok = torch.randint(0, m, (p,), device=device, dtype=torch.int32)
    lid = torch.randint(0, e, (p,), device=device, dtype=torch.int32)
    got = ternary_grouped(xr, tok, lid, codes, scales, out_dtype=torch.bfloat16)
    ref = ternary_grouped_ref(xr, tok, lid, codes, scales, out_dtype=torch.bfloat16)
    assert _rel_l2(got, ref) < 2e-2


@pytest.mark.parametrize("k", [2304, 5120])
def test_ternary_grouped_neg_lid(k):
    device = torch.device("cuda")
    e, n, m = 4, 128, 16
    xr = torch.randn(m, k, device=device, dtype=torch.float32)
    codes = _random_codes(e, n, k, device)
    scales = _random_scales(e, n, k, device)
    p = 16
    tok = torch.randint(0, m, (p,), device=device, dtype=torch.int32)
    lid = torch.full((p,), -1, device=device, dtype=torch.int32)
    got = ternary_grouped(xr, tok, lid, codes, scales, out_dtype=torch.float32)
    assert torch.all(got == 0)


def test_ternary_grouped_p_zero():
    device = torch.device("cuda")
    k, n, e, m = 5120, 64, 4, 8
    xr = torch.randn(m, k, device=device, dtype=torch.float32)
    codes = _random_codes(e, n, k, device)
    scales = _random_scales(e, n, k, device)
    tok = torch.empty(0, device=device, dtype=torch.int32)
    lid = torch.empty(0, device=device, dtype=torch.int32)
    got = ternary_grouped(xr, tok, lid, codes, scales, out_dtype=torch.bfloat16)
    assert got.shape == (0, n)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_ternary_grouped_cuda_graph_padded():
    device = torch.device("cuda")
    k, n, e, m = 5120, 64, 4, 16
    p_max = _p_max()
    p = 8
    xr = torch.randn(m, k, device=device, dtype=torch.float32)
    codes = _random_codes(e, n, k, device)
    scales = _random_scales(e, n, k, device)
    tok_pad = torch.zeros(p_max, device=device, dtype=torch.int32)
    lid_pad = torch.full((p_max,), -1, device=device, dtype=torch.int32)
    out_buf = torch.empty(p_max, n, device=device, dtype=torch.bfloat16)
    tok_pad[:p] = torch.randint(0, m, (p,), device=device, dtype=torch.int32)
    lid_pad[:p] = torch.randint(0, e, (p,), device=device, dtype=torch.int32)
    g = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(3):
            ternary_grouped(
                xr, tok_pad, lid_pad, codes, scales, out_dtype=torch.bfloat16, out=out_buf
            )
        stream.synchronize()
        with torch.cuda.graph(g, stream=stream):
            ternary_grouped(
                xr, tok_pad, lid_pad, codes, scales, out_dtype=torch.bfloat16, out=out_buf
            )
    g.replay()
    ref = ternary_grouped_ref(
        xr, tok_pad[:p], lid_pad[:p], codes, scales, out_dtype=torch.bfloat16
    )
    assert _rel_l2(out_buf[:p], ref) < 2e-2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_ternary_grouped_cuda_graph_live_p():
    """SGLang captures with live pair counts, not TERNARY_P_MAX pads."""
    device = torch.device("cuda")
    k, n, e, m, p = 5120, 64, 4, 16, 48
    xr = torch.randn(m, k, device=device, dtype=torch.float32)
    codes = _random_codes(e, n, k, device)
    scales = _random_scales(e, n, k, device)
    tok = torch.randint(0, m, (p,), device=device, dtype=torch.int32)
    lid = torch.randint(0, e, (p,), device=device, dtype=torch.int32)
    g = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(3):
            out = ternary_grouped(xr, tok, lid, codes, scales, out_dtype=torch.bfloat16)
        stream.synchronize()
        with torch.cuda.graph(g, stream=stream):
            out = ternary_grouped(xr, tok, lid, codes, scales, out_dtype=torch.bfloat16)
    g.replay()
    ref = ternary_grouped_ref(xr, tok, lid, codes, scales, out_dtype=torch.bfloat16)
    assert _rel_l2(out, ref) < 2e-2


def _fb_ref(x, w, coef):
    xs = x if x.dim() == 3 else x.unsqueeze(0).expand(w.shape[0], -1, -1)
    out = torch.stack([torch.nn.functional.linear(xs[f], w[f]) for f in range(w.shape[0])])
    return out * (coef != 0).any(dim=0).view(-1, 1, 1)


@pytest.mark.parametrize("t", [1, 8, 13])
@pytest.mark.parametrize("k,n,per_expert", [(5120, 4608, False), (2304, 5120, True)])
def test_fb_gemv_matches_linear(t, k, n, per_expert):
    from ternary_runtime.kernels import fb_gemv

    device = torch.device("cuda")
    torch.manual_seed(t + k)
    f = 3
    w = (torch.randn(f, n, k, device=device) * 0.02).to(torch.bfloat16)
    shape = (f, t, k) if per_expert else (t, k)
    x = torch.randn(*shape, device=device).to(torch.bfloat16)
    coef = torch.zeros(t, f, device=device)
    coef[0, 0] = 0.3
    coef[t - 1, 2] = 0.7
    got = fb_gemv(x, w, coef)
    ref = _fb_ref(x, w, coef)
    assert torch.all(got[1] == 0)
    assert _rel_l2(got, ref) < 1e-2


def test_fb_gemv_cuda_graph_skip_follows_coef():
    from ternary_runtime.kernels import fb_gemv

    device = torch.device("cuda")
    torch.manual_seed(0)
    w = (torch.randn(4, 4608, 5120, device=device) * 0.02).to(torch.bfloat16)
    x = torch.randn(8, 5120, device=device).to(torch.bfloat16)
    coef = torch.zeros(8, 4, device=device)
    coef[:, 1] = 1.0
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        fb_gemv(x, w, coef)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=stream):
        out = fb_gemv(x, w, coef)
    for col in (1, 3):
        coef.zero_()
        coef[2, col] = 0.5
        g.replay()
        torch.cuda.synchronize()
        assert _rel_l2(out, _fb_ref(x, w, coef)) < 1e-2
        assert all(torch.all(out[f] == 0) for f in range(4) if f != col)


def _act_quant_ref(x):
    flat = x.reshape(-1, 32).float()
    amax = flat.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    scale = torch.pow(2.0, torch.ceil(torch.log2(amax / 448.0)))
    q = (flat / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return (q.float() * scale).reshape_as(x).to(torch.bfloat16)


def test_act_quant_bit_exact():
    from ternary_runtime.kernels import act_quant

    torch.manual_seed(0)
    x = (torch.randn(13, 5120, device="cuda") * torch.logspace(-6, 3, 5120, device="cuda")).to(torch.bfloat16)
    x[3] = 0
    assert torch.equal(act_quant(x), _act_quant_ref(x))


def test_fb_swiglu_quant_matches_torch():
    from ternary_runtime.kernels import fb_swiglu_quant

    torch.manual_seed(0)
    f, t, i, limit = 3, 8, 2304, 10.0
    gu = (torch.randn(f, t, 2 * i, device="cuda") * 6).to(torch.bfloat16)
    coef = torch.rand(t, f, device="cuda")
    coef[:, 1] = 0
    g, u = gu.float().split(i, dim=2)
    h = (coef.t().unsqueeze(-1) * (torch.nn.functional.silu(g.clamp(max=limit)) * u.clamp(-limit, limit))).to(torch.bfloat16)
    ref = _act_quant_ref(h)
    got = fb_swiglu_quant(gu, coef, limit)
    assert torch.all(got[1] == 0)
    assert (got != ref).float().mean().item() < 1e-3
    assert _rel_l2(got, ref) < 1e-3
