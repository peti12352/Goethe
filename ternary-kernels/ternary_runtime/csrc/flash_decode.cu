#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>

using bf16 = __nv_bfloat16;

constexpr int kThreads = 256;
constexpr int kWarpsPerCta = kThreads / 32;
constexpr int kWordStride = 17;
constexpr int kMaxTopK = 32;
constexpr int kHcMaxRows = 16;
constexpr int kDownPrefetch = 16;

__device__ __forceinline__ int sidx(int elem) {
    return (elem >> 4) * kWordStride + (elem & 15);
}

template <int EPL>
__device__ __forceinline__ void warp_fwht(float (&v)[EPL], int lane) {
    // In-warp Walsh-Hadamard over 32*EPL values; lane owns indices lane*EPL .. lane*EPL+EPL-1.
#pragma unroll
    for (int h = 1; h < EPL; h <<= 1) {
#pragma unroll
        for (int j = 0; j < EPL; ++j) {
            if ((j & h) == 0) {
                float a = v[j];
                float b = v[j + h];
                v[j] = a + b;
                v[j + h] = a - b;
            }
        }
    }
#pragma unroll
    for (int hl = 1; hl < 32; hl <<= 1) {
        bool upper = (lane & hl) != 0;
#pragma unroll
        for (int j = 0; j < EPL; ++j) {
            float p = __shfl_xor_sync(0xffffffffu, v[j], hl);
            v[j] = upper ? (p - v[j]) : (v[j] + p);
        }
    }
}

template <int EPL>
__device__ __forceinline__ void load_bf16_vec(const bf16* __restrict__ p, float (&v)[EPL]) {
    static_assert(EPL % 4 == 0, "EPL must be a multiple of 4");
    if constexpr (EPL % 8 == 0) {
#pragma unroll
        for (int q = 0; q < EPL / 8; ++q) {
            uint4 raw = __ldg(reinterpret_cast<const uint4*>(p) + q);
            const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float2 f = __bfloat1622float2(b2[j]);
                v[q * 8 + 2 * j] = f.x;
                v[q * 8 + 2 * j + 1] = f.y;
            }
        }
    } else {
#pragma unroll
        for (int q = 0; q < EPL / 4; ++q) {
            uint2 raw = __ldg(reinterpret_cast<const uint2*>(p) + q);
            const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                float2 f = __bfloat1622float2(b2[j]);
                v[q * 4 + 2 * j] = f.x;
                v[q * 4 + 2 * j + 1] = f.y;
            }
        }
    }
}

__device__ __forceinline__ float trit_dot(int packed, const float (&xv)[16]) {
    // (2^23 + c) - (2^23 + 1) == c - 1 exactly; avoids quarter-rate I2F per trit.
    const unsigned u = static_cast<unsigned>(packed);
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        const float t = __uint_as_float(((u >> (2 * i)) & 3u) | 0x4B000000u) - 8388609.0f;
        s = fmaf(t, xv[i], s);
    }
    return s;
}

__device__ __forceinline__ float bf16_dot16(const bf16* __restrict__ w, const float (&xv)[16]) {
    float wf[16];
    load_bf16_vec<16>(w, wf);
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        s = fmaf(wf[i], xv[i], s);
    }
    return s;
}

__device__ __forceinline__ void load_word(const float* __restrict__ xs, int word, float (&xv)[16]) {
    const float* p = xs + word * kWordStride;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        xv[i] = p[i];
    }
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int s = 16; s > 0; s >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, s);
    }
    return v;
}

template <typename IdT, int K, int I, int BH, int JPW>
__global__ void __launch_bounds__(kThreads) moe_gate_up_kernel(
    const bf16* __restrict__ x,
    const IdT* __restrict__ ids,
    const float* __restrict__ topw,
    const int* __restrict__ emap,
    const float* __restrict__ signs,
    const int* __restrict__ codes,
    const __half* __restrict__ scales,
    const bf16* __restrict__ fbw,
    float* __restrict__ h,
    int topk,
    float limit) {
    // h[pair, j] = w[pair] * silu(gate_j) * up_j for one (token, expert) pair; ternary experts
    // use R x (block Hadamard in registers), bf16 fallbacks use x; unrouted pairs exit.
    constexpr int WORDS = K / 16;
    constexpr int NS = K / 128;
    constexpr int EPL = BH / 32;
    constexpr int R = 2 * JPW;
    static_assert(WORDS % 32 == 0, "K/16 must be a multiple of 32");
    static_assert(K % BH == 0 && EPL >= 4, "bad Hadamard block");
    __shared__ float xs[WORDS * kWordStride];
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;");
    const int pair = blockIdx.y;
    const int t = pair / topk;
    const int code = __ldg(emap + static_cast<int>(ids[pair]));
    if (code == -1) {
        return;
    }
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const bf16* xrow = x + static_cast<int64_t>(t) * K;
    const int j0 = (static_cast<int>(blockIdx.x) * kWarpsPerCta + warp) * JPW;
    constexpr int IT = WORDS / 32;
    int pk[IT][R];
    float sc[IT][R];
    if (code >= 0 && j0 < I) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int row = r < JPW ? j0 + r : I + j0 + (r - JPW);
            const int64_t base = static_cast<int64_t>(code) * (2 * I) + row;
#pragma unroll
            for (int it = 0; it < IT; ++it) {
                const int word = it * 32 + lane;
                pk[it][r] = __ldg(codes + base * WORDS + word);
                sc[it][r] = __half2float(__ldg(scales + base * NS + (word >> 3)));
            }
        }
#pragma unroll
        for (int it = 0; it < IT; ++it) {
#pragma unroll
            for (int r = 0; r < R; ++r) {
                asm volatile("" ::"r"(pk[it][r]), "f"(sc[it][r]));
            }
        }
    }
    if (code >= 0) {
        const float inv = rsqrtf(static_cast<float>(BH));
        constexpr int NBLK = K / BH;
#pragma unroll
        for (int b0 = 0; b0 < NBLK; b0 += kWarpsPerCta) {
            const int b = b0 + warp;
            const bool ok = b < NBLK;
            const int base = (ok ? b : 0) * BH + lane * EPL;
            float v[EPL];
            load_bf16_vec<EPL>(xrow + base, v);
#pragma unroll
            for (int j = 0; j < EPL; ++j) {
                v[j] *= __ldg(signs + lane * EPL + j) * inv;
            }
            warp_fwht<EPL>(v, lane);
            if (ok) {
#pragma unroll
                for (int j = 0; j < EPL; ++j) {
                    xs[sidx(base + j)] = v[j];
                }
            }
        }
    } else {
        for (int e = threadIdx.x; e < K; e += kThreads) {
            xs[sidx(e)] = __bfloat162float(xrow[e]);
        }
    }
    __syncthreads();
    if (j0 >= I) {
        return;
    }
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        acc[r] = 0.f;
    }
    if (code >= 0) {
#pragma unroll
        for (int it = 0; it < IT; ++it) {
            float xv[16];
            load_word(xs, it * 32 + lane, xv);
#pragma unroll
            for (int r = 0; r < R; ++r) {
                acc[r] = fmaf(trit_dot(pk[it][r], xv), sc[it][r], acc[r]);
            }
        }
    } else {
        const int f = -code - 2;
        const bf16* wr[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int row = r < JPW ? j0 + r : I + j0 + (r - JPW);
            wr[r] = fbw + (static_cast<int64_t>(f) * (2 * I) + row) * K;
        }
#pragma unroll
        for (int it = 0; it < WORDS / 32; ++it) {
            const int word = it * 32 + lane;
            float xv[16];
            load_word(xs, word, xv);
#pragma unroll
            for (int r = 0; r < R; ++r) {
                acc[r] += bf16_dot16(wr[r] + word * 16, xv);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        acc[r] = warp_sum(acc[r]);
    }
    if (lane == 0) {
        const float wgt = topw[pair];
#pragma unroll
        for (int jj = 0; jj < JPW; ++jj) {
            float g = acc[jj];
            float u = acc[JPW + jj];
            if (limit > 0.f) {
                u = fminf(fmaxf(u, -limit), limit);
                g = fminf(g, limit);
            }
            h[static_cast<int64_t>(pair) * I + j0 + jj] = wgt * (g / (1.f + __expf(-g))) * u;
        }
    }
}

template <typename IdT, int N, int KI, int BI>
__global__ void __launch_bounds__(kThreads) moe_down_kernel(
    const float* __restrict__ h,
    const IdT* __restrict__ ids,
    const int* __restrict__ emap,
    const float* __restrict__ signs,
    const int* __restrict__ codes,
    const __half* __restrict__ scales,
    const bf16* __restrict__ fbw,
    bf16* __restrict__ out,
    int topk) {
    // out[t, n] = sum over routed pairs of W2[e] (R h) (ternary) or W2_bf16[f] h (fallback).
    constexpr int WORDS = KI / 16;
    constexpr int NS = KI / 128;
    constexpr int EPL = BI / 32;
    constexpr int NB = KI / BI;
    constexpr int RPW = 2;
    constexpr int TASKS = RPW * WORDS;
    static_assert(KI % BI == 0 && EPL >= 4, "bad Hadamard block");
    extern __shared__ float hs[];
    __shared__ int pcode[kMaxTopK];
    __shared__ int pslot[kMaxTopK];
    __shared__ int nact;
    const int t = blockIdx.y;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    if (warp == 0) {
        int c = -1;
        if (lane < topk) {
            c = __ldg(emap + static_cast<int>(ids[t * topk + lane]));
        }
        const unsigned live = __ballot_sync(0xffffffffu, c != -1);
        if (c != -1) {
            const int pos = __popc(live & ((1u << lane) - 1u));
            pcode[pos] = c;
            pslot[pos] = lane;
        }
        if (lane == 0) {
            nact = __popc(live);
        }
    }
    __syncthreads();
    const int na = nact;
    const int n0 = (static_cast<int>(blockIdx.x) * kWarpsPerCta + warp) * RPW;
    const int total = na * TASKS;
    int pk[kDownPrefetch];
    float sc[kDownPrefetch];
#pragma unroll
    for (int i = 0; i < kDownPrefetch; ++i) {
        const int task = lane + 32 * i;
        pk[i] = 0x55555555;
        sc[i] = 0.f;
        if (task < total) {
            const int a = task / TASKS;
            const int rem = task - a * TASKS;
            const int r = rem / WORDS;
            const int w = rem - r * WORDS;
            const int c = pcode[a];
            if (c >= 0) {
                const int64_t row = static_cast<int64_t>(c) * N + n0 + r;
                pk[i] = __ldg(codes + row * WORDS + w);
                sc[i] = __half2float(__ldg(scales + row * NS + (w >> 3)));
            }
        }
    }
#pragma unroll
    for (int i = 0; i < kDownPrefetch; ++i) {
        asm volatile("" ::"r"(pk[i]), "f"(sc[i]));
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
    const float inv = rsqrtf(static_cast<float>(BI));
    float sg[EPL];
#pragma unroll
    for (int j = 0; j < EPL; ++j) {
        sg[j] = __ldg(signs + lane * EPL + j) * inv;
    }
    constexpr int UB = 4;
    for (int task0 = warp; task0 < na * NB; task0 += kWarpsPerCta * UB) {
        float v[UB][EPL];
#pragma unroll
        for (int u = 0; u < UB; ++u) {
            const int task = task0 + u * kWarpsPerCta;
            if (task < na * NB) {
                const int a = task / NB;
                const int b = task - a * NB;
                const float* src = h + static_cast<int64_t>(t * topk + pslot[a]) * KI + b * BI + lane * EPL;
#pragma unroll
                for (int q = 0; q < EPL / 4; ++q) {
                    float4 f = __ldcg(reinterpret_cast<const float4*>(src + 4 * q));
                    v[u][4 * q] = f.x;
                    v[u][4 * q + 1] = f.y;
                    v[u][4 * q + 2] = f.z;
                    v[u][4 * q + 3] = f.w;
                }
            }
        }
#pragma unroll
        for (int u = 0; u < UB; ++u) {
            const int task = task0 + u * kWarpsPerCta;
            if (task < na * NB) {
                const int a = task / NB;
                const int b = task - a * NB;
                if (pcode[a] >= 0) {
#pragma unroll
                    for (int j = 0; j < EPL; ++j) {
                        v[u][j] *= sg[j];
                    }
                    warp_fwht<EPL>(v[u], lane);
                }
                float* dst = hs + a * (WORDS * kWordStride);
                const int base = b * BI + lane * EPL;
#pragma unroll
                for (int j = 0; j < EPL; ++j) {
                    dst[sidx(base + j)] = v[u][j];
                }
            }
        }
    }
    __syncthreads();
    float acc0 = 0.f;
    float acc1 = 0.f;
#pragma unroll
    for (int i = 0; i < kDownPrefetch; ++i) {
        const int task = lane + 32 * i;
        if (task < total) {
            const int a = task / TASKS;
            const int rem = task - a * TASKS;
            const int r = rem / WORDS;
            const int w = rem - r * WORDS;
            const int c = pcode[a];
            float xv[16];
            load_word(hs + a * (WORDS * kWordStride), w, xv);
            float d;
            if (c >= 0) {
                d = trit_dot(pk[i], xv) * sc[i];
            } else {
                d = bf16_dot16(fbw + (static_cast<int64_t>(-c - 2) * N + n0 + r) * KI + w * 16, xv);
            }
            if (r == 0) {
                acc0 += d;
            } else {
                acc1 += d;
            }
        }
    }
#pragma unroll 4
    for (int task = lane + 32 * kDownPrefetch; task < total; task += 32) {
        const int a = task / TASKS;
        const int rem = task - a * TASKS;
        const int r = rem / WORDS;
        const int w = rem - r * WORDS;
        const int n = n0 + r;
        const int c = pcode[a];
        float xv[16];
        load_word(hs + a * (WORDS * kWordStride), w, xv);
        float d;
        if (c >= 0) {
            const int64_t row = static_cast<int64_t>(c) * N + n;
            const int pk = __ldg(codes + row * WORDS + w);
            const float sc = __half2float(__ldg(scales + row * NS + (w >> 3)));
            d = trit_dot(pk, xv) * sc;
        } else {
            d = bf16_dot16(fbw + (static_cast<int64_t>(-c - 2) * N + n) * KI + w * 16, xv);
        }
        if (r == 0) {
            acc0 += d;
        } else {
            acc1 += d;
        }
    }
    acc0 = warp_sum(acc0);
    acc1 = warp_sum(acc1);
    if (lane == 0) {
        out[static_cast<int64_t>(t) * N + n0] = __float2bfloat16(acc0);
        out[static_cast<int64_t>(t) * N + n0 + 1] = __float2bfloat16(acc1);
    }
}

void check_cuda_launch() {
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));
}

template <typename IdT, int K, int I, int BH, int BI>
void launch_moe(
    const torch::Tensor& x,
    const torch::Tensor& ids,
    const torch::Tensor& topw,
    const torch::Tensor& emap,
    const torch::Tensor& signs_h,
    const torch::Tensor& signs_i,
    const torch::Tensor& w13c,
    const torch::Tensor& w13s,
    const torch::Tensor& w2c,
    const torch::Tensor& w2s,
    const torch::Tensor& fb13,
    const torch::Tensor& fb2,
    torch::Tensor& h,
    torch::Tensor& out,
    int T,
    int topk,
    float limit) {
    auto stream = at::cuda::getCurrentCUDAStream();
    const bf16* fb13p = fb13.numel() ? reinterpret_cast<const bf16*>(fb13.data_ptr<at::BFloat16>()) : nullptr;
    const bf16* fb2p = fb2.numel() ? reinterpret_cast<const bf16*>(fb2.data_ptr<at::BFloat16>()) : nullptr;
    dim3 g1(I / (kWarpsPerCta * 2), T * topk);
    static const bool pdl_gu = [] {
        const char* v = std::getenv("TERNARY_PDL");
        return v == nullptr || std::strcmp(v, "0") != 0;
    }();
    cudaLaunchConfig_t cfg1 = {};
    cfg1.gridDim = g1;
    cfg1.blockDim = dim3(kThreads);
    cfg1.stream = stream;
    cudaLaunchAttribute attr1[1];
    attr1[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr1[0].val.programmaticStreamSerializationAllowed = 1;
    cfg1.attrs = attr1;
    cfg1.numAttrs = pdl_gu ? 1 : 0;
    cudaError_t egu = cudaLaunchKernelEx(
        &cfg1,
        moe_gate_up_kernel<IdT, K, I, BH, 2>,
        reinterpret_cast<const bf16*>(x.data_ptr<at::BFloat16>()),
        ids.data_ptr<IdT>(),
        topw.data_ptr<float>(),
        emap.data_ptr<int>(),
        signs_h.data_ptr<float>(),
        w13c.data_ptr<int>(),
        reinterpret_cast<const __half*>(w13s.data_ptr<at::Half>()),
        fb13p,
        h.data_ptr<float>(),
        topk,
        limit);
    TORCH_CHECK(egu == cudaSuccess, cudaGetErrorString(egu));
    check_cuda_launch();
    dim3 g2(K / (kWarpsPerCta * 2), T);
    size_t smem = static_cast<size_t>(topk) * (I / 16) * kWordStride * sizeof(float);
    static const bool pdl = [] {
        const char* v = std::getenv("TERNARY_PDL");
        return v == nullptr || std::strcmp(v, "0") != 0;
    }();
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = g2;
    cfg.blockDim = dim3(kThreads);
    cfg.dynamicSmemBytes = smem;
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = pdl ? 1 : 0;
    cudaError_t err = cudaLaunchKernelEx(
        &cfg,
        moe_down_kernel<IdT, K, I, BI>,
        static_cast<const float*>(h.data_ptr<float>()),
        static_cast<const IdT*>(ids.data_ptr<IdT>()),
        static_cast<const int*>(emap.data_ptr<int>()),
        static_cast<const float*>(signs_i.data_ptr<float>()),
        static_cast<const int*>(w2c.data_ptr<int>()),
        reinterpret_cast<const __half*>(w2s.data_ptr<at::Half>()),
        fb2p,
        reinterpret_cast<bf16*>(out.data_ptr<at::BFloat16>()),
        topk);
    TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));
    check_cuda_launch();
}

torch::Tensor moe_decode(
    torch::Tensor x,
    torch::Tensor ids,
    torch::Tensor topw,
    torch::Tensor emap,
    torch::Tensor signs_h,
    torch::Tensor signs_i,
    torch::Tensor w13c,
    torch::Tensor w13s,
    torch::Tensor w2c,
    torch::Tensor w2s,
    torch::Tensor fb13,
    torch::Tensor fb2,
    double limit) {
    // x [T, K] bf16; ids [T, k] int32/int64; topw [T, k] fp32; emap [E] int32
    // (slot >= 0 ternary, -1 unrouted, -(f+2) bf16 fallback f). Returns [T, K] bf16.
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kBFloat16 && x.dim() == 2, "x must be contiguous CUDA bf16 [T, K]");
    TORCH_CHECK(ids.is_contiguous() && ids.dim() == 2 && ids.size(0) == x.size(0), "ids must be [T, k]");
    TORCH_CHECK(topw.is_contiguous() && topw.scalar_type() == torch::kFloat32 && topw.sizes() == ids.sizes(), "topw must be fp32 [T, k]");
    TORCH_CHECK(emap.is_contiguous() && emap.scalar_type() == torch::kInt32, "emap must be int32");
    TORCH_CHECK(w13c.is_contiguous() && w13s.is_contiguous() && w2c.is_contiguous() && w2s.is_contiguous(), "weights must be contiguous");
    TORCH_CHECK(signs_h.scalar_type() == torch::kFloat32 && signs_i.scalar_type() == torch::kFloat32, "signs must be fp32");
    const int T = static_cast<int>(x.size(0));
    const int K = static_cast<int>(x.size(1));
    const int topk = static_cast<int>(ids.size(1));
    const int I = static_cast<int>(w2c.size(2) * 16);
    TORCH_CHECK(topk >= 1 && topk <= kMaxTopK, "top_k must be in 1..32");
    TORCH_CHECK(w13c.size(1) == 2 * I && w13c.size(2) * 16 == K && w2c.size(1) == K, "weight shapes do not match x");
    if (fb13.numel()) {
        TORCH_CHECK(fb13.is_contiguous() && fb2.is_contiguous() && fb13.scalar_type() == torch::kBFloat16 && fb2.scalar_type() == torch::kBFloat16, "fallbacks must be contiguous bf16");
        TORCH_CHECK(fb13.size(1) == 2 * I && fb13.size(2) == K && fb2.size(1) == K && fb2.size(2) == I, "fallback shapes do not match");
    }
    auto out = torch::empty({T, K}, x.options());
    if (T == 0) {
        return out;
    }
    auto h = torch::empty({static_cast<int64_t>(T) * topk, I}, x.options().dtype(torch::kFloat32));
    const int bh = static_cast<int>(signs_h.numel());
    const int bi = static_cast<int>(signs_i.numel());
    const float lim = static_cast<float>(limit);
    const bool i64 = ids.scalar_type() == torch::kInt64;
    TORCH_CHECK(i64 || ids.scalar_type() == torch::kInt32, "ids must be int32 or int64");
    if (K == 2560 && I == 640 && bh == 512 && bi == 128) {
        if (i64) {
            launch_moe<int64_t, 2560, 640, 512, 128>(x, ids, topw, emap, signs_h, signs_i, w13c, w13s, w2c, w2s, fb13, fb2, h, out, T, topk, lim);
        } else {
            launch_moe<int32_t, 2560, 640, 512, 128>(x, ids, topw, emap, signs_h, signs_i, w13c, w13s, w2c, w2s, fb13, fb2, h, out, T, topk, lim);
        }
    } else {
        TORCH_CHECK(false, "moe_decode: no specialization for K=", K, " I=", I, " BH=", bh, " BI=", bi);
    }
    return out;
}

template <int MM>
__global__ void __launch_bounds__(kThreads) hc_down_kernel(
    const bf16* __restrict__ x,
    const bf16* __restrict__ wd,
    bf16* __restrict__ tout,
    int M,
    int KH,
    int R,
    float inv_hc) {
    // tout[m, n] = bf16(silu(inv_hc * x[m] . wd[n])); two rows per CTA, four warps per row.
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;");
    __shared__ float red[2][4][MM];
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int half = warp >> 2;
    const int q = warp & 3;
    const int n = blockIdx.x * 2 + half;
    const int kc = KH / 4;
    const bf16* wr = wd + static_cast<int64_t>(n) * KH + q * kc;
    const bf16* xb = x + q * kc;
    float acc[MM];
#pragma unroll
    for (int m = 0; m < MM; ++m) {
        acc[m] = 0.f;
    }
#pragma unroll 4
    for (int k = lane * 8; k < kc; k += 256) {
        float wf[8];
        uint4 raw = __ldg(reinterpret_cast<const uint4*>(wr + k));
        const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 f = __bfloat1622float2(b2[j]);
            wf[2 * j] = f.x;
            wf[2 * j + 1] = f.y;
        }
#pragma unroll
        for (int m = 0; m < MM; ++m) {
            if (m < M) {
                uint4 xr = __ldg(reinterpret_cast<const uint4*>(xb + static_cast<int64_t>(m) * KH + k));
                const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(&xr);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float2 f = __bfloat1622float2(x2[j]);
                    acc[m] = fmaf(wf[2 * j], f.x, acc[m]);
                    acc[m] = fmaf(wf[2 * j + 1], f.y, acc[m]);
                }
            }
        }
    }
#pragma unroll
    for (int m = 0; m < MM; ++m) {
        float v = warp_sum(acc[m]);
        if (lane == 0) {
            red[half][q][m] = v;
        }
    }
    __syncthreads();
    if (threadIdx.x < 2 * MM) {
        const int hh = threadIdx.x / MM;
        const int m = threadIdx.x - hh * MM;
        const int nn = blockIdx.x * 2 + hh;
        if (m < M && nn < R) {
            float a = (red[hh][0][m] + red[hh][1][m] + red[hh][2][m] + red[hh][3][m]) * inv_hc;
            tout[static_cast<int64_t>(m) * R + nn] = __float2bfloat16(a / (1.f + __expf(-a)));
        }
    }
}

template <int MM>
__global__ void __launch_bounds__(kThreads) hc_up_kernel(
    const bf16* __restrict__ tin,
    const bf16* __restrict__ wu,
    const bf16* __restrict__ x,
    bf16* __restrict__ out,
    int M,
    int R,
    int HS,
    float inv_hc) {
    // out[m, j] = inv_hc * sum_g sigmoid(wu[g*HS+j] . t[m]) * x[m, g*HS+j]; HC = 4.
    extern __shared__ float ts[];
    asm volatile("griddepcontrol.wait;");
    for (int q = threadIdx.x; q < M * R; q += kThreads) {
        ts[q] = __bfloat162float(tin[q]);
    }
    __syncthreads();
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int g = lane >> 3;
    const int c0 = lane & 7;
    const int chunks = R / 8;
#pragma unroll
    for (int jj = 0; jj < 2; ++jj) {
        const int j = (blockIdx.x * kWarpsPerCta + warp) * 2 + jj;
        if (j >= HS) {
            break;
        }
        const bf16* wrow = wu + (static_cast<int64_t>(g) * HS + j) * R;
        float acc[MM];
#pragma unroll
        for (int m = 0; m < MM; ++m) {
            acc[m] = 0.f;
        }
        for (int c = c0; c < chunks; c += 8) {
            float wf[8];
            uint4 raw = __ldg(reinterpret_cast<const uint4*>(wrow + c * 8));
            const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                float2 f = __bfloat1622float2(b2[q]);
                wf[2 * q] = f.x;
                wf[2 * q + 1] = f.y;
            }
#pragma unroll
            for (int m = 0; m < MM; ++m) {
                if (m < M) {
                    const float* tv = ts + m * R + c * 8;
#pragma unroll
                    for (int q = 0; q < 8; ++q) {
                        acc[m] = fmaf(wf[q], tv[q], acc[m]);
                    }
                }
            }
        }
#pragma unroll
        for (int m = 0; m < MM; ++m) {
            float v = acc[m];
            v += __shfl_xor_sync(0xffffffffu, v, 1);
            v += __shfl_xor_sync(0xffffffffu, v, 2);
            v += __shfl_xor_sync(0xffffffffu, v, 4);
            float y = 0.f;
            if (m < M) {
                const float gate = 1.f / (1.f + __expf(-v));
                y = gate * __bfloat162float(x[static_cast<int64_t>(m) * 4 * HS + g * HS + j]);
            }
            y += __shfl_xor_sync(0xffffffffu, y, 8);
            y += __shfl_xor_sync(0xffffffffu, y, 16);
            if (lane == 0 && m < M) {
                out[static_cast<int64_t>(m) * HS + j] = __float2bfloat16(y * inv_hc);
            }
        }
    }
}

template <int MM>
void launch_hc(const torch::Tensor& x, const torch::Tensor& wd, const torch::Tensor& wu, torch::Tensor& part, torch::Tensor& out, int M, int KH, int R, int HS, float inv_hc) {
    auto stream = at::cuda::getCurrentCUDAStream();
    static const bool pdl = [] {
        const char* v = std::getenv("TERNARY_PDL");
        return v == nullptr || std::strcmp(v, "0") != 0;
    }();
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t cfg1 = {};
    cfg1.gridDim = dim3(R / 2);
    cfg1.blockDim = dim3(kThreads);
    cfg1.stream = stream;
    cfg1.attrs = attr;
    cfg1.numAttrs = pdl ? 1 : 0;
    cudaError_t e1 = cudaLaunchKernelEx(
        &cfg1,
        hc_down_kernel<MM>,
        reinterpret_cast<const bf16*>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<const bf16*>(wd.data_ptr<at::BFloat16>()),
        reinterpret_cast<bf16*>(part.data_ptr<at::BFloat16>()),
        M, KH, R, inv_hc);
    TORCH_CHECK(e1 == cudaSuccess, cudaGetErrorString(e1));
    check_cuda_launch();
    dim3 g2((HS + kWarpsPerCta * 2 - 1) / (kWarpsPerCta * 2));
    size_t smem = static_cast<size_t>(M) * R * sizeof(float);
    cudaLaunchConfig_t cfg2 = {};
    cfg2.gridDim = g2;
    cfg2.blockDim = dim3(kThreads);
    cfg2.dynamicSmemBytes = smem;
    cfg2.stream = stream;
    cfg2.attrs = attr;
    cfg2.numAttrs = pdl ? 1 : 0;
    cudaError_t e2 = cudaLaunchKernelEx(
        &cfg2,
        hc_up_kernel<MM>,
        reinterpret_cast<const bf16*>(part.data_ptr<at::BFloat16>()),
        reinterpret_cast<const bf16*>(wu.data_ptr<at::BFloat16>()),
        reinterpret_cast<const bf16*>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<bf16*>(out.data_ptr<at::BFloat16>()),
        M, R, HS, inv_hc);
    TORCH_CHECK(e2 == cudaSuccess, cudaGetErrorString(e2));
    check_cuda_launch();
}

torch::Tensor hc_mix(torch::Tensor x, torch::Tensor wd, torch::Tensor wu, int64_t hc, int64_t hs) {
    // x [M, 4*HS] bf16, wd [R, 4*HS] bf16, wu [4*HS, R] bf16 -> [M, HS] bf16; M <= 16.
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && wd.is_contiguous() && wu.is_contiguous(), "contiguous CUDA tensors required");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16 && wd.scalar_type() == torch::kBFloat16 && wu.scalar_type() == torch::kBFloat16, "bf16 required");
    TORCH_CHECK(hc == 4, "hc_mix: hc must be 4");
    const int M = static_cast<int>(x.size(0));
    const int KH = static_cast<int>(x.size(1));
    const int R = static_cast<int>(wd.size(0));
    const int HS = static_cast<int>(hs);
    TORCH_CHECK(KH == 4 * HS && wd.size(1) == KH && wu.size(0) == KH && wu.size(1) == R, "hc_mix shape mismatch");
    TORCH_CHECK(M <= kHcMaxRows, "hc_mix: M must be <= 16");
    TORCH_CHECK(KH % 1024 == 0 && R % 64 == 0, "hc_mix: KH % 1024 and R % 64 required");
    auto out = torch::empty({M, HS}, x.options());
    if (M == 0) {
        return out;
    }
    auto part = torch::empty({M, R}, x.options());
    const float inv_hc = 1.f / static_cast<float>(hc);
    if (M == 1) {
        launch_hc<1>(x, wd, wu, part, out, M, KH, R, HS, inv_hc);
    } else if (M <= 2) {
        launch_hc<2>(x, wd, wu, part, out, M, KH, R, HS, inv_hc);
    } else if (M <= 4) {
        launch_hc<4>(x, wd, wu, part, out, M, KH, R, HS, inv_hc);
    } else if (M <= 8) {
        launch_hc<8>(x, wd, wu, part, out, M, KH, R, HS, inv_hc);
    } else {
        launch_hc<16>(x, wd, wu, part, out, M, KH, R, HS, inv_hc);
    }
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("moe_decode", &moe_decode);
    m.def("hc_mix", &hc_mix);
}
