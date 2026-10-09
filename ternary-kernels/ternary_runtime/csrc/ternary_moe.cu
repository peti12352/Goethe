#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>
#include <cstring>

constexpr int kWarps = 8;
constexpr int kRowsPerWarp = 8;
constexpr int kRowsPerBlock = kWarps * kRowsPerWarp;
constexpr int kChunk = 512;
constexpr int kXStride = 17;
constexpr int kChunkStride = 32 * kXStride;

__device__ __forceinline__ int x_index(int elem) {
    int chunk = elem >> 9;
    int inn = elem & (kChunk - 1);
    int owner = inn >> 4;
    int sub = inn & 15;
    return chunk * kChunkStride + owner * kXStride + sub;
}

__device__ __forceinline__ int unpack_w0(int w) {
    return w & 0x03030303;
}

__device__ __forceinline__ int unpack_w1(int w) {
    return (w >> 2) & 0x03030303;
}

__device__ __forceinline__ int unpack_w2(int w) {
    return (w >> 4) & 0x03030303;
}

__device__ __forceinline__ int unpack_w3(int w) {
    return (w >> 6) & 0x03030303;
}

__device__ __forceinline__ int dp4a_word(int w, int x0, int x1, int x2, int x3) {
    return __dp4a(unpack_w0(w), x0, 0) + __dp4a(unpack_w1(w), x1, 0) + __dp4a(unpack_w2(w), x2, 0) +
           __dp4a(unpack_w3(w), x3, 0);
}

__device__ __forceinline__ void store_out(void* out, int64_t idx, float v, int kind) {
    if (kind == 0) {
        reinterpret_cast<float*>(out)[idx] = v;
    } else if (kind == 1) {
        reinterpret_cast<__half*>(out)[idx] = __float2half(v);
    } else {
        reinterpret_cast<__nv_bfloat16*>(out)[idx] = __float2bfloat16(v);
    }
}

template <bool kBf16>
__device__ __forceinline__ float load_act(const void* x, int64_t idx) {
    if constexpr (kBf16) {
        return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(x)[idx]);
    } else {
        return reinterpret_cast<const float*>(x)[idx];
    }
}

template <bool kBf16>
__global__ void rotate_kernel(const void* __restrict__ x, const float* __restrict__ signs, float* __restrict__ y, int K, int block) {
    // One CTA per (row, Hadamard block) with block/2 threads; one butterfly per thread per stage.
    extern __shared__ float smem[];
    int half = block >> 1;
    int t = threadIdx.x;
    int64_t base = static_cast<int64_t>(blockIdx.x) * K + static_cast<int64_t>(blockIdx.y) * block;
    smem[t] = load_act<kBf16>(x, base + t) * signs[t];
    smem[t + half] = load_act<kBf16>(x, base + t + half) * signs[t + half];
    __syncthreads();
    for (int h = 1; h < block; h <<= 1) {
        int i = ((t & ~(h - 1)) << 1) | (t & (h - 1));
        float a = smem[i];
        float b = smem[i + h];
        smem[i] = a + b;
        smem[i + h] = a - b;
        __syncthreads();
    }
    float inv = 1.0f / sqrtf(static_cast<float>(block));
    y[base + t] = smem[t] * inv;
    y[base + t + half] = smem[t + half] * inv;
}

__device__ __forceinline__ int quant_one(float v, float scale) {
    return scale > 0.f ? static_cast<int>(roundf(fminf(fmaxf(v / scale, -127.f), 127.f))) : 0;
}

template <int kSpec>
__global__ void quant_x_kernel(
    const float* __restrict__ xr,
    int* __restrict__ xq,
    float* __restrict__ xs,
    int* __restrict__ xsum,
    int Kruntime) {
    // One warp per 128-element group; lane L owns elements 4L..4L+3. Output word
    // 32g+L holds byte b = element (L&3) of lane (L&~3)+b, the dp4a x permutation.
    const int K = kSpec > 0 ? kSpec : Kruntime;
    const int groups = K >> 7;
    int lane = threadIdx.x;
    int g = static_cast<int>(blockIdx.x) * kWarps + static_cast<int>(threadIdx.y);
    int64_t row = blockIdx.y;
    if (g >= groups) {
        return;
    }
    float4 v = *reinterpret_cast<const float4*>(xr + row * K + (g << 7) + (lane << 2));
    float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off));
    }
    float scale = amax > 0.f ? amax / 127.f : 0.f;
    int q0 = quant_one(v.x, scale);
    int q1 = quant_one(v.y, scale);
    int q2 = quant_one(v.z, scale);
    int q3 = quant_one(v.w, scale);
    int sum = q0 + q1 + q2 + q3;
    sum += __shfl_xor_sync(0xffffffff, sum, 1);
    sum += __shfl_xor_sync(0xffffffff, sum, 2);
    sum += __shfl_xor_sync(0xffffffff, sum, 4);
    if ((lane & 7) == 0) {
        xsum[row * (K >> 5) + (g << 2) + (lane >> 3)] = sum;
    }
    if (lane == 0) {
        xs[row * groups + g] = scale;
    }
    int packed = (q0 & 0xff) | ((q1 & 0xff) << 8) | ((q2 & 0xff) << 16) | ((q3 & 0xff) << 24);
    int base = lane & ~3;
    unsigned j = static_cast<unsigned>(lane & 3);
    int p0 = __shfl_sync(0xffffffff, packed, base);
    int p1 = __shfl_sync(0xffffffff, packed, base + 1);
    int p2 = __shfl_sync(0xffffffff, packed, base + 2);
    int p3 = __shfl_sync(0xffffffff, packed, base + 3);
    unsigned sel = j | ((j + 4) << 4);
    int lo = static_cast<int>(__byte_perm(p0, p1, sel));
    int hi = static_cast<int>(__byte_perm(p2, p3, sel));
    xq[row * (K >> 2) + (g << 5) + lane] = static_cast<int>(__byte_perm(lo, hi, 0x5410));
}

template <typename TokT, int kSpec>
__global__ void gemv_dp4a_kernel(
    const int* __restrict__ xq,
    const float* __restrict__ xs,
    const int* __restrict__ xsum,
    const TokT* __restrict__ tok,
    const int* __restrict__ lid,
    const int* __restrict__ codes,
    const __half* __restrict__ scales,
    void* __restrict__ out,
    int N,
    int Kruntime,
    int pair0,
    int kind) {
    const int K = kSpec > 0 ? kSpec : Kruntime;
    const int words = K >> 4;
    const int xq_words = K >> 2;
    const int xsum_words = K >> 5;
    const int nscale = K >> 7;
    const int nchunk = K >> 5;
    int pair = pair0 + static_cast<int>(blockIdx.y);
    int lane = threadIdx.x;
    int warp = threadIdx.y;
    int row0 = (static_cast<int>(blockIdx.x) * kWarps + warp) * kRowsPerWarp;
    int lid_v = __ldg(lid + pair);
    if (lid_v < 0) {
        if (lane == 0) {
            #pragma unroll
            for (int r = 0; r < kRowsPerWarp; ++r) {
                int row = row0 + r;
                if (row < N) {
                    store_out(out, static_cast<int64_t>(pair) * N + row, 0.f, kind);
                }
            }
        }
        return;
    }
    int64_t token = static_cast<int64_t>(tok[pair]);
    const int* xq_base = xq + token * xq_words;
    const float* xs_base = xs + token * nscale;
    const int* xsum_base = xsum + token * xsum_words;
    const int* wrow[kRowsPerWarp];
    const __half* srow[kRowsPerWarp];
    float acc[kRowsPerWarp];
    #pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
        int row = row0 + r;
        acc[r] = 0.f;
        if (row < N) {
            wrow[r] = codes + (static_cast<int64_t>(lid_v) * N + row) * words;
            srow[r] = scales + (static_cast<int64_t>(lid_v) * N + row) * nscale;
        } else {
            wrow[r] = codes;
            srow[r] = scales;
        }
    }
    for (int c = lane; c < nchunk; c += 32) {
        int xq_off = c << 3;
        int4 xpack0 = *reinterpret_cast<const int4*>(xq_base + xq_off);
        int4 xpack1 = *reinterpret_cast<const int4*>(xq_base + xq_off + 4);
        int sq = __ldg(xsum_base + c);
        float sx = __ldg(xs_base + (c >> 2));
        int wc0 = c << 1;
        int wc1 = wc0 + 1;
        #pragma unroll
        for (int r = 0; r < kRowsPerWarp; ++r) {
            if (row0 + r >= N) {
                continue;
            }
            int w0 = __ldg(wrow[r] + wc0);
            int w1 = __ldg(wrow[r] + wc1);
            float sc = __half2float(__ldg(srow[r] + (wc0 >> 3)));
            int a0 = dp4a_word(w0, xpack0.x, xpack0.y, xpack0.z, xpack0.w);
            int a1 = dp4a_word(w1, xpack1.x, xpack1.y, xpack1.z, xpack1.w);
            acc[r] = fmaf(__int2float_rn(a0 + a1 - sq), sx * sc, acc[r]);
        }
    }
    #pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
        float sum = acc[r];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, off);
        }
        if (lane == 0 && row0 + r < N) {
            store_out(out, static_cast<int64_t>(pair) * N + (row0 + r), sum, kind);
        }
    }
}

template <typename TokT, int kSpec>
__global__ void gemv_fp32_kernel(
    const float* __restrict__ xr,
    const TokT* __restrict__ tok,
    const int* __restrict__ lid,
    const int* __restrict__ codes,
    const __half* __restrict__ scales,
    void* __restrict__ out,
    int N,
    int Kruntime,
    int pair0,
    int kind) {
    const int K = kSpec > 0 ? kSpec : Kruntime;
    const int words = K >> 4;
    const int nscale = K >> 7;
    extern __shared__ float smem[];
    int pair = pair0 + static_cast<int>(blockIdx.y);
    int lane = threadIdx.x;
    int warp = threadIdx.y;
    int row0 = (static_cast<int>(blockIdx.x) * kWarps + warp) * kRowsPerWarp;
    int lid_v = __ldg(lid + pair);
    if (lid_v < 0) {
        if (lane == 0) {
            #pragma unroll
            for (int r = 0; r < kRowsPerWarp; ++r) {
                int row = row0 + r;
                if (row < N) {
                    store_out(out, static_cast<int64_t>(pair) * N + row, 0.f, kind);
                }
            }
        }
        return;
    }
    int64_t token = static_cast<int64_t>(tok[pair]);
    const float* x = xr + token * K;
    int tid = warp * 32 + lane;
    for (int elem = tid; elem < K; elem += kWarps * 32) {
        smem[x_index(elem)] = __ldg(x + elem);
    }
    __syncthreads();
    const int* wrow[kRowsPerWarp];
    const __half* srow[kRowsPerWarp];
    float acc[kRowsPerWarp];
    #pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
        int row = row0 + r;
        acc[r] = 0.f;
        if (row < N) {
            wrow[r] = codes + (static_cast<int64_t>(lid_v) * N + row) * words;
            srow[r] = scales + (static_cast<int64_t>(lid_v) * N + row) * nscale;
        } else {
            wrow[r] = codes;
            srow[r] = scales;
        }
    }
    const int niter = (words + 31) >> 5;
    for (int iter = 0; iter < niter; ++iter) {
        int word = (iter << 5) + lane;
        bool live = word < words;
        const float* xp = smem + iter * kChunkStride + lane * kXStride;
        float xv[16];
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            xv[i] = xp[i];
        }
        if (!live) {
            continue;
        }
        int packed[kRowsPerWarp];
        float scale[kRowsPerWarp];
        #pragma unroll
        for (int r = 0; r < kRowsPerWarp; ++r) {
            if (row0 + r < N) {
                packed[r] = __ldg(wrow[r] + word);
                scale[r] = __half2float(__ldg(srow[r] + (word >> 3)));
            } else {
                packed[r] = 0;
                scale[r] = 0.f;
            }
        }
        float partial[kRowsPerWarp];
        #pragma unroll
        for (int r = 0; r < kRowsPerWarp; ++r) {
            partial[r] = 0.f;
        }
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            float xval = xv[i];
            #pragma unroll
            for (int r = 0; r < kRowsPerWarp; ++r) {
                float trit = static_cast<float>(((packed[r] >> (i * 2)) & 3) - 1);
                partial[r] = fmaf(trit, xval, partial[r]);
            }
        }
        #pragma unroll
        for (int r = 0; r < kRowsPerWarp; ++r) {
            acc[r] = fmaf(partial[r], scale[r], acc[r]);
        }
    }
    #pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
        float sum = acc[r];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, off);
        }
        if (lane == 0 && row0 + r < N) {
            store_out(out, static_cast<int64_t>(pair) * N + (row0 + r), sum, kind);
        }
    }
}

__device__ __forceinline__ void store_word(__half*& ptr, int packed, float scale, int N) {
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        float trit = static_cast<float>(((packed >> (i * 2)) & 3) - 1);
        *ptr = __float2half(trit * scale);
        ptr += N;
    }
}

__global__ void dequant_kn_kernel(
    const int* __restrict__ codes,
    const __half* __restrict__ scales,
    __half* __restrict__ out,
    int N,
    int K) {
    int n = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    int kg = static_cast<int>(blockIdx.y);
    int e = static_cast<int>(blockIdx.z);
    if (n >= N) {
        return;
    }
    int words = K >> 4;
    int nscale = K >> 7;
    const int* row = codes + (static_cast<int64_t>(e) * N + n) * words + (kg << 3);
    float scale = __half2float(__ldg(scales + (static_cast<int64_t>(e) * N + n) * nscale + kg));
    int4 a = *reinterpret_cast<const int4*>(row);
    int4 b = *reinterpret_cast<const int4*>(row + 4);
    __half* ptr = out + (static_cast<int64_t>(e) * K + (static_cast<int64_t>(kg) << 7)) * N + n;
    store_word(ptr, a.x, scale, N);
    store_word(ptr, a.y, scale, N);
    store_word(ptr, a.z, scale, N);
    store_word(ptr, a.w, scale, N);
    store_word(ptr, b.x, scale, N);
    store_word(ptr, b.y, scale, N);
    store_word(ptr, b.z, scale, N);
    store_word(ptr, b.w, scale, N);
}

constexpr int kFbRows = 1;
constexpr int kFbWarps = 4;
constexpr int kFbMaxT = 8;

__global__ void __launch_bounds__(kFbWarps * 32) fb_gemv_kernel(
    const __nv_bfloat16* __restrict__ x,
    long long x_fstride,
    const __nv_bfloat16* __restrict__ w,
    const float* __restrict__ coef,
    __nv_bfloat16* __restrict__ out,
    int F,
    int T,
    int N,
    int K) {
    // out[f, t, n] = bf16(sum_k x[f?, t, k] * w[f, n, k]); zeros when coef[:, f] is all zero.
    const int f = blockIdx.y;
    const int lane = threadIdx.x;
    const int tid = threadIdx.y * 32 + lane;
    const int row0 = (blockIdx.x * kFbWarps + threadIdx.y) * kFbRows;
    int hit = 0;
    for (int t = tid; t < T; t += kFbWarps * 32) {
        hit |= coef[static_cast<long long>(t) * F + f] != 0.f;
    }
    if (!__syncthreads_or(hit)) {
        for (int i = lane; i < T * kFbRows; i += 32) {
            out[(static_cast<long long>(f) * T + i / kFbRows) * N + row0 + i % kFbRows] = __float2bfloat16(0.f);
        }
        return;
    }
    const __nv_bfloat16* xf = x + f * x_fstride;
    const __nv_bfloat16* wr = w + (static_cast<long long>(f) * N + row0) * K;
    for (int t0 = 0; t0 < T; t0 += kFbMaxT) {
        const int tn = min(kFbMaxT, T - t0);
        float acc[kFbRows][kFbMaxT];
#pragma unroll
        for (int r = 0; r < kFbRows; ++r) {
#pragma unroll
            for (int t = 0; t < kFbMaxT; ++t) {
                acc[r][t] = 0.f;
            }
        }
#pragma unroll 2
        for (int k = lane * 8; k < K; k += 256) {
            float wf[kFbRows][8];
#pragma unroll
            for (int r = 0; r < kFbRows; ++r) {
                uint4 v = __ldg(reinterpret_cast<const uint4*>(wr + static_cast<long long>(r) * K + k));
                const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    float2 q = __bfloat1622float2(p[j]);
                    wf[r][2 * j] = q.x;
                    wf[r][2 * j + 1] = q.y;
                }
            }
#pragma unroll
            for (int t = 0; t < kFbMaxT; ++t) {
                if (t < tn) {
                    uint4 v = __ldg(reinterpret_cast<const uint4*>(xf + static_cast<long long>(t0 + t) * K + k));
                    const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        float2 q = __bfloat1622float2(p[j]);
#pragma unroll
                        for (int r = 0; r < kFbRows; ++r) {
                            acc[r][t] = fmaf(wf[r][2 * j], q.x, acc[r][t]);
                            acc[r][t] = fmaf(wf[r][2 * j + 1], q.y, acc[r][t]);
                        }
                    }
                }
            }
        }
#pragma unroll
        for (int r = 0; r < kFbRows; ++r) {
#pragma unroll
            for (int t = 0; t < kFbMaxT; ++t) {
                float v = acc[r][t];
#pragma unroll
                for (int s = 16; s > 0; s >>= 1) {
                    v += __shfl_xor_sync(0xffffffffu, v, s);
                }
                if (lane == r * kFbMaxT + t && t < tn) {
                    out[(static_cast<long long>(f) * T + t0 + t) * N + row0 + r] = __float2bfloat16(v);
                }
            }
        }
    }
}

__device__ __forceinline__ float fp8_group_emulate(float v) {
    // Matches experts.act_quant_emulate: UE8M0 scale per 32 lanes, e4m3 round trip.
    float a = fabsf(v);
#pragma unroll
    for (int s = 16; s > 0; s >>= 1) {
        a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, s));
    }
    a = fmaxf(a, 1e-12f);
    float scale = exp2f(ceilf(log2f(a / 448.f)));
    float q = fminf(fmaxf(v / scale, -448.f), 448.f);
    return static_cast<float>(__nv_fp8_e4m3(q)) * scale;
}

__global__ void act_quant_kernel(const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ y, long long groups) {
    long long g = static_cast<long long>(blockIdx.x) * kFbWarps + threadIdx.y;
    if (g >= groups) {
        return;
    }
    long long i = g * 32 + threadIdx.x;
    y[i] = __float2bfloat16(fp8_group_emulate(__bfloat162float(x[i])));
}

__global__ void fb_swiglu_quant_kernel(
    const __nv_bfloat16* __restrict__ gu,
    const float* __restrict__ coef,
    __nv_bfloat16* __restrict__ h,
    int F,
    int T,
    int I,
    float limit) {
    // h[f, t, :] = fp8_emulate(bf16(coef[t, f] * silu(min(gate, L)) * clamp(up, -L, L))).
    long long g = static_cast<long long>(blockIdx.x) * kFbWarps + threadIdx.y;
    int per_row = I / 32;
    if (g >= static_cast<long long>(F) * T * per_row) {
        return;
    }
    long long row = g / per_row;
    int c = static_cast<int>(g % per_row) * 32 + threadIdx.x;
    int f = static_cast<int>(row / T);
    int t = static_cast<int>(row % T);
    const __nv_bfloat16* r = gu + row * 2 * I;
    float gate = fminf(__bfloat162float(r[c]), limit);
    float up = fminf(fmaxf(__bfloat162float(r[I + c]), -limit), limit);
    float v = coef[static_cast<long long>(t) * F + f] * ((gate / (1.f + expf(-gate))) * up);
    v = __bfloat162float(__float2bfloat16(v));
    h[row * I + c] = __float2bfloat16(fp8_group_emulate(v));
}

void check_launch() {
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, cudaGetErrorString(err));
}

int out_kind(const torch::Tensor& out) {
    if (out.scalar_type() == torch::kFloat32) {
        return 0;
    }
    if (out.scalar_type() == torch::kFloat16) {
        return 1;
    }
    TORCH_CHECK(out.scalar_type() == torch::kBFloat16, "out dtype must be fp32, fp16, or bf16");
    return 2;
}

void check_grouped_args(const torch::Tensor& xr, const torch::Tensor& tok, const torch::Tensor& lid, const torch::Tensor& codes, const torch::Tensor& scales) {
    TORCH_CHECK(xr.is_cuda() && tok.is_cuda() && lid.is_cuda() && codes.is_cuda() && scales.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(xr.is_contiguous() && tok.is_contiguous() && lid.is_contiguous() && codes.is_contiguous() && scales.is_contiguous(), "tensors must be contiguous");
    TORCH_CHECK(xr.scalar_type() == torch::kFloat32, "xr must be fp32");
    TORCH_CHECK(xr.dim() == 2, "xr must be [M, K]");
    TORCH_CHECK(tok.dim() == 1 && lid.dim() == 1 && tok.size(0) == lid.size(0), "tok and lid must be [P]");
    TORCH_CHECK(tok.scalar_type() == torch::kInt32 || tok.scalar_type() == torch::kInt64, "tok must be int32 or int64");
    TORCH_CHECK(lid.scalar_type() == torch::kInt32, "lid must be int32");
    TORCH_CHECK(codes.dim() == 3 && scales.dim() == 3, "codes and scales must be rank 3");
    TORCH_CHECK(codes.scalar_type() == torch::kInt32, "codes must be int32");
    TORCH_CHECK(scales.scalar_type() == torch::kFloat16, "scales must be fp16");
    TORCH_CHECK(codes.size(0) == scales.size(0) && codes.size(1) == scales.size(1), "codes and scales expert/row dims differ");
    int K = static_cast<int>(codes.size(2) * 16);
    TORCH_CHECK(xr.size(1) == K, "xr K does not match codes");
    TORCH_CHECK(K % 128 == 0, "K must be a multiple of 128");
    TORCH_CHECK(scales.size(2) * 128 == K, "scales group dim must be K/128");
    TORCH_CHECK(xr.get_device() == codes.get_device(), "tensors must share a device");
}

bool decode_use_int8() {
    // int8 activations + dp4a; lossy vs the exact fp32 path, opt-in via TERNARY_DECODE=int8.
    const char* mode = std::getenv("TERNARY_DECODE");
    return mode != nullptr && std::strcmp(mode, "int8") == 0;
}

void launch_quant_x(const float* xr, int* xq, float* xs, int* xsum, int M, int K) {
    int groups = K >> 7;
    dim3 block(32, kWarps);
    dim3 grid((groups + kWarps - 1) / kWarps, M);
    auto stream = at::cuda::getCurrentCUDAStream();
    // Specialized Ks: deepseek (5120,2304), flash_next (2560,640), bonsai (5120,17408).
    if (K == 5120) {
        quant_x_kernel<5120><<<grid, block, 0, stream>>>(xr, xq, xs, xsum, K);
    } else if (K == 2304) {
        quant_x_kernel<2304><<<grid, block, 0, stream>>>(xr, xq, xs, xsum, K);
    } else if (K == 2560) {
        quant_x_kernel<2560><<<grid, block, 0, stream>>>(xr, xq, xs, xsum, K);
    } else if (K == 640) {
        quant_x_kernel<640><<<grid, block, 0, stream>>>(xr, xq, xs, xsum, K);
    } else if (K == 17408) {
        quant_x_kernel<17408><<<grid, block, 0, stream>>>(xr, xq, xs, xsum, K);
    } else {
        quant_x_kernel<0><<<grid, block, 0, stream>>>(xr, xq, xs, xsum, K);
    }
    check_launch();
}

template <typename TokT>
void launch_gemv_fp32(const torch::Tensor& xr, const torch::Tensor& tok, const torch::Tensor& lid, const torch::Tensor& codes, const torch::Tensor& scales, torch::Tensor& out, int K) {
    int N = static_cast<int>(codes.size(1));
    int P = static_cast<int>(tok.size(0));
    int kind = out_kind(out);
    int tiles = (N + kRowsPerBlock - 1) / kRowsPerBlock;
    int chunks = (K + kChunk - 1) >> 9;
    size_t smem = static_cast<size_t>(chunks) * kChunkStride * sizeof(float);
    dim3 block(32, kWarps);
    auto stream = at::cuda::getCurrentCUDAStream();
    int pair0 = 0;
    while (pair0 < P) {
        int batch = std::min(P - pair0, 65535);
        dim3 grid(tiles, batch);
        auto go = [&](auto kernel) {
            kernel<<<grid, block, smem, stream>>>(
                xr.data_ptr<float>(),
                tok.data_ptr<TokT>() + pair0,
                lid.data_ptr<int>() + pair0,
                codes.data_ptr<int>(),
                reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()),
                out.data_ptr(),
                N,
                K,
                pair0,
                kind);
        };
        if (K == 5120) {
            go(gemv_fp32_kernel<TokT, 5120>);
        } else if (K == 2304) {
            go(gemv_fp32_kernel<TokT, 2304>);
        } else if (K == 2560) {
            go(gemv_fp32_kernel<TokT, 2560>);
        } else if (K == 640) {
            go(gemv_fp32_kernel<TokT, 640>);
        } else if (K == 17408) {
            go(gemv_fp32_kernel<TokT, 17408>);
        } else {
            go(gemv_fp32_kernel<TokT, 0>);
        }
        check_launch();
        pair0 += batch;
    }
}

template <typename TokT>
void launch_gemv_dp4a(
    const int* xq,
    const float* xs,
    const int* xsum,
    const torch::Tensor& tok,
    const torch::Tensor& lid,
    const torch::Tensor& codes,
    const torch::Tensor& scales,
    torch::Tensor& out,
    int K) {
    int N = static_cast<int>(codes.size(1));
    int P = static_cast<int>(tok.size(0));
    int kind = out_kind(out);
    int tiles = (N + kRowsPerBlock - 1) / kRowsPerBlock;
    dim3 block(32, kWarps);
    auto stream = at::cuda::getCurrentCUDAStream();
    int pair0 = 0;
    while (pair0 < P) {
        int batch = std::min(P - pair0, 65535);
        dim3 grid(tiles, batch);
        auto go = [&](auto kernel) {
            kernel<<<grid, block, 0, stream>>>(
                xq,
                xs,
                xsum,
                tok.data_ptr<TokT>() + pair0,
                lid.data_ptr<int>() + pair0,
                codes.data_ptr<int>(),
                reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()),
                out.data_ptr(),
                N,
                K,
                pair0,
                kind);
        };
        if (K == 5120) {
            go(gemv_dp4a_kernel<TokT, 5120>);
        } else if (K == 2304) {
            go(gemv_dp4a_kernel<TokT, 2304>);
        } else if (K == 2560) {
            go(gemv_dp4a_kernel<TokT, 2560>);
        } else if (K == 640) {
            go(gemv_dp4a_kernel<TokT, 640>);
        } else if (K == 17408) {
            go(gemv_dp4a_kernel<TokT, 17408>);
        } else {
            go(gemv_dp4a_kernel<TokT, 0>);
        }
        check_launch();
        pair0 += batch;
    }
}

void grouped_decode(
    torch::Tensor xr,
    torch::Tensor tok,
    torch::Tensor lid,
    torch::Tensor codes,
    torch::Tensor scales,
    torch::Tensor out) {
    check_grouped_args(xr, tok, lid, codes, scales);
    TORCH_CHECK(out.is_cuda() && out.is_contiguous(), "out must be contiguous CUDA");
    TORCH_CHECK(out.size(0) == tok.size(0) && out.size(1) == codes.size(1), "out shape must be [P, N]");
    int P = static_cast<int>(tok.size(0));
    int M = static_cast<int>(xr.size(0));
    int K = static_cast<int>(xr.size(1));
    if (P == 0 || codes.size(1) == 0) {
        return;
    }
    if (!decode_use_int8()) {
        if (tok.scalar_type() == torch::kInt32) {
            launch_gemv_fp32<int32_t>(xr, tok, lid, codes, scales, out, K);
        } else {
            launch_gemv_fp32<int64_t>(xr, tok, lid, codes, scales, out, K);
        }
        return;
    }
    TORCH_CHECK(M > 0 && M <= 65535, "int8 decode needs 0 < M <= 65535");
    int64_t xq_words = static_cast<int64_t>(M) * (K >> 2);
    int64_t xs_words = static_cast<int64_t>(M) * (K >> 7);
    int64_t xsum_words = static_cast<int64_t>(M) * (K >> 5);
    auto ws = torch::empty({xq_words + xs_words + xsum_words}, xr.options().dtype(torch::kInt32));
    int* xq = ws.data_ptr<int>();
    float* xs = reinterpret_cast<float*>(xq + xq_words);
    int* xsum = xq + xq_words + xs_words;
    launch_quant_x(xr.data_ptr<float>(), xq, xs, xsum, M, K);
    if (tok.scalar_type() == torch::kInt32) {
        launch_gemv_dp4a<int32_t>(xq, xs, xsum, tok, lid, codes, scales, out, K);
    } else {
        launch_gemv_dp4a<int64_t>(xq, xs, xsum, tok, lid, codes, scales, out, K);
    }
}

void launch_dequant_kn(const torch::Tensor& codes, const torch::Tensor& scales, torch::Tensor& out) {
    TORCH_CHECK(codes.is_cuda() && scales.is_cuda(), "codes and scales must be CUDA");
    TORCH_CHECK(codes.is_contiguous() && scales.is_contiguous(), "codes and scales must be contiguous");
    TORCH_CHECK(out.is_cuda() && out.is_contiguous(), "out must be contiguous CUDA");
    TORCH_CHECK(codes.scalar_type() == torch::kInt32 && scales.scalar_type() == torch::kFloat16, "codes int32, scales fp16");
    TORCH_CHECK(out.scalar_type() == torch::kFloat16, "out must be fp16");
    TORCH_CHECK(codes.dim() == 3 && scales.dim() == 3 && out.dim() == 3, "codes, scales, out must be rank 3");
    int E = static_cast<int>(codes.size(0));
    int N = static_cast<int>(codes.size(1));
    int K = static_cast<int>(codes.size(2) * 16);
    TORCH_CHECK(K % 128 == 0 && scales.size(2) * 128 == K, "group 128 scales required");
    TORCH_CHECK(out.size(0) == E && out.size(1) == K && out.size(2) == N, "out shape must be [E, K, N]");
    TORCH_CHECK(E <= 65535, "E exceeds launch grid");
    if (E == 0 || N == 0 || K == 0) {
        return;
    }
    dim3 block(128);
    dim3 grid((N + 127) / 128, K / 128, E);
    dequant_kn_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        codes.data_ptr<int>(),
        reinterpret_cast<const __half*>(scales.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        N,
        K);
    check_launch();
}

torch::Tensor dequant_kn(torch::Tensor codes, torch::Tensor scales) {
    int E = static_cast<int>(codes.size(0));
    int N = static_cast<int>(codes.size(1));
    int K = static_cast<int>(codes.size(2) * 16);
    auto out = torch::empty({E, K, N}, codes.options().dtype(torch::kFloat16));
    launch_dequant_kn(codes, scales, out);
    return out;
}

void dequant_kn_into(torch::Tensor codes, torch::Tensor scales, torch::Tensor out) {
    launch_dequant_kn(codes, scales, out);
}

torch::Tensor rotate_cuda(torch::Tensor x, torch::Tensor signs, int64_t block) {
    TORCH_CHECK(x.is_cuda() && signs.is_cuda(), "x and signs must be CUDA");
    TORCH_CHECK(x.is_contiguous() && signs.is_contiguous(), "x and signs must be contiguous");
    TORCH_CHECK(x.dim() == 2, "x must be [M, K]");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32 || x.scalar_type() == torch::kBFloat16, "x must be fp32 or bf16");
    TORCH_CHECK(signs.scalar_type() == torch::kFloat32 && signs.dim() == 1, "signs must be fp32 [block]");
    int K = static_cast<int>(x.size(1));
    int M = static_cast<int>(x.size(0));
    int b = static_cast<int>(block);
    TORCH_CHECK(b >= 2 && b <= 1024 && (b & (b - 1)) == 0, "block must be a power of two in 2..1024");
    TORCH_CHECK(signs.numel() == b, "signs length must equal block");
    TORCH_CHECK(K % b == 0, "K must be divisible by block");
    auto y = torch::empty({M, K}, x.options().dtype(torch::kFloat32));
    if (M == 0) {
        return y;
    }
    auto stream = at::cuda::getCurrentCUDAStream();
    size_t smem = static_cast<size_t>(b) * sizeof(float);
    dim3 grid(M, K / b);
    if (x.scalar_type() == torch::kBFloat16) {
        rotate_kernel<true><<<grid, b / 2, smem, stream>>>(x.data_ptr(), signs.data_ptr<float>(), y.data_ptr<float>(), K, b);
    } else {
        rotate_kernel<false><<<grid, b / 2, smem, stream>>>(x.data_ptr(), signs.data_ptr<float>(), y.data_ptr<float>(), K, b);
    }
    check_launch();
    return y;
}

torch::Tensor fb_gemv(torch::Tensor x, torch::Tensor w, torch::Tensor coef) {
    // x [T, K] shared or [F, T, K] per expert; w [F, N, K]; coef [T, F]; returns [F, T, N] bf16.
    TORCH_CHECK(x.is_cuda() && w.is_cuda() && coef.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && coef.is_contiguous(), "tensors must be contiguous");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16 && w.scalar_type() == torch::kBFloat16, "x and w must be bf16");
    TORCH_CHECK(coef.scalar_type() == torch::kFloat32 && coef.dim() == 2, "coef must be fp32 [T, F]");
    TORCH_CHECK(w.dim() == 3, "w must be [F, N, K]");
    int F = static_cast<int>(w.size(0));
    int N = static_cast<int>(w.size(1));
    int K = static_cast<int>(w.size(2));
    int T = static_cast<int>(coef.size(0));
    TORCH_CHECK(coef.size(1) == F, "coef must be [T, F]");
    TORCH_CHECK(x.size(-1) == K && x.size(-2) == T, "x must be [.., T, K]");
    long long x_fstride = 0;
    if (x.dim() == 3) {
        TORCH_CHECK(x.size(0) == F, "per-expert x must be [F, T, K]");
        x_fstride = static_cast<long long>(T) * K;
    } else {
        TORCH_CHECK(x.dim() == 2, "x must be [T, K] or [F, T, K]");
    }
    TORCH_CHECK(K % 256 == 0, "K must be a multiple of 256");
    TORCH_CHECK(N % (kFbRows * kFbWarps) == 0, "N must be a multiple of 32");
    auto out = torch::empty({F, T, N}, x.options());
    if (F == 0 || T == 0) {
        return out;
    }
    dim3 grid(N / (kFbRows * kFbWarps), F);
    dim3 block(32, kFbWarps);
    fb_gemv_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        x_fstride,
        reinterpret_cast<const __nv_bfloat16*>(w.data_ptr<at::BFloat16>()),
        coef.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        F,
        T,
        N,
        K);
    check_launch();
    return out;
}

torch::Tensor act_quant(torch::Tensor x) {
    // bf16 -> bf16 FP8 round trip per 32 contiguous values.
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kBFloat16, "x must be contiguous CUDA bf16");
    TORCH_CHECK(x.numel() % 32 == 0, "numel must be a multiple of 32");
    auto y = torch::empty_like(x);
    long long groups = x.numel() / 32;
    if (groups == 0) {
        return y;
    }
    dim3 block(32, kFbWarps);
    act_quant_kernel<<<static_cast<unsigned>((groups + kFbWarps - 1) / kFbWarps), block, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(y.data_ptr<at::BFloat16>()),
        groups);
    check_launch();
    return y;
}

torch::Tensor fb_swiglu_quant(torch::Tensor gu, torch::Tensor coef, double limit) {
    // gu [F, T, 2I] bf16, coef [T, F] fp32 -> [F, T, I] bf16.
    TORCH_CHECK(gu.is_cuda() && gu.is_contiguous() && gu.scalar_type() == torch::kBFloat16 && gu.dim() == 3, "gu must be contiguous CUDA bf16 [F, T, 2I]");
    TORCH_CHECK(coef.is_cuda() && coef.is_contiguous() && coef.scalar_type() == torch::kFloat32, "coef must be contiguous CUDA fp32");
    int F = static_cast<int>(gu.size(0));
    int T = static_cast<int>(gu.size(1));
    int I = static_cast<int>(gu.size(2) / 2);
    TORCH_CHECK(gu.size(2) == 2 * I && I % 32 == 0, "2I must be even and I a multiple of 32");
    TORCH_CHECK(coef.dim() == 2 && coef.size(0) == T && coef.size(1) == F, "coef must be [T, F]");
    auto h = torch::empty({F, T, I}, gu.options());
    long long groups = static_cast<long long>(F) * T * (I / 32);
    if (groups == 0) {
        return h;
    }
    dim3 block(32, kFbWarps);
    fb_swiglu_quant_kernel<<<static_cast<unsigned>((groups + kFbWarps - 1) / kFbWarps), block, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(gu.data_ptr<at::BFloat16>()),
        coef.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(h.data_ptr<at::BFloat16>()),
        F,
        T,
        I,
        static_cast<float>(limit));
    check_launch();
    return h;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fb_gemv", &fb_gemv);
    m.def("act_quant", &act_quant);
    m.def("fb_swiglu_quant", &fb_swiglu_quant);
    m.def("rotate", &rotate_cuda);
    m.def("grouped_decode", &grouped_decode);
    m.def("dequant_kn", &dequant_kn);
    m.def("dequant_kn_into", &dequant_kn_into);
}
