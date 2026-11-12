#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Kernels for the batched (prompt) attention block, replacing the per-row
// RoPE / KV-store launches and the one-key-at-a-time batched attention:
//
//   apply_rope_f32_rows: apply_rope_f32_inplace (attn_fused.cu) for n rows at
//     positions start_pos.., one launch; per-element math is identical.
//   store_kv_f16_rows: store_kv_f16_row for rows row0..n-1 of K and V, one
//     launch; per-element conversion is identical.
//   attention_prefill_f16: causal (optionally sliding-window) attention for
//     n query rows. Positions before start_pos come from the f16 cache;
//     positions in the chunk come from the chunk's own K/V rows, rounded
//     through f16 so they hold exactly what the cache would. Because no key is
//     read from a slot the chunk writes, the cache store can follow the
//     attention, and a chunk longer than a sliding-window ring needs no
//     per-row interleaving.

static __device__ __forceinline__ float prefill_f16_round(float x) {
    return __half2float(__float2half(x));
}

__global__ void apply_rope_f32_rows_kernel(
    float* __restrict__ x,
    const float* __restrict__ inv_freq,
    int start_pos,
    float attn_scale,
    int head_dim,
    int half,
    int n_heads)
{
    const int head = blockIdx.x;
    const int row = blockIdx.y;
    float* xh = x + ((size_t)row * n_heads + head) * head_dim;
    const float fpos = static_cast<float>(start_pos + row);
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        float angle = fpos * inv_freq[i];
        float c, s;
        __sincosf(angle, &s, &c);
        c *= attn_scale;
        s *= attn_scale;
        float x0 = xh[i];
        float x1 = xh[i + half];
        xh[i] = x0 * c - x1 * s;
        xh[i + half] = x0 * s + x1 * c;
    }
}

extern "C" __host__ int mantleCudaApplyRoPERowsF32(
    float* x,
    const float* inv_freq,
    int start_pos,
    float attn_scale,
    int head_dim,
    int half,
    int n_heads,
    int n_rows,
    cudaStream_t stream)
{
    if (!x || !inv_freq || head_dim <= 0 || half <= 0 || n_heads <= 0 || n_rows <= 0 || start_pos < 0) {
        return cudaErrorInvalidValue;
    }
    if (half * 2 > head_dim) {
        return cudaErrorInvalidValue;
    }
    if (attn_scale == 0.0f) attn_scale = 1.0f;
    int block = 128;
    if (half < 128) block = 64;
    if (half < 64) block = 32;
    dim3 grid(n_heads, n_rows);
    apply_rope_f32_rows_kernel<<<grid, block, 0, stream>>>(x, inv_freq, start_pos, attn_scale, head_dim, half, n_heads);
    return cudaGetLastError();
}

// blockIdx.y selects K (0) or V (1).
__global__ void store_kv_f16_rows_kernel(
    uint16_t* __restrict__ cache_k,
    uint16_t* __restrict__ cache_v,
    const float* __restrict__ k,
    const float* __restrict__ v,
    int start_pos,
    int row0,
    int kv_stride,
    int modulo_len)
{
    const int row = row0 + blockIdx.x;
    const bool isV = blockIdx.y == 1;
    uint16_t* cache = isV ? cache_v : cache_k;
    const float* src = (isV ? v : k) + (size_t)row * kv_stride;
    uint16_t* dst = cache + (size_t)((start_pos + row) % modulo_len) * kv_stride;
    for (int i = threadIdx.x; i < kv_stride; i += blockDim.x) {
        dst[i] = __half_as_ushort(__float2half(src[i]));
    }
}

extern "C" __host__ int mantleCudaStoreKVF16Rows(
    uint16_t* cache_k,
    uint16_t* cache_v,
    const float* k,
    const float* v,
    int start_pos,
    int row0,
    int n_rows,
    int kv_stride,
    int modulo_len,
    cudaStream_t stream)
{
    if (!cache_k || !cache_v || !k || !v || start_pos < 0 || row0 < 0 || n_rows <= row0 || kv_stride <= 0 ||
        modulo_len <= 0 || n_rows - row0 > modulo_len) {
        return cudaErrorInvalidValue;
    }
    int block = 256;
    if (kv_stride < 256) block = 128;
    if (kv_stride < 128) block = 64;
    dim3 grid(n_rows - row0, 2);
    store_kv_f16_rows_kernel<<<grid, block, 0, stream>>>(cache_k, cache_v, k, v, start_pos, row0, kv_stride, modulo_len);
    return cudaGetLastError();
}

#define PREFILL_WARPS 4
#define PREFILL_QPW 8                          // query vectors per warp
#define PREFILL_QV (PREFILL_WARPS * PREFILL_QPW) // query vectors per block
#define PREFILL_KT 32                          // keys per tile, one per lane

// One block per (row tile, kv head). The block's PREFILL_QV query vectors are
// the group's query heads for PREFILL_QV/group consecutive rows; each warp
// owns eight of them. Per 32-key tile, lane j scores key j against all eight
// of its warp's vectors (each K element loaded once for eight FMAs), then each
// vector updates its online softmax and accumulates V, every lane owning the
// head dims lane, lane+32, ...
//
// DPL is head_dim / 32.
template <int DPL>
__global__ void __launch_bounds__(PREFILL_WARPS * 32) attention_prefill_f16_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k_new,
    const float* __restrict__ v_new,
    const uint16_t* __restrict__ cache_k,
    const uint16_t* __restrict__ cache_v,
    float* __restrict__ out,
    int n,
    int start_pos,
    int kv_stride,
    int n_head,
    int group,
    int cache_len,
    float scale,
    float softcap,
    int window)
{
    constexpr int HD = DPL * 32;
    extern __shared__ float smem[];
    float* qs = smem;                          // [PREFILL_QV][HD]
    float* ks = qs + PREFILL_QV * HD;          // [PREFILL_KT][HD + 1]
    float* vs = ks + PREFILL_KT * (HD + 1);    // [PREFILL_KT][HD]

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int rowsPerBlock = PREFILL_QV / group;
    const int r0 = blockIdx.x * rowsPerBlock;
    const int kvHead = blockIdx.y;
    const int qStride = n_head * HD;

    // Stage the block's query vectors (zero for rows past n).
    for (int idx = tid; idx < PREFILL_QV * HD; idx += blockDim.x) {
        const int qv = idx / HD;
        const int d = idx % HD;
        const int r = r0 + qv / group;
        const int h = kvHead * group + qv % group;
        qs[idx] = r < n ? q[(size_t)r * qStride + (size_t)h * HD + d] : 0.0f;
    }

    // Per-vector causal bounds and softmax state.
    int vPos[PREFILL_QPW], vStart[PREFILL_QPW];
    float m[PREFILL_QPW], l[PREFILL_QPW], acc[PREFILL_QPW][DPL];
#pragma unroll
    for (int i = 0; i < PREFILL_QPW; i++) {
        const int qv = warp * PREFILL_QPW + i;
        const int r = r0 + qv / group;
        if (r < n) {
            vPos[i] = start_pos + r;
            vStart[i] = window > 0 ? max(vPos[i] - window + 1, 0) : 0;
        } else {
            vPos[i] = -1;
            vStart[i] = 0;
        }
        m[i] = -INFINITY;
        l[i] = 0.0f;
#pragma unroll
        for (int k = 0; k < DPL; k++) acc[i][k] = 0.0f;
    }

    const int rLast = min(r0 + rowsPerBlock, n) - 1;
    const int keyEnd = start_pos + rLast;
    const int keyBegin = window > 0 ? max(start_pos + r0 - window + 1, 0) : 0;
    const size_t headOff = (size_t)kvHead * HD;

    for (int t0 = keyBegin; t0 <= keyEnd; t0 += PREFILL_KT) {
        __syncthreads();
        for (int idx = tid; idx < PREFILL_KT * HD; idx += blockDim.x) {
            const int j = idx / HD;
            const int d = idx % HD;
            const int t = t0 + j;
            float kv = 0.0f, vv = 0.0f;
            if (t <= keyEnd) {
                if (t < start_pos) {
                    const size_t off = (size_t)(t % cache_len) * kv_stride + headOff + d;
                    kv = __half2float(__ushort_as_half(cache_k[off]));
                    vv = __half2float(__ushort_as_half(cache_v[off]));
                } else {
                    const size_t off = (size_t)(t - start_pos) * kv_stride + headOff + d;
                    kv = prefill_f16_round(k_new[off]);
                    vv = prefill_f16_round(v_new[off]);
                }
            }
            ks[j * (HD + 1) + d] = kv;
            vs[j * HD + d] = vv;
        }
        __syncthreads();

        const int t = t0 + lane;
        float s[PREFILL_QPW];
#pragma unroll
        for (int i = 0; i < PREFILL_QPW; i++) s[i] = 0.0f;
        const float* qw = qs + warp * PREFILL_QPW * HD;
        const float* kr = ks + lane * (HD + 1);
        for (int d = 0; d < HD; d++) {
            const float kd = kr[d];
#pragma unroll
            for (int i = 0; i < PREFILL_QPW; i++) s[i] = fmaf(qw[i * HD + d], kd, s[i]);
        }

#pragma unroll
        for (int i = 0; i < PREFILL_QPW; i++) {
            if (vPos[i] < 0 || t0 > vPos[i] || t0 + PREFILL_KT <= vStart[i]) continue;
            const bool valid = t >= vStart[i] && t <= vPos[i];
            float sc = s[i] * scale;
            if (softcap > 0.0f) sc = tanhf(sc / softcap) * softcap;
            sc = valid ? sc : -INFINITY;
            float tmax = sc;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, o));
            const float mNew = fmaxf(m[i], tmax);
            const float alpha = __expf(m[i] - mNew);
            const float p = valid ? __expf(sc - mNew) : 0.0f;
            float psum = p;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) psum += __shfl_xor_sync(0xffffffffu, psum, o);
            l[i] = l[i] * alpha + psum;
            m[i] = mNew;
#pragma unroll
            for (int k = 0; k < DPL; k++) acc[i][k] *= alpha;
            for (int j = 0; j < PREFILL_KT; j++) {
                const float pj = __shfl_sync(0xffffffffu, p, j);
#pragma unroll
                for (int k = 0; k < DPL; k++) acc[i][k] = fmaf(pj, vs[j * HD + lane + 32 * k], acc[i][k]);
            }
        }
    }

#pragma unroll
    for (int i = 0; i < PREFILL_QPW; i++) {
        if (vPos[i] < 0) continue;
        const int qv = warp * PREFILL_QPW + i;
        const int r = r0 + qv / group;
        const int h = kvHead * group + qv % group;
        float* oh = out + (size_t)r * qStride + (size_t)h * HD;
        const float inv = 1.0f / (l[i] + 1e-12f);
#pragma unroll
        for (int k = 0; k < DPL; k++) oh[lane + 32 * k] = acc[i][k] * inv;
    }
}

static size_t prefillSmemBytes(int headDim) {
    return sizeof(float) * ((size_t)PREFILL_QV * headDim + (size_t)PREFILL_KT * (headDim + 1) + (size_t)PREFILL_KT * headDim);
}

template <int DPL>
static int launchPrefill(const float* q, const float* kNew, const float* vNew, const uint16_t* cacheK,
                         const uint16_t* cacheV, float* out, int n, int startPos, int kvStride, int nHead,
                         int group, int cacheLen, float scale, float softcap, int window, cudaStream_t stream) {
    const size_t smem = prefillSmemBytes(DPL * 32);
    cudaError_t e = cudaFuncSetAttribute(attention_prefill_f16_kernel<DPL>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    if (e != cudaSuccess) return e;
    const int rowsPerBlock = PREFILL_QV / group;
    dim3 grid((n + rowsPerBlock - 1) / rowsPerBlock, nHead / group);
    attention_prefill_f16_kernel<DPL><<<grid, PREFILL_WARPS * 32, smem, stream>>>(
        q, kNew, vNew, cacheK, cacheV, out, n, startPos, kvStride, nHead, group, cacheLen, scale, softcap, window);
    return cudaGetLastError();
}

// mantleCudaAttentionPrefillF16Supported reports whether
// mantleCudaAttentionPrefillF16 takes this head layout and fits the device's
// per-block shared memory.
extern "C" __host__ int mantleCudaAttentionPrefillF16Supported(int headDim, int nHead, int kvHeads) {
    if (headDim <= 0 || nHead <= 0 || kvHeads <= 0 || nHead % kvHeads != 0) return 0;
    if (headDim != 32 && headDim != 64 && headDim != 128 && headDim != 256) return 0;
    const int group = nHead / kvHeads;
    if (group > PREFILL_QV || PREFILL_QV % group != 0) return 0;
    int dev = 0, optin = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return 0;
    if (cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) != cudaSuccess) return 0;
    return prefillSmemBytes(headDim) <= (size_t)optin ? 1 : 0;
}

// mantleCudaAttentionPrefillF16 computes attention for n query rows at
// positions start_pos..start_pos+n-1. q and out are [n, nHead*headDim];
// kNew and vNew are the chunk's [n, kvStride] rows after RoPE; the f16 cache
// supplies positions below start_pos at slot t % cacheLen. Row r attends
// [max(pos-window+1, 0), pos] (window 0 means no window).
extern "C" __host__ int mantleCudaAttentionPrefillF16(
    const float* q,
    const float* kNew,
    const float* vNew,
    const uint16_t* cacheK,
    const uint16_t* cacheV,
    float* out,
    int n,
    int startPos,
    int kvStride,
    int headDim,
    int nHead,
    int kvHeads,
    int cacheLen,
    float scale,
    float softcap,
    int window,
    cudaStream_t stream)
{
    if (!q || !kNew || !vNew || !cacheK || !cacheV || !out || n <= 0 || startPos < 0 || kvStride <= 0 ||
        cacheLen <= 0 || !mantleCudaAttentionPrefillF16Supported(headDim, nHead, kvHeads)) {
        return cudaErrorInvalidValue;
    }
    const int group = nHead / kvHeads;
    switch (headDim) {
    case 32:
        return launchPrefill<1>(q, kNew, vNew, cacheK, cacheV, out, n, startPos, kvStride, nHead, group, cacheLen, scale, softcap, window, stream);
    case 64:
        return launchPrefill<2>(q, kNew, vNew, cacheK, cacheV, out, n, startPos, kvStride, nHead, group, cacheLen, scale, softcap, window, stream);
    case 128:
        return launchPrefill<4>(q, kNew, vNew, cacheK, cacheV, out, n, startPos, kvStride, nHead, group, cacheLen, scale, softcap, window, stream);
    default:
        return launchPrefill<8>(q, kNew, vNew, cacheK, cacheV, out, n, startPos, kvStride, nHead, group, cacheLen, scale, softcap, window, stream);
    }
}

