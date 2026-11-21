#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

// Batched quantized matvec (GEMM) kernels supporting three on-device weight
// layouts:
//   - CachedInt8Weights (quantMatFormatCachedInt8): q holds totalBlocks*32
//     int8 values, scales one f32 per 32-element block.
//   - K4RawWeights (quantMatFormatK4Raw): packed 4-bit two's-complement nibbles
//     (16 bytes per 32-element block) + f16 super-scales (1 per 8 blocks) +
//     6-bit sub-scales (u6 / 32.0f).
//   - Q4RawWeights (quantMatFormatQ4Raw): packed 4-bit two's-complement nibbles
//     (16 bytes per 32-element block) + f16 block scales.
//
// All three layouts unpack into identical int8 weight tiles and f32 per-block
// scales in shared memory, so every kernel (f32 activations, single-term dp4a,
// single-term sm_80+ MMA, and two-term 14-bit hi/lo sm_80+ MMA) produces
// bit-identical results across raw K4/Q4 and cached-int8 without allocating a
// second cached-int8 weight copy in VRAM.

#define TILE_R 32
#define TILE_N 16
#define CHUNK 256
#define CHUNKS_PER_BLOCK (CHUNK / 32)

#define MMA_BM 64
#define MMA_BN 64
#define MMA_BK 128
#define MMA_KB (MMA_BK / 32)
#define MMA_LDS (MMA_BK + 16) // 144-byte rows: fragment loads hit 32 distinct banks
#define MMA_THREADS 128

#define MMA_HL_BM 64
#define MMA_HL_BN 64
#define MMA_HL_BK 64
#define MMA_HL_KB (MMA_HL_BK / 32)
#define MMA_HL_LDS (MMA_HL_BK + 16)
#define MMA_HL_THREADS 128

struct CachedInt8Weights {
    const int8_t* q;
    const float* scales;
};

struct K4RawWeights {
    const uint8_t* q;
    const uint16_t* superScales;
    const uint8_t* subScales;
    int superBlocksPerRow;
};

struct Q4RawWeights {
    const uint8_t* q;
    const uint16_t* scalesF16;
};

__device__ __forceinline__ float qb_f16_to_f32(uint16_t raw) {
    return __half2float(__ushort_as_half(raw));
}

// Unpacks 4 packed bytes (8 two's-complement 4-bit nibbles, low nibble first)
// into 8 signed int8 values in two uint32_t words.
// For any nibble n in [0..15], u = n ^ 8 is in [0..15], and ((u + 0x78) ^ 0x80)
// yields the exact signed int8 in [-8..+7] with zero cross-byte carries.
__device__ __forceinline__ void qb_unpack_s4x8_to_s8x8(uint32_t w, uint32_t& out0, uint32_t& out1) {
    const uint32_t u = w ^ 0x88888888u;
    const uint32_t e = u & 0x0F0F0F0Fu;
    const uint32_t o = (u >> 4) & 0x0F0F0F0Fu;
    const uint32_t p0 = __byte_perm(e, o, 0x5140);
    const uint32_t p1 = __byte_perm(e, o, 0x7362);
    out0 = (p0 + 0x78787878u) ^ 0x80808080u;
    out1 = (p1 + 0x78787878u) ^ 0x80808080u;
}

__device__ __forceinline__ void qb_zero_block32(int8_t* dst32) {
    const uint4 z = make_uint4(0u, 0u, 0u, 0u);
    *reinterpret_cast<uint4*>(dst32) = z;
    *reinterpret_cast<uint4*>(dst32 + 16) = z;
}

__device__ __forceinline__ void qb_unpack_block4(const uint8_t* __restrict__ src16, int8_t* dst32) {
    uint4 qv;
    if ((reinterpret_cast<uintptr_t>(src16) & 15) == 0) {
        qv = __ldg(reinterpret_cast<const uint4*>(src16));
    } else {
        auto loadU32 = [&](int off) -> uint32_t {
            return (uint32_t)__ldg(src16 + off) |
                   ((uint32_t)__ldg(src16 + off + 1) << 8) |
                   ((uint32_t)__ldg(src16 + off + 2) << 16) |
                   ((uint32_t)__ldg(src16 + off + 3) << 24);
        };
        qv = make_uint4(loadU32(0), loadU32(4), loadU32(8), loadU32(12));
    }
    uint4 lo, hi;
    qb_unpack_s4x8_to_s8x8(qv.x, lo.x, lo.y);
    qb_unpack_s4x8_to_s8x8(qv.y, lo.z, lo.w);
    qb_unpack_s4x8_to_s8x8(qv.z, hi.x, hi.y);
    qb_unpack_s4x8_to_s8x8(qv.w, hi.z, hi.w);
    *reinterpret_cast<uint4*>(dst32) = lo;
    *reinterpret_cast<uint4*>(dst32 + 16) = hi;
}

__device__ __forceinline__ void load_w_tile(
    const CachedInt8Weights& W,
    int8_t wsh[TILE_R][CHUNK + 16],
    float wsc[TILE_R][CHUNKS_PER_BLOCK + 1],
    int tid,
    int mBase,
    int cBase,
    int cLen,
    int rows,
    int blocksPerRow
) {
    for (int idx = tid; idx < TILE_R * cLen; idx += blockDim.x) {
        const int lrl = idx / cLen;
        const int c = idx % cLen;
        const int rr = mBase + lrl;
        wsh[lrl][c] = (rr < rows) ? W.q[(size_t)rr * blocksPerRow * 32 + (size_t)(cBase + c)] : (int8_t)0;
    }
    for (int idx = tid; idx < TILE_R * CHUNKS_PER_BLOCK; idx += blockDim.x) {
        const int lrl = idx / CHUNKS_PER_BLOCK;
        const int b = idx % CHUNKS_PER_BLOCK;
        const int rr = mBase + lrl;
        const int gb = (cBase + b * 32) / 32;
        wsc[lrl][b] = (rr < rows && gb < blocksPerRow) ? W.scales[(size_t)rr * blocksPerRow + gb] : 0.0f;
    }
}

__device__ __forceinline__ void load_w_tile(
    const K4RawWeights& W,
    int8_t wsh[TILE_R][CHUNK + 16],
    float wsc[TILE_R][CHUNKS_PER_BLOCK + 1],
    int tid,
    int mBase,
    int cBase,
    int cLen,
    int rows,
    int blocksPerRow
) {
    const int kb0 = cBase >> 5;
    for (int idx = tid; idx < TILE_R * CHUNKS_PER_BLOCK; idx += blockDim.x) {
        const int lrl = idx / CHUNKS_PER_BLOCK;
        const int b = idx % CHUNKS_PER_BLOCK;
        const int rr = mBase + lrl;
        const int gb = kb0 + b;
        int8_t* dst32 = &wsh[lrl][b * 32];
        if (rr < rows && gb < blocksPerRow && b * 32 < cLen) {
            const size_t blk = (size_t)rr * blocksPerRow + gb;
            const uint8_t u6 = __ldg(W.subScales + blk) & 0x3Fu;
            const float super = qb_f16_to_f32(__ldg(W.superScales + (size_t)rr * W.superBlocksPerRow + (gb >> 3)));
            if (super != 0.0f && u6 != 0) {
                wsc[lrl][b] = super * (static_cast<float>(u6) * (1.0f / 32.0f));
                qb_unpack_block4(W.q + blk * 16, dst32);
                continue;
            }
        }
        wsc[lrl][b] = 0.0f;
        qb_zero_block32(dst32);
    }
}

__device__ __forceinline__ void load_w_tile(
    const Q4RawWeights& W,
    int8_t wsh[TILE_R][CHUNK + 16],
    float wsc[TILE_R][CHUNKS_PER_BLOCK + 1],
    int tid,
    int mBase,
    int cBase,
    int cLen,
    int rows,
    int blocksPerRow
) {
    const int kb0 = cBase >> 5;
    for (int idx = tid; idx < TILE_R * CHUNKS_PER_BLOCK; idx += blockDim.x) {
        const int lrl = idx / CHUNKS_PER_BLOCK;
        const int b = idx % CHUNKS_PER_BLOCK;
        const int rr = mBase + lrl;
        const int gb = kb0 + b;
        int8_t* dst32 = &wsh[lrl][b * 32];
        if (rr < rows && gb < blocksPerRow && b * 32 < cLen) {
            const size_t blk = (size_t)rr * blocksPerRow + gb;
            const float sc = qb_f16_to_f32(__ldg(W.scalesF16 + blk));
            if (sc != 0.0f) {
                wsc[lrl][b] = sc;
                qb_unpack_block4(W.q + blk * 16, dst32);
                continue;
            }
        }
        wsc[lrl][b] = 0.0f;
        qb_zero_block32(dst32);
    }
}

template <typename WType>
__global__ void quant_matvec_batch_f32_kernel(
    WType W,
    const float* __restrict__ x,
    float* __restrict__ y,
    int rows,
    int n,
    int cols,
    int blocksPerRow
) {
    __shared__ __align__(16) int8_t wsh[TILE_R][CHUNK + 16];
    __shared__ float wsc[TILE_R][CHUNKS_PER_BLOCK + 1];
    __shared__ float xsh[TILE_N][CHUNK + 4];

    const int tid = threadIdx.x;
    const int r = blockIdx.x * TILE_R + (tid / TILE_N);
    const int ni = blockIdx.y * TILE_N + (tid % TILE_N);
    const bool active = (r < rows) && (ni < n);
    const int rl = tid / TILE_N;
    const int nl = tid % TILE_N;

    const int nChunks = (cols + CHUNK - 1) / CHUNK;
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int ch = 0; ch < nChunks; ch++) {
        const int cBase = ch * CHUNK;
        const int cLen = min(CHUNK, cols - cBase);
        if (cLen <= 0) break;
        load_w_tile(W, wsh, wsc, tid, blockIdx.x * TILE_R, cBase, cLen, rows, blocksPerRow);
        for (int idx = tid; idx < TILE_N * cLen; idx += blockDim.x) {
            int lnl = idx / cLen;
            int c = idx % cLen;
            int nn = blockIdx.y * TILE_N + lnl;
            xsh[lnl][c] = (nn < n) ? x[(size_t)nn * cols + (size_t)(cBase + c)] : 0.0f;
        }
        __syncthreads();
        if (active) {
            const int8_t* wrow = wsh[rl];
            const float* xrow = xsh[nl];
            for (int b = 0; b < CHUNKS_PER_BLOCK; b++) {
                const int c = b * 32;
                if (c >= cLen) break;
                const float s = wsc[rl][b];
                if (s == 0.0f) continue;
                const int len = min(32, cLen - c);
                float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
                float d4 = 0.0f, d5 = 0.0f, d6 = 0.0f, d7 = 0.0f;
                int i = 0;
                for (; i + 8 <= len; i += 8) {
                    const char4 w0 = *reinterpret_cast<const char4*>(wrow + c + i);
                    const char4 w1 = *reinterpret_cast<const char4*>(wrow + c + i + 4);
                    const float4 x0 = *reinterpret_cast<const float4*>(xrow + c + i);
                    const float4 x1 = *reinterpret_cast<const float4*>(xrow + c + i + 4);
                    d0 = fmaf((float)w0.x, x0.x, d0);
                    d1 = fmaf((float)w0.y, x0.y, d1);
                    d2 = fmaf((float)w0.z, x0.z, d2);
                    d3 = fmaf((float)w0.w, x0.w, d3);
                    d4 = fmaf((float)w1.x, x1.x, d4);
                    d5 = fmaf((float)w1.y, x1.y, d5);
                    d6 = fmaf((float)w1.z, x1.z, d6);
                    d7 = fmaf((float)w1.w, x1.w, d7);
                }
                float dot = ((d0 + d1) + (d2 + d3)) + ((d4 + d5) + (d6 + d7));
                for (; i < len; i++) {
                    dot = fmaf((float)wrow[c + i], xrow[c + i], dot);
                }
                acc[0] = fmaf(s, dot, acc[0]);
            }
        }
        __syncthreads();
    }
    if (active) {
        y[(size_t)ni * rows + r] = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
    }
}

// Quantizes x [n, cols] to int8 per 32-element block with an f32 scale per
// block (maxAbs/127), producing xq [n, cols] and xsc [n, blocksPerRow].
__global__ void quantize_x_blocks_kernel(
    const float* __restrict__ x,
    int8_t* __restrict__ xq,
    float* __restrict__ xsc,
    int n,
    int cols,
    int blocksPerRow
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n * blocksPerRow;
    if (idx >= total) return;
    const int row = idx / blocksPerRow;
    const int b = idx % blocksPerRow;
    const int base = row * cols + b * 32;
    const int len = min(32, cols - b * 32);
    float maxAbs = 0.0f;
    for (int i = 0; i < len; i++) maxAbs = fmaxf(maxAbs, fabsf(x[base + i]));
    const float scale = maxAbs / 127.0f;
    xsc[idx] = scale;
    for (int i = 0; i < len; i++) {
        float v = (scale > 0.0f) ? x[base + i] / scale : 0.0f;
        v = fminf(fmaxf(v, -127.0f), 127.0f);
        xq[base + i] = (int8_t)__float2int_rn(v);
    }
}

// Two-term variant of quantize_x_blocks_kernel: each activation value is split
// into x = s1*q1 + s2*q2 with q1 and q2 int8 and both scales f32 per
// 32-element block.
__global__ void quantize_x_blocks_hi_lo_kernel(
    const float* __restrict__ x,
    int8_t* __restrict__ xq,
    float* __restrict__ xsc,
    int8_t* __restrict__ xq2,
    float* __restrict__ xsc2,
    int n,
    int cols,
    int blocksPerRow
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n * blocksPerRow;
    if (idx >= total) return;
    const int row = idx / blocksPerRow;
    const int b = idx % blocksPerRow;
    const int base = row * cols + b * 32;
    const int len = min(32, cols - b * 32);
    float maxAbs = 0.0f;
    for (int i = 0; i < len; i++) maxAbs = fmaxf(maxAbs, fabsf(x[base + i]));
    const float scale1 = maxAbs / 127.0f;
    float maxRes = 0.0f;
    for (int i = 0; i < len; i++) {
        const float v = x[base + i];
        int q = 0;
        if (scale1 > 0.0f) {
            const float s = fminf(fmaxf(v / scale1, -127.0f), 127.0f);
            q = __float2int_rn(s);
        }
        xq[base + i] = (int8_t)q;
        maxRes = fmaxf(maxRes, fabsf(v - scale1 * (float)q));
    }
    const float scale2 = maxRes / 127.0f;
    xsc[idx] = scale1;
    xsc2[idx] = scale2;
    for (int i = 0; i < len; i++) {
        const float v = x[base + i];
        int q1 = 0;
        if (scale1 > 0.0f) {
            const float s = fminf(fmaxf(v / scale1, -127.0f), 127.0f);
            q1 = __float2int_rn(s);
        }
        int q2 = 0;
        if (scale2 > 0.0f) {
            const float r = v - scale1 * (float)q1;
            const float s = fminf(fmaxf(r / scale2, -127.0f), 127.0f);
            q2 = __float2int_rn(s);
        }
        xq2[base + i] = (int8_t)q2;
    }
}

template <typename WType>
__global__ void quant_matvec_batch_dp4a_kernel(
    WType W,
    const int8_t* __restrict__ xq,
    const float* __restrict__ xsc,
    float* __restrict__ y,
    int rows,
    int n,
    int cols,
    int blocksPerRow
) {
    // +16 padding keeps every row 16-byte aligned for uint4 block stores and int32 dp4a loads
    __shared__ __align__(16) int8_t wsh[TILE_R][CHUNK + 16];
    __shared__ float wsc[TILE_R][CHUNKS_PER_BLOCK + 1];
    __shared__ int8_t xsh[TILE_N][CHUNK + 4];
    __shared__ float xscsh[TILE_N][CHUNKS_PER_BLOCK + 1];

    const int tid = threadIdx.x;
    const int r = blockIdx.x * TILE_R + (tid / TILE_N);
    const int ni = blockIdx.y * TILE_N + (tid % TILE_N);
    const bool active = (r < rows) && (ni < n);
    const int rl = tid / TILE_N;
    const int nl = tid % TILE_N;

    const int nChunks = (cols + CHUNK - 1) / CHUNK;
    float acc = 0.0f;
    for (int ch = 0; ch < nChunks; ch++) {
        const int cBase = ch * CHUNK;
        const int cLen = min(CHUNK, cols - cBase);
        if (cLen <= 0) break;
        load_w_tile(W, wsh, wsc, tid, blockIdx.x * TILE_R, cBase, cLen, rows, blocksPerRow);
        for (int idx = tid; idx < TILE_N * cLen; idx += blockDim.x) {
            int lnl = idx / cLen;
            int c = idx % cLen;
            int nn = blockIdx.y * TILE_N + lnl;
            xsh[lnl][c] = (nn < n) ? xq[(size_t)nn * cols + (size_t)(cBase + c)] : (int8_t)0;
        }
        for (int idx = tid; idx < TILE_N * CHUNKS_PER_BLOCK; idx += blockDim.x) {
            int lnl = idx / CHUNKS_PER_BLOCK;
            int b = idx % CHUNKS_PER_BLOCK;
            int nn = blockIdx.y * TILE_N + lnl;
            int gb = (cBase + b * 32) / 32;
            xscsh[lnl][b] = (nn < n && gb < blocksPerRow) ? xsc[(size_t)nn * blocksPerRow + gb] : 0.0f;
        }
        // zero the shared padding beyond cLen so dp4a never reads stale data
        for (int idx = tid; idx < TILE_R * (CHUNK - cLen); idx += blockDim.x) {
            int lrl = idx / (CHUNK - cLen);
            int c = cLen + idx % (CHUNK - cLen);
            wsh[lrl][c] = (int8_t)0;
        }
        for (int idx = tid; idx < TILE_N * (CHUNK - cLen); idx += blockDim.x) {
            int lnl = idx / (CHUNK - cLen);
            int c = cLen + idx % (CHUNK - cLen);
            xsh[lnl][c] = (int8_t)0;
        }
        __syncthreads();
        if (active) {
#if __CUDA_ARCH__ >= 610
            const int nBlocks = (cLen + 31) / 32;
            for (int b = 0; b < nBlocks; b++) {
                int d = 0;
                const int c0 = b * 32;
                const int rem = min(32, cLen - c0);
                const int n4 = min(8, (rem + 3) / 4);
                for (int j = 0; j < n4; j++) {
                    const int c = c0 + j * 4;
                    int w4 = *(const int*)&wsh[rl][c];
                    int x4 = *(const int*)&xsh[nl][c];
                    d = __dp4a(w4, x4, d);
                }
                for (int c = c0 + n4 * 4; c < c0 + rem; c++) {
                    d += (int)wsh[rl][c] * (int)xsh[nl][c];
                }
                acc += wsc[rl][b] * xscsh[nl][b] * (float)d;
            }
#else
            for (int c = 0; c < cLen; c++) {
                acc += wsc[rl][c / 32] * (float)wsh[rl][c] * (float)xsh[nl][c] * xscsh[nl][c / 32];
            }
#endif
        }
        __syncthreads();
    }
    if (active) {
        y[(size_t)ni * rows + r] = acc;
    }
}

#if __CUDA_ARCH__ >= 800
__device__ __forceinline__ void mma_cp_async16(void* smem, const void* gmem, bool valid) {
    unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    int size = valid ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(size));
}

__device__ __forceinline__ void mma_cp_async4(void* smem, const void* gmem, bool valid) {
    unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    int size = valid ? 4 : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(s), "l"(gmem), "r"(size));
}

template <int BM, int BK, int KB, int LDS, int THREADS>
__device__ __forceinline__ void load_mma_w_stage(
    const CachedInt8Weights& W,
    int8_t ws[BM][LDS],
    float wsc[BM][KB],
    int tid,
    int mBase,
    int kb0,
    int rows,
    int cols,
    int blocksPerRow
) {
    const int kByte = kb0 * 32;
    const size_t wRowBytes = (size_t)blocksPerRow * 32;
    constexpr int chunksPerRow = BK / 16;
    for (int c = tid; c < BM * chunksPerRow; c += THREADS) {
        const int r = c / chunksPerRow;
        const int off = (c % chunksPerRow) * 16;
        const bool inK = kByte + off < cols;
        const int gr = mBase + r;
        const bool wv = inK && gr < rows;
        mma_cp_async16(&ws[r][off], wv ? W.q + (size_t)gr * wRowBytes + kByte + off : W.q, wv);
    }
    for (int c = tid; c < BM * KB; c += THREADS) {
        const int r = c / KB;
        const int j = c % KB;
        const bool inK = kb0 + j < blocksPerRow;
        const int gr = mBase + r;
        const bool wv = inK && gr < rows;
        mma_cp_async4(&wsc[r][j], wv ? W.scales + (size_t)gr * blocksPerRow + kb0 + j : W.scales, wv);
    }
}

template <int BM, int BK, int KB, int LDS, int THREADS>
__device__ __forceinline__ void load_mma_w_stage(
    const K4RawWeights& W,
    int8_t ws[BM][LDS],
    float wsc[BM][KB],
    int tid,
    int mBase,
    int kb0,
    int rows,
    int cols,
    int blocksPerRow
) {
    (void)cols;
    for (int c = tid; c < BM * KB; c += THREADS) {
        const int r = c / KB;
        const int j = c % KB;
        const int gr = mBase + r;
        const int gb = kb0 + j;
        int8_t* dst32 = &ws[r][j * 32];
        if (gr < rows && gb < blocksPerRow) {
            const size_t blk = (size_t)gr * blocksPerRow + gb;
            const uint8_t u6 = __ldg(W.subScales + blk) & 0x3Fu;
            const float super = qb_f16_to_f32(__ldg(W.superScales + (size_t)gr * W.superBlocksPerRow + (gb >> 3)));
            if (super != 0.0f && u6 != 0) {
                wsc[r][j] = super * (static_cast<float>(u6) * (1.0f / 32.0f));
                qb_unpack_block4(W.q + blk * 16, dst32);
                continue;
            }
        }
        wsc[r][j] = 0.0f;
        qb_zero_block32(dst32);
    }
}

template <int BM, int BK, int KB, int LDS, int THREADS>
__device__ __forceinline__ void load_mma_w_stage(
    const Q4RawWeights& W,
    int8_t ws[BM][LDS],
    float wsc[BM][KB],
    int tid,
    int mBase,
    int kb0,
    int rows,
    int cols,
    int blocksPerRow
) {
    (void)cols;
    for (int c = tid; c < BM * KB; c += THREADS) {
        const int r = c / KB;
        const int j = c % KB;
        const int gr = mBase + r;
        const int gb = kb0 + j;
        int8_t* dst32 = &ws[r][j * 32];
        if (gr < rows && gb < blocksPerRow) {
            const size_t blk = (size_t)gr * blocksPerRow + gb;
            const float sc = qb_f16_to_f32(__ldg(W.scalesF16 + blk));
            if (sc != 0.0f) {
                wsc[r][j] = sc;
                qb_unpack_block4(W.q + blk * 16, dst32);
                continue;
            }
        }
        wsc[r][j] = 0.0f;
        qb_zero_block32(dst32);
    }
}
#endif

template <typename WType>
__global__ void __launch_bounds__(MMA_THREADS) quant_matmul_mma_kernel(
    WType W,
    const int8_t* __restrict__ xq,
    const float* __restrict__ xsc,
    float* __restrict__ y,
    int rows,
    int n,
    int cols,
    int blocksPerRow
) {
#if __CUDA_ARCH__ >= 800
    __shared__ __align__(16) int8_t ws[2][MMA_BM][MMA_LDS];
    __shared__ __align__(16) int8_t xs[2][MMA_BN][MMA_LDS];
    __shared__ float wsc[2][MMA_BM][MMA_KB];
    __shared__ float xss[2][MMA_BN][MMA_KB];

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int t = lane & 3;
    const int warpM = (warp >> 1) * 32;
    const int warpN = (warp & 1) * 32;
    const int mBase = blockIdx.x * MMA_BM;
    const int nBase = blockIdx.y * MMA_BN;
    const int nTiles = (blocksPerRow + MMA_KB - 1) / MMA_KB;

    auto loadStage = [&](int stage, int kt) {
        const int kb0 = kt * MMA_KB;
        const int kByte = kb0 * 32;
        load_mma_w_stage<MMA_BM, MMA_BK, MMA_KB, MMA_LDS, MMA_THREADS>(
            W, ws[stage], wsc[stage], tid, mBase, kb0, rows, cols, blocksPerRow);
        for (int c = tid; c < MMA_BN * (MMA_BK / 16); c += MMA_THREADS) {
            const int r = c >> 3;
            const int off = (c & 7) * 16;
            const bool inK = kByte + off < cols;
            const int gn = nBase + r;
            const bool xv = inK && gn < n;
            mma_cp_async16(&xs[stage][r][off], xv ? xq + (size_t)gn * cols + kByte + off : xq, xv);
        }
        for (int c = tid; c < MMA_BN * MMA_KB; c += MMA_THREADS) {
            const int r = c / MMA_KB;
            const int j = c % MMA_KB;
            const bool inK = kb0 + j < blocksPerRow;
            const int gn = nBase + r;
            const bool xv = inK && gn < n;
            mma_cp_async4(&xss[stage][r][j], xv ? xsc + (size_t)gn * blocksPerRow + kb0 + j : xsc, xv);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };

    float acc[2][4][4];
#pragma unroll
    for (int mi = 0; mi < 2; mi++)
#pragma unroll
        for (int ni = 0; ni < 4; ni++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[mi][ni][e] = 0.0f;

    loadStage(0, 0);
    for (int kt = 0; kt < nTiles; kt++) {
        const int s = kt & 1;
        if (kt + 1 < nTiles) {
            loadStage(s ^ 1, kt + 1);
            asm volatile("cp.async.wait_group 1;\n" ::);
        } else {
            asm volatile("cp.async.wait_group 0;\n" ::);
        }
        __syncthreads();

        const int kbValid = min(MMA_KB, blocksPerRow - kt * MMA_KB);
        for (int j = 0; j < kbValid; j++) {
            const int koff = j * 32 + t * 4;
            unsigned a[2][4], b[4][2];
            float wsv[2][2], xsv[4][2];
#pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                const int r0 = warpM + mi * 16 + g;
                a[mi][0] = *reinterpret_cast<const unsigned*>(&ws[s][r0][koff]);
                a[mi][1] = *reinterpret_cast<const unsigned*>(&ws[s][r0 + 8][koff]);
                a[mi][2] = *reinterpret_cast<const unsigned*>(&ws[s][r0][koff + 16]);
                a[mi][3] = *reinterpret_cast<const unsigned*>(&ws[s][r0 + 8][koff + 16]);
                wsv[mi][0] = wsc[s][r0][j];
                wsv[mi][1] = wsc[s][r0 + 8][j];
            }
#pragma unroll
            for (int ni = 0; ni < 4; ni++) {
                const int n0 = warpN + ni * 8;
                b[ni][0] = *reinterpret_cast<const unsigned*>(&xs[s][n0 + g][koff]);
                b[ni][1] = *reinterpret_cast<const unsigned*>(&xs[s][n0 + g][koff + 16]);
                xsv[ni][0] = xss[s][n0 + t * 2][j];
                xsv[ni][1] = xss[s][n0 + t * 2 + 1][j];
            }
#pragma unroll
            for (int mi = 0; mi < 2; mi++) {
#pragma unroll
                for (int ni = 0; ni < 4; ni++) {
                    int c0, c1, c2, c3;
                    asm volatile(
                        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
                        : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3)
                        : "r"(a[mi][0]), "r"(a[mi][1]), "r"(a[mi][2]), "r"(a[mi][3]),
                          "r"(b[ni][0]), "r"(b[ni][1]), "r"(0));
                    acc[mi][ni][0] = fmaf(wsv[mi][0] * xsv[ni][0], (float)c0, acc[mi][ni][0]);
                    acc[mi][ni][1] = fmaf(wsv[mi][0] * xsv[ni][1], (float)c1, acc[mi][ni][1]);
                    acc[mi][ni][2] = fmaf(wsv[mi][1] * xsv[ni][0], (float)c2, acc[mi][ni][2]);
                    acc[mi][ni][3] = fmaf(wsv[mi][1] * xsv[ni][1], (float)c3, acc[mi][ni][3]);
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int mi = 0; mi < 2; mi++) {
#pragma unroll
        for (int ni = 0; ni < 4; ni++) {
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int m = mBase + warpM + mi * 16 + g + (e >> 1) * 8;
                const int nn = nBase + warpN + ni * 8 + t * 2 + (e & 1);
                if (m < rows && nn < n) {
                    y[(size_t)nn * rows + m] = acc[mi][ni][e];
                }
            }
        }
    }
#else
    __trap();
#endif
}

template <typename WType>
__global__ void __launch_bounds__(MMA_HL_THREADS) quant_matmul_mma_hilo_kernel(
    WType W,
    const int8_t* __restrict__ xq,
    const float* __restrict__ xsc,
    const int8_t* __restrict__ xq2,
    const float* __restrict__ xsc2,
    float* __restrict__ y,
    int rows,
    int n,
    int cols,
    int blocksPerRow
) {
#if __CUDA_ARCH__ >= 800
    __shared__ __align__(16) int8_t ws[2][MMA_HL_BM][MMA_HL_LDS];
    __shared__ __align__(16) int8_t xs[2][MMA_HL_BN][MMA_HL_LDS];
    __shared__ __align__(16) int8_t xs2[2][MMA_HL_BN][MMA_HL_LDS];
    __shared__ float wsc[2][MMA_HL_BM][MMA_HL_KB];
    __shared__ float xss[2][MMA_HL_BN][MMA_HL_KB];
    __shared__ float xss2[2][MMA_HL_BN][MMA_HL_KB];

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int t = lane & 3;
    const int warpM = (warp >> 1) * 32;
    const int warpN = (warp & 1) * 32;
    const int mBase = blockIdx.x * MMA_HL_BM;
    const int nBase = blockIdx.y * MMA_HL_BN;
    const int nTiles = (blocksPerRow + MMA_HL_KB - 1) / MMA_HL_KB;

    auto loadStage = [&](int stage, int kt) {
        const int kb0 = kt * MMA_HL_KB;
        const int kByte = kb0 * 32;
        load_mma_w_stage<MMA_HL_BM, MMA_HL_BK, MMA_HL_KB, MMA_HL_LDS, MMA_HL_THREADS>(
            W, ws[stage], wsc[stage], tid, mBase, kb0, rows, cols, blocksPerRow);
        for (int c = tid; c < MMA_HL_BN * (MMA_HL_BK / 16); c += MMA_HL_THREADS) {
            const int r = c >> 2;
            const int off = (c & 3) * 16;
            const bool inK = kByte + off < cols;
            const int gn = nBase + r;
            const bool xv = inK && gn < n;
            mma_cp_async16(&xs[stage][r][off], xv ? xq + (size_t)gn * cols + kByte + off : xq, xv);
            mma_cp_async16(&xs2[stage][r][off], xv ? xq2 + (size_t)gn * cols + kByte + off : xq2, xv);
        }
        for (int c = tid; c < MMA_HL_BN * MMA_HL_KB; c += MMA_HL_THREADS) {
            const int r = c / MMA_HL_KB;
            const int j = c % MMA_HL_KB;
            const bool inK = kb0 + j < blocksPerRow;
            const int gn = nBase + r;
            const bool xv = inK && gn < n;
            mma_cp_async4(&xss[stage][r][j], xv ? xsc + (size_t)gn * blocksPerRow + kb0 + j : xsc, xv);
            mma_cp_async4(&xss2[stage][r][j], xv ? xsc2 + (size_t)gn * blocksPerRow + kb0 + j : xsc2, xv);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };

    float acc[2][4][4];
#pragma unroll
    for (int mi = 0; mi < 2; mi++)
#pragma unroll
        for (int ni = 0; ni < 4; ni++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[mi][ni][e] = 0.0f;

    loadStage(0, 0);
    for (int kt = 0; kt < nTiles; kt++) {
        const int s = kt & 1;
        if (kt + 1 < nTiles) {
            loadStage(s ^ 1, kt + 1);
            asm volatile("cp.async.wait_group 1;\n" ::);
        } else {
            asm volatile("cp.async.wait_group 0;\n" ::);
        }
        __syncthreads();

        const int kbValid = min(MMA_HL_KB, blocksPerRow - kt * MMA_HL_KB);
        for (int j = 0; j < kbValid; j++) {
            const int koff = j * 32 + t * 4;
            unsigned a[2][4], b[4][2], b2[4][2];
            float wsv[2][2], xsv[4][2], xsv2[4][2];
#pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                const int r0 = warpM + mi * 16 + g;
                a[mi][0] = *reinterpret_cast<const unsigned*>(&ws[s][r0][koff]);
                a[mi][1] = *reinterpret_cast<const unsigned*>(&ws[s][r0 + 8][koff]);
                a[mi][2] = *reinterpret_cast<const unsigned*>(&ws[s][r0][koff + 16]);
                a[mi][3] = *reinterpret_cast<const unsigned*>(&ws[s][r0 + 8][koff + 16]);
                wsv[mi][0] = wsc[s][r0][j];
                wsv[mi][1] = wsc[s][r0 + 8][j];
            }
#pragma unroll
            for (int ni = 0; ni < 4; ni++) {
                const int n0 = warpN + ni * 8;
                b[ni][0] = *reinterpret_cast<const unsigned*>(&xs[s][n0 + g][koff]);
                b[ni][1] = *reinterpret_cast<const unsigned*>(&xs[s][n0 + g][koff + 16]);
                b2[ni][0] = *reinterpret_cast<const unsigned*>(&xs2[s][n0 + g][koff]);
                b2[ni][1] = *reinterpret_cast<const unsigned*>(&xs2[s][n0 + g][koff + 16]);
                xsv[ni][0] = xss[s][n0 + t * 2][j];
                xsv[ni][1] = xss[s][n0 + t * 2 + 1][j];
                xsv2[ni][0] = xss2[s][n0 + t * 2][j];
                xsv2[ni][1] = xss2[s][n0 + t * 2 + 1][j];
            }
#pragma unroll
            for (int mi = 0; mi < 2; mi++) {
#pragma unroll
                for (int ni = 0; ni < 4; ni++) {
                    int c0, c1, c2, c3;
                    int d0, d1, d2, d3;
                    asm volatile(
                        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
                        : "=r"(c0), "=r"(c1), "=r"(c2), "=r"(c3)
                        : "r"(a[mi][0]), "r"(a[mi][1]), "r"(a[mi][2]), "r"(a[mi][3]),
                          "r"(b[ni][0]), "r"(b[ni][1]), "r"(0));
                    asm volatile(
                        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
                        : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
                        : "r"(a[mi][0]), "r"(a[mi][1]), "r"(a[mi][2]), "r"(a[mi][3]),
                          "r"(b2[ni][0]), "r"(b2[ni][1]), "r"(0));
                    acc[mi][ni][0] = fmaf(wsv[mi][0] * xsv[ni][0], (float)c0, fmaf(wsv[mi][0] * xsv2[ni][0], (float)d0, acc[mi][ni][0]));
                    acc[mi][ni][1] = fmaf(wsv[mi][0] * xsv[ni][1], (float)c1, fmaf(wsv[mi][0] * xsv2[ni][1], (float)d1, acc[mi][ni][1]));
                    acc[mi][ni][2] = fmaf(wsv[mi][1] * xsv[ni][0], (float)c2, fmaf(wsv[mi][1] * xsv2[ni][0], (float)d2, acc[mi][ni][2]));
                    acc[mi][ni][3] = fmaf(wsv[mi][1] * xsv[ni][1], (float)c3, fmaf(wsv[mi][1] * xsv2[ni][1], (float)d3, acc[mi][ni][3]));
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int mi = 0; mi < 2; mi++) {
#pragma unroll
        for (int ni = 0; ni < 4; ni++) {
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int m = mBase + warpM + mi * 16 + g + (e >> 1) * 8;
                const int nn = nBase + warpN + ni * 8 + t * 2 + (e & 1);
                if (m < rows && nn < n) {
                    y[(size_t)nn * rows + m] = acc[mi][ni][e];
                }
            }
        }
    }
#else
    __trap();
#endif
}

template <typename WType>
static int launchBatchF32(WType W, const float* x, float* y, int rows, int n, int cols, int blocksPerRow, cudaStream_t stream) {
    const int blockSize = TILE_R * TILE_N;
    dim3 block(blockSize);
    dim3 grid((rows + TILE_R - 1) / TILE_R, (n + TILE_N - 1) / TILE_N);
    quant_matvec_batch_f32_kernel<<<grid, block, 0, stream>>>(W, x, y, rows, n, cols, blocksPerRow);
    return cudaGetLastError();
}

template <typename WType>
static int launchBatchDp4a(WType W, const float* x, int8_t* xq, float* xsc, float* y, int rows, int n, int cols, int blocksPerRow, cudaStream_t stream) {
    const int total = n * blocksPerRow;
    const int qThreads = 256;
    quantize_x_blocks_kernel<<<(total + qThreads - 1) / qThreads, qThreads, 0, stream>>>(x, xq, xsc, n, cols, blocksPerRow);
    cudaError_t qe = cudaGetLastError();
    if (qe != cudaSuccess) return qe;

    const int blockSize = TILE_R * TILE_N;
    dim3 block(blockSize);
    dim3 grid((rows + TILE_R - 1) / TILE_R, (n + TILE_N - 1) / TILE_N);
    quant_matvec_batch_dp4a_kernel<<<grid, block, 0, stream>>>(W, xq, xsc, y, rows, n, cols, blocksPerRow);
    return cudaGetLastError();
}

extern "C" {

// mantleCudaQuantBatchMMASupported reports whether the current device runs the
// tensor-core kernel (compute capability 8.0 or later).
__host__ int mantleCudaQuantBatchMMASupported(void) {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return 0;
    int major = 0;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess) return 0;
    return major >= 8 ? 1 : 0;
}

} // extern "C"

template <typename WType>
static int launchBatchMMA(WType W, const void* wPtr, const float* x, int8_t* xq, float* xsc, float* y, int rows, int n, int cols, int blocksPerRow, cudaStream_t stream) {
    if (cols != blocksPerRow * 32 || ((reinterpret_cast<uintptr_t>(wPtr) | reinterpret_cast<uintptr_t>(xq)) & 15) != 0 ||
        !mantleCudaQuantBatchMMASupported()) {
        return cudaErrorNotSupported;
    }
    const int total = n * blocksPerRow;
    const int qThreads = 256;
    quantize_x_blocks_kernel<<<(total + qThreads - 1) / qThreads, qThreads, 0, stream>>>(x, xq, xsc, n, cols, blocksPerRow);
    cudaError_t qe = cudaGetLastError();
    if (qe != cudaSuccess) return qe;

    dim3 grid((rows + MMA_BM - 1) / MMA_BM, (n + MMA_BN - 1) / MMA_BN);
    quant_matmul_mma_kernel<<<grid, MMA_THREADS, 0, stream>>>(W, xq, xsc, y, rows, n, cols, blocksPerRow);
    return cudaGetLastError();
}

template <typename WType>
static int launchBatchMMAHilo(WType W, const void* wPtr, const float* x, int8_t* xq, float* xsc, int8_t* xq2, float* xsc2, float* y, int rows, int n, int cols, int blocksPerRow, cudaStream_t stream) {
    if (cols != blocksPerRow * 32 ||
        ((reinterpret_cast<uintptr_t>(wPtr) | reinterpret_cast<uintptr_t>(xq) | reinterpret_cast<uintptr_t>(xq2)) & 15) != 0 ||
        !mantleCudaQuantBatchMMASupported()) {
        return cudaErrorNotSupported;
    }
    const int total = n * blocksPerRow;
    const int qThreads = 256;
    quantize_x_blocks_hi_lo_kernel<<<(total + qThreads - 1) / qThreads, qThreads, 0, stream>>>(
        x, xq, xsc, xq2, xsc2, n, cols, blocksPerRow);
    cudaError_t qe = cudaGetLastError();
    if (qe != cudaSuccess) return qe;

    dim3 grid((rows + MMA_HL_BM - 1) / MMA_HL_BM, (n + MMA_HL_BN - 1) / MMA_HL_BN);
    quant_matmul_mma_hilo_kernel<<<grid, MMA_HL_THREADS, 0, stream>>>(
        W, xq, xsc, xq2, xsc2, y, rows, n, cols, blocksPerRow);
    return cudaGetLastError();
}

extern "C" {

// --- Cached-int8 batched launchers ---

__host__ int mantleCudaQuantMatVecCachedBatch(
    const int8_t* q,
    const float* scales,
    const float* x,
    float* y,
    int rows,
    int n,
    int cols,
    int blocksPerRow,
    cudaStream_t stream
) {
    if (!q || !scales || !x || !y || rows <= 0 || n <= 0 || cols <= 0 || blocksPerRow <= 0) {
        return cudaErrorInvalidValue;
    }
    return launchBatchF32(CachedInt8Weights{q, scales}, x, y, rows, n, cols, blocksPerRow, stream);
}

__host__ int mantleCudaQuantMatVecCachedBatchDp4a(
    const int8_t* q,
    const float* scales,
    const float* x,
    int8_t* xq,
    float* xsc,
    float* y,
    int rows,
    int n,
    int cols,
    int blocksPerRow,
    cudaStream_t stream
) {
    if (!q || !scales || !x || !xq || !xsc || !y || rows <= 0 || n <= 0 || cols <= 0 || blocksPerRow <= 0) {
        return cudaErrorInvalidValue;
    }
    return launchBatchDp4a(CachedInt8Weights{q, scales}, x, xq, xsc, y, rows, n, cols, blocksPerRow, stream);
}

__host__ int mantleCudaQuantMatVecCachedBatchMMA(
    const int8_t* q,
    const float* scales,
    const float* x,
    int8_t* xq,
    float* xsc,
    float* y,
    int rows,
    int n,
    int cols,
    int blocksPerRow,
    cudaStream_t stream
) {
    if (!q || !scales || !x || !xq || !xsc || !y || rows <= 0 || n <= 0 || cols <= 0 || blocksPerRow <= 0) {
        return cudaErrorInvalidValue;
    }
    return launchBatchMMA(CachedInt8Weights{q, scales}, q, x, xq, xsc, y, rows, n, cols, blocksPerRow, stream);
}

__host__ int mantleCudaQuantMatVecCachedBatchMMAHilo(
    const int8_t* q,
    const float* scales,
    const float* x,
    int8_t* xq,
    float* xsc,
    int8_t* xq2,
    float* xsc2,
    float* y,
    int rows,
    int n,
    int cols,
    int blocksPerRow,
    cudaStream_t stream
) {
    if (!q || !scales || !x || !xq || !xsc || !xq2 || !xsc2 || !y || rows <= 0 || n <= 0 || cols <= 0 ||
        blocksPerRow <= 0) {
        return cudaErrorInvalidValue;
    }
    return launchBatchMMAHilo(CachedInt8Weights{q, scales}, q, x, xq, xsc, xq2, xsc2, y, rows, n, cols, blocksPerRow, stream);
}

// --- Raw K4 batched launchers ---

__host__ int mantleCudaQuantMatVecK4Batch(
    const uint8_t* qData,
    const uint16_t* superScalesF16,
    const uint8_t* subScales,
    const float* x,
    float* y,
    int rows,
    int n,
    int cols,
    int blocksPerRow,
    cudaStream_t stream
) {
    if (!qData || !superScalesF16 || !subScales || !x || !y || rows <= 0 || n <= 0 || cols <= 0 || blocksPerRow <= 0) {
        return cudaErrorInvalidValue;
    }
    const K4RawWeights W{qData, superScalesF16, subScales, (blocksPerRow + 7) / 8};
    return launchBatchF32(W, x, y, rows, n, cols, blocksPerRow, stream);
}

__host__ int mantleCudaQuantMatVecK4BatchDp4a(
    const uint8_t* qData,
    const uint16_t* superScalesF16,
    const uint8_t* subScales,
    const float* x,
    int8_t* xq,
    float* xsc,
    float* y,
    int rows,
    int n,
    int cols,
    int blocksPerRow,
    cudaStream_t stream
) {
    if (!qData || !superScalesF16 || !subScales || !x || !xq || !xsc || !y || rows <= 0 || n <= 0 || cols <= 0 || blocksPerRow <= 0) {
        return cudaErrorInvalidValue;
    }
    const K4RawWeights W{qData, superScalesF16, subScales, (blocksPerRow + 7) / 8};
    return launchBatchDp4a(W, x, xq, xsc, y, rows, n, cols, blocksPerRow, stream);
}

__host__ int mantleCudaQuantMatVecK4BatchMMA(
    const uint8_t* qData,
    const uint16_t* superScalesF16,
    const uint8_t* subScales,
    const float* x,
    int8_t* xq,
    float* xsc,
    float* y,
    int rows,
    int n,
    int cols,
    int blocksPerRow,
    cudaStream_t stream
) {
    if (!qData || !superScalesF16 || !subScales || !x || !xq || !xsc || !y || rows <= 0 || n <= 0 || cols <= 0 || blocksPerRow <= 0) {
        return cudaErrorInvalidValue;
    }
    const K4RawWeights W{qData, superScalesF16, subScales, (blocksPerRow + 7) / 8};
    return launchBatchMMA(W, qData, x, xq, xsc, y, rows, n, cols, blocksPerRow, stream);
}

__host__ int mantleCudaQuantMatVecK4BatchMMAHilo(
    const uint8_t* qData,
    const uint16_t* superScalesF16,
    const uint8_t* subScales,
    const float* x,
    int8_t* xq,
    float* xsc,
    int8_t* xq2,
    float* xsc2,
    float* y,
    int rows,
    int n,
    int cols,
    int blocksPerRow,
    cudaStream_t stream
) {
    if (!qData || !superScalesF16 || !subScales || !x || !xq || !xsc || !xq2 || !xsc2 || !y ||
        rows <= 0 || n <= 0 || cols <= 0 || blocksPerRow <= 0) {
        return cudaErrorInvalidValue;
    }
    const K4RawWeights W{qData, superScalesF16, subScales, (blocksPerRow + 7) / 8};
    return launchBatchMMAHilo(W, qData, x, xq, xsc, xq2, xsc2, y, rows, n, cols, blocksPerRow, stream);
}

// --- Raw Q4 batched launchers ---

__host__ int mantleCudaQuantMatVecQ4Batch(
    const uint8_t* qData,
    const uint16_t* scalesF16,
    const float* x,
    float* y,
    int rows,
    int n,
    int cols,
    int blocksPerRow,
    cudaStream_t stream
) {
    if (!qData || !scalesF16 || !x || !y || rows <= 0 || n <= 0 || cols <= 0 || blocksPerRow <= 0) {
        return cudaErrorInvalidValue;
    }
    return launchBatchF32(Q4RawWeights{qData, scalesF16}, x, y, rows, n, cols, blocksPerRow, stream);
}

__host__ int mantleCudaQuantMatVecQ4BatchDp4a(
    const uint8_t* qData,
    const uint16_t* scalesF16,
    const float* x,
    int8_t* xq,
    float* xsc,
    float* y,
    int rows,
    int n,
    int cols,
    int blocksPerRow,
    cudaStream_t stream
) {
    if (!qData || !scalesF16 || !x || !xq || !xsc || !y || rows <= 0 || n <= 0 || cols <= 0 || blocksPerRow <= 0) {
        return cudaErrorInvalidValue;
    }
    return launchBatchDp4a(Q4RawWeights{qData, scalesF16}, x, xq, xsc, y, rows, n, cols, blocksPerRow, stream);
}

__host__ int mantleCudaQuantMatVecQ4BatchMMA(
    const uint8_t* qData,
    const uint16_t* scalesF16,
    const float* x,
    int8_t* xq,
    float* xsc,
    float* y,
    int rows,
    int n,
    int cols,
    int blocksPerRow,
    cudaStream_t stream
) {
    if (!qData || !scalesF16 || !x || !xq || !xsc || !y || rows <= 0 || n <= 0 || cols <= 0 || blocksPerRow <= 0) {
        return cudaErrorInvalidValue;
    }
    return launchBatchMMA(Q4RawWeights{qData, scalesF16}, qData, x, xq, xsc, y, rows, n, cols, blocksPerRow, stream);
}

__host__ int mantleCudaQuantMatVecQ4BatchMMAHilo(
    const uint8_t* qData,
    const uint16_t* scalesF16,
    const float* x,
    int8_t* xq,
    float* xsc,
    int8_t* xq2,
    float* xsc2,
    float* y,
    int rows,
    int n,
    int cols,
    int blocksPerRow,
    cudaStream_t stream
) {
    if (!qData || !scalesF16 || !x || !xq || !xsc || !xq2 || !xsc2 || !y ||
        rows <= 0 || n <= 0 || cols <= 0 || blocksPerRow <= 0) {
        return cudaErrorInvalidValue;
    }
    return launchBatchMMAHilo(Q4RawWeights{qData, scalesF16}, qData, x, xq, xsc, xq2, xsc2, y, rows, n, cols, blocksPerRow, stream);
}

} // extern "C"
