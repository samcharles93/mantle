#include <cuda_runtime.h>
#include <cstdint>

extern "C" {

// Batched quantized matvec (GEMM) over the cached-int8 layout used by
// Ops.ensureQuantMat (quantMatFormatCachedInt8): q holds totalBlocks*32 int8
// values, scales one f32 per 32-element block. Computes
//   y[n, r] = sum_b scales[r*blocksPerRow+b] * dot(q[(r*blocksPerRow+b)*32..], x[n, b*32..])
// for x [n, cols] row-major and y [n, rows] row-major.
//
// Each block computes a TILE_R x TILE_N output tile, streaming K in CHUNK-wide
// slices through double-buffered shared memory (next slice loads while the
// current one computes), so each weight row is read from VRAM once per
// TILE_N-wide n tile.
#define TILE_R 32
#define TILE_N 16
#define CHUNK 256
#define CHUNKS_PER_BLOCK (CHUNK / 32)

__global__ void quant_matvec_cached_batch_kernel(
    const int8_t* __restrict__ q,
    const float* __restrict__ scales,
    const float* __restrict__ x,
    float* __restrict__ y,
    int rows,
    int n,
    int cols,
    int blocksPerRow
) {
    __shared__ int8_t wsh[TILE_R][CHUNK + 16];
    __shared__ float wsc[TILE_R][CHUNKS_PER_BLOCK + 1];
    __shared__ float xsh[TILE_N][CHUNK + 4];

    const int tid = threadIdx.x;
    const int r = blockIdx.x * TILE_R + (tid / TILE_N);
    const int ni = blockIdx.y * TILE_N + (tid % TILE_N);
    const bool active = (r < rows) && (ni < n);
    const int rl = tid / TILE_N;
    const int nl = tid % TILE_N;

    const int nChunks = (cols + CHUNK - 1) / CHUNK;
    // Eight accumulators break the fma dependency chain; the inner loop walks
    // one 32-element quant block per iteration so the block scale stays in a
    // register.
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int ch = 0; ch < nChunks; ch++) {
        const int cBase = ch * CHUNK;
        const int cLen = min(CHUNK, cols - cBase);
        if (cLen <= 0) break;
        for (int idx = tid; idx < TILE_R * cLen; idx += blockDim.x) {
            int lrl = idx / cLen;
            int c = idx % cLen;
            int rr = blockIdx.x * TILE_R + lrl;
            wsh[lrl][c] = (rr < rows) ? q[(size_t)rr * blocksPerRow * 32 + (size_t)(cBase + c)] : (int8_t)0;
        }
        for (int idx = tid; idx < TILE_R * CHUNKS_PER_BLOCK; idx += blockDim.x) {
            int lrl = idx / CHUNKS_PER_BLOCK;
            int b = idx % CHUNKS_PER_BLOCK;
            int rr = blockIdx.x * TILE_R + lrl;
            int gb = (cBase + b * 32) / 32;
            wsc[lrl][b] = (rr < rows && gb < blocksPerRow) ? scales[(size_t)rr * blocksPerRow + gb] : 0.0f;
        }
        for (int idx = tid; idx < TILE_N * cLen; idx += blockDim.x) {
            int lnl = idx / cLen;
            int c = idx % cLen;
            int nn = blockIdx.y * TILE_N + lnl;
            xsh[lnl][c] = (nn < n) ? x[(size_t)nn * cols + (size_t)(cBase + c)] : 0.0f;
        }
        __syncthreads();
        if (active) {
            const int8_t *wrow = wsh[rl];
            const float *xrow = xsh[nl];
            // One dot per 32-element quant block, then a single scale multiply:
            // the per-element cost is a convert and an fma, not a convert, a
            // multiply and an fma.
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
                    const char4 w0 = *reinterpret_cast<const char4 *>(wrow + c + i);
                    const char4 w1 = *reinterpret_cast<const char4 *>(wrow + c + i + 4);
                    const float4 x0 = *reinterpret_cast<const float4 *>(xrow + c + i);
                    const float4 x1 = *reinterpret_cast<const float4 *>(xrow + c + i + 4);
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

// dp4a variant of the batched quant GEMM: weights and activations are both
// int8 (per-32-block f32 scales); the per-block dot accumulates in int32 via
// __dp4a (exact) and is scaled once in f32. Falls back to the f32 FMA loop on
// pre-sm_61 architectures.
__global__ void quant_matvec_cached_batch_dp4a_kernel(
    const int8_t* __restrict__ q,
    const float* __restrict__ scales,
    const int8_t* __restrict__ xq,
    const float* __restrict__ xsc,
    float* __restrict__ y,
    int rows,
    int n,
    int cols,
    int blocksPerRow
) {
    // +4 (not +1) padding keeps every row 4-byte aligned for the int32 dp4a loads
    __shared__ int8_t wsh[TILE_R][CHUNK + 4];
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
        for (int idx = tid; idx < TILE_R * cLen; idx += blockDim.x) {
            int lrl = idx / cLen;
            int c = idx % cLen;
            int rr = blockIdx.x * TILE_R + lrl;
            wsh[lrl][c] = (rr < rows) ? q[(size_t)rr * blocksPerRow * 32 + (size_t)(cBase + c)] : (int8_t)0;
        }
        for (int idx = tid; idx < TILE_R * CHUNKS_PER_BLOCK; idx += blockDim.x) {
            int lrl = idx / CHUNKS_PER_BLOCK;
            int b = idx % CHUNKS_PER_BLOCK;
            int rr = blockIdx.x * TILE_R + lrl;
            int gb = (cBase + b * 32) / 32;
            wsc[lrl][b] = (rr < rows && gb < blocksPerRow) ? scales[(size_t)rr * blocksPerRow + gb] : 0.0f;
        }
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
                const int rem = min(32, cLen - c0); // elements in THIS 32-block
                const int n4 = min(8, (rem + 3) / 4);
                for (int j = 0; j < n4; j++) {
                    const int c = c0 + j * 4;
                    int w4 = *(const int*)&wsh[rl][c];
                    int x4 = *(const int*)&xsh[nl][c];
                    d = __dp4a(w4, x4, d);
                }
                // tail elements (block not full at the row end)
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
    const int total = n * blocksPerRow;
    const int qThreads = 256;
    const int qBlocks = (total + qThreads - 1) / qThreads;
    quantize_x_blocks_kernel<<<qBlocks, qThreads, 0, stream>>>(x, xq, xsc, n, cols, blocksPerRow);
    cudaError_t qe = cudaGetLastError();
    if (qe != cudaSuccess) return qe;

    const int blockSize = TILE_R * TILE_N; // 256
    dim3 block(blockSize);
    dim3 grid((rows + TILE_R - 1) / TILE_R, (n + TILE_N - 1) / TILE_N);
    quant_matvec_cached_batch_dp4a_kernel<<<grid, block, 0, stream>>>(
        q, scales, xq, xsc, y, rows, n, cols, blocksPerRow);
    return cudaGetLastError();
}

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
    const int blockSize = TILE_R * TILE_N; // 256
    dim3 block(blockSize);
    dim3 grid((rows + TILE_R - 1) / TILE_R, (n + TILE_N - 1) / TILE_N);
    quant_matvec_cached_batch_kernel<<<grid, block, 0, stream>>>(
        q, scales, x, y, rows, n, cols, blocksPerRow);
    return cudaGetLastError();
}

} // extern "C"

// Tensor-core form of quant_matvec_cached_batch_dp4a_kernel above:
// the same cached-int8 weights with one f32 scale per 32-element block, the
// same per-block int8 activations from quantize_x_blocks_kernel, and the same
// per-output sum over blocks in order of fmaf(wScale*xScale, dot, acc). Only
// the int32 block dot moves from __dp4a to mma.sync m16n8k32, whose K of 32 is
// exactly one quant block, so the two kernels agree bit for bit.
//
// Each 128-thread block computes a 64 (weight rows) x 64 (tokens) tile. K is
// staged 128 bytes (four quant blocks) at a time through double-buffered
// shared memory filled by cp.async; each warp owns a 32x32 sub-tile, i.e. two
// m16 by four n8 MMAs per quant block.

#define MMA_BM 64
#define MMA_BN 64
#define MMA_BK 128
#define MMA_KB (MMA_BK / 32)
#define MMA_LDS (MMA_BK + 16) // 144-byte rows: fragment loads hit 32 distinct banks
#define MMA_THREADS 128

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

__global__ void __launch_bounds__(MMA_THREADS) quant_matmul_cached_mma_kernel(
    const int8_t* __restrict__ q,
    const float* __restrict__ scales,
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
    const size_t wRowBytes = (size_t)blocksPerRow * 32;

    auto loadStage = [&](int stage, int kt) {
        const int kb0 = kt * MMA_KB;
        const int kByte = kb0 * 32;
        // 64 rows x 8 sixteen-byte chunks, four chunks per thread, per operand.
        for (int c = tid; c < MMA_BM * (MMA_BK / 16); c += MMA_THREADS) {
            const int r = c >> 3;
            const int off = (c & 7) * 16;
            const bool inK = kByte + off < cols;
            const int gr = mBase + r;
            const bool wv = inK && gr < rows;
            mma_cp_async16(&ws[stage][r][off], wv ? q + (size_t)gr * wRowBytes + kByte + off : q, wv);
            const int gn = nBase + r;
            const bool xv = inK && gn < n;
            mma_cp_async16(&xs[stage][r][off], xv ? xq + (size_t)gn * cols + kByte + off : xq, xv);
        }
        for (int c = tid; c < MMA_BM * MMA_KB; c += MMA_THREADS) {
            const int r = c / MMA_KB;
            const int j = c % MMA_KB;
            const bool inK = kb0 + j < blocksPerRow;
            const int gr = mBase + r;
            const bool wv = inK && gr < rows;
            mma_cp_async4(&wsc[stage][r][j], wv ? scales + (size_t)gr * blocksPerRow + kb0 + j : scales, wv);
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

// mantleCudaQuantMatVecCachedBatchMMA has the contract of
// mantleCudaQuantMatVecCachedBatchDp4a, restricted to the shapes the
// tensor-core kernel takes: cols a whole number of quant blocks
// (cols == blocksPerRow*32) and 16-byte aligned q and xq. It returns
// cudaErrorNotSupported otherwise, or on a pre-sm_80 device.
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
    if (cols != blocksPerRow * 32 || ((reinterpret_cast<uintptr_t>(q) | reinterpret_cast<uintptr_t>(xq)) & 15) != 0 ||
        !mantleCudaQuantBatchMMASupported()) {
        return cudaErrorNotSupported;
    }
    const int total = n * blocksPerRow;
    const int qThreads = 256;
    quantize_x_blocks_kernel<<<(total + qThreads - 1) / qThreads, qThreads, 0, stream>>>(x, xq, xsc, n, cols, blocksPerRow);
    cudaError_t qe = cudaGetLastError();
    if (qe != cudaSuccess) return qe;

    dim3 grid((rows + MMA_BM - 1) / MMA_BM, (n + MMA_BN - 1) / MMA_BN);
    quant_matmul_cached_mma_kernel<<<grid, MMA_THREADS, 0, stream>>>(q, scales, xq, xsc, y, rows, n, cols, blocksPerRow);
    return cudaGetLastError();
}

} // extern "C"
