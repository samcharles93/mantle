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
#define TILE_R 16
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
    __shared__ int8_t wsh[TILE_R][CHUNK + 1];
    __shared__ float wsc[TILE_R][CHUNKS_PER_BLOCK + 1];
    __shared__ float xsh[TILE_N][CHUNK + 1];

    const int tid = threadIdx.x;
    const int r = blockIdx.x * TILE_R + (tid / TILE_N);
    const int ni = blockIdx.y * TILE_N + (tid % TILE_N);
    const bool active = (r < rows) && (ni < n);
    const int rl = tid / TILE_N;
    const int nl = tid % TILE_N;

    const int nChunks = (cols + CHUNK - 1) / CHUNK;
    float acc0 = 0.0f, acc1 = 0.0f;
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
            int c = 0;
            for (; c + 1 < cLen; c += 2) {
                acc0 += wsc[rl][c / 32] * (float)wsh[rl][c] * xsh[nl][c];
                acc1 += wsc[rl][(c + 1) / 32] * (float)wsh[rl][c + 1] * xsh[nl][c + 1];
            }
            for (; c < cLen; c++) {
                acc0 += wsc[rl][c / 32] * (float)wsh[rl][c] * xsh[nl][c];
            }
        }
        __syncthreads();
    }
    if (active) {
        y[(size_t)ni * rows + r] = acc0 + acc1;
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
