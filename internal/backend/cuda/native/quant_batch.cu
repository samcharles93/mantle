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
