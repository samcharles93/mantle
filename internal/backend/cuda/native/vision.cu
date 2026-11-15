#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

// Vision-tower kernels for the device-side LFM3-VL encoder
// (internal/backend/cuda/vision_lfm3vl.go). All kernels are F32 and mirror the
// host reference in internal/backend/simd/vision_lfm3vl.go.

#define VISION_ATTN_WARPS 8  // warps per block; each warp owns one query row
#define VISION_ATTN_BK 32    // key/value tile
#define VISION_MAX_HEAD_DIM 128
#define VISION_ATTN_LANES_PER_DIM (VISION_MAX_HEAD_DIM / 32) // <= 4

__device__ __forceinline__ float warpSumF32(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float warpMaxF32(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

extern "C" {

// ---------------------------------------------------------------------------
// Axial 2D RoPE apply: x is [n][heads][headDim], cosT/sinT are [n][headDim]
// with both halves duplicated (host-built tables). For each pair (d, d+half):
//   x[d]        = x1*cos[d]        - x2*sin[d]
//   x[d+half]   = x2*cos[half+d]   + x1*sin[half+d]
// ---------------------------------------------------------------------------
__global__ void vision_rope_f32_kernel(
    float* __restrict__ x,
    const float* __restrict__ cosT,
    const float* __restrict__ sinT,
    int n,
    int heads,
    int headDim)
{
    const int half = headDim / 2;
    const long long total = (long long)n * heads * half;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += (long long)gridDim.x * blockDim.x) {
        const int d = (int)(idx % half);
        const long long rem = idx / half;
        const int hi = (int)(rem % heads);
        const int p = (int)(rem / heads);
        const long long base = ((long long)p * heads + hi) * headDim;
        const float* cos = cosT + (long long)p * headDim;
        const float* sin = sinT + (long long)p * headDim;
        const float x1 = x[base + d];
        const float x2 = x[base + half + d];
        x[base + d] = x1 * cos[d] - x2 * sin[d];
        x[base + half + d] = x2 * cos[half + d] + x1 * sin[half + d];
    }
}

int mantleCudaVisionRoPEF32(
    float* x,
    const float* cosT,
    const float* sinT,
    int n,
    int heads,
    int headDim,
    cudaStream_t stream)
{
    if (!x || !cosT || !sinT || n <= 0 || heads <= 0 || headDim <= 0 || headDim % 2 != 0) {
        return cudaErrorInvalidValue;
    }
    const long long total = (long long)n * heads * (headDim / 2);
    const int block = 256;
    const int grid = (int)((total + block - 1) / block);
    vision_rope_f32_kernel<<<grid, block, 0, stream>>>(x, cosT, sinT, n, heads, headDim);
    return cudaGetLastError();
}

// ---------------------------------------------------------------------------
// Segment-wise bidirectional (full) attention with GQA, tiled with an online
// softmax so no O(L^2) score matrix is materialized. q: [n][heads][headDim],
// k/v: [n][kvHeads][headDim], out: [n][heads][headDim]. segStart/segLen delimit
// nSeg token ranges (one per frame). Grid: (ceil(n/BQ), heads, nSeg); each
// block handles BQ query rows of one (segment, head).
// ---------------------------------------------------------------------------
__global__ void vision_attn_segments_f32_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ out,
    const int* __restrict__ segStart,
    const int* __restrict__ segLen,
    int n,
    int heads,
    int kvHeads,
    int headDim,
    float scale)
{
    extern __shared__ float smem[];
    const int BK = VISION_ATTN_BK;
    float* sk = smem;              // BK * headDim
    float* sv = sk + BK * headDim; // BK * headDim

    const int seg = blockIdx.z;
    const int head = blockIdx.y;
    const int kvh = head / (heads / kvHeads);
    const int start = segStart[seg];
    const int len = segLen[seg];
    const int qTileBase = blockIdx.x * VISION_ATTN_WARPS;
    if (qTileBase >= len || qTileBase >= n) {
        return; // uniform per block: no syncthreads above this point
    }
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int qi = qTileBase + warp; // this warp's query row within the segment
    const bool active = qi < len;

    // Each lane holds the slice of its query row it owns: dims lane, lane+32, ...
    float qreg[VISION_ATTN_LANES_PER_DIM];
    const int qd = (headDim + 31) / 32; // dims per lane
    if (active) {
        const float* qrow = q + ((long long)(start + qi) * heads + head) * headDim;
        for (int j = 0; j < qd; j++) {
            const int d = lane + 32 * j;
            qreg[j] = (d < headDim) ? qrow[d] : 0.0f;
        }
    }

    float acc[VISION_ATTN_LANES_PER_DIM];
    for (int j = 0; j < qd; j++) acc[j] = 0.0f;
    float m = -1e30f;
    float l = 0.0f;

    const int nKB = (len + BK - 1) / BK;
    for (int kb = 0; kb < nKB; kb++) {
        const int kBase = kb * BK;
        for (int i = threadIdx.x; i < BK * headDim; i += blockDim.x) {
            const int r = i / headDim;
            const int c = i % headDim;
            const int col = kBase + r;
            float kk = 0.0f;
            float vv = 0.0f;
            if (col < len) {
                const long long off = ((long long)(start + col) * kvHeads + kvh) * headDim + c;
                kk = k[off];
                vv = v[off];
            }
            sk[i] = kk;
            sv[i] = vv;
        }
        __syncthreads();
        if (active) {
            float s[BK];
            for (int ki = 0; ki < BK; ki++) {
                const float* krow = sk + ki * headDim;
                float dot = 0.0f;
                for (int j = 0; j < qd; j++) {
                    const int d = lane + 32 * j;
                    if (d < headDim) dot += qreg[j] * krow[d];
                }
                dot = warpSumF32(dot);
                const int col = kBase + ki;
                s[ki] = (col < len) ? dot * scale : -1e30f;
            }
            float mNew = m;
            for (int ki = 0; ki < BK; ki++) mNew = fmaxf(mNew, s[ki]);
            mNew = warpMaxF32(mNew);
            const float corr = __expf(m - mNew); // 0 on the first tile
            // Every lane holds the full row of scores (post warp-reduce), so
            // pSum is already the complete sum — do not warp-reduce it again.
            float pSum = 0.0f;
            for (int ki = 0; ki < BK; ki++) {
                s[ki] = __expf(s[ki] - mNew);
                pSum += s[ki];
            }
            l = l * corr + pSum;
            for (int j = 0; j < qd; j++) acc[j] *= corr;
            for (int ki = 0; ki < BK; ki++) {
                const float* vrow = sv + ki * headDim;
                const float pv = s[ki];
                for (int j = 0; j < qd; j++) {
                    const int d = lane + 32 * j;
                    if (d < headDim) acc[j] += pv * vrow[d];
                }
            }
            m = mNew;
        }
        __syncthreads(); // sk/sv reused next iteration
    }
    if (active) {
        const float inv = 1.0f / l;
        float* orow = out + ((long long)(start + qi) * heads + head) * headDim;
        for (int j = 0; j < qd; j++) {
            const int d = lane + 32 * j;
            if (d < headDim) orow[d] = acc[j] * inv;
        }
    }
}

int mantleCudaVisionAttnSegmentsF32(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    const int* segStart,
    const int* segLen,
    int nSeg,
    int n,
    int heads,
    int kvHeads,
    int headDim,
    float scale,
    cudaStream_t stream)
{
    if (!q || !k || !v || !out || !segStart || !segLen || nSeg <= 0 || n <= 0 ||
        heads <= 0 || kvHeads <= 0 || heads % kvHeads != 0 || headDim <= 0 ||
        headDim > VISION_MAX_HEAD_DIM) {
        return cudaErrorInvalidValue;
    }
    dim3 grid((unsigned)((n + VISION_ATTN_WARPS - 1) / VISION_ATTN_WARPS), (unsigned)heads, (unsigned)nSeg);
    const size_t shmem = (size_t)(2 * VISION_ATTN_BK) * headDim * sizeof(float);
    vision_attn_segments_f32_kernel<<<grid, VISION_ATTN_WARPS * 32, shmem, stream>>>(
        q, k, v, out, segStart, segLen, n, heads, kvHeads, headDim, scale);
    return cudaGetLastError();
}

// ---------------------------------------------------------------------------
// Gated depthwise 2D convolution over merge-block-major tokens. in: [n][3*C]
// (B, C, x chunks), convW: [C][kSize][kSize], out: [n][C]. Grid:
// (ceil(maxFrameTokens*C/256), nFrames). Token p of frame f maps to raster
// (y, x) = blockMajorToRaster(p - start, H, W, merge); the gated input
// B[c]*x[c] of the neighbours is convolved with zero padding, then scaled by
// the C chunk of token p.
// ---------------------------------------------------------------------------
__global__ void vision_conv2d_gated_f32_kernel(
    const float* __restrict__ in,
    const float* __restrict__ convW,
    float* __restrict__ out,
    const int* __restrict__ frameStart,
    const int* __restrict__ frameH,
    const int* __restrict__ frameW,
    int nFrames,
    int n,
    int C,
    int kSize,
    int merge)
{
    const int f = blockIdx.y;
    const int start = frameStart[f];
    const int end = (f + 1 < nFrames) ? frameStart[f + 1] : n;
    const int H = frameH[f];
    const int W = frameW[f];
    const int pad = kSize / 2;
    const int mm = merge * merge;
    const int Wm = W / merge;
    const long long frameTotal = (long long)(end - start) * C;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x; idx < frameTotal;
         idx += (long long)gridDim.x * blockDim.x) {
        const int p = start + (int)(idx / C);
        const int c = (int)(idx % C);
        const int i = p - start;
        const int bi = i / mm;
        const int wi = i % mm;
        const int y = (bi / Wm) * merge + wi / merge;
        const int x = (bi % Wm) * merge + wi % merge;
        const float* kc = convW + (long long)c * kSize * kSize;
        float acc = 0.0f;
        for (int ky = 0; ky < kSize; ky++) {
            const int iy = y + ky - pad;
            if (iy < 0 || iy >= H) continue;
            for (int kx = 0; kx < kSize; kx++) {
                const int ix = x + kx - pad;
                if (ix < 0 || ix >= W) continue;
                const int pn = start + ((iy / merge) * Wm + (ix / merge)) * mm +
                               (iy % merge) * merge + (ix % merge);
                const long long base = (long long)pn * 3 * C + c;
                acc += kc[ky * kSize + kx] * (in[base] * in[base + 2 * C]);
            }
        }
        out[(long long)start * C + idx] = acc * in[(long long)p * 3 * C + C + c];
    }
}

int mantleCudaVisionConv2dGatedF32(
    const float* in,
    const float* convW,
    float* out,
    const int* frameStart,
    const int* frameH,
    const int* frameW,
    int nFrames,
    int n,
    int maxFrameTokens,
    int channels,
    int kSize,
    int merge,
    cudaStream_t stream)
{
    if (!in || !convW || !out || !frameStart || !frameH || !frameW || nFrames <= 0 ||
        n <= 0 || maxFrameTokens <= 0 || channels <= 0 || kSize <= 0 ||
        kSize % 2 == 0 || merge <= 0) {
        return cudaErrorInvalidValue;
    }
    const int block = 256;
    const long long perFrame = (long long)maxFrameTokens * channels;
    const int gridX = (int)((perFrame + block - 1) / block);
    dim3 grid((unsigned)gridX, (unsigned)nFrames, 1);
    vision_conv2d_gated_f32_kernel<<<grid, block, 0, stream>>>(
        in, convW, out, frameStart, frameH, frameW, nFrames, n, channels, kSize, merge);
    return cudaGetLastError();
}

// ---------------------------------------------------------------------------
// Row-wise LayerNorm: one block per row of the [rows][cols] matrix.
// out[i] = (x[i]-mean) * (1/sqrt(var+eps)) * weight[i] + bias[i] (bias optional).
// Safe in place (out may alias x).
// ---------------------------------------------------------------------------
__global__ void layer_norm_rows_f32_kernel(
    float* __restrict__ out,
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float eps,
    int rows,
    int cols)
{
    __shared__ float shmem[256];
    __shared__ float scaleS;
    const int row = blockIdx.x;
    if (row >= rows) return;
    const float* xr = x + (long long)row * cols;
    float* orow = out + (long long)row * cols;
    const int tid = threadIdx.x;

    float sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) sum += xr[i];
    shmem[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shmem[tid] += shmem[tid + s];
        __syncthreads();
    }
    const float mean = shmem[0] / (float)cols;
    __syncthreads();

    float varAcc = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        const float d = xr[i] - mean;
        varAcc += d * d;
    }
    shmem[tid] = varAcc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shmem[tid] += shmem[tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        scaleS = 1.0f / sqrtf(shmem[0] / (float)cols + eps);
    }
    __syncthreads();

    for (int i = tid; i < cols; i += blockDim.x) {
        const float b = bias ? bias[i] : 0.0f;
        orow[i] = (xr[i] - mean) * scaleS * weight[i] + b;
    }
}

int mantleCudaLayerNormRowsF32(
    float* out,
    const float* x,
    const float* weight,
    const float* bias,
    float eps,
    int rows,
    int cols,
    cudaStream_t stream)
{
    if (!out || !x || !weight || rows <= 0 || cols <= 0) {
        return cudaErrorInvalidValue;
    }
    const int block = 256;
    layer_norm_rows_f32_kernel<<<rows, block, 0, stream>>>(out, x, weight, bias, eps, rows, cols);
    return cudaGetLastError();
}

// ---------------------------------------------------------------------------
// Exact (erf) GELU in place over [rows][cols], with optional per-column bias:
// x[i] = gelu(x[i] + bias[i % cols]). erf is evaluated in double to match the
// host's math.Erf(float64) path.
// ---------------------------------------------------------------------------
__global__ void gelu_exact_bias_f32_kernel(
    float* __restrict__ x,
    const float* __restrict__ bias,
    int rows,
    int cols)
{
    const long long total = (long long)rows * cols;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += (long long)gridDim.x * blockDim.x) {
        float v = x[idx];
        if (bias) v += bias[idx % cols];
        x[idx] = 0.5f * v * (1.0f + (float)erf((double)v * 0.70710678118654752440));
    }
}

int mantleCudaGeluExactBiasF32(
    float* x,
    const float* bias,
    int rows,
    int cols,
    cudaStream_t stream)
{
    if (!x || rows <= 0 || cols <= 0) {
        return cudaErrorInvalidValue;
    }
    const long long total = (long long)rows * cols;
    const int block = 256;
    const int grid = (int)((total + block - 1) / block);
    gelu_exact_bias_f32_kernel<<<grid, block, 0, stream>>>(x, bias, rows, cols);
    return cudaGetLastError();
}

// ---------------------------------------------------------------------------
// Row-wise bias add in place: out[i] += bias[i % cols] over [rows][cols].
// ---------------------------------------------------------------------------
__global__ void add_bias_rows_f32_kernel(
    float* __restrict__ out,
    const float* __restrict__ bias,
    int rows,
    int cols)
{
    const long long total = (long long)rows * cols;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += (long long)gridDim.x * blockDim.x) {
        out[idx] += bias[idx % cols];
    }
}

int mantleCudaAddBiasRowsF32(
    float* out,
    const float* bias,
    int rows,
    int cols,
    cudaStream_t stream)
{
    if (!out || !bias || rows <= 0 || cols <= 0) {
        return cudaErrorInvalidValue;
    }
    const long long total = (long long)rows * cols;
    const int block = 256;
    const int grid = (int)((total + block - 1) / block);
    add_bias_rows_f32_kernel<<<grid, block, 0, stream>>>(out, bias, rows, cols);
    return cudaGetLastError();
}

} // extern "C"
