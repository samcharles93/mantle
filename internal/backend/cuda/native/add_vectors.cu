#include <cuda_runtime.h>

extern "C" {

// Element-wise addition: dst[i] += src[i] for i in [0, n)
__global__ void add_vectors_f32_kernel(
    float* __restrict__ dst,
    const float* __restrict__ src,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] += src[idx];
    }
}

// Wrapper function that launches the kernel with appropriate grid/block size.
__host__ int mantleCudaAddVectorsF32(
    float* dst,
    const float* src,
    int n,
    cudaStream_t stream
) {
    if (!dst || !src || n <= 0) return cudaErrorInvalidValue;
    const int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    add_vectors_f32_kernel<<<gridSize, blockSize, 0, stream>>>(dst, src, n);
    return cudaGetLastError();
}


// Scaled addition: dst[i] += src[i] * scale (residual multiplier).
__global__ void scale_add_f32_kernel(
    float* __restrict__ dst,
    const float* __restrict__ src,
    float scale,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] += src[idx] * scale;
    }
}

__host__ int mantleCudaScaleAddF32(
    float* dst,
    const float* src,
    float scale,
    int n,
    cudaStream_t stream
) {
    if (!dst || !src || n <= 0) return cudaErrorInvalidValue;
    const int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    scale_add_f32_kernel<<<gridSize, blockSize, 0, stream>>>(dst, src, scale, n);
    return cudaGetLastError();
}

// Attention output gate: attn[row, i] *= sigmoid(gate[row, ...]).
// perHead != 0: gate has nHead values per row, one per head (LFM3).
// perHead == 0: gate has qDim values per row, one per element.
__global__ void attn_gate_scale_f32_kernel(
    float* __restrict__ attn,        // [nRows, qDim]
    const float* __restrict__ gate,  // [nRows, nHead] or [nRows, qDim]
    int nRows,
    int qDim,
    int nHead,
    int headDim,
    int perHead
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nRows * qDim;
    if (idx >= total) return;
    int row = idx / qDim;
    int i = idx % qDim;
    float g = perHead ? gate[row * nHead + (i / headDim)] : gate[idx];
    attn[idx] *= 1.0f / (1.0f + __expf(-g));
}

__host__ int mantleCudaAttnGateScaleF32(
    float* attn,
    const float* gate,
    int nRows,
    int qDim,
    int nHead,
    int headDim,
    int perHead,
    cudaStream_t stream
) {
    if (!attn || !gate || nRows <= 0 || qDim <= 0) return cudaErrorInvalidValue;
    int total = nRows * qDim;
    const int blockSize = 256;
    int gridSize = (total + blockSize - 1) / blockSize;
    attn_gate_scale_f32_kernel<<<gridSize, blockSize, 0, stream>>>(
        attn, gate, nRows, qDim, nHead, headDim, perHead);
    return cudaGetLastError();
}

// Transpose [d0, d1, d2] -> [d1, d0, d2] (used to lay out batched per-layer
// inputs [N, layers, pleDim] as contiguous [layers, N, pleDim] slices).
__global__ void transpose_3d_f32_kernel(
    float* __restrict__ dst,
    const float* __restrict__ src,
    int d0,
    int d1,
    int d2
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)d0 * d1 * d2;
    if (idx >= total) return;
    size_t k = idx % d2;
    size_t tmp = idx / d2;
    size_t j = tmp % d1;
    size_t i = tmp / d1;
    dst[(j * d0 + i) * d2 + k] = src[idx];
}

__host__ int mantleCudaTranspose3DF32(
    float* dst,
    const float* src,
    int d0,
    int d1,
    int d2,
    cudaStream_t stream
) {
    if (!dst || !src || d0 <= 0 || d1 <= 0 || d2 <= 0) return cudaErrorInvalidValue;
    size_t total = (size_t)d0 * d1 * d2;
    const int blockSize = 256;
    int gridSize = (int)((total + blockSize - 1) / blockSize);
    transpose_3d_f32_kernel<<<gridSize, blockSize, 0, stream>>>(dst, src, d0, d1, d2);
    return cudaGetLastError();
}

} // extern "C"

