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
    if (n <= 0) return 0;
    const int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    add_vectors_f32_kernel<<<gridSize, blockSize, 0, stream>>>(dst, src, n);
    return cudaGetLastError();
}

} // extern "C"