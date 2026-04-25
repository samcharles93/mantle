#include <cuda_runtime.h>

extern "C" {

// In-place SiLU activation: x[i] = x[i] * sigmoid(x[i]).
// Mirrors the post-conv SiLU in internal/backend/simd/mamba.go
// (fastSiluVec + scalar Silu). Grid-stride loop so small launch
// configs still cover large buffers.

__global__ void silu_f32_inplace_kernel(float *__restrict__ x, int n) {
    const int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        const float v = x[i];
        const float s = 1.0f / (1.0f + __expf(-v));
        x[i] = v * s;
    }
}

int mantleCudaSiluF32InPlace(float *x, int n, cudaStream_t stream) {
    if (n <= 0) {
        return cudaSuccess;
    }
    constexpr int kBlock = 256;
    int blocks = (n + kBlock - 1) / kBlock;
    if (blocks > 4096) {
        blocks = 4096;
    }
    silu_f32_inplace_kernel<<<blocks, kBlock, 0, stream>>>(x, n);
    return cudaGetLastError();
}

} // extern "C"
