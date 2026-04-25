#include <cuda_runtime.h>

extern "C" {

__global__ void moe_accumulate_f32_kernel(
    float* __restrict__ accum,
    const float* __restrict__ src,
    float w,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        accum[i] = fmaf(w, src[i], accum[i]);
    }
}

int mantleCudaMoEAccumulateF32(
    float* accum,
    const float* src,
    float w,
    int n,
    cudaStream_t stream
) {
    if (n <= 0) {
        return cudaSuccess;
    }
    if (accum == nullptr || src == nullptr) {
        return cudaErrorInvalidValue;
    }
    int block = 256;
    int grid = (n + block - 1) / block;
    if (grid > 1024) grid = 1024;
    moe_accumulate_f32_kernel<<<grid, block, 0, stream>>>(accum, src, w, n);
    return cudaGetLastError();
}

} // extern "C"
