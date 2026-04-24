#include <cuda_runtime.h>

extern "C" {

__global__ void scale_round_bf16_inplace_f32_kernel(float* __restrict__ data, float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float scaled = data[idx] * scale;
        unsigned int bits = __float_as_uint(scaled);
        unsigned int lsb  = (bits >> 16) & 1;
        unsigned int bias = 0x7FFFu + lsb;
        data[idx] = __uint_as_float((bits + bias) & 0xFFFF0000u);
    }
}

__host__ int mantleCudaScaleRoundBF16InPlaceF32(
    float* data,
    float scale,
    int n,
    cudaStream_t stream
) {
    if (!data || n <= 0) return cudaErrorInvalidValue;
    const int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    scale_round_bf16_inplace_f32_kernel<<<gridSize, blockSize, 0, stream>>>(data, scale, n);
    return cudaGetLastError();
}

} // extern "C"
