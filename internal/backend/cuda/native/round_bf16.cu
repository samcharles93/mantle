#include <cuda_runtime.h>

extern "C" {

// In-place BF16 rounding: truncates each float32 element to BF16 precision
// using round-to-nearest-even (banker's rounding).
// This matches the Go implementation in internal/model/float.go:RoundFloat32ToBF16.
__global__ void round_bf16_inplace_f32_kernel(float* __restrict__ data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        unsigned int bits = __float_as_uint(data[idx]);
        unsigned int lsb  = (bits >> 16) & 1;
        unsigned int bias = 0x7FFFu + lsb;
        data[idx] = __uint_as_float((bits + bias) & 0xFFFF0000u);
    }
}

// Wrapper function that launches the kernel with appropriate grid/block size.
__host__ int mantleCudaRoundBF16InPlaceF32(
    float* data,
    int n,
    cudaStream_t stream
) {
    if (!data || n <= 0) return cudaErrorInvalidValue;
    const int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    round_bf16_inplace_f32_kernel<<<gridSize, blockSize, 0, stream>>>(data, n);
    return cudaGetLastError();
}

} // extern "C"
