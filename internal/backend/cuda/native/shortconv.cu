#include <cuda_runtime.h>

extern "C" {

// ShortConv depthwise convolution kernel.
// Given InProj output (proj = [b|c|xg], each embd elements):
//   1. bx[i] = b[i] * xg[i]
//   2. convOut[i] = dot(conv_weights[i*klen..], [state[0..klen-2], bx[i]])
//   3. state update: shift left, append bx[i]
//   4. out[i] = c[i] * convOut[i]
//
// Each thread handles one channel.
__global__ void shortconv_depthwise_kernel(
    const float* __restrict__ proj,       // [3*embd] InProj output: b|c|xg
    const float* __restrict__ conv_w,     // [embd*klen] conv kernel weights (row-major)
    float*       __restrict__ state,      // [embd*(klen-1)] conv state, updated in-place
    float*       __restrict__ out,        // [embd] output for OutProj
    int embd,
    int klen
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= embd) return;

    float b  = proj[i];
    float c  = proj[embd + i];
    float xg = proj[2 * embd + i];
    float bx = b * xg;

    // Depthwise 1D convolution
    const float* kw = conv_w + i * klen;
    float sum = 0.0f;
    for (int k = 0; k < klen - 1; k++) {
        sum += kw[k] * state[k * embd + i];
    }
    sum += kw[klen - 1] * bx;

    // State update: shift left, append bx
    for (int k = 0; k < klen - 2; k++) {
        state[k * embd + i] = state[(k + 1) * embd + i];
    }
    if (klen > 1) {
        state[(klen - 2) * embd + i] = bx;
    }

    // Output: c * convOut
    out[i] = c * sum;
}

__host__ int mantleCudaShortConvDepthwise(
    const float* proj,
    const float* conv_w,
    float* state,
    float* out,
    int embd,
    int klen,
    cudaStream_t stream
) {
    if (embd <= 0 || klen <= 0) return 0;
    const int blockSize = 256;
    int gridSize = (embd + blockSize - 1) / blockSize;
    shortconv_depthwise_kernel<<<gridSize, blockSize, 0, stream>>>(
        proj, conv_w, state, out, embd, klen);
    return cudaGetLastError();
}

} // extern "C"
