#include <cuda_runtime.h>

extern "C" {

// Mamba depthwise 1D convolution kernel.
// Mirrors mambaDepthwiseConv in internal/backend/simd/mamba.go.
//
// For each channel c in [0, channels):
//   sum = bias[c] (if has_bias != 0, else 0)
//   for k in [0, klen-1): sum += conv_w[c*klen + k] * state[k*channels + c]
//   sum += conv_w[c*klen + (klen-1)] * in[c]
//   out[c] = sum
//
// Then state is shifted left by one slot along the time axis and in[c] is
// appended at position (klen-2), mirroring the CPU reference exactly. When
// klen == 1 no state update is required. When klen == 2 the state is
// overwritten with the current inputs.
//
// One thread per channel. State is updated in-place on device; inputs must
// have been read into registers before any writes because state[(klen-2)*channels+c]
// is one of the read positions when klen > 1.
__global__ void mamba_depthwise_conv_kernel(
    const float* __restrict__ in,         // [channels]
    const float* __restrict__ conv_w,     // [channels * klen] row-major
    const float* __restrict__ bias,       // [channels] or nullptr
    float*       __restrict__ state,      // [channels * (klen - 1)]
    float*       __restrict__ out,        // [channels]
    int channels,
    int klen,
    int has_bias
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;

    const float* kw = conv_w + c * klen;
    float xc = in[c];

    float sum = (has_bias != 0) ? bias[c] : 0.0f;

    // Convolution with stored history followed by the current sample.
    // Read all state slots into registers/sum before writing to avoid
    // races with the in-place state shift below.
    for (int k = 0; k < klen - 1; ++k) {
        sum += kw[k] * state[k * channels + c];
    }
    sum += kw[klen - 1] * xc;

    // State update: shift left by one time step and append xc.
    // Because all reads above are complete, we can now overwrite state[k*channels+c].
    for (int k = 0; k < klen - 2; ++k) {
        state[k * channels + c] = state[(k + 1) * channels + c];
    }
    if (klen > 1) {
        state[(klen - 2) * channels + c] = xc;
    }

    out[c] = sum;
}

__host__ int mantleCudaMambaDepthwiseConv(
    const float* in,
    const float* conv_w,
    const float* bias,
    float*       state,
    float*       out,
    int channels,
    int klen,
    int has_bias,
    cudaStream_t stream
) {
    if (!in || !conv_w || !state || !out || channels <= 0 || klen <= 0) {
        return cudaErrorInvalidValue;
    }
    if (has_bias != 0 && !bias) {
        return cudaErrorInvalidValue;
    }
    const int blockSize = 256;
    int gridSize = (channels + blockSize - 1) / blockSize;
    mamba_depthwise_conv_kernel<<<gridSize, blockSize, 0, stream>>>(
        in, conv_w, bias, state, out, channels, klen, has_bias);
    return cudaGetLastError();
}

} // extern "C"
