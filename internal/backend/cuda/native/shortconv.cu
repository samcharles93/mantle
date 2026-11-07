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
    if (!proj || !conv_w || !state || !out || embd <= 0 || klen <= 0) return cudaErrorInvalidValue;
    const int blockSize = 256;
    int gridSize = (embd + blockSize - 1) / blockSize;
    shortconv_depthwise_kernel<<<gridSize, blockSize, 0, stream>>>(
        proj, conv_w, state, out, embd, klen);
    return cudaGetLastError();
}


// Batched depthwise short-conv for a prefill chunk of N rows.
// proj is [N, 3*embd] (b|c|xg per row); out is [N, embd] (c * convOut, the
// OutProj input). Tap k of row r (weight kw[k], input at global position
// pos-(klen-1)+k) reads chunk row r-klen+1+k when that is inside the chunk,
// else state slot r+k (state is oldest-first: state[k] = bx at
// startPos-(klen-2)+k). The final tap reads the row's own bx. State is NOT
// rotated here; see shortconv_state_rotate_kernel.
__global__ void shortconv_depthwise_batch_kernel(
    const float* __restrict__ proj,   // [N, 3*embd]
    const float* __restrict__ conv_w, // [embd*klen]
    const float* __restrict__ state,  // [embd*(klen-1)], read-only here
    float*       __restrict__ out,    // [N, embd]
    float*       __restrict__ bxAll,  // [N, embd] gated conv inputs (for state rotate)
    int N,
    int embd,
    int klen
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // channel
    int r = blockIdx.y;                            // chunk row
    if (i >= embd || r >= N) return;

    const float* prow = proj + (size_t)r * 3 * embd;
    float bx = prow[i] * prow[2 * embd + i];
    const float* kw = conv_w + i * klen;
    float sum = 0.0f;
    for (int k = 0; k < klen - 1; k++) {
        int j = r - klen + 1 + k;
        if (j >= 0) {
            const float* pj = proj + (size_t)j * 3 * embd;
            sum += kw[k] * (pj[i] * pj[2 * embd + i]);
        } else {
            sum += kw[k] * state[(r + k) * embd + i];
        }
    }
    sum += kw[klen - 1] * bx;
    out[(size_t)r * embd + i] = prow[embd + i] * sum; // c * convOut
    if (bxAll) bxAll[(size_t)r * embd + i] = bx;
}

// Rotates the conv state forward by N chunk rows: afterwards state[k] holds
// bx at global position startPos+N-klen+1+k, i.e. chunk row N-klen+1+k when
// inside the chunk, else old state slot N+k. Reads old state, writes newState
// (caller copies newState back over state to avoid read/write races).
__global__ void shortconv_state_rotate_kernel(
    const float* __restrict__ state,     // [embd*(klen-1)] old
    const float* __restrict__ bxAll,     // [N, embd] gated conv inputs (b*xg)
    float*       __restrict__ newState,  // [embd*(klen-1)]
    int N,
    int embd,
    int klen
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = (klen - 1) * embd;
    if (idx >= total) return;
    int k = idx / embd;
    int i = idx % embd;
    int row = N - klen + 1 + k;
    newState[idx] = (row >= 0) ? bxAll[(size_t)row * embd + i]
                               : state[(N + k) * embd + i];
}

__host__ int mantleCudaShortConvDepthwiseBatch(
    const float* proj,
    const float* conv_w,
    const float* state,
    float* out,
    float* bxAll,
    int N,
    int embd,
    int klen,
    cudaStream_t stream
) {
    if (!proj || !conv_w || !state || !out || N <= 0 || embd <= 0 || klen <= 0) return cudaErrorInvalidValue;
    const int blockSize = 256;
    dim3 block(blockSize);
    dim3 grid((embd + blockSize - 1) / blockSize, N);
    shortconv_depthwise_batch_kernel<<<grid, block, 0, stream>>>(
        proj, conv_w, state, out, bxAll, N, embd, klen);
    return cudaGetLastError();
}

__host__ int mantleCudaShortConvStateRotate(
    const float* state,
    const float* bxAll,
    float* newState,
    int N,
    int embd,
    int klen,
    cudaStream_t stream
) {
    if (!state || !bxAll || !newState || N <= 0 || embd <= 0 || klen <= 1) return cudaErrorInvalidValue;
    int total = (klen - 1) * embd;
    const int blockSize = 256;
    int gridSize = (total + blockSize - 1) / blockSize;
    shortconv_state_rotate_kernel<<<gridSize, blockSize, 0, stream>>>(
        state, bxAll, newState, N, embd, klen);
    return cudaGetLastError();
}

} // extern "C"

