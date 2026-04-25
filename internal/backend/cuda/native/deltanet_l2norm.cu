#include <cuda_runtime.h>

extern "C" {

// Per-head L2 normalization in-place: x[i] *= rsqrt(sum(x^2) + eps).
// One block per head; head_dim elements per head; n_heads heads.
// Matches simd/deltanet.go l2norm semantics (sum-of-squares, not mean).
__global__ void deltanet_l2norm_f32_kernel(
    float* __restrict__ x,
    float eps,
    int head_dim,
    int n_heads
) {
    int head = blockIdx.x;
    if (head >= n_heads) return;

    float* x_h = x + head * head_dim;
    int tid = threadIdx.x;

    __shared__ float shmem[256];

    float sum_sq = 0.0f;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        float val = x_h[i];
        sum_sq += val * val;
    }
    shmem[tid] = sum_sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shmem[tid] += shmem[tid + s];
        __syncthreads();
    }

    __shared__ float scale;
    if (tid == 0) {
        scale = rsqrtf(shmem[0] + eps);
    }
    __syncthreads();

    for (int i = tid; i < head_dim; i += blockDim.x) {
        x_h[i] = x_h[i] * scale;
    }
}

int mantleCudaDeltaNetL2NormF32(
    float* x,
    float eps,
    int head_dim,
    int n_heads,
    cudaStream_t stream
) {
    int blockSize = 256;
    if (head_dim < 256) blockSize = 128;
    if (head_dim < 128) blockSize = 64;
    if (head_dim < 64) blockSize = 32;
    deltanet_l2norm_f32_kernel<<<n_heads, blockSize, 0, stream>>>(x, eps, head_dim, n_heads);
    return cudaGetLastError();
}

} // extern "C"
