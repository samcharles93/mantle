#include <cuda_runtime.h>

#define WARP_SIZE 32

extern "C" {

// Standalone RMSNorm kernel: out[i] = x[i] * rsqrt(mean(x^2) + eps) * weight[i]
// Single block processes the entire vector - no CPU roundtrip for the scalar reduction.
__global__ void rmsnorm_f32_kernel(
    float* __restrict__ out,
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float eps,
    int n
) {
    __shared__ float shmem[256];
    int tid = threadIdx.x;

    // Parallel sum-of-squares reduction
    float sum_sq = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float val = x[i];
        sum_sq += val * val;
    }
    shmem[tid] = sum_sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shmem[tid] += shmem[tid + s];
        __syncthreads();
    }

    // Scale computed entirely on GPU
    __shared__ float scale;
    if (tid == 0) {
        scale = rsqrtf(shmem[0] / float(n) + eps);
    }
    __syncthreads();

    // Apply normalization + weight
    for (int i = tid; i < n; i += blockDim.x) {
        out[i] = x[i] * scale * weight[i];
    }
}

// Batched RMSNorm: one block per head, processes nHeads vectors of headDim each.
// All heads share the same weight vector (len=headDim).
__global__ void rmsnorm_batched_f32_kernel(
    float* __restrict__ out,
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float eps,
    int head_dim,
    int n_heads
) {
    int head = blockIdx.x;
    if (head >= n_heads) return;

    const float* x_h = x + head * head_dim;
    float* out_h = out + head * head_dim;
    int tid = threadIdx.x;

    __shared__ float shmem[256];

    // Sum-of-squares for this head
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
        scale = rsqrtf(shmem[0] / float(head_dim) + eps);
    }
    __syncthreads();

    for (int i = tid; i < head_dim; i += blockDim.x) {
        out_h[i] = x_h[i] * scale * weight[i];
    }
}

// Launch wrapper for standalone RMSNorm
int mantleCudaRMSNormF32(
    float* out,
    const float* x,
    const float* weight,
    float eps,
    int n,
    cudaStream_t stream
) {
    int blockSize = 256;
    if (n < 256) blockSize = 128;
    if (n < 128) blockSize = 64;
    // Single block for reduction
    rmsnorm_f32_kernel<<<1, blockSize, 0, stream>>>(out, x, weight, eps, n);
    return cudaGetLastError();
}

// Launch wrapper for batched RMSNorm (one block per head)
int mantleCudaRMSNormBatchedF32(
    float* out,
    const float* x,
    const float* weight,
    float eps,
    int head_dim,
    int n_heads,
    cudaStream_t stream
) {
    int blockSize = 256;
    if (head_dim < 256) blockSize = 128;
    if (head_dim < 128) blockSize = 64;
    rmsnorm_batched_f32_kernel<<<n_heads, blockSize, 0, stream>>>(out, x, weight, eps, head_dim, n_heads);
    return cudaGetLastError();
}

} // extern "C"
