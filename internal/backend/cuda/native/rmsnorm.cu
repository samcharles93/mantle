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


// Batched RMSNorm over two tensors in one launch: grid = nHeadsQ + nHeadsK,
// blocks [0, nHeadsQ) normalize Q heads with wQ, the rest normalize K heads
// with wK. Per-block math is identical to rmsnorm_batched_f32_kernel, so the
// result is bit-identical to two separate launches.
__global__ void rmsnorm_batched_pair_f32_kernel(
    float* __restrict__ out_q,
    const float* __restrict__ x_q,
    const float* __restrict__ w_q,
    float* __restrict__ out_k,
    const float* __restrict__ x_k,
    const float* __restrict__ w_k,
    float eps,
    int head_dim,
    int n_heads_q,
    int n_heads_k)
{
    const int head = blockIdx.x;
    float* out;
    const float* x;
    const float* w;
    int h;
    if (head < n_heads_q) {
        out = out_q; x = x_q; w = w_q; h = head;
    } else {
        out = out_k; x = x_k; w = w_k; h = head - n_heads_q;
    }
    const float* x_h = x + h * head_dim;
    float* out_h = out + h * head_dim;
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
        scale = rsqrtf(shmem[0] / float(head_dim) + eps);
    }
    __syncthreads();

    for (int i = tid; i < head_dim; i += blockDim.x) {
        out_h[i] = x_h[i] * scale * w[i];
    }
}

int mantleCudaRMSNormBatchedPairF32(
    float* out_q,
    const float* x_q,
    const float* w_q,
    float* out_k,
    const float* x_k,
    const float* w_k,
    float eps,
    int head_dim,
    int n_heads_q,
    int n_heads_k,
    cudaStream_t stream
) {
    if (!out_q || !x_q || !w_q || !out_k || !x_k || !w_k || head_dim <= 0 ||
        n_heads_q <= 0 || n_heads_k <= 0) {
        return cudaErrorInvalidValue;
    }
    int blockSize = 256;
    if (head_dim < 256) blockSize = 128;
    if (head_dim < 128) blockSize = 64;
    rmsnorm_batched_pair_f32_kernel<<<n_heads_q + n_heads_k, blockSize, 0, stream>>>(
        out_q, x_q, w_q, out_k, x_k, w_k, eps, head_dim, n_heads_q, n_heads_k);
    return cudaGetLastError();
}


// Fused RMSNorm + residual add: out[i] = out[i] + x[i]*scale*w[i], where
// scale = rsqrtf(mean(x^2) + eps) computed with the exact same reduction as
// rmsnorm_f32_kernel. Bit-identical to running rmsnorm_f32_kernel into a
// temporary and then add_vectors_f32_kernel (dst += tmp), because
// out[i] + x[i]*scale*w[i] associates as out[i] + ((x[i]*scale)*w[i]) — the
// same two roundings the two-kernel sequence performs.
__global__ void rmsnorm_residual_f32_kernel(
    float* __restrict__ out,
    const float* __restrict__ x,
    const float* __restrict__ weight,
    float eps,
    int n
) {
    __shared__ float shmem[256];
    int tid = threadIdx.x;

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

    __shared__ float scale;
    if (tid == 0) {
        scale = rsqrtf(shmem[0] / float(n) + eps);
    }
    __syncthreads();

    for (int i = tid; i < n; i += blockDim.x) {
        // Explicit roundings: the two-kernel sequence computes
        // tmp = RN(RN(x*scale)*w) then out = RN(out+tmp); prevent the compiler
        // from contracting the expression into a single FMA.
        const float t = __fmul_rn(__fmul_rn(x[i], scale), weight[i]);
        out[i] = __fadd_rn(out[i], t);
    }
}

int mantleCudaRMSNormResidualF32(
    float* out,
    const float* x,
    const float* weight,
    float eps,
    int n,
    cudaStream_t stream
) {
    if (!out || !x || !weight || n <= 0) {
        return cudaErrorInvalidValue;
    }
    int blockSize = 256;
    if (n < 256) blockSize = 128;
    if (n < 128) blockSize = 64;
    rmsnorm_residual_f32_kernel<<<1, blockSize, 0, stream>>>(out, x, weight, eps, n);
    return cudaGetLastError();
}

} // extern "C"
