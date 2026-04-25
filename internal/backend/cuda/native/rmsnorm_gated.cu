#include <cuda_runtime.h>

__device__ __forceinline__ float silu_fast(float x) {
    return x / (1.0f + __expf(-x));
}

template <bool NormBeforeGate>
__global__ void rmsnorm_gated_f32_kernel(
    float *__restrict__ out,
    const float *__restrict__ y,
    const float *__restrict__ z,
    const float *__restrict__ weight,
    float eps,
    int n) {
    __shared__ float shmem[256];
    __shared__ float scale;
    const int tid = threadIdx.x;

    if (NormBeforeGate) {
        float sum_sq = 0.0f;
        for (int i = tid; i < n; i += blockDim.x) {
            float v = y[i];
            sum_sq += v * v;
        }
        shmem[tid] = sum_sq;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) shmem[tid] += shmem[tid + s];
            __syncthreads();
        }
        if (tid == 0) {
            scale = rsqrtf(shmem[0] / float(n) + eps);
        }
        __syncthreads();
        for (int i = tid; i < n; i += blockDim.x) {
            float yn = y[i] * scale * weight[i];
            out[i] = yn * silu_fast(z[i]);
        }
    } else {
        float sum_sq = 0.0f;
        for (int i = tid; i < n; i += blockDim.x) {
            float t = y[i] * silu_fast(z[i]);
            out[i] = t;
            sum_sq += t * t;
        }
        shmem[tid] = sum_sq;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) shmem[tid] += shmem[tid + s];
            __syncthreads();
        }
        if (tid == 0) {
            scale = rsqrtf(shmem[0] / float(n) + eps);
        }
        __syncthreads();
        for (int i = tid; i < n; i += blockDim.x) {
            out[i] = out[i] * scale * weight[i];
        }
    }
}

extern "C" {

int mantleCudaRMSNormGatedF32(
    float *out,
    const float *y,
    const float *z,
    const float *weight,
    float eps,
    int n,
    int norm_before_gate,
    cudaStream_t stream) {
    if (n <= 0) {
        return cudaSuccess;
    }
    int blockSize = 256;
    if (n < 256) blockSize = 128;
    if (n < 128) blockSize = 64;
    if (norm_before_gate) {
        rmsnorm_gated_f32_kernel<true>
            <<<1, blockSize, 0, stream>>>(out, y, z, weight, eps, n);
    } else {
        rmsnorm_gated_f32_kernel<false>
            <<<1, blockSize, 0, stream>>>(out, y, z, weight, eps, n);
    }
    return cudaGetLastError();
}

}
