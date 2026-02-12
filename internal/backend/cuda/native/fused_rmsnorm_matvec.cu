#include <cuda_runtime.h>
#include <cuda_bf16.h>

extern "C" {

// Fused RMSNorm + MatVec for BF16 weights
// Computes: out = matmul(W, rmsnorm(x, weight, eps))
// Avoids writing RMSNorm output to memory
__global__ void fusedRMSNormMatVecBF16(
    float* out,              // output vector [rows]
    const __nv_bfloat16* W,  // weight matrix [rows x cols], row-major
    const float* x,          // input vector [cols]
    const float* normWeight, // RMSNorm weight [cols]
    float eps,
    int rows,
    int cols
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    // Step 1: Compute RMS of input (shared across all threads)
    // Use shared memory for reduction
    __shared__ float shmem[256];
    float sum_sq = 0.0f;

    // Each thread computes partial sum
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float val = x[i];
        sum_sq += val * val;
    }

    shmem[threadIdx.x] = sum_sq;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shmem[threadIdx.x] += shmem[threadIdx.x + s];
        }
        __syncthreads();
    }

    float rms_scale;
    if (threadIdx.x == 0) {
        float mean = shmem[0] / cols;
        rms_scale = rsqrtf(mean + eps);
        shmem[0] = rms_scale;
    }
    __syncthreads();
    rms_scale = shmem[0];

    // Step 2: Compute dot product with normalized input (on-the-fly)
    // out[row] = sum(W[row, i] * (x[i] * rms_scale * normWeight[i]))
    float acc = 0.0f;
    const __nv_bfloat16* W_row = W + row * cols;

    for (int i = 0; i < cols; i++) {
        float x_normalized = x[i] * rms_scale * normWeight[i];
        float w_val = __bfloat162float(W_row[i]);
        acc += w_val * x_normalized;
    }

    out[row] = acc;
}

// Fused RMSNorm + MatVec for F32 weights
__global__ void fusedRMSNormMatVecF32(
    float* out,
    const float* W,
    const float* x,
    const float* normWeight,
    float eps,
    int rows,
    int cols
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    __shared__ float shmem[256];
    float sum_sq = 0.0f;

    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float val = x[i];
        sum_sq += val * val;
    }

    shmem[threadIdx.x] = sum_sq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shmem[threadIdx.x] += shmem[threadIdx.x + s];
        }
        __syncthreads();
    }

    float rms_scale;
    if (threadIdx.x == 0) {
        float mean = shmem[0] / cols;
        rms_scale = rsqrtf(mean + eps);
        shmem[0] = rms_scale;
    }
    __syncthreads();
    rms_scale = shmem[0];

    float acc = 0.0f;
    const float* W_row = W + row * cols;

    for (int i = 0; i < cols; i++) {
        float x_normalized = x[i] * rms_scale * normWeight[i];
        acc += W_row[i] * x_normalized;
    }

    out[row] = acc;
}

// Launch wrappers
int launchFusedRMSNormMatVecBF16(
    float* out,
    const __nv_bfloat16* W,
    const float* x,
    const float* normWeight,
    float eps,
    int rows,
    int cols,
    cudaStream_t stream
) {
    int blockSize = 256;
    int gridSize = (rows + blockSize - 1) / blockSize;
    fusedRMSNormMatVecBF16<<<gridSize, blockSize, 0, stream>>>(out, W, x, normWeight, eps, rows, cols);
    return cudaGetLastError();
}

int launchFusedRMSNormMatVecF32(
    float* out,
    const float* W,
    const float* x,
    const float* normWeight,
    float eps,
    int rows,
    int cols,
    cudaStream_t stream
) {
    int blockSize = 256;
    int gridSize = (rows + blockSize - 1) / blockSize;
    fusedRMSNormMatVecF32<<<gridSize, blockSize, 0, stream>>>(out, W, x, normWeight, eps, rows, cols);
    return cudaGetLastError();
}

} // extern "C"
