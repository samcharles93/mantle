#include <cuda_runtime.h>

extern "C" {

// dt[i] = clamp(softplus(dt[i] + bias[i]), t_min, t_max), then if
// (t_floor > 0 && dt[i] < t_floor) dt[i] = t_floor.
// Mirrors the dt preprocessing in internal/backend/simd/mamba.go before
// the selective-SSM scan. Grid-stride loop tolerates small launch configs.
//
// softplus(x) = log1p(exp(x)) computed stably: for x > 20, softplus(x) ~= x.

__global__ void mamba_dt_softplus_clamp_f32_kernel(
    float *__restrict__ dt,
    const float *__restrict__ bias,
    int n,
    float t_min,
    float t_max,
    float t_floor) {
    const int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        float v = dt[i] + bias[i];
        float sp;
        if (v > 20.0f) {
            sp = v;
        } else if (v < -20.0f) {
            sp = __expf(v);
        } else {
            sp = log1pf(__expf(v));
        }
        if (sp < t_min) sp = t_min;
        if (sp > t_max) sp = t_max;
        if (t_floor > 0.0f && sp < t_floor) sp = t_floor;
        dt[i] = sp;
    }
}

int mantleCudaMambaDtSoftplusClampF32(
    float *dt,
    const float *bias,
    int n,
    float t_min,
    float t_max,
    float t_floor,
    cudaStream_t stream) {
    if (n <= 0) {
        return cudaSuccess;
    }
    constexpr int kBlock = 256;
    int blocks = (n + kBlock - 1) / kBlock;
    if (blocks > 4096) {
        blocks = 4096;
    }
    mamba_dt_softplus_clamp_f32_kernel<<<blocks, kBlock, 0, stream>>>(
        dt, bias, n, t_min, t_max, t_floor);
    return cudaGetLastError();
}

__global__ void scale_f32_inplace_kernel(float *__restrict__ x, int n, float scale) {
    const int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        x[i] = x[i] * scale;
    }
}

int mantleCudaScaleF32InPlace(float *x, int n, float scale, cudaStream_t stream) {
    if (n <= 0 || scale == 1.0f) {
        return cudaSuccess;
    }
    constexpr int kBlock = 256;
    int blocks = (n + kBlock - 1) / kBlock;
    if (blocks > 4096) {
        blocks = 4096;
    }
    scale_f32_inplace_kernel<<<blocks, kBlock, 0, stream>>>(x, n, scale);
    return cudaGetLastError();
}

} // extern "C"
