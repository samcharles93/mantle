#include <cuda_runtime.h>

extern "C" {

__global__ void deltanet_recurrent_f32_kernel(
    float* __restrict__ state,
    const float* __restrict__ q,
    const float* __restrict__ k_buf,
    const float* __restrict__ v,
    const float* __restrict__ a_log,
    const float* __restrict__ delta_a,
    const float* __restrict__ delta_b,
    const float* __restrict__ dt_bias,
    float* __restrict__ out,
    float scale,
    int head_key_dim,
    int head_value_dim,
    int n_value_heads,
    int n_key_heads,
    int group_size
) {
    int hv = blockIdx.x;
    if (hv >= n_value_heads) return;
    int hk = (group_size > 1) ? (hv / group_size) : hv;

    int kvSize = head_key_dim * head_value_dim;
    float* state_h = state + hv * kvSize;
    const float* q_h = q + hk * head_key_dim;
    const float* k_h = k_buf + hk * head_key_dim;
    const float* v_h = v + hv * head_value_dim;
    float* out_h = out + hv * head_value_dim;

    extern __shared__ float shmem[];
    float* delta = shmem;
    float* s_scalars = shmem + head_value_dim; // [0]=decay, [1]=beta

    int tid = threadIdx.x;

    // Phase 0: thread 0 fuses decay/beta from raw weights.
    // decay = exp(-exp(a_log) * softplus(delta_a + dt_bias))
    // beta  = sigmoid(delta_b)
    if (tid == 0) {
        float a = a_log[hv];
        float da = delta_a[hv] + dt_bias[hv];
        float sp;
        if (da > 20.0f) {
            sp = da;
        } else if (da < -20.0f) {
            sp = expf(da);
        } else {
            sp = log1pf(expf(da));
        }
        s_scalars[0] = expf(-expf(a) * sp);
        float db = delta_b[hv];
        s_scalars[1] = 1.0f / (1.0f + expf(-db));
    }
    __syncthreads();

    float dec = s_scalars[0];
    float bet = s_scalars[1];

    for (int idx = tid; idx < kvSize; idx += blockDim.x) {
        state_h[idx] *= dec;
    }
    __syncthreads();

    for (int v_idx = tid; v_idx < head_value_dim; v_idx += blockDim.x) {
        float kvMem = 0.0f;
        for (int kk = 0; kk < head_key_dim; kk++) {
            kvMem += state_h[kk * head_value_dim + v_idx] * k_h[kk];
        }
        delta[v_idx] = (v_h[v_idx] - kvMem) * bet;
    }
    __syncthreads();

    for (int idx = tid; idx < kvSize; idx += blockDim.x) {
        int kk = idx / head_value_dim;
        int v_idx = idx - kk * head_value_dim;
        state_h[idx] += k_h[kk] * delta[v_idx];
    }
    __syncthreads();

    for (int v_idx = tid; v_idx < head_value_dim; v_idx += blockDim.x) {
        float sum = 0.0f;
        for (int kk = 0; kk < head_key_dim; kk++) {
            sum += state_h[kk * head_value_dim + v_idx] * q_h[kk];
        }
        out_h[v_idx] = sum * scale;
    }
}

int mantleCudaDeltaNetRecurrentF32(
    float* state,
    const float* q,
    const float* k_buf,
    const float* v,
    const float* a_log,
    const float* delta_a,
    const float* delta_b,
    const float* dt_bias,
    float* out,
    float scale,
    int head_key_dim,
    int head_value_dim,
    int n_value_heads,
    int n_key_heads,
    int group_size,
    cudaStream_t stream
) {
    int blockSize = 128;
    if (head_value_dim >= 256) blockSize = 256;
    size_t shmemBytes = (size_t)(head_value_dim + 2) * sizeof(float);
    deltanet_recurrent_f32_kernel<<<n_value_heads, blockSize, shmemBytes, stream>>>(
        state, q, k_buf, v, a_log, delta_a, delta_b, dt_bias, out, scale,
        head_key_dim, head_value_dim, n_value_heads, n_key_heads, group_size);
    return cudaGetLastError();
}

} // extern "C"
