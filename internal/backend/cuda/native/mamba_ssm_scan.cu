#include <cuda_runtime.h>

extern "C" {

// Mamba selective SSM scan kernel.
// Mirrors mambaScan in internal/backend/simd/mamba.go.
//
// For each head h in [0, head_count), position p in [0, head_dim):
//   group  = h / group_size
//   a      = -exp(a_log[h])
//   dA     = exp(a * dt[h])
//   xhp    = x[h*head_dim + p]
//   for n in [0, d_state):
//     idx  = (h*head_dim + p) * d_state + n
//     state[idx] = state[idx] * dA + dt[h] * b[group*d_state + n] * xhp
//     sum += c[group*d_state + n] * state[idx]
//   out[h*head_dim + p] = sum + d_vec[h] * xhp
//
// Launch shape:
//   grid  = (head_count, head_dim)
//   block = d_state threads
// Each thread owns one state dimension n; the block computes sum via shared
// memory reduction and thread 0 writes the per-(h,p) output scalar.
__global__ void mamba_ssm_scan_kernel(
    const float* __restrict__ x,        // [head_count * head_dim]
    const float* __restrict__ dt,       // [head_count]
    const float* __restrict__ b,        // [groups * d_state]
    const float* __restrict__ c,        // [groups * d_state]
    const float* __restrict__ a_log,    // [head_count]
    const float* __restrict__ d_vec,    // [head_count]
    float*       __restrict__ state,    // [head_count * head_dim * d_state]
    float*       __restrict__ out,      // [head_count * head_dim]
    int head_dim,
    int d_state,
    int group_size
) {
    int h = blockIdx.x;
    int p = blockIdx.y;
    int n = threadIdx.x;

    int group = h / group_size;
    float dtH  = dt[h];
    float a    = -__expf(a_log[h]);
    float dA   = __expf(a * dtH);
    float xhp  = x[h * head_dim + p];

    int idx = (h * head_dim + p) * d_state + n;
    float bn = b[group * d_state + n];
    float cn = c[group * d_state + n];

    float s = state[idx] * dA + dtH * bn * xhp;
    state[idx] = s;
    float partial = cn * s;

    extern __shared__ float smem[];
    smem[n] = partial;
    __syncthreads();

    // Tree reduction. d_state is typically a power of two (64, 128, 256);
    // the loop handles non-power-of-two sizes correctly because the upper
    // half is always within bounds when offset < blockDim.x.
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (n < offset && n + offset < blockDim.x) {
            smem[n] += smem[n + offset];
        }
        __syncthreads();
    }
    // Tail accumulation when blockDim.x is not a power of two. The loop
    // above reduces to smem[0] the largest power-of-two prefix; any leftover
    // elements in [largest_pow2, blockDim.x) must be added in separately.
    // We handle this by having thread 0 walk the residual once.
    if (n == 0) {
        float sum = smem[0];
        // Detect non-power-of-two: find highest set bit of blockDim.x.
        int bd = blockDim.x;
        int pow2 = 1;
        while ((pow2 << 1) <= bd) pow2 <<= 1;
        // If blockDim.x is a power of two, the loop above fully reduced.
        // Otherwise elements [pow2, blockDim.x) were never folded in.
        // The reduction above guards with (n + offset < blockDim.x), so
        // for non-power-of-two sizes smem[pow2 .. blockDim.x) still hold
        // their original partials. Fold them now.
        if (bd != pow2) {
            for (int k = pow2; k < bd; ++k) {
                sum += smem[k];
            }
        }
        out[h * head_dim + p] = sum + d_vec[h] * xhp;
    }
}

__host__ int mantleCudaMambaSSMScan(
    const float* x,
    const float* dt,
    const float* b,
    const float* c,
    const float* a_log,
    const float* d_vec,
    float*       state,
    float*       out,
    int head_count,
    int head_dim,
    int d_state,
    int group_size,
    cudaStream_t stream
) {
    if (!x || !dt || !b || !c || !a_log || !d_vec || !state || !out) {
        return cudaErrorInvalidValue;
    }
    if (head_count <= 0 || head_dim <= 0 || d_state <= 0 || group_size <= 0) {
        return cudaErrorInvalidValue;
    }
    if (d_state > 1024) {
        // Block dim limit; dState in real Mamba configs is <= 256.
        return cudaErrorInvalidValue;
    }
    dim3 grid(head_count, head_dim, 1);
    dim3 block(d_state, 1, 1);
    size_t smem = d_state * sizeof(float);
    mamba_ssm_scan_kernel<<<grid, block, smem, stream>>>(
        x, dt, b, c, a_log, d_vec, state, out,
        head_dim, d_state, group_size);
    return cudaGetLastError();
}

} // extern "C"
