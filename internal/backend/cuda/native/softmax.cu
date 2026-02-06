#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>
#include <climits>
#include <cstdint>

#define WARP_SIZE 32

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return __shfl_sync(0xffffffff, val, 0);
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return __shfl_sync(0xffffffff, val, 0);
}

__device__ __forceinline__ int8_t decode_q4_nibble(uint8_t packed, int idx_in_block) {
    const uint8_t nib = (idx_in_block & 1) ? (packed >> 4) : (packed & 0x0F);
    return (nib < 8) ? static_cast<int8_t>(nib) : static_cast<int8_t>(static_cast<int>(nib) - 16);
}

__device__ __forceinline__ float f16_to_f32(uint16_t raw) {
    return __half2float(__ushort_as_half(raw));
}

extern "C" {

__global__ void quant_matvec_int8_blocks_f32_kernel(
    const int8_t* __restrict__ q,
    const float* __restrict__ scales,
    const float* __restrict__ x,
    float* __restrict__ y,
    int rows,
    int blocks_per_row,
    int cols) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= rows) return;

    float sum = 0.0f;
    const int row_block_base = row * blocks_per_row;
    for (int b = 0; b < blocks_per_row; b++) {
        const float s = scales[row_block_base + b];
        if (s == 0.0f) {
            continue;
        }
        const int col_base = b * 32;
        const int n = min(32, cols - col_base);
        if (n <= 0) break;

        const int8_t* qb = q + (row_block_base + b) * 32;
        for (int i = tid; i < n; i += blockDim.x) {
            sum += (s * static_cast<float>(qb[i])) * x[col_base + i];
        }
    }

    __shared__ float shared[256];
    shared[tid] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        y[row] = shared[0];
    }
}

__global__ void softmax_rows_optimized_kernel(float* __restrict__ __align__(16) data, int rows, int cols) {
    __shared__ float shared_max[8];
    __shared__ float shared_sum[8];

    const int row = blockIdx.x;
    if (row >= rows) return;

    const int tid = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int warp = tid / WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE;

    float* row_ptr = data + static_cast<size_t>(row) * cols;

    float local_max = -INFINITY;
    for (int i = tid * 4; i < cols; i += blockDim.x * 4) {
        if (i + 3 < cols) {
            float4 vals = reinterpret_cast<float4*>(&row_ptr[i])[0];
            local_max = fmaxf(local_max, fmaxf(fmaxf(vals.x, vals.y), fmaxf(vals.z, vals.w)));
        } else {
            #pragma unroll
            for (int j = 0; j < 4 && (i + j) < cols; j++) {
                local_max = fmaxf(local_max, row_ptr[i + j]);
            }
        }
    }

    float warp_max = warp_reduce_max(local_max);
    if (lane == 0) shared_max[warp] = warp_max;
    __syncthreads();

    if (warp == 0) {
        float val = (lane < warps_per_block) ? shared_max[lane] : -INFINITY;
        float block_max = warp_reduce_max(val);
        if (lane == 0) shared_max[0] = block_max;
    }
    __syncthreads();

    const float row_max = shared_max[0];

    float local_sum = 0.0f;
    for (int i = tid * 4; i < cols; i += blockDim.x * 4) {
        if (i + 3 < cols) {
            float4 vals = reinterpret_cast<float4*>(&row_ptr[i])[0];
            vals.x = __expf(vals.x - row_max);
            vals.y = __expf(vals.y - row_max);
            vals.z = __expf(vals.z - row_max);
            vals.w = __expf(vals.w - row_max);
            local_sum += vals.x + vals.y + vals.z + vals.w;
            reinterpret_cast<float4*>(&row_ptr[i])[0] = vals;
        } else {
            #pragma unroll
            for (int j = 0; j < 4 && (i + j) < cols; j++) {
                float e = __expf(row_ptr[i + j] - row_max);
                row_ptr[i + j] = e;
                local_sum += e;
            }
        }
    }

    float warp_sum = warp_reduce_sum(local_sum);
    if (lane == 0) shared_sum[warp] = warp_sum;
    __syncthreads();

    if (warp == 0) {
        float val = (lane < warps_per_block) ? shared_sum[lane] : 0.0f;
        float block_sum = warp_reduce_sum(val);
        if (lane == 0) shared_sum[0] = block_sum;
    }
    __syncthreads();

    float inv_sum = 1.0f / (shared_sum[0] + FLT_EPSILON);

    for (int i = tid * 4; i < cols; i += blockDim.x * 4) {
        if (i + 3 < cols) {
            float4 vals = reinterpret_cast<float4*>(&row_ptr[i])[0];
            vals.x *= inv_sum;
            vals.y *= inv_sum;
            vals.z *= inv_sum;
            vals.w *= inv_sum;
            reinterpret_cast<float4*>(&row_ptr[i])[0] = vals;
        } else {
            #pragma unroll
            for (int j = 0; j < 4 && (i + j) < cols; j++) {
                row_ptr[i + j] *= inv_sum;
            }
        }
    }
}

__global__ void quant_matvec_q4_f32_kernel(
    const uint8_t* __restrict__ q_data,
    const uint16_t* __restrict__ scales_f16,
    const float* __restrict__ x,
    float* __restrict__ y,
    int rows,
    int blocks_per_row,
    int cols) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= rows) return;

    float sum = 0.0f;
    const int row_block_base = row * blocks_per_row;
    for (int b = 0; b < blocks_per_row; b++) {
        const float s = f16_to_f32(scales_f16[row_block_base + b]);
        if (s == 0.0f) {
            continue;
        }
        const int col_base = b * 32;
        const int n = min(32, cols - col_base);
        if (n <= 0) break;

        const uint8_t* qb = q_data + (row_block_base + b) * 16;
        for (int i = tid; i < n; i += blockDim.x) {
            const uint8_t packed = qb[i >> 1];
            const int8_t qv = decode_q4_nibble(packed, i);
            sum += (s * static_cast<float>(qv)) * x[col_base + i];
        }
    }

    __shared__ float shared[256];
    shared[tid] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        y[row] = shared[0];
    }
}

__global__ void quant_matvec_k4_f32_kernel(
    const uint8_t* __restrict__ q_data,
    const uint16_t* __restrict__ super_scales_f16,
    const uint8_t* __restrict__ sub_scales,
    const float* __restrict__ x,
    float* __restrict__ y,
    int rows,
    int blocks_per_row,
    int cols) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= rows) return;

    const int super_blocks_per_row = (blocks_per_row + 7) / 8;
    float sum = 0.0f;
    const int row_block_base = row * blocks_per_row;
    const int row_super_base = row * super_blocks_per_row;
    for (int b = 0; b < blocks_per_row; b++) {
        const float super_scale = f16_to_f32(super_scales_f16[row_super_base + (b / 8)]);
        const uint8_t u6 = sub_scales[row_block_base + b] & 0x3F;
        if (super_scale == 0.0f || u6 == 0) {
            continue;
        }
        const float s = super_scale * (static_cast<float>(u6) * (1.0f / 32.0f));
        const int col_base = b * 32;
        const int n = min(32, cols - col_base);
        if (n <= 0) break;

        const uint8_t* qb = q_data + (row_block_base + b) * 16;
        for (int i = tid; i < n; i += blockDim.x) {
            const uint8_t packed = qb[i >> 1];
            const int8_t qv = decode_q4_nibble(packed, i);
            sum += (s * static_cast<float>(qv)) * x[col_base + i];
        }
    }

    __shared__ float shared[256];
    shared[tid] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        y[row] = shared[0];
    }
}

__global__ void dequantize_q4_to_f16_kernel(
    const uint8_t* __restrict__ q_data,
    const uint16_t* __restrict__ scales_f16,
    uint16_t* __restrict__ out_f16,
    int rows,
    int blocks_per_row,
    int cols) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * cols;
    if (idx >= total) return;

    const int row = idx / cols;
    const int col = idx - row * cols;
    const int block = col >> 5; // /32
    const int in_block = col & 31;

    const int row_block_base = row * blocks_per_row;
    const float s = f16_to_f32(scales_f16[row_block_base + block]);
    if (s == 0.0f) {
        out_f16[idx] = __half_as_ushort(__float2half(0.0f));
        return;
    }

    const uint8_t packed = q_data[(row_block_base + block) * 16 + (in_block >> 1)];
    const int8_t qv = decode_q4_nibble(packed, in_block);
    out_f16[idx] = __half_as_ushort(__float2half(s * static_cast<float>(qv)));
}

__global__ void dequantize_k4_to_f16_kernel(
    const uint8_t* __restrict__ q_data,
    const uint16_t* __restrict__ super_scales_f16,
    const uint8_t* __restrict__ sub_scales,
    uint16_t* __restrict__ out_f16,
    int rows,
    int blocks_per_row,
    int cols) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * cols;
    if (idx >= total) return;

    const int row = idx / cols;
    const int col = idx - row * cols;
    const int block = col >> 5; // /32
    const int in_block = col & 31;

    const int row_block_base = row * blocks_per_row;
    const int super_blocks_per_row = (blocks_per_row + 7) / 8;
    const int row_super_base = row * super_blocks_per_row;

    const float super_scale = f16_to_f32(super_scales_f16[row_super_base + (block / 8)]);
    const uint8_t u6 = sub_scales[row_block_base + block] & 0x3F;
    if (super_scale == 0.0f || u6 == 0) {
        out_f16[idx] = __half_as_ushort(__float2half(0.0f));
        return;
    }
    const float s = super_scale * (static_cast<float>(u6) * (1.0f / 32.0f));
    const uint8_t packed = q_data[(row_block_base + block) * 16 + (in_block >> 1)];
    const int8_t qv = decode_q4_nibble(packed, in_block);
    out_f16[idx] = __half_as_ushort(__float2half(s * static_cast<float>(qv)));
}

__global__ void silu_mul_f32_kernel(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ out,
    int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const float g = gate[idx];
    const float u = up[idx];
    const float s = 1.0f / (1.0f + __expf(-g));
    out[idx] = (g * s) * u;
}

__global__ void convert_f32_to_f16_kernel(
    const float* __restrict__ in,
    uint16_t* __restrict__ out,
    int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = __half_as_ushort(__float2half(in[idx]));
}

__global__ void attention_inner_f16_cache_f32_kernel(
    const float* __restrict__ q,
    const uint16_t* __restrict__ cache_k,
    const uint16_t* __restrict__ cache_v,
    float* __restrict__ out,
    int pos,
    int start,
    int kv_stride,
    int head_dim,
    int n_head,
    int kv_heads,
    int cache_len,
    float scale) {
    const int h = blockIdx.x;
    const int tid = threadIdx.x;
    if (h >= n_head) return;

    const int kv_head = (h * kv_heads) / n_head;
    const float* qh = q + h * head_dim;
    float* out_h = out + h * head_dim;

    for (int d = tid; d < head_dim; d += blockDim.x) {
        out_h[d] = 0.0f;
    }
    __syncthreads();

    __shared__ float partial[256];
    __shared__ float s_m;
    __shared__ float s_l;
    __shared__ float s_alpha;
    __shared__ float s_beta;

    if (tid == 0) {
        s_m = -INFINITY;
        s_l = 0.0f;
    }
    __syncthreads();

    const bool use_ring = cache_len > 0 && cache_len < (pos + 1);

    for (int t = start; t <= pos; t++) {
        const int cache_pos = use_ring ? (t % cache_len) : t;
        const int k_base = cache_pos * kv_stride + kv_head * head_dim;

        float dot_part = 0.0f;
        for (int d = tid; d < head_dim; d += blockDim.x) {
            const float kf = f16_to_f32(cache_k[k_base + d]);
            dot_part += qh[d] * kf;
        }
        partial[tid] = dot_part;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                partial[tid] += partial[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0) {
            const float s = partial[0] * scale;
            const float m_new = fmaxf(s_m, s);
            s_alpha = __expf(s_m - m_new);
            s_beta = __expf(s - m_new);
            s_l = s_l * s_alpha + s_beta;
            s_m = m_new;
        }
        __syncthreads();

        const int v_base = cache_pos * kv_stride + kv_head * head_dim;
        for (int d = tid; d < head_dim; d += blockDim.x) {
            const float vf = f16_to_f32(cache_v[v_base + d]);
            out_h[d] = out_h[d]*s_alpha + s_beta*vf;
        }
        __syncthreads();
    }

    for (int d = tid; d < head_dim; d += blockDim.x) {
        out_h[d] = out_h[d] / (s_l + 1e-12f);
    }
}

int mantleCudaSoftmaxRowsF32(float* data, int rows, int cols, cudaStream_t stream) {
    if (!data || rows <= 0 || cols <= 0) return 0;
    if (rows > INT_MAX / cols) return cudaErrorInvalidValue;

    const int threads = 256;
    dim3 block(threads);
    dim3 grid(rows);

    softmax_rows_optimized_kernel<<<grid, block, 0, stream>>>(data, rows, cols);

    cudaError_t err = cudaGetLastError();
    return (int)err;
}

int mantleCudaQuantMatVecInt8BlocksF32(
    const int8_t* q,
    const float* scales,
    const float* x,
    float* y,
    int rows,
    int blocksPerRow,
    int cols,
    cudaStream_t stream) {
    if (!q || !scales || !x || !y || rows <= 0 || blocksPerRow <= 0 || cols <= 0) return cudaErrorInvalidValue;
    const int threads = 256;
    dim3 block(threads);
    dim3 grid(rows);
    quant_matvec_int8_blocks_f32_kernel<<<grid, block, 0, stream>>>(q, scales, x, y, rows, blocksPerRow, cols);
    return (int)cudaGetLastError();
}

int mantleCudaQuantMatVecQ4F32(
    const uint8_t* qData,
    const uint16_t* scalesF16,
    const float* x,
    float* y,
    int rows,
    int blocksPerRow,
    int cols,
    cudaStream_t stream) {
    if (!qData || !scalesF16 || !x || !y || rows <= 0 || blocksPerRow <= 0 || cols <= 0) return cudaErrorInvalidValue;
    const int threads = 256;
    dim3 block(threads);
    dim3 grid(rows);
    quant_matvec_q4_f32_kernel<<<grid, block, 0, stream>>>(qData, scalesF16, x, y, rows, blocksPerRow, cols);
    return (int)cudaGetLastError();
}

int mantleCudaQuantMatVecK4F32(
    const uint8_t* qData,
    const uint16_t* superScalesF16,
    const uint8_t* subScales,
    const float* x,
    float* y,
    int rows,
    int blocksPerRow,
    int cols,
    cudaStream_t stream) {
    if (!qData || !superScalesF16 || !subScales || !x || !y || rows <= 0 || blocksPerRow <= 0 || cols <= 0) return cudaErrorInvalidValue;
    const int threads = 256;
    dim3 block(threads);
    dim3 grid(rows);
    quant_matvec_k4_f32_kernel<<<grid, block, 0, stream>>>(qData, superScalesF16, subScales, x, y, rows, blocksPerRow, cols);
    return (int)cudaGetLastError();
}

int mantleCudaDequantizeQ4ToF16(
    const uint8_t* qData,
    const uint16_t* scalesF16,
    uint16_t* outF16,
    int rows,
    int blocksPerRow,
    int cols,
    cudaStream_t stream) {
    if (!qData || !scalesF16 || !outF16 || rows <= 0 || blocksPerRow <= 0 || cols <= 0) return cudaErrorInvalidValue;
    const int total = rows * cols;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    dequantize_q4_to_f16_kernel<<<blocks, threads, 0, stream>>>(qData, scalesF16, outF16, rows, blocksPerRow, cols);
    return (int)cudaGetLastError();
}

int mantleCudaDequantizeK4ToF16(
    const uint8_t* qData,
    const uint16_t* superScalesF16,
    const uint8_t* subScales,
    uint16_t* outF16,
    int rows,
    int blocksPerRow,
    int cols,
    cudaStream_t stream) {
    if (!qData || !superScalesF16 || !subScales || !outF16 || rows <= 0 || blocksPerRow <= 0 || cols <= 0) return cudaErrorInvalidValue;
    const int total = rows * cols;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    dequantize_k4_to_f16_kernel<<<blocks, threads, 0, stream>>>(qData, superScalesF16, subScales, outF16, rows, blocksPerRow, cols);
    return (int)cudaGetLastError();
}

int mantleCudaSiluMulF32(
    const float* gate,
    const float* up,
    float* out,
    int n,
    cudaStream_t stream) {
    if (!gate || !up || !out || n <= 0) return cudaErrorInvalidValue;
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    silu_mul_f32_kernel<<<blocks, threads, 0, stream>>>(gate, up, out, n);
    return (int)cudaGetLastError();
}

int mantleCudaConvertF32ToF16(
    const float* in,
    uint16_t* out,
    int n,
    cudaStream_t stream) {
    if (!in || !out || n <= 0) return cudaErrorInvalidValue;
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    convert_f32_to_f16_kernel<<<blocks, threads, 0, stream>>>(in, out, n);
    return (int)cudaGetLastError();
}

int mantleCudaAttentionInnerF16CacheF32(
    const float* q,
    const uint16_t* cacheK,
    const uint16_t* cacheV,
    float* out,
    int pos,
    int start,
    int kvStride,
    int headDim,
    int nHead,
    int kvHeads,
    int cacheLen,
    float scale,
    cudaStream_t stream) {
    if (!q || !cacheK || !cacheV || !out || pos < 0 || start < 0 || start > pos || kvStride <= 0 || headDim <= 0 || nHead <= 0 || kvHeads <= 0 || cacheLen <= 0) {
        return cudaErrorInvalidValue;
    }
    const int threads = 256;
    dim3 block(threads);
    dim3 grid(nHead);
    attention_inner_f16_cache_f32_kernel<<<grid, block, 0, stream>>>(
        q, cacheK, cacheV, out, pos, start, kvStride, headDim, nHead, kvHeads, cacheLen, scale);
    return (int)cudaGetLastError();
}

}
