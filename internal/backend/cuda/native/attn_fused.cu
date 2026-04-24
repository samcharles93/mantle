#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Kernels used to keep the fused attention pipeline fully on the device:
//   1. apply_rope_f32_inplace: rotary positional embedding, in-place on
//      [nHead, headDim] F32 tensor, using a pre-uploaded F32 inv_freq table.
//   2. store_kv_f16_row: F32 -> F16 conversion of one KV vector, written
//      at cachePos * kvStride in the device KV cache.
//   3. store_kv_q8_row_broadcast: absmax Q8 quantization of one KV vector
//      with the row's scale broadcast to every block entry (matches the
//      existing host `broadcastQ8Scale` semantics so the mixed attention
//      kernel reads a uniform scale per row).
//
// Matches the host semantics in internal/backend/simd/ops.go
// (applyRoPEScalar) and internal/backend/cuda/ops.go (storeQ8 /
// broadcastQ8Scale).

extern "C" {

// One block per head. Each thread handles multiple (i) pairs.
__global__ void apply_rope_f32_inplace_kernel(
    float* __restrict__ x,
    const float* __restrict__ inv_freq,
    int pos,
    float attn_scale,
    int head_dim,
    int half,
    int n_heads)
{
  const int head = blockIdx.x;
  if (head >= n_heads) return;
  const int tid = threadIdx.x;
  float* xh = x + head * head_dim;
  const float fpos = static_cast<float>(pos);
  for (int i = tid; i < half; i += blockDim.x) {
    float angle = fpos * inv_freq[i];
    float c, s;
    __sincosf(angle, &s, &c);
    c *= attn_scale;
    s *= attn_scale;
    const int i0 = i;
    const int i1 = i + half;
    float x0 = xh[i0];
    float x1 = xh[i1];
    xh[i0] = x0 * c - x1 * s;
    xh[i1] = x0 * s + x1 * c;
  }
}

__host__ int mantleCudaApplyRoPEInplaceF32(
    float* x,
    const float* inv_freq,
    int pos,
    float attn_scale,
    int head_dim,
    int half,
    int n_heads,
    cudaStream_t stream)
{
  if (!x || !inv_freq || head_dim <= 0 || half <= 0 || n_heads <= 0) {
    return cudaErrorInvalidValue;
  }
  if (half * 2 > head_dim) {
    return cudaErrorInvalidValue;
  }
  if (attn_scale == 0.0f) attn_scale = 1.0f;
  int block = 128;
  if (half < 128) block = 64;
  if (half < 64) block = 32;
  apply_rope_f32_inplace_kernel<<<n_heads, block, 0, stream>>>(
      x, inv_freq, pos, attn_scale, head_dim, half, n_heads);
  return cudaGetLastError();
}

// One block. Each thread converts multiple elements.
__global__ void store_kv_f16_row_kernel(
    uint16_t* __restrict__ cache,
    const float* __restrict__ src,
    int cache_pos,
    int kv_stride)
{
  const int tid = threadIdx.x;
  uint16_t* dst = cache + (size_t)cache_pos * (size_t)kv_stride;
  for (int i = tid; i < kv_stride; i += blockDim.x) {
    __half h = __float2half(src[i]);
    dst[i] = __half_as_ushort(h);
  }
}

__host__ int mantleCudaStoreKVF16Row(
    uint16_t* cache,
    const float* src,
    int cache_pos,
    int kv_stride,
    cudaStream_t stream)
{
  if (!cache || !src || cache_pos < 0 || kv_stride <= 0) {
    return cudaErrorInvalidValue;
  }
  int block = 256;
  if (kv_stride < 256) block = 128;
  if (kv_stride < 128) block = 64;
  store_kv_f16_row_kernel<<<1, block, 0, stream>>>(
      cache, src, cache_pos, kv_stride);
  return cudaGetLastError();
}

// Single-block Q8 row quantizer. Computes absmax across `kv_stride`
// elements, derives `scale = absmax / 127` (0 if absmax == 0), quantizes
// each element to int8, and broadcasts the single scale value to every
// block entry of this row in the scales buffer.
__global__ void store_kv_q8_row_broadcast_kernel(
    int8_t* __restrict__ q_cache,
    float* __restrict__ scales,
    const float* __restrict__ src,
    int cache_pos,
    int kv_stride,
    int blocks_per_row)
{
  extern __shared__ float shmem[];
  const int tid = threadIdx.x;
  const int bs = blockDim.x;

  // Phase 1: parallel absmax reduction.
  float local = 0.0f;
  for (int i = tid; i < kv_stride; i += bs) {
    float v = fabsf(src[i]);
    if (v > local) local = v;
  }
  shmem[tid] = local;
  __syncthreads();
  for (int s = bs / 2; s > 0; s >>= 1) {
    if (tid < s) {
      float a = shmem[tid];
      float b = shmem[tid + s];
      shmem[tid] = (a > b) ? a : b;
    }
    __syncthreads();
  }

  __shared__ float scale_s;
  __shared__ float inv_scale_s;
  if (tid == 0) {
    float absmax = shmem[0];
    float scale = (absmax == 0.0f) ? 0.0f : (absmax / 127.0f);
    scale_s = scale;
    inv_scale_s = (scale == 0.0f) ? 0.0f : (127.0f / absmax);
  }
  __syncthreads();

  // Phase 2: quantize.
  int8_t* qdst = q_cache + (size_t)cache_pos * (size_t)kv_stride;
  const float inv_scale = inv_scale_s;
  if (inv_scale == 0.0f) {
    for (int i = tid; i < kv_stride; i += bs) qdst[i] = 0;
  } else {
    for (int i = tid; i < kv_stride; i += bs) {
      float v = src[i];
      float scaled = v * inv_scale;
      // Round toward zero-symmetric half-up, matching host storeQ8.
      float rounded = (v >= 0.0f) ? (scaled + 0.5f) : (scaled - 0.5f);
      int q = static_cast<int>(rounded);
      if (q > 127) q = 127;
      else if (q < -127) q = -127;
      qdst[i] = static_cast<int8_t>(q);
    }
  }

  // Phase 3: broadcast the scale across this row's block scale strip.
  float* sdst = scales + (size_t)cache_pos * (size_t)blocks_per_row;
  const float scale = scale_s;
  for (int i = tid; i < blocks_per_row; i += bs) {
    sdst[i] = scale;
  }
}

__host__ int mantleCudaStoreKVQ8RowBroadcast(
    int8_t* q_cache,
    float* scales,
    const float* src,
    int cache_pos,
    int kv_stride,
    int blocks_per_row,
    cudaStream_t stream)
{
  if (!q_cache || !scales || !src || cache_pos < 0 || kv_stride <= 0 ||
      blocks_per_row <= 0) {
    return cudaErrorInvalidValue;
  }
  int block = 256;
  if (kv_stride < 256) block = 128;
  if (kv_stride < 128) block = 64;
  size_t shmem_bytes = block * sizeof(float);
  store_kv_q8_row_broadcast_kernel<<<1, block, shmem_bytes, stream>>>(
      q_cache, scales, src, cache_pos, kv_stride, blocks_per_row);
  return cudaGetLastError();
}

} // extern "C"
