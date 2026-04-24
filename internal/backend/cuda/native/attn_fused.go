//go:build cuda

package native

/*
#include <stdint.h>
typedef void* cudaStream_t;
typedef int cudaError_t;

extern const char* cudaGetErrorString(cudaError_t err);

extern int mantleCudaApplyRoPEInplaceF32(
	float* x,
	const float* inv_freq,
	int pos,
	float attn_scale,
	int head_dim,
	int half,
	int n_heads,
	cudaStream_t stream);

extern int mantleCudaStoreKVF16Row(
	unsigned short* cache,
	const float* src,
	int cache_pos,
	int kv_stride,
	cudaStream_t stream);

extern int mantleCudaStoreKVQ8RowBroadcast(
	signed char* q_cache,
	float* scales,
	const float* src,
	int cache_pos,
	int kv_stride,
	int blocks_per_row,
	cudaStream_t stream);

static int mantleCudaApplyRoPEInplaceF32Wrapper(
	float* x,
	const float* inv_freq,
	int pos,
	float attn_scale,
	int head_dim,
	int half,
	int n_heads,
	cudaStream_t stream) {
	return mantleCudaApplyRoPEInplaceF32(x, inv_freq, pos, attn_scale, head_dim, half, n_heads, stream);
}

static int mantleCudaStoreKVF16RowWrapper(
	unsigned short* cache,
	const float* src,
	int cache_pos,
	int kv_stride,
	cudaStream_t stream) {
	return mantleCudaStoreKVF16Row(cache, src, cache_pos, kv_stride, stream);
}

static int mantleCudaStoreKVQ8RowBroadcastWrapper(
	signed char* q_cache,
	float* scales,
	const float* src,
	int cache_pos,
	int kv_stride,
	int blocks_per_row,
	cudaStream_t stream) {
	return mantleCudaStoreKVQ8RowBroadcast(q_cache, scales, src, cache_pos, kv_stride, blocks_per_row, stream);
}
*/
import "C"

import "fmt"

// ApplyRoPEInplaceF32 applies rotary positional embeddings in place to a
// [nHeads, headDim] F32 tensor on the device. invFreq is an F32 device
// buffer of length `half` (= rotary dim / 2). attnScale == 0 is treated
// as 1.0. Mirrors the CPU semantics in simd.ApplyRoPE.
func ApplyRoPEInplaceF32(x, invFreq DeviceBuffer, pos int, attnScale float32, headDim, half, nHeads int, stream Stream) error {
	if x.ptr == nil || invFreq.ptr == nil {
		return fmt.Errorf("native.ApplyRoPEInplaceF32: buffer is nil")
	}
	if headDim <= 0 || half <= 0 || nHeads <= 0 {
		return fmt.Errorf("native.ApplyRoPEInplaceF32: dimensions must be > 0")
	}
	if half*2 > headDim {
		return fmt.Errorf("native.ApplyRoPEInplaceF32: half*2 (%d) exceeds headDim (%d)", half*2, headDim)
	}
	if pos < 0 {
		return fmt.Errorf("native.ApplyRoPEInplaceF32: pos must be >= 0")
	}
	if err := cudaErr(C.mantleCudaApplyRoPEInplaceF32Wrapper(
		(*C.float)(x.ptr),
		(*C.float)(invFreq.ptr),
		C.int(pos),
		C.float(attnScale),
		C.int(headDim),
		C.int(half),
		C.int(nHeads),
		stream.ptr,
	)); err != nil {
		return fmt.Errorf("native.ApplyRoPEInplaceF32(pos=%d, nHeads=%d, headDim=%d, half=%d): %w", pos, nHeads, headDim, half, err)
	}
	return nil
}

// StoreKVF16Row converts one F32 vector of kvStride elements to F16 and
// writes it at cachePos * kvStride in the device cache buffer.
func StoreKVF16Row(cache, src DeviceBuffer, cachePos, kvStride int, stream Stream) error {
	if cache.ptr == nil || src.ptr == nil {
		return fmt.Errorf("native.StoreKVF16Row: buffer is nil")
	}
	if cachePos < 0 {
		return fmt.Errorf("native.StoreKVF16Row: cachePos must be >= 0")
	}
	if kvStride <= 0 {
		return fmt.Errorf("native.StoreKVF16Row: kvStride must be > 0")
	}
	if err := cudaErr(C.mantleCudaStoreKVF16RowWrapper(
		(*C.ushort)(cache.ptr),
		(*C.float)(src.ptr),
		C.int(cachePos),
		C.int(kvStride),
		stream.ptr,
	)); err != nil {
		return fmt.Errorf("native.StoreKVF16Row(cachePos=%d, kvStride=%d): %w", cachePos, kvStride, err)
	}
	return nil
}

// StoreKVQ8RowBroadcast absmax-quantizes one F32 vector of kvStride
// elements to int8 at cachePos * kvStride in qCache, and broadcasts the
// single row scale (absmax / 127, or 0 if absmax == 0) to every
// block entry of this row's scale strip in scales
// ([cachePos * blocksPerRow .. +blocksPerRow]). Matches the host Q8
// layout produced by quantizeQ8 + broadcastQ8Scale.
func StoreKVQ8RowBroadcast(qCache, scales, src DeviceBuffer, cachePos, kvStride, blocksPerRow int, stream Stream) error {
	if qCache.ptr == nil || scales.ptr == nil || src.ptr == nil {
		return fmt.Errorf("native.StoreKVQ8RowBroadcast: buffer is nil")
	}
	if cachePos < 0 {
		return fmt.Errorf("native.StoreKVQ8RowBroadcast: cachePos must be >= 0")
	}
	if kvStride <= 0 || blocksPerRow <= 0 {
		return fmt.Errorf("native.StoreKVQ8RowBroadcast: kvStride and blocksPerRow must be > 0")
	}
	if err := cudaErr(C.mantleCudaStoreKVQ8RowBroadcastWrapper(
		(*C.schar)(qCache.ptr),
		(*C.float)(scales.ptr),
		(*C.float)(src.ptr),
		C.int(cachePos),
		C.int(kvStride),
		C.int(blocksPerRow),
		stream.ptr,
	)); err != nil {
		return fmt.Errorf("native.StoreKVQ8RowBroadcast(cachePos=%d, kvStride=%d, blocks=%d): %w", cachePos, kvStride, blocksPerRow, err)
	}
	return nil
}
