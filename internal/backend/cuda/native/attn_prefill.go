//go:build cuda

package native

/*
#include <stdint.h>
typedef void* cudaStream_t;

extern int mantleCudaApplyRoPERowsF32(
	float* x,
	const float* inv_freq,
	int start_pos,
	float attn_scale,
	int head_dim,
	int half,
	int n_heads,
	int n_rows,
	cudaStream_t stream);

extern int mantleCudaStoreKVF16Rows(
	unsigned short* cache_k,
	unsigned short* cache_v,
	const float* k,
	const float* v,
	int start_pos,
	int row0,
	int n_rows,
	int kv_stride,
	int modulo_len,
	cudaStream_t stream);

extern int mantleCudaAttentionPrefillF16Supported(int headDim, int nHead, int kvHeads);

extern int mantleCudaAttentionPrefillF16(
	const float* q,
	const float* kNew,
	const float* vNew,
	const unsigned short* cacheK,
	const unsigned short* cacheV,
	float* out,
	int n,
	int startPos,
	int kvStride,
	int headDim,
	int nHead,
	int kvHeads,
	int cacheLen,
	float scale,
	float softcap,
	int window,
	cudaStream_t stream);
*/
import "C"

import "fmt"

// ApplyRoPERowsF32 is ApplyRoPEInplaceF32 for nRows consecutive [nHeads,
// headDim] rows of x at positions startPos, startPos+1, ..., in one launch
// with identical per-element results.
func ApplyRoPERowsF32(x, invFreq DeviceBuffer, startPos int, attnScale float32, headDim, half, nHeads, nRows int, stream Stream) error {
	if x.ptr == nil || invFreq.ptr == nil {
		return fmt.Errorf("native.ApplyRoPERowsF32: buffer is nil")
	}
	if headDim <= 0 || half <= 0 || nHeads <= 0 || nRows <= 0 || half*2 > headDim || startPos < 0 {
		return fmt.Errorf("native.ApplyRoPERowsF32: invalid dims (headDim=%d half=%d heads=%d rows=%d start=%d)", headDim, half, nHeads, nRows, startPos)
	}
	return cudaErr(C.mantleCudaApplyRoPERowsF32(
		(*C.float)(x.ptr), (*C.float)(invFreq.ptr), C.int(startPos), C.float(attnScale),
		C.int(headDim), C.int(half), C.int(nHeads), C.int(nRows), stream.ptr))
}

// StoreKVF16Rows stores rows row0..nRows-1 of the [nRows, kvStride] K and V
// blocks into the f16 caches at slot (startPos+row) % moduloLen, converting
// exactly as StoreKVF16Row. At most moduloLen rows may be stored, so no two
// rows of one call share a slot.
func StoreKVF16Rows(cacheK, cacheV, k, v DeviceBuffer, startPos, row0, nRows, kvStride, moduloLen int, stream Stream) error {
	if cacheK.ptr == nil || cacheV.ptr == nil || k.ptr == nil || v.ptr == nil {
		return fmt.Errorf("native.StoreKVF16Rows: buffer is nil")
	}
	if startPos < 0 || row0 < 0 || nRows <= row0 || kvStride <= 0 || moduloLen <= 0 || nRows-row0 > moduloLen {
		return fmt.Errorf("native.StoreKVF16Rows: invalid rows (start=%d row0=%d rows=%d modulo=%d)", startPos, row0, nRows, moduloLen)
	}
	return cudaErr(C.mantleCudaStoreKVF16Rows(
		(*C.ushort)(cacheK.ptr), (*C.ushort)(cacheV.ptr), (*C.float)(k.ptr), (*C.float)(v.ptr),
		C.int(startPos), C.int(row0), C.int(nRows), C.int(kvStride), C.int(moduloLen), stream.ptr))
}

// AttentionPrefillF16Supported reports whether AttentionPrefillF16 takes this
// head layout on the current device.
func AttentionPrefillF16Supported(headDim, nHead, kvHeads int) bool {
	return C.mantleCudaAttentionPrefillF16Supported(C.int(headDim), C.int(nHead), C.int(kvHeads)) != 0
}

// AttentionPrefillF16 computes causal attention for n query rows at positions
// startPos..startPos+n-1. q and out are [n, nHead*headDim]; kNew and vNew are
// the chunk's own [n, kvStride] K/V rows (after RoPE), read rounded through
// f16; positions below startPos come from the f16 caches at slot t%cacheLen.
// Row r attends [max(pos-window+1, 0), pos], or [0, pos] when window is 0.
// The caches are only read, so the chunk's rows can be stored afterwards.
func AttentionPrefillF16(q, kNew, vNew, cacheK, cacheV, out DeviceBuffer, n, startPos, kvStride, headDim, nHead, kvHeads, cacheLen, window int, scale, softcap float32, stream Stream) error {
	if q.ptr == nil || kNew.ptr == nil || vNew.ptr == nil || cacheK.ptr == nil || cacheV.ptr == nil || out.ptr == nil {
		return fmt.Errorf("native.AttentionPrefillF16: buffer is nil")
	}
	if n <= 0 || startPos < 0 || kvStride <= 0 || cacheLen <= 0 {
		return fmt.Errorf("native.AttentionPrefillF16: invalid dims (n=%d start=%d kvStride=%d cacheLen=%d)", n, startPos, kvStride, cacheLen)
	}
	return cudaErr(C.mantleCudaAttentionPrefillF16(
		(*C.float)(q.ptr), (*C.float)(kNew.ptr), (*C.float)(vNew.ptr),
		(*C.ushort)(cacheK.ptr), (*C.ushort)(cacheV.ptr), (*C.float)(out.ptr),
		C.int(n), C.int(startPos), C.int(kvStride), C.int(headDim), C.int(nHead), C.int(kvHeads),
		C.int(cacheLen), C.float(scale), C.float(softcap), C.int(window), stream.ptr))
}
