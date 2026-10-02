//go:build cuda

package native

/*
typedef void* cudaStream_t;
typedef int cudaError_t;

extern const char* cudaGetErrorString(cudaError_t err);
extern int mantleCudaAttentionInnerF16CacheF32(
	const float* q,
	const unsigned short* cacheK,
	const unsigned short* cacheV,
	float* out,
	int pos,
	int start,
	int kvStride,
	int headDim,
	int nHead,
	int kvHeads,
	int cacheLen,
	float scale,
	float softcap,
	cudaStream_t stream);
extern int mantleCudaAttentionInnerMixedCacheF32(
	const float* q,
	const unsigned short* cacheKF16,
	const unsigned short* cacheVF16,
	const signed char* cacheKQ8,
	const signed char* cacheVQ8,
	const float* cacheKScales,
	const float* cacheVScales,
	float* out,
	int useQ8K,
	int useQ8V,
	int pos,
	int start,
	int kvStride,
	int headDim,
	int nHead,
	int kvHeads,
	int cacheLen,
	float scale,
	float softcap,
	cudaStream_t stream);
extern int mantleCudaAttentionInnerF16CacheF32Batch(
	const float* q,
	const unsigned short* cacheK,
	const unsigned short* cacheV,
	float* out,
	int nTokens,
	int startPos,
	int kvStride,
	int headDim,
	int nHead,
	int kvHeads,
	int cacheLen,
	float scale,
	float softcap,
	cudaStream_t stream);
extern int mantleCudaAttentionInnerMixedCacheF32Batch(
	const float* q,
	const unsigned short* cacheKF16,
	const unsigned short* cacheVF16,
	const signed char* cacheKQ8,
	const signed char* cacheVQ8,
	const float* cacheKScales,
	const float* cacheVScales,
	float* out,
	int useQ8K,
	int useQ8V,
	int nTokens,
	int startPos,
	int kvStride,
	int headDim,
	int nHead,
	int kvHeads,
	int cacheLen,
	float scale,
	float softcap,
	cudaStream_t stream);

static int mantleCudaAttentionInnerF16CacheF32Wrapper(
	const float* q,
	const unsigned short* cacheK,
	const unsigned short* cacheV,
	float* out,
	int pos,
	int start,
	int kvStride,
	int headDim,
	int nHead,
	int kvHeads,
	int cacheLen,
	float scale,
	float softcap,
	cudaStream_t stream) {
	return mantleCudaAttentionInnerF16CacheF32(q, cacheK, cacheV, out, pos, start, kvStride, headDim, nHead, kvHeads, cacheLen, scale, softcap, stream);
}

static int mantleCudaAttentionInnerMixedCacheF32Wrapper(
	const float* q,
	const unsigned short* cacheKF16,
	const unsigned short* cacheVF16,
	const signed char* cacheKQ8,
	const signed char* cacheVQ8,
	const float* cacheKScales,
	const float* cacheVScales,
	float* out,
	int useQ8K,
	int useQ8V,
	int pos,
	int start,
	int kvStride,
	int headDim,
	int nHead,
	int kvHeads,
	int cacheLen,
	float scale,
	float softcap,
	cudaStream_t stream) {
	return mantleCudaAttentionInnerMixedCacheF32(
		q,
		cacheKF16,
		cacheVF16,
		cacheKQ8,
		cacheVQ8,
		cacheKScales,
		cacheVScales,
		out,
		useQ8K,
		useQ8V,
		pos,
		start,
		kvStride,
		headDim,
		nHead,
		kvHeads,
		cacheLen,
		scale,
		softcap,
		stream);
}

static int mantleCudaAttentionInnerF16CacheF32BatchWrapper(
	const float* q,
	const unsigned short* cacheK,
	const unsigned short* cacheV,
	float* out,
	int nTokens,
	int startPos,
	int kvStride,
	int headDim,
	int nHead,
	int kvHeads,
	int cacheLen,
	float scale,
	float softcap,
	cudaStream_t stream) {
	return mantleCudaAttentionInnerF16CacheF32Batch(q, cacheK, cacheV, out, nTokens, startPos, kvStride, headDim, nHead, kvHeads, cacheLen, scale, softcap, stream);
}

static int mantleCudaAttentionInnerMixedCacheF32BatchWrapper(
	const float* q,
	const unsigned short* cacheKF16,
	const unsigned short* cacheVF16,
	const signed char* cacheKQ8,
	const signed char* cacheVQ8,
	const float* cacheKScales,
	const float* cacheVScales,
	float* out,
	int useQ8K,
	int useQ8V,
	int nTokens,
	int startPos,
	int kvStride,
	int headDim,
	int nHead,
	int kvHeads,
	int cacheLen,
	float scale,
	float softcap,
	cudaStream_t stream) {
	return mantleCudaAttentionInnerMixedCacheF32Batch(
		q,
		cacheKF16,
		cacheVF16,
		cacheKQ8,
		cacheVQ8,
		cacheKScales,
		cacheVScales,
		out,
		useQ8K,
		useQ8V,
		nTokens,
		startPos,
		kvStride,
		headDim,
		nHead,
		kvHeads,
		cacheLen,
		scale,
		softcap,
		stream);
}
*/
import "C"

import (
	"fmt"
)

func AttentionInnerF16CacheF32(q, cacheK, cacheV, out DeviceBuffer, pos, start, kvStride, headDim, nHead, kvHeads, cacheLen int, scale, softcap float32, stream Stream) error {
	if q.ptr == nil || cacheK.ptr == nil || cacheV.ptr == nil || out.ptr == nil {
		return fmt.Errorf("attention inner buffer is nil")
	}
	if pos < 0 || start < 0 || start > pos {
		return fmt.Errorf("attention inner invalid position/window")
	}
	if kvStride <= 0 || headDim <= 0 || nHead <= 0 || kvHeads <= 0 || cacheLen <= 0 {
		return fmt.Errorf("attention inner dimensions must be > 0")
	}
	return cudaErr(C.mantleCudaAttentionInnerF16CacheF32Wrapper(
		(*C.float)(q.ptr),
		(*C.ushort)(cacheK.ptr),
		(*C.ushort)(cacheV.ptr),
		(*C.float)(out.ptr),
		C.int(pos),
		C.int(start),
		C.int(kvStride),
		C.int(headDim),
		C.int(nHead),
		C.int(kvHeads),
		C.int(cacheLen),
		C.float(scale),
		C.float(softcap),
		stream.ptr,
	))
}

func AttentionInnerMixedCacheF32(
	q DeviceBuffer,
	cacheKF16 DeviceBuffer,
	cacheVF16 DeviceBuffer,
	cacheKQ8 DeviceBuffer,
	cacheVQ8 DeviceBuffer,
	cacheKScales DeviceBuffer,
	cacheVScales DeviceBuffer,
	out DeviceBuffer,
	useQ8K bool,
	useQ8V bool,
	pos, start, kvStride, headDim, nHead, kvHeads, cacheLen int,
	scale, softcap float32,
	stream Stream,
) error {
	if q.ptr == nil || out.ptr == nil {
		return fmt.Errorf("attention inner buffer is nil")
	}
	if useQ8K {
		if cacheKQ8.ptr == nil || cacheKScales.ptr == nil {
			return fmt.Errorf("attention inner q8 k cache buffers are nil")
		}
	} else if cacheKF16.ptr == nil {
		return fmt.Errorf("attention inner f16 k cache buffer is nil")
	}
	if useQ8V {
		if cacheVQ8.ptr == nil || cacheVScales.ptr == nil {
			return fmt.Errorf("attention inner q8 v cache buffers are nil")
		}
	} else if cacheVF16.ptr == nil {
		return fmt.Errorf("attention inner f16 v cache buffer is nil")
	}
	if pos < 0 || start < 0 || start > pos {
		return fmt.Errorf("attention inner invalid position/window")
	}
	if kvStride <= 0 || headDim <= 0 || nHead <= 0 || kvHeads <= 0 || cacheLen <= 0 {
		return fmt.Errorf("attention inner dimensions must be > 0")
	}
	useQ8KC := C.int(0)
	if useQ8K {
		useQ8KC = 1
	}
	useQ8VC := C.int(0)
	if useQ8V {
		useQ8VC = 1
	}
	return cudaErr(C.mantleCudaAttentionInnerMixedCacheF32Wrapper(
		(*C.float)(q.ptr),
		(*C.ushort)(cacheKF16.ptr),
		(*C.ushort)(cacheVF16.ptr),
		(*C.schar)(cacheKQ8.ptr),
		(*C.schar)(cacheVQ8.ptr),
		(*C.float)(cacheKScales.ptr),
		(*C.float)(cacheVScales.ptr),
		(*C.float)(out.ptr),
		useQ8KC,
		useQ8VC,
		C.int(pos),
		C.int(start),
		C.int(kvStride),
		C.int(headDim),
		C.int(nHead),
		C.int(kvHeads),
		C.int(cacheLen),
		C.float(scale),
		C.float(softcap),
		stream.ptr,
	))
}

// AttentionInnerF16CacheF32Batch runs the F16-cache inner attention for
// nTokens query rows in one kernel launch. q and out are laid out as
// [nTokens, nHead, headDim] with row i at offset i*nHead*headDim, and row i
// attends at pos = startPos + i with a zero window start (the caller rejects
// sliding windows, so start is always 0). The result is numerically identical
// to calling AttentionInnerF16CacheF32 once per row.
func AttentionInnerF16CacheF32Batch(q, cacheK, cacheV, out DeviceBuffer, nTokens, startPos, kvStride, headDim, nHead, kvHeads, cacheLen int, scale, softcap float32, stream Stream) error {
	if q.ptr == nil || cacheK.ptr == nil || cacheV.ptr == nil || out.ptr == nil {
		return fmt.Errorf("attention inner buffer is nil")
	}
	if nTokens <= 0 || startPos < 0 {
		return fmt.Errorf("attention inner invalid token count/start position")
	}
	if kvStride <= 0 || headDim <= 0 || nHead <= 0 || kvHeads <= 0 || cacheLen <= 0 {
		return fmt.Errorf("attention inner dimensions must be > 0")
	}
	return cudaErr(C.mantleCudaAttentionInnerF16CacheF32BatchWrapper(
		(*C.float)(q.ptr),
		(*C.ushort)(cacheK.ptr),
		(*C.ushort)(cacheV.ptr),
		(*C.float)(out.ptr),
		C.int(nTokens),
		C.int(startPos),
		C.int(kvStride),
		C.int(headDim),
		C.int(nHead),
		C.int(kvHeads),
		C.int(cacheLen),
		C.float(scale),
		C.float(softcap),
		stream.ptr,
	))
}

// AttentionInnerMixedCacheF32Batch is the batched form of
// AttentionInnerMixedCacheF32 with the same [nTokens, nHead, headDim] query
// layout and startPos + i position mapping described on
// AttentionInnerF16CacheF32Batch.
func AttentionInnerMixedCacheF32Batch(
	q DeviceBuffer,
	cacheKF16 DeviceBuffer,
	cacheVF16 DeviceBuffer,
	cacheKQ8 DeviceBuffer,
	cacheVQ8 DeviceBuffer,
	cacheKScales DeviceBuffer,
	cacheVScales DeviceBuffer,
	out DeviceBuffer,
	useQ8K bool,
	useQ8V bool,
	nTokens, startPos, kvStride, headDim, nHead, kvHeads, cacheLen int,
	scale, softcap float32,
	stream Stream,
) error {
	if q.ptr == nil || out.ptr == nil {
		return fmt.Errorf("attention inner buffer is nil")
	}
	if useQ8K {
		if cacheKQ8.ptr == nil || cacheKScales.ptr == nil {
			return fmt.Errorf("attention inner q8 k cache buffers are nil")
		}
	} else if cacheKF16.ptr == nil {
		return fmt.Errorf("attention inner f16 k cache buffer is nil")
	}
	if useQ8V {
		if cacheVQ8.ptr == nil || cacheVScales.ptr == nil {
			return fmt.Errorf("attention inner q8 v cache buffers are nil")
		}
	} else if cacheVF16.ptr == nil {
		return fmt.Errorf("attention inner f16 v cache buffer is nil")
	}
	if nTokens <= 0 || startPos < 0 {
		return fmt.Errorf("attention inner invalid token count/start position")
	}
	if kvStride <= 0 || headDim <= 0 || nHead <= 0 || kvHeads <= 0 || cacheLen <= 0 {
		return fmt.Errorf("attention inner dimensions must be > 0")
	}
	useQ8KC := C.int(0)
	if useQ8K {
		useQ8KC = 1
	}
	useQ8VC := C.int(0)
	if useQ8V {
		useQ8VC = 1
	}
	return cudaErr(C.mantleCudaAttentionInnerMixedCacheF32BatchWrapper(
		(*C.float)(q.ptr),
		(*C.ushort)(cacheKF16.ptr),
		(*C.ushort)(cacheVF16.ptr),
		(*C.schar)(cacheKQ8.ptr),
		(*C.schar)(cacheVQ8.ptr),
		(*C.float)(cacheKScales.ptr),
		(*C.float)(cacheVScales.ptr),
		(*C.float)(out.ptr),
		useQ8KC,
		useQ8VC,
		C.int(nTokens),
		C.int(startPos),
		C.int(kvStride),
		C.int(headDim),
		C.int(nHead),
		C.int(kvHeads),
		C.int(cacheLen),
		C.float(scale),
		C.float(softcap),
		stream.ptr,
	))
}
