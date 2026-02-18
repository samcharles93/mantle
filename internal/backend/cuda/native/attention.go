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
