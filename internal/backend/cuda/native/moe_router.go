//go:build cuda

package native

/*
#include <stdint.h>
typedef void* cudaStream_t;
typedef int cudaError_t;

extern int mantleCudaMoERouterF32(
	float* raw,
	const float* bias,
	int biasLen,
	int k,
	float routeScale,
	int numExperts,
	int* idxOut,
	float* weightsOut,
	cudaStream_t stream);

static int mantleCudaMoERouterF32Wrapper(
	float* raw,
	const float* bias,
	int biasLen,
	int k,
	float routeScale,
	int numExperts,
	int* idxOut,
	float* weightsOut,
	cudaStream_t stream) {
	return mantleCudaMoERouterF32(raw, bias, biasLen, k, routeScale, numExperts, idxOut, weightsOut, stream);
}
*/
import "C"

import (
	"fmt"
)

// MoERouterF32 runs the fused router pipeline:
//
//	raw[i]      = sigmoid(raw[i])           (in-place, returned via raw)
//	sel[i]      = raw[i] + bias[i]          (bias zero-padded to numExperts)
//	idxOut[0:k] = argmax^k over sel with lower-index tiebreak
//	wOut[j]     = (raw[idx[j]] / sum_j raw[idx[j]]) * routeScale
//
// raw must hold numExperts contiguous f32; bias may be empty (biasLen==0) or len numExperts.
// idxOut must be a device int32 buffer of length k; weightsOut a device f32 buffer of length k.
func MoERouterF32(
	raw DeviceBuffer,
	bias DeviceBuffer,
	biasLen int,
	k int,
	routeScale float32,
	numExperts int,
	idxOut DeviceBuffer,
	weightsOut DeviceBuffer,
	stream Stream,
) error {
	if raw.ptr == nil {
		return fmt.Errorf("moe router raw buffer is nil")
	}
	if idxOut.ptr == nil || weightsOut.ptr == nil {
		return fmt.Errorf("moe router output buffers are nil")
	}
	if numExperts <= 0 {
		return fmt.Errorf("moe router numExperts must be > 0")
	}
	if k <= 0 || k > numExperts {
		return fmt.Errorf("moe router k=%d out of range [1,%d]", k, numExperts)
	}
	if biasLen < 0 || biasLen > numExperts {
		return fmt.Errorf("moe router biasLen=%d out of range [0,%d]", biasLen, numExperts)
	}
	var biasPtr *C.float
	if biasLen > 0 {
		if bias.ptr == nil {
			return fmt.Errorf("moe router bias buffer is nil but biasLen=%d", biasLen)
		}
		biasPtr = (*C.float)(bias.ptr)
	}
	return cudaErr(C.mantleCudaMoERouterF32Wrapper(
		(*C.float)(raw.ptr),
		biasPtr,
		C.int(biasLen),
		C.int(k),
		C.float(routeScale),
		C.int(numExperts),
		(*C.int)(idxOut.ptr),
		(*C.float)(weightsOut.ptr),
		stream.ptr,
	))
}
