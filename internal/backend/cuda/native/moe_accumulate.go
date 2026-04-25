//go:build cuda

package native

/*
#include <stdint.h>
typedef void* cudaStream_t;
typedef int cudaError_t;

extern int mantleCudaMoEAccumulateF32(
	float* accum,
	const float* src,
	float w,
	int n,
	cudaStream_t stream);

static int mantleCudaMoEAccumulateF32Wrapper(
	float* accum,
	const float* src,
	float w,
	int n,
	cudaStream_t stream) {
	return mantleCudaMoEAccumulateF32(accum, src, w, n, stream);
}
*/
import "C"
import (
	"fmt"
)

// MoEAccumulateF32 computes accum[i] = fma(w, src[i], accum[i]) for i in [0,n).
// Both buffers must be device f32 of length >= n. n==0 is a no-op.
func MoEAccumulateF32(
	accum DeviceBuffer,
	src DeviceBuffer,
	w float32,
	n int,
	stream Stream,
) error {
	if n == 0 {
		return nil
	}
	if n < 0 {
		return fmt.Errorf("moe accumulate n=%d must be >= 0", n)
	}
	if accum.ptr == nil || src.ptr == nil {
		return fmt.Errorf("moe accumulate buffers must be non-nil")
	}
	return cudaErr(C.mantleCudaMoEAccumulateF32Wrapper(
		(*C.float)(accum.ptr),
		(*C.float)(src.ptr),
		C.float(w),
		C.int(n),
		stream.ptr,
	))
}
