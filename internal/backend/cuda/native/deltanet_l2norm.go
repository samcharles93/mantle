//go:build cuda

package native

/*
#include <stdint.h>
typedef void* cudaStream_t;
typedef int cudaError_t;

extern int mantleCudaDeltaNetL2NormF32(
	float* x,
	float eps,
	int head_dim,
	int n_heads,
	cudaStream_t stream);

static int mantleCudaDeltaNetL2NormF32Wrapper(
	float* x,
	float eps,
	int head_dim,
	int n_heads,
	cudaStream_t stream) {
	return mantleCudaDeltaNetL2NormF32(x, eps, head_dim, n_heads, stream);
}
*/
import "C"
import (
	"fmt"
)

// DeltaNetL2NormF32 normalizes each head row of x in-place: x_h *= rsqrt(sum(x_h^2)+eps).
// x must contain n_heads * head_dim contiguous floats.
func DeltaNetL2NormF32(x DeviceBuffer, eps float32, headDim, nHeads int, stream Stream) error {
	if x.ptr == nil {
		return fmt.Errorf("deltanet l2norm buffer is nil")
	}
	if headDim <= 0 || nHeads <= 0 {
		return fmt.Errorf("deltanet l2norm dimensions must be > 0")
	}
	return cudaErr(C.mantleCudaDeltaNetL2NormF32Wrapper(
		(*C.float)(x.ptr),
		C.float(eps),
		C.int(headDim),
		C.int(nHeads),
		stream.ptr,
	))
}
