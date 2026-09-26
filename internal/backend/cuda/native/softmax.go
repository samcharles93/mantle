//go:build cuda

package native

/*
#include <stdint.h>
typedef void* cudaStream_t;
typedef int cudaError_t;

extern const char* cudaGetErrorString(cudaError_t err);

extern int mantleCudaSoftmaxRowsF32(
	float* data,
	int rows,
	int cols,
	cudaStream_t stream);

extern int mantleCudaLogitSoftcapF32(
	float* data,
	float softcap,
	int n,
	cudaStream_t stream);

static int mantleCudaSoftmaxRowsF32Wrapper(
	float* data,
	int rows,
	int cols,
	cudaStream_t stream) {
	return mantleCudaSoftmaxRowsF32(data, rows, cols, stream);
}

static int mantleCudaLogitSoftcapF32Wrapper(
	float* data,
	float softcap,
	int n,
	cudaStream_t stream) {
	return mantleCudaLogitSoftcapF32(data, softcap, n, stream);
}
*/
import "C"

import (
	"fmt"
	"math"
)

func SoftmaxRowsF32(data DeviceBuffer, rows, cols int, stream Stream) error {
	if data.ptr == nil {
		return fmt.Errorf("native.SoftmaxRowsF32: softmax buffer is nil")
	}
	if rows <= 0 || cols <= 0 || rows > math.MaxInt32 || cols > math.MaxInt32 {
		return fmt.Errorf("native.SoftmaxRowsF32: dimensions must be in [1, %d]", math.MaxInt32)
	}
	return cudaErr(C.mantleCudaSoftmaxRowsF32Wrapper(
		(*C.float)(data.ptr),
		C.int(rows),
		C.int(cols),
		stream.ptr,
	))
}

func LogitSoftcapF32(data DeviceBuffer, softcap float32, n int, stream Stream) error {
	if data.ptr == nil {
		return fmt.Errorf("logit softcap buffer is nil")
	}
	if n <= 0 {
		return fmt.Errorf("logit softcap n must be > 0")
	}
	if softcap <= 0 {
		return nil // Nothing to do
	}
	return cudaErr(C.mantleCudaLogitSoftcapF32Wrapper(
		(*C.float)(data.ptr),
		C.float(softcap),
		C.int(n),
		stream.ptr,
	))
}
