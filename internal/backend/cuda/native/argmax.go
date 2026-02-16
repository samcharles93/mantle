//go:build cuda

package native

/*
#include <stdint.h>
typedef void* cudaStream_t;
typedef int cudaError_t;

extern const char* cudaGetErrorString(cudaError_t err);

extern int mantleCudaArgMaxF32(
	const float* x,
	int n,
	int* outIdx,
	cudaStream_t stream);

static int mantleCudaArgMaxF32Wrapper(
	const float* x,
	int n,
	int* outIdx,
	cudaStream_t stream) {
	return mantleCudaArgMaxF32(x, n, outIdx, stream);
}
*/
import "C"
import (
	"fmt"
	"unsafe"
)

func ArgMaxF32(x DeviceBuffer, n int, stream Stream) (int, error) {
	if x.ptr == nil {
		return 0, fmt.Errorf("argmax input buffer is nil")
	}
	if n <= 0 {
		return 0, fmt.Errorf("argmax n must be > 0")
	}
	idxDev, err := AllocDevice(int64(C.sizeof_int))
	if err != nil {
		return 0, err
	}
	defer idxDev.Free()

	if err := ArgMaxF32To(x, n, idxDev, stream); err != nil {
		return 0, err
	}

	var idx C.int
	if err := MemcpyD2H(unsafe.Pointer(&idx), idxDev, int64(C.sizeof_int)); err != nil {
		return 0, err
	}
	return int(idx), nil
}

func ArgMaxF32To(x DeviceBuffer, n int, outIdx DeviceBuffer, stream Stream) error {
	if x.ptr == nil || outIdx.ptr == nil {
		return fmt.Errorf("argmax buffer is nil")
	}
	if n <= 0 {
		return fmt.Errorf("argmax n must be > 0")
	}
	return cudaErr(C.mantleCudaArgMaxF32Wrapper(
		(*C.float)(x.ptr),
		C.int(n),
		(*C.int)(outIdx.ptr),
		stream.ptr,
	))
}
