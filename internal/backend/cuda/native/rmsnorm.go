//go:build cuda

package native

/*
#include <stdint.h>
typedef void* cudaStream_t;
typedef int cudaError_t;

extern const char* cudaGetErrorString(cudaError_t err);

extern int mantleCudaRMSNormF32(
	float* out,
	const float* x,
	const float* weight,
	float eps,
	int n,
	cudaStream_t stream);
extern int mantleCudaRMSNormBatchedF32(
	float* out,
	const float* x,
	const float* weight,
	float eps,
	int head_dim,
	int n_heads,
	cudaStream_t stream);

static int mantleCudaRMSNormF32Wrapper(
	float* out,
	const float* x,
	const float* weight,
	float eps,
	int n,
	cudaStream_t stream) {
	return mantleCudaRMSNormF32(out, x, weight, eps, n, stream);
}

static int mantleCudaRMSNormBatchedF32Wrapper(
	float* out,
	const float* x,
	const float* weight,
	float eps,
	int head_dim,
	int n_heads,
	cudaStream_t stream) {
	return mantleCudaRMSNormBatchedF32(out, x, weight, eps, head_dim, n_heads, stream);
}
*/
import "C"
import (
	"fmt"
)

func RMSNormF32(out, x, weight DeviceBuffer, eps float32, n int, stream Stream) error {
	if out.ptr == nil || x.ptr == nil || weight.ptr == nil {
		return fmt.Errorf("rmsnorm buffer is nil")
	}
	if n <= 0 {
		return fmt.Errorf("rmsnorm n must be > 0")
	}
	return cudaErr(C.mantleCudaRMSNormF32Wrapper(
		(*C.float)(out.ptr),
		(*C.float)(x.ptr),
		(*C.float)(weight.ptr),
		C.float(eps),
		C.int(n),
		stream.ptr,
	))
}

func RMSNormBatchedF32(out, x, weight DeviceBuffer, eps float32, headDim, nHeads int, stream Stream) error {
	if out.ptr == nil || x.ptr == nil || weight.ptr == nil {
		return fmt.Errorf("batched rmsnorm buffer is nil")
	}
	if headDim <= 0 || nHeads <= 0 {
		return fmt.Errorf("batched rmsnorm dimensions must be > 0")
	}
	return cudaErr(C.mantleCudaRMSNormBatchedF32Wrapper(
		(*C.float)(out.ptr),
		(*C.float)(x.ptr),
		(*C.float)(weight.ptr),
		C.float(eps),
		C.int(headDim),
		C.int(nHeads),
		stream.ptr,
	))
}
