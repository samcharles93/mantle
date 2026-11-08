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
extern int mantleCudaRMSNormResidualF32(
	float* out,
	const float* x,
	const float* weight,
	float eps,
	int n,
	cudaStream_t stream);
extern int mantleCudaRMSNormBatchedPairF32(
	float* out_q,
	const float* x_q,
	const float* w_q,
	float* out_k,
	const float* x_k,
	const float* w_k,
	float eps,
	int head_dim,
	int n_heads_q,
	int n_heads_k,
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

static int mantleCudaRMSNormResidualF32Wrapper(
	float* out,
	const float* x,
	const float* weight,
	float eps,
	int n,
	cudaStream_t stream) {
	return mantleCudaRMSNormResidualF32(out, x, weight, eps, n, stream);
}

static int mantleCudaRMSNormBatchedPairF32Wrapper(
	float* out_q,
	const float* x_q,
	const float* w_q,
	float* out_k,
	const float* x_k,
	const float* w_k,
	float eps,
	int head_dim,
	int n_heads_q,
	int n_heads_k,
	cudaStream_t stream) {
	return mantleCudaRMSNormBatchedPairF32(out_q, x_q, w_q, out_k, x_k, w_k, eps, head_dim, n_heads_q, n_heads_k, stream);
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

// RMSNormBatchedPairF32 normalizes the Q tensor [nHeadsQ, headDim] with wQ
// and the K tensor [nHeadsK, headDim] with wK in a single kernel launch.
// Bit-identical to two RMSNormBatchedF32 calls.
func RMSNormBatchedPairF32(outQ, xQ, wQ, outK, xK, wK DeviceBuffer, eps float32, headDim, nHeadsQ, nHeadsK int, stream Stream) error {
	if outQ.ptr == nil || xQ.ptr == nil || wQ.ptr == nil || outK.ptr == nil || xK.ptr == nil || wK.ptr == nil {
		return fmt.Errorf("batched rmsnorm pair buffer is nil")
	}
	if headDim <= 0 || nHeadsQ <= 0 || nHeadsK <= 0 {
		return fmt.Errorf("batched rmsnorm pair dimensions must be > 0")
	}
	return cudaErr(C.mantleCudaRMSNormBatchedPairF32Wrapper(
		(*C.float)(outQ.ptr),
		(*C.float)(xQ.ptr),
		(*C.float)(wQ.ptr),
		(*C.float)(outK.ptr),
		(*C.float)(xK.ptr),
		(*C.float)(wK.ptr),
		C.float(eps),
		C.int(headDim),
		C.int(nHeadsQ),
		C.int(nHeadsK),
		stream.ptr,
	))
}

// RMSNormResidualF32 computes out[i] = out[i] + x[i]*scale*w[i] in one kernel,
// with scale from the same reduction as RMSNormF32. Bit-identical to
// RMSNormF32 into a temporary followed by AddVectorsF32.
func RMSNormResidualF32(out, x, weight DeviceBuffer, eps float32, n int, stream Stream) error {
	if out.ptr == nil || x.ptr == nil || weight.ptr == nil {
		return fmt.Errorf("rmsnorm residual buffer is nil")
	}
	if n <= 0 {
		return fmt.Errorf("rmsnorm residual n must be > 0")
	}
	return cudaErr(C.mantleCudaRMSNormResidualF32Wrapper(
		(*C.float)(out.ptr),
		(*C.float)(x.ptr),
		(*C.float)(weight.ptr),
		C.float(eps),
		C.int(n),
		stream.ptr,
	))
}
