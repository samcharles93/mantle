//go:build cuda

package native

/*
#include <stdint.h>
typedef void* cudaStream_t;

extern int mantleCudaDeltaNetRecurrentF32(
	float* state,
	const float* q,
	const float* k_buf,
	const float* v,
	const float* a_log,
	const float* delta_a,
	const float* delta_b,
	const float* dt_bias,
	float* out,
	float scale,
	int head_key_dim,
	int head_value_dim,
	int n_value_heads,
	int n_key_heads,
	int group_size,
	cudaStream_t stream);

static int mantleCudaDeltaNetRecurrentF32Wrapper(
	float* state,
	const float* q,
	const float* k_buf,
	const float* v,
	const float* a_log,
	const float* delta_a,
	const float* delta_b,
	const float* dt_bias,
	float* out,
	float scale,
	int head_key_dim,
	int head_value_dim,
	int n_value_heads,
	int n_key_heads,
	int group_size,
	cudaStream_t stream) {
	return mantleCudaDeltaNetRecurrentF32(state, q, k_buf, v, a_log, delta_a, delta_b, dt_bias, out, scale,
		head_key_dim, head_value_dim, n_value_heads, n_key_heads, group_size, stream);
}
*/
import "C"
import "fmt"

// DeltaNetRecurrentF32 runs one DeltaNet recurrent step for all value heads.
// Fuses decay = exp(-exp(aLog) * softplus(deltaA + dtBias)) and
// beta = sigmoid(deltaB) on-device so the host avoids per-step H2D copies
// and the launch is CUDA Graph capturable.
//
// state: [n_value_heads * head_key_dim * head_value_dim] (k-major within head).
// q, k_buf: [n_key_heads * head_key_dim].
// v: [n_value_heads * head_value_dim].
// aLog, deltaA, deltaB, dtBias: [n_value_heads].
// out: [n_value_heads * head_value_dim] (overwritten).
func DeltaNetRecurrentF32(
	state, q, kBuf, v, aLog, deltaA, deltaB, dtBias, out DeviceBuffer,
	scale float32,
	headKeyDim, headValueDim, nValueHeads, nKeyHeads, groupSize int,
	stream Stream,
) error {
	if state.ptr == nil || q.ptr == nil || kBuf.ptr == nil || v.ptr == nil ||
		aLog.ptr == nil || deltaA.ptr == nil || deltaB.ptr == nil || dtBias.ptr == nil ||
		out.ptr == nil {
		return fmt.Errorf("deltanet recurrent buffer is nil")
	}
	if headKeyDim <= 0 || headValueDim <= 0 || nValueHeads <= 0 || nKeyHeads <= 0 || groupSize <= 0 {
		return fmt.Errorf("deltanet recurrent dimensions must be > 0")
	}
	return cudaErr(C.mantleCudaDeltaNetRecurrentF32Wrapper(
		(*C.float)(state.ptr),
		(*C.float)(q.ptr),
		(*C.float)(kBuf.ptr),
		(*C.float)(v.ptr),
		(*C.float)(aLog.ptr),
		(*C.float)(deltaA.ptr),
		(*C.float)(deltaB.ptr),
		(*C.float)(dtBias.ptr),
		(*C.float)(out.ptr),
		C.float(scale),
		C.int(headKeyDim),
		C.int(headValueDim),
		C.int(nValueHeads),
		C.int(nKeyHeads),
		C.int(groupSize),
		stream.ptr,
	))
}
