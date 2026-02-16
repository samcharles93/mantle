//go:build cuda

package native

/*
#include <stdint.h>
typedef void* cudaStream_t;
typedef int cudaError_t;

extern const char* cudaGetErrorString(cudaError_t err);

extern int mantleCudaSoftmaxRowsF32(float* data, int rows, int cols, cudaStream_t stream);
extern int mantleCudaSiluMulF32(
	const float* gate,
	const float* up,
	float* out,
	int n,
	cudaStream_t stream);
extern int mantleCudaAddVectorsF32(
	float* dst,
	const float* src,
	int n,
	cudaStream_t stream);
extern int mantleCudaConvertF32ToF16(
	const float* in,
	unsigned short* out,
	int n,
	cudaStream_t stream);
extern int mantleCudaShortConvDepthwise(
	const float* proj,
	const float* conv_w,
	float* state,
	float* out,
	int embd,
	int klen,
	cudaStream_t stream);

static int mantleCudaSoftmaxRowsF32Wrapper(float* data, int rows, int cols, cudaStream_t stream) {
	return mantleCudaSoftmaxRowsF32(data, rows, cols, stream);
}

static int mantleCudaSiluMulF32Wrapper(
	const float* gate,
	const float* up,
	float* out,
	int n,
	cudaStream_t stream) {
	return mantleCudaSiluMulF32(gate, up, out, n, stream);
}

static int mantleCudaAddVectorsF32Wrapper(
	float* dst,
	const float* src,
	int n,
	cudaStream_t stream) {
	return mantleCudaAddVectorsF32(dst, src, n, stream);
}

static int mantleCudaConvertF32ToF16Wrapper(
	const float* in,
	unsigned short* out,
	int n,
	cudaStream_t stream) {
	return mantleCudaConvertF32ToF16(in, out, n, stream);
}

static int mantleCudaShortConvDepthwiseWrapper(
	const float* proj,
	const float* conv_w,
	float* state,
	float* out,
	int embd,
	int klen,
	cudaStream_t stream) {
	return mantleCudaShortConvDepthwise(proj, conv_w, state, out, embd, klen, stream);
}
*/
import "C"
import (
	"fmt"
)

func SoftmaxRowsF32(buf DeviceBuffer, rows, cols int, stream Stream) error {
	if buf.ptr == nil {
		return fmt.Errorf("softmax buffer is nil")
	}
	if rows <= 0 || cols <= 0 {
		return fmt.Errorf("softmax rows/cols must be > 0")
	}
	return cudaErr(C.mantleCudaSoftmaxRowsF32Wrapper((*C.float)(buf.ptr), C.int(rows), C.int(cols), stream.ptr))
}

func SiluMulF32(gate, up, out DeviceBuffer, n int, stream Stream) error {
	if gate.ptr == nil || up.ptr == nil || out.ptr == nil {
		return fmt.Errorf("silu mul buffer is nil")
	}
	if n <= 0 {
		return fmt.Errorf("silu mul n must be > 0")
	}
	return cudaErr(C.mantleCudaSiluMulF32Wrapper(
		(*C.float)(gate.ptr),
		(*C.float)(up.ptr),
		(*C.float)(out.ptr),
		C.int(n),
		stream.ptr,
	))
}

func AddVectorsF32(dst, src DeviceBuffer, n int, stream Stream) error {
	if dst.ptr == nil || src.ptr == nil {
		return fmt.Errorf("add vectors buffer is nil")
	}
	if n <= 0 {
		return fmt.Errorf("add vectors n must be > 0")
	}
	return cudaErr(C.mantleCudaAddVectorsF32Wrapper(
		(*C.float)(dst.ptr),
		(*C.float)(src.ptr),
		C.int(n),
		stream.ptr,
	))
}

func ShortConvDepthwise(proj, convW, state, out DeviceBuffer, embd, klen int, stream Stream) error {
	if proj.ptr == nil || convW.ptr == nil || state.ptr == nil || out.ptr == nil {
		return fmt.Errorf("shortconv buffer is nil")
	}
	if embd <= 0 || klen <= 0 {
		return fmt.Errorf("shortconv embd/klen must be > 0")
	}
	return cudaErr(C.mantleCudaShortConvDepthwiseWrapper(
		(*C.float)(proj.ptr),
		(*C.float)(convW.ptr),
		(*C.float)(state.ptr),
		(*C.float)(out.ptr),
		C.int(embd),
		C.int(klen),
		stream.ptr,
	))
}

func ConvertF32ToF16(in, out DeviceBuffer, n int, stream Stream) error {
	if in.ptr == nil || out.ptr == nil {
		return fmt.Errorf("f32->f16 convert buffer is nil")
	}
	if n <= 0 {
		return fmt.Errorf("f32->f16 convert n must be > 0")
	}
	return cudaErr(C.mantleCudaConvertF32ToF16Wrapper(
		(*C.float)(in.ptr),
		(*C.ushort)(out.ptr),
		C.int(n),
		stream.ptr,
	))
}
