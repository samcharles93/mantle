//go:build cuda

package native

/*
#include <stdint.h>
typedef void* cudaStream_t;
typedef int cudaError_t;

extern const char* cudaGetErrorString(cudaError_t err);


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
extern int mantleCudaConvertF32ToBF16(
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

static int mantleCudaConvertF32ToBF16Wrapper(
	const float* in,
	unsigned short* out,
	int n,
	cudaStream_t stream) {
	return mantleCudaConvertF32ToBF16(in, out, n, stream);
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
	"math"
)

func checkedPositiveCInt(name string, v int) (C.int, error) {
	if v <= 0 {
		return 0, fmt.Errorf("%s must be > 0", name)
	}
	if v > math.MaxInt32 {
		return 0, fmt.Errorf("%s exceeds int32 max: %d", name, v)
	}
	return C.int(v), nil
}

func SiluMulF32(gate, up, out DeviceBuffer, n int, stream Stream) error {
	if gate.ptr == nil || up.ptr == nil || out.ptr == nil {
		return fmt.Errorf("native.SiluMulF32: buffer is nil")
	}
	nC, err := checkedPositiveCInt("n", n)
	if err != nil {
		return fmt.Errorf("native.SiluMulF32: %w", err)
	}
	if err := cudaErr(C.mantleCudaSiluMulF32Wrapper(
		(*C.float)(gate.ptr),
		(*C.float)(up.ptr),
		(*C.float)(out.ptr),
		nC,
		stream.ptr,
	)); err != nil {
		return fmt.Errorf("native.SiluMulF32(n=%d): %w", n, err)
	}
	return nil
}

func AddVectorsF32(dst, src DeviceBuffer, n int, stream Stream) error {
	if dst.ptr == nil || src.ptr == nil {
		return fmt.Errorf("native.AddVectorsF32: buffer is nil")
	}
	nC, err := checkedPositiveCInt("n", n)
	if err != nil {
		return fmt.Errorf("native.AddVectorsF32: %w", err)
	}
	if err := cudaErr(C.mantleCudaAddVectorsF32Wrapper(
		(*C.float)(dst.ptr),
		(*C.float)(src.ptr),
		nC,
		stream.ptr,
	)); err != nil {
		return fmt.Errorf("native.AddVectorsF32(n=%d): %w", n, err)
	}
	return nil
}

func ShortConvDepthwise(proj, convW, state, out DeviceBuffer, embd, klen int, stream Stream) error {
	if proj.ptr == nil || convW.ptr == nil || state.ptr == nil || out.ptr == nil {
		return fmt.Errorf("native.ShortConvDepthwise: buffer is nil")
	}
	embdC, err := checkedPositiveCInt("embd", embd)
	if err != nil {
		return fmt.Errorf("native.ShortConvDepthwise: %w", err)
	}
	klenC, err := checkedPositiveCInt("klen", klen)
	if err != nil {
		return fmt.Errorf("native.ShortConvDepthwise: %w", err)
	}
	if err := cudaErr(C.mantleCudaShortConvDepthwiseWrapper(
		(*C.float)(proj.ptr),
		(*C.float)(convW.ptr),
		(*C.float)(state.ptr),
		(*C.float)(out.ptr),
		embdC,
		klenC,
		stream.ptr,
	)); err != nil {
		return fmt.Errorf("native.ShortConvDepthwise(embd=%d, klen=%d): %w", embd, klen, err)
	}
	return nil
}

func ConvertF32ToF16(in, out DeviceBuffer, n int, stream Stream) error {
	if in.ptr == nil || out.ptr == nil {
		return fmt.Errorf("native.ConvertF32ToF16: buffer is nil")
	}
	nC, err := checkedPositiveCInt("n", n)
	if err != nil {
		return fmt.Errorf("native.ConvertF32ToF16: %w", err)
	}
	if err := cudaErr(C.mantleCudaConvertF32ToF16Wrapper(
		(*C.float)(in.ptr),
		(*C.ushort)(out.ptr),
		nC,
		stream.ptr,
	)); err != nil {
		return fmt.Errorf("native.ConvertF32ToF16(n=%d): %w", n, err)
	}
	return nil
}

func ConvertF32ToBF16(in, out DeviceBuffer, n int, stream Stream) error {
	if in.ptr == nil || out.ptr == nil {
		return fmt.Errorf("native.ConvertF32ToBF16: buffer is nil")
	}
	nC, err := checkedPositiveCInt("n", n)
	if err != nil {
		return fmt.Errorf("native.ConvertF32ToBF16: %w", err)
	}
	if err := cudaErr(C.mantleCudaConvertF32ToBF16Wrapper(
		(*C.float)(in.ptr),
		(*C.ushort)(out.ptr),
		nC,
		stream.ptr,
	)); err != nil {
		return fmt.Errorf("native.ConvertF32ToBF16(n=%d): %w", n, err)
	}
	return nil
}
