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
extern int mantleCudaGeluMulF32(
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
extern int mantleCudaMambaDepthwiseConv(
	const float* in,
	const float* conv_w,
	const float* bias,
	float* state,
	float* out,
	int channels,
	int klen,
	int has_bias,
	cudaStream_t stream);
extern int mantleCudaSiluF32InPlace(
	float* x,
	int n,
	cudaStream_t stream);
extern int mantleCudaMambaSSMScan(
	const float* x,
	const float* dt,
	const float* b,
	const float* c,
	const float* a_log,
	const float* d_vec,
	float* state,
	float* out,
	int head_count,
	int head_dim,
	int d_state,
	int group_size,
	cudaStream_t stream);
extern int mantleCudaRoundBF16InPlaceF32(
	float* data,
	int n,
	cudaStream_t stream);
extern int mantleCudaScaleRoundBF16InPlaceF32(
	float* data,
	float scale,
	int n,
	cudaStream_t stream);
extern int mantleCudaMambaDtSoftplusClampF32(
	float* dt,
	const float* bias,
	int n,
	float t_min,
	float t_max,
	float t_floor,
	cudaStream_t stream);
extern int mantleCudaRMSNormGatedF32(
	float* out,
	const float* y,
	const float* z,
	const float* weight,
	float eps,
	int n,
	int norm_before_gate,
	cudaStream_t stream);
extern int mantleCudaScaleF32InPlace(
	float* x,
	int n,
	float scale,
	cudaStream_t stream);



static int mantleCudaSiluMulF32Wrapper(
	const float* gate,
	const float* up,
	float* out,
	int n,
	cudaStream_t stream) {
	return mantleCudaSiluMulF32(gate, up, out, n, stream);
}

static int mantleCudaGeluMulF32Wrapper(
	const float* gate,
	const float* up,
	float* out,
	int n,
	cudaStream_t stream) {
	return mantleCudaGeluMulF32(gate, up, out, n, stream);
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

static int mantleCudaMambaDepthwiseConvWrapper(
	const float* in,
	const float* conv_w,
	const float* bias,
	float* state,
	float* out,
	int channels,
	int klen,
	int has_bias,
	cudaStream_t stream) {
	return mantleCudaMambaDepthwiseConv(in, conv_w, bias, state, out, channels, klen, has_bias, stream);
}

static int mantleCudaSiluF32InPlaceWrapper(
	float* x,
	int n,
	cudaStream_t stream) {
	return mantleCudaSiluF32InPlace(x, n, stream);
}

static int mantleCudaMambaSSMScanWrapper(
	const float* x,
	const float* dt,
	const float* b,
	const float* c,
	const float* a_log,
	const float* d_vec,
	float* state,
	float* out,
	int head_count,
	int head_dim,
	int d_state,
	int group_size,
	cudaStream_t stream) {
	return mantleCudaMambaSSMScan(x, dt, b, c, a_log, d_vec, state, out, head_count, head_dim, d_state, group_size, stream);
}

static int mantleCudaRoundBF16InPlaceF32Wrapper(
	float* data,
	int n,
	cudaStream_t stream) {
	return mantleCudaRoundBF16InPlaceF32(data, n, stream);
}

static int mantleCudaScaleRoundBF16InPlaceF32Wrapper(
	float* data,
	float scale,
	int n,
	cudaStream_t stream) {
	return mantleCudaScaleRoundBF16InPlaceF32(data, scale, n, stream);
}

static int mantleCudaMambaDtSoftplusClampF32Wrapper(
	float* dt,
	const float* bias,
	int n,
	float t_min,
	float t_max,
	float t_floor,
	cudaStream_t stream) {
	return mantleCudaMambaDtSoftplusClampF32(dt, bias, n, t_min, t_max, t_floor, stream);
}

static int mantleCudaScaleF32InPlaceWrapper(
	float* x,
	int n,
	float scale,
	cudaStream_t stream) {
	return mantleCudaScaleF32InPlace(x, n, scale, stream);
}

static int mantleCudaRMSNormGatedF32Wrapper(
	float* out,
	const float* y,
	const float* z,
	const float* weight,
	float eps,
	int n,
	int norm_before_gate,
	cudaStream_t stream) {
	return mantleCudaRMSNormGatedF32(out, y, z, weight, eps, n, norm_before_gate, stream);
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

func GeluMulF32(gate, up, out DeviceBuffer, n int, stream Stream) error {
	if gate.ptr == nil || up.ptr == nil || out.ptr == nil {
		return fmt.Errorf("native.GeluMulF32: buffer is nil")
	}
	nC, err := checkedPositiveCInt("n", n)
	if err != nil {
		return fmt.Errorf("native.GeluMulF32: %w", err)
	}
	if err := cudaErr(C.mantleCudaGeluMulF32Wrapper(
		(*C.float)(gate.ptr),
		(*C.float)(up.ptr),
		(*C.float)(out.ptr),
		nC,
		stream.ptr,
	)); err != nil {
		return fmt.Errorf("native.GeluMulF32(n=%d): %w", n, err)
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

// MambaDepthwiseConv applies a per-channel 1D convolution over a rolling
// history state plus the current input sample, matching the CPU reference
// in internal/backend/simd/mamba.go (mambaDepthwiseConv).
//
// Dimensions:
//   - in:     [channels]
//   - convW:  [channels * klen], row-major per channel
//   - bias:   [channels] or zero-valued DeviceBuffer (ptr == nil) when absent
//   - state:  [channels * (klen - 1)]
//   - out:    [channels]
//
// The state buffer is updated in place: shifted left by one time step along
// the time axis and the current input is appended at the last slot. When
// klen == 1 the state buffer is not accessed.
func MambaDepthwiseConv(in, convW, bias, state, out DeviceBuffer, channels, klen int, stream Stream) error {
	if in.ptr == nil || convW.ptr == nil || state.ptr == nil || out.ptr == nil {
		return fmt.Errorf("native.MambaDepthwiseConv: buffer is nil")
	}
	channelsC, err := checkedPositiveCInt("channels", channels)
	if err != nil {
		return fmt.Errorf("native.MambaDepthwiseConv: %w", err)
	}
	klenC, err := checkedPositiveCInt("klen", klen)
	if err != nil {
		return fmt.Errorf("native.MambaDepthwiseConv: %w", err)
	}
	var biasPtr *C.float
	var hasBias C.int
	if bias.ptr != nil {
		biasPtr = (*C.float)(bias.ptr)
		hasBias = 1
	}
	if err := cudaErr(C.mantleCudaMambaDepthwiseConvWrapper(
		(*C.float)(in.ptr),
		(*C.float)(convW.ptr),
		biasPtr,
		(*C.float)(state.ptr),
		(*C.float)(out.ptr),
		channelsC,
		klenC,
		hasBias,
		stream.ptr,
	)); err != nil {
		return fmt.Errorf("native.MambaDepthwiseConv(channels=%d, klen=%d): %w", channels, klen, err)
	}
	return nil
}

func SiluF32InPlace(x DeviceBuffer, n int, stream Stream) error {
	if x.ptr == nil {
		return fmt.Errorf("native.SiluF32InPlace: buffer is nil")
	}
	nC, err := checkedPositiveCInt("n", n)
	if err != nil {
		return fmt.Errorf("native.SiluF32InPlace: %w", err)
	}
	if err := cudaErr(C.mantleCudaSiluF32InPlaceWrapper(
		(*C.float)(x.ptr),
		nC,
		stream.ptr,
	)); err != nil {
		return fmt.Errorf("native.SiluF32InPlace(n=%d): %w", n, err)
	}
	return nil
}

// MambaSSMScan performs the selective-SSM recurrence update for a single
// time step across all heads and positions, mirroring mambaScan in
// internal/backend/simd/mamba.go.
//
// Dimensions:
//   - x:     [headCount * headDim]
//   - dt:    [headCount]
//   - b:     [groups * dState] where groups = headCount / groupSize
//   - c:     [groups * dState]
//   - aLog:  [headCount]
//   - dVec:  [headCount]
//   - state: [headCount * headDim * dState] (updated in place)
//   - out:   [headCount * headDim]
//
// groupSize must be > 0 and must divide headCount (headCount % groupSize == 0).
func MambaSSMScan(x, dt, b, c, aLog, dVec, state, out DeviceBuffer, headCount, headDim, dState, groupSize int, stream Stream) error {
	if x.ptr == nil || dt.ptr == nil || b.ptr == nil || c.ptr == nil ||
		aLog.ptr == nil || dVec.ptr == nil || state.ptr == nil || out.ptr == nil {
		return fmt.Errorf("native.MambaSSMScan: buffer is nil")
	}
	headCountC, err := checkedPositiveCInt("headCount", headCount)
	if err != nil {
		return fmt.Errorf("native.MambaSSMScan: %w", err)
	}
	headDimC, err := checkedPositiveCInt("headDim", headDim)
	if err != nil {
		return fmt.Errorf("native.MambaSSMScan: %w", err)
	}
	dStateC, err := checkedPositiveCInt("dState", dState)
	if err != nil {
		return fmt.Errorf("native.MambaSSMScan: %w", err)
	}
	groupSizeC, err := checkedPositiveCInt("groupSize", groupSize)
	if err != nil {
		return fmt.Errorf("native.MambaSSMScan: %w", err)
	}
	if headCount%groupSize != 0 {
		return fmt.Errorf("native.MambaSSMScan: headCount (%d) must be divisible by groupSize (%d)", headCount, groupSize)
	}
	if dState > 1024 {
		return fmt.Errorf("native.MambaSSMScan: dState (%d) exceeds block-dim limit 1024", dState)
	}
	if err := cudaErr(C.mantleCudaMambaSSMScanWrapper(
		(*C.float)(x.ptr),
		(*C.float)(dt.ptr),
		(*C.float)(b.ptr),
		(*C.float)(c.ptr),
		(*C.float)(aLog.ptr),
		(*C.float)(dVec.ptr),
		(*C.float)(state.ptr),
		(*C.float)(out.ptr),
		headCountC,
		headDimC,
		dStateC,
		groupSizeC,
		stream.ptr,
	)); err != nil {
		return fmt.Errorf("native.MambaSSMScan(headCount=%d, headDim=%d, dState=%d, groupSize=%d): %w", headCount, headDim, dState, groupSize, err)
	}
	return nil
}

// MambaDtSoftplusClampF32 applies dt[i] = clamp(softplus(dt[i]+bias[i]), tMin, tMax);
// if tFloor > 0 and result < tFloor, result = tFloor. Mirrors the dt preprocessing
// in internal/backend/simd/mamba.go before the selective-SSM scan.
func MambaDtSoftplusClampF32(dt, bias DeviceBuffer, n int, tMin, tMax, tFloor float32, stream Stream) error {
	if dt.ptr == nil || bias.ptr == nil {
		return fmt.Errorf("native.MambaDtSoftplusClampF32: buffer is nil")
	}
	nC, err := checkedPositiveCInt("n", n)
	if err != nil {
		return fmt.Errorf("native.MambaDtSoftplusClampF32: %w", err)
	}
	if err := cudaErr(C.mantleCudaMambaDtSoftplusClampF32Wrapper(
		(*C.float)(dt.ptr),
		(*C.float)(bias.ptr),
		nC,
		C.float(tMin),
		C.float(tMax),
		C.float(tFloor),
		stream.ptr,
	)); err != nil {
		return fmt.Errorf("native.MambaDtSoftplusClampF32(n=%d): %w", n, err)
	}
	return nil
}

// RMSNormGatedF32 fuses RMSNorm with a SiLU gate. When normBeforeGate is
// true, out = rms_norm(y, weight, eps) * silu(z). Otherwise, out =
// rms_norm(y * silu(z), weight, eps). Generic kernel reused by Mamba and
// DeltaNet; mirrors RMSNormGated in internal/backend/simd/mamba.go.
func RMSNormGatedF32(out, y, z, weight DeviceBuffer, n int, eps float32, normBeforeGate bool, stream Stream) error {
	if out.ptr == nil || y.ptr == nil || z.ptr == nil || weight.ptr == nil {
		return fmt.Errorf("native.RMSNormGatedF32: buffer is nil")
	}
	nC, err := checkedPositiveCInt("n", n)
	if err != nil {
		return fmt.Errorf("native.RMSNormGatedF32: %w", err)
	}
	var flag C.int
	if normBeforeGate {
		flag = 1
	}
	if err := cudaErr(C.mantleCudaRMSNormGatedF32Wrapper(
		(*C.float)(out.ptr),
		(*C.float)(y.ptr),
		(*C.float)(z.ptr),
		(*C.float)(weight.ptr),
		C.float(eps),
		nC,
		flag,
		stream.ptr,
	)); err != nil {
		return fmt.Errorf("native.RMSNormGatedF32(n=%d, normBeforeGate=%v): %w", n, normBeforeGate, err)
	}
	return nil
}

func ScaleF32InPlace(x DeviceBuffer, n int, scale float32, stream Stream) error {
	if x.ptr == nil {
		return fmt.Errorf("native.ScaleF32InPlace: buffer is nil")
	}
	nC, err := checkedPositiveCInt("n", n)
	if err != nil {
		return fmt.Errorf("native.ScaleF32InPlace: %w", err)
	}
	if err := cudaErr(C.mantleCudaScaleF32InPlaceWrapper(
		(*C.float)(x.ptr),
		nC,
		C.float(scale),
		stream.ptr,
	)); err != nil {
		return fmt.Errorf("native.ScaleF32InPlace(n=%d, scale=%g): %w", n, scale, err)
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

func RoundBF16InPlaceF32(buf DeviceBuffer, n int, stream Stream) error {
	if buf.ptr == nil {
		return fmt.Errorf("native.RoundBF16InPlaceF32: buffer is nil")
	}
	nC, err := checkedPositiveCInt("n", n)
	if err != nil {
		return fmt.Errorf("native.RoundBF16InPlaceF32: %w", err)
	}
	if err := cudaErr(C.mantleCudaRoundBF16InPlaceF32Wrapper(
		(*C.float)(buf.ptr),
		nC,
		stream.ptr,
	)); err != nil {
		return fmt.Errorf("native.RoundBF16InPlaceF32(n=%d): %w", n, err)
	}
	return nil
}

func ScaleRoundBF16InPlaceF32(buf DeviceBuffer, scale float32, n int, stream Stream) error {
	if buf.ptr == nil {
		return fmt.Errorf("native.ScaleRoundBF16InPlaceF32: buffer is nil")
	}
	nC, err := checkedPositiveCInt("n", n)
	if err != nil {
		return fmt.Errorf("native.ScaleRoundBF16InPlaceF32: %w", err)
	}
	if err := cudaErr(C.mantleCudaScaleRoundBF16InPlaceF32Wrapper(
		(*C.float)(buf.ptr),
		C.float(scale),
		nC,
		stream.ptr,
	)); err != nil {
		return fmt.Errorf("native.ScaleRoundBF16InPlaceF32(n=%d): %w", n, err)
	}
	return nil
}
