//go:build cuda

package native

/*
#include <stdint.h>
typedef void* cudaStream_t;
typedef int cudaError_t;

extern const char* cudaGetErrorString(cudaError_t err);

extern int mantleCudaQuantMatVecInt8BlocksF32(
	const signed char* q,
	const float* scales,
	const float* x,
	float* y,
	int rows,
	int blocksPerRow,
	int cols,
	cudaStream_t stream);
extern int mantleCudaQuantMatVecInt8BlocksF32Legacy(
	const signed char* q,
	const float* scales,
	const float* x,
	float* y,
	int rows,
	int blocksPerRow,
	int cols,
	cudaStream_t stream);
extern int mantleCudaQuantMatVecInt8BlocksF32Multi(
	const signed char* q0,
	const float* s0,
	float* y0,
	int rows0,
	const signed char* q1,
	const float* s1,
	float* y1,
	int rows1,
	const signed char* q2,
	const float* s2,
	float* y2,
	int rows2,
	const signed char* q3,
	const float* s3,
	float* y3,
	int rows3,
	int nSegs,
	const float* xUp,
	const float* xGate,
	int act,
	int cols,
	int blocksPerRow,
	cudaStream_t stream);
extern int mantleCudaQuantMatVecQ4F32(
	const unsigned char* qData,
	const unsigned short* scalesF16,
	const float* x,
	float* y,
	int rows,
	int blocksPerRow,
	int cols,
	cudaStream_t stream);
extern int mantleCudaQuantMatVecCachedBatchDp4a(
	const signed char* q,
	const float* scales,
	const float* x,
	signed char* xq,
	float* xsc,
	float* y,
	int rows,
	int n,
	int cols,
	int blocksPerRow,
	cudaStream_t stream
);

extern int mantleCudaQuantMatVecCachedBatch(
	const signed char* q,
	const float* scales,
	const float* x,
	float* y,
	int rows,
	int n,
	int cols,
	int blocksPerRow,
	cudaStream_t stream);
extern int mantleCudaQuantMatVecK4F32(
	const unsigned char* qData,
	const unsigned short* superScalesF16,
	const unsigned char* subScales,
	const float* x,
	float* y,
	int rows,
	int blocksPerRow,
	int cols,
	cudaStream_t stream);
extern int mantleCudaDequantizeQ4ToF16(
	const unsigned char* qData,
	const unsigned short* scalesF16,
	unsigned short* outF16,
	int rows,
	int blocksPerRow,
	int cols,
	cudaStream_t stream);
extern int mantleCudaDequantizeK4ToF16(
	const unsigned char* qData,
	const unsigned short* superScalesF16,
	const unsigned char* subScales,
	unsigned short* outF16,
	int rows,
	int blocksPerRow,
	int cols,
	cudaStream_t stream);
extern int launchFusedRMSNormMatVecBF16(
	float* out,
	const unsigned short* W,
	const float* x,
	const float* normWeight,
	float eps,
	int rows,
	int cols,
	cudaStream_t stream);
extern int launchFusedRMSNormMatVecF32(
	float* out,
	const float* W,
	const float* x,
	const float* normWeight,
	float eps,
	int rows,
	int cols,
	cudaStream_t stream);

static int mantleCudaQuantMatVecInt8BlocksF32Wrapper(
	const signed char* q,
	const float* scales,
	const float* x,
	float* y,
	int rows,
	int blocksPerRow,
	int cols,
	cudaStream_t stream) {
	return mantleCudaQuantMatVecInt8BlocksF32(q, scales, x, y, rows, blocksPerRow, cols, stream);
}

static int mantleCudaQuantMatVecInt8BlocksF32LegacyWrapper(
	const signed char* q,
	const float* scales,
	const float* x,
	float* y,
	int rows,
	int blocksPerRow,
	int cols,
	cudaStream_t stream) {
	return mantleCudaQuantMatVecInt8BlocksF32Legacy(q, scales, x, y, rows, blocksPerRow, cols, stream);
}

static int mantleCudaQuantMatVecInt8BlocksF32MultiWrapper(
	const signed char* q0,
	const float* s0,
	float* y0,
	int rows0,
	const signed char* q1,
	const float* s1,
	float* y1,
	int rows1,
	const signed char* q2,
	const float* s2,
	float* y2,
	int rows2,
	const signed char* q3,
	const float* s3,
	float* y3,
	int rows3,
	int nSegs,
	const float* xUp,
	const float* xGate,
	int act,
	int cols,
	int blocksPerRow,
	cudaStream_t stream) {
	return mantleCudaQuantMatVecInt8BlocksF32Multi(
		q0, s0, y0, rows0,
		q1, s1, y1, rows1,
		q2, s2, y2, rows2,
		q3, s3, y3, rows3,
		nSegs, xUp, xGate, act, cols, blocksPerRow, stream);
}

static int mantleCudaQuantMatVecCachedBatchDp4aWrapper(
	const signed char* q,
	const float* scales,
	const float* x,
	signed char* xq,
	float* xsc,
	float* y,
	int rows,
	int n,
	int cols,
	int blocksPerRow,
	cudaStream_t stream
) {
	return mantleCudaQuantMatVecCachedBatchDp4a(q, scales, x, xq, xsc, y, rows, n, cols, blocksPerRow, stream);
}

static int mantleCudaQuantMatVecCachedBatchWrapper(
	const signed char* q,
	const float* scales,
	const float* x,
	float* y,
	int rows,
	int n,
	int cols,
	int blocksPerRow,
	cudaStream_t stream) {
	return mantleCudaQuantMatVecCachedBatch(q, scales, x, y, rows, n, cols, blocksPerRow, stream);
}

static int mantleCudaQuantMatVecQ4F32Wrapper(
	const unsigned char* qData,
	const unsigned short* scalesF16,
	const float* x,
	float* y,
	int rows,
	int blocksPerRow,
	int cols,
	cudaStream_t stream) {
	return mantleCudaQuantMatVecQ4F32(qData, scalesF16, x, y, rows, blocksPerRow, cols, stream);
}

static int mantleCudaQuantMatVecK4F32Wrapper(
	const unsigned char* qData,
	const unsigned short* superScalesF16,
	const unsigned char* subScales,
	const float* x,
	float* y,
	int rows,
	int blocksPerRow,
	int cols,
	cudaStream_t stream) {
	return mantleCudaQuantMatVecK4F32(qData, superScalesF16, subScales, x, y, rows, blocksPerRow, cols, stream);
}

static int mantleCudaDequantizeQ4ToF16Wrapper(
	const unsigned char* qData,
	const unsigned short* scalesF16,
	unsigned short* outF16,
	int rows,
	int blocksPerRow,
	int cols,
	cudaStream_t stream) {
	return mantleCudaDequantizeQ4ToF16(qData, scalesF16, outF16, rows, blocksPerRow, cols, stream);
}

static int mantleCudaDequantizeK4ToF16Wrapper(
	const unsigned char* qData,
	const unsigned short* superScalesF16,
	const unsigned char* subScales,
	unsigned short* outF16,
	int rows,
	int blocksPerRow,
	int cols,
	cudaStream_t stream) {
	return mantleCudaDequantizeK4ToF16(qData, superScalesF16, subScales, outF16, rows, blocksPerRow, cols, stream);
}

static int mantleFusedRMSNormMatVecBF16(
	float* out,
	const unsigned short* W,
	const float* x,
	const float* normWeight,
	float eps,
	int rows,
	int cols,
	cudaStream_t stream) {
	return launchFusedRMSNormMatVecBF16(out, W, x, normWeight, eps, rows, cols, stream);
}

static int mantleFusedRMSNormMatVecF32(
	float* out,
	const float* W,
	const float* x,
	const float* normWeight,
	float eps,
	int rows,
	int cols,
	cudaStream_t stream) {
	return launchFusedRMSNormMatVecF32(out, W, x, normWeight, eps, rows, cols, stream);
}

*/
import "C"

import (
	"fmt"
	"unsafe"
)

func QuantMatVecInt8BlocksF32(q, scales, x, y DeviceBuffer, rows, blocksPerRow, cols int, stream Stream) error {
	if q.ptr == nil || scales.ptr == nil || x.ptr == nil || y.ptr == nil {
		return fmt.Errorf("quant matvec buffer is nil")
	}
	if rows <= 0 || blocksPerRow <= 0 || cols <= 0 {
		return fmt.Errorf("quant matvec rows/blocks/cols must be > 0")
	}
	return cudaErr(C.mantleCudaQuantMatVecInt8BlocksF32Wrapper(
		(*C.schar)(q.ptr),
		(*C.float)(scales.ptr),
		(*C.float)(x.ptr),
		(*C.float)(y.ptr),
		C.int(rows),
		C.int(blocksPerRow),
		C.int(cols),
		stream.ptr,
	))
}

// QuantMatVecInt8BlocksF32Legacy runs the original per-lane int8-blocks
// matvec kernels. It is bit-identical to QuantMatVecInt8BlocksF32 but keeps
// the legacy code path reachable for parity tests and explicit fallbacks.
func QuantMatVecInt8BlocksF32Legacy(q, scales, x, y DeviceBuffer, rows, blocksPerRow, cols int, stream Stream) error {
	if q.ptr == nil || scales.ptr == nil || x.ptr == nil || y.ptr == nil {
		return fmt.Errorf("quant matvec legacy buffer is nil")
	}
	if rows <= 0 || blocksPerRow <= 0 || cols <= 0 {
		return fmt.Errorf("quant matvec legacy rows/blocks/cols must be > 0")
	}
	return cudaErr(C.mantleCudaQuantMatVecInt8BlocksF32LegacyWrapper(
		(*C.schar)(q.ptr),
		(*C.float)(scales.ptr),
		(*C.float)(x.ptr),
		(*C.float)(y.ptr),
		C.int(rows),
		C.int(blocksPerRow),
		C.int(cols),
		stream.ptr,
	))
}

// Activation modes for QuantMatVecInt8BlocksF32Multi (must match the .cu).
const (
	QuantActNone = 0
	QuantActSilu = 1
	QuantActGelu = 2
)

// QuantMatVecSegment describes one quantized weight matrix (cached-int8
// layout) and its output buffer within a fused multi-segment matvec.
type QuantMatVecSegment struct {
	Q      DeviceBuffer
	Scales DeviceBuffer
	Y      DeviceBuffer
	Rows   int
}

// QuantMatVecInt8BlocksF32Multi computes, in a single kernel launch, one
// matvec per segment over the shared input xUp: y[seg] = W[seg]^T * x. When
// act != QuantActNone, the effective input is act(xGate[c]) * xUp[c], using
// the exact expressions of SiluMulF32/GeluMulF32, so the launch is
// bit-identical to running the activation kernel and then a matvec per
// segment. All segments must share cols and blocksPerRow. Returns an error
// when len(segs) is out of range or the shared-memory budget does not fit
// (caller should fall back to per-segment QuantMatVecInt8BlocksF32 calls).
func QuantMatVecInt8BlocksF32Multi(segs []QuantMatVecSegment, xUp, xGate DeviceBuffer, act int, cols, blocksPerRow int, stream Stream) error {
	if len(segs) < 1 || len(segs) > 4 {
		return fmt.Errorf("quant matvec multi: need 1..4 segments, got %d", len(segs))
	}
	if xUp.ptr == nil {
		return fmt.Errorf("quant matvec multi: xUp is nil")
	}
	if act != QuantActNone && xGate.ptr == nil {
		return fmt.Errorf("quant matvec multi: act=%d requires xGate", act)
	}
	if cols <= 0 || blocksPerRow <= 0 {
		return fmt.Errorf("quant matvec multi: cols/blocksPerRow must be > 0")
	}
	var q, s, y [4]unsafe.Pointer
	var rows [4]C.int
	for i, seg := range segs {
		if seg.Q.ptr == nil || seg.Scales.ptr == nil || seg.Y.ptr == nil {
			return fmt.Errorf("quant matvec multi: segment %d has a nil buffer", i)
		}
		if seg.Rows <= 0 {
			return fmt.Errorf("quant matvec multi: segment %d rows must be > 0", i)
		}
		q[i] = seg.Q.ptr
		s[i] = seg.Scales.ptr
		y[i] = seg.Y.ptr
		rows[i] = C.int(seg.Rows)
	}
	return cudaErr(C.mantleCudaQuantMatVecInt8BlocksF32MultiWrapper(
		(*C.schar)(q[0]), (*C.float)(s[0]), (*C.float)(y[0]), rows[0],
		(*C.schar)(q[1]), (*C.float)(s[1]), (*C.float)(y[1]), rows[1],
		(*C.schar)(q[2]), (*C.float)(s[2]), (*C.float)(y[2]), rows[2],
		(*C.schar)(q[3]), (*C.float)(s[3]), (*C.float)(y[3]), rows[3],
		C.int(len(segs)),
		(*C.float)(xUp.ptr),
		(*C.float)(xGate.ptr),
		C.int(act),
		C.int(cols),
		C.int(blocksPerRow),
		stream.ptr,
	))
}

// QuantMatVecCachedBatchDp4a is QuantMatVecCachedBatch with an int8 dp4a inner
// product: x is quantized per 32-element block into xq/xsc (device scratch)
// and the dot accumulates in int32 via __dp4a, scaled once per block in f32.
func QuantMatVecCachedBatchDp4a(q, scales, x, xq, xsc, y DeviceBuffer, rows, n, cols, blocksPerRow int, stream Stream) error {
	if q.ptr == nil || scales.ptr == nil || x.ptr == nil || xq.ptr == nil || xsc.ptr == nil || y.ptr == nil {
		return fmt.Errorf("quant batch dp4a buffer is nil")
	}
	if rows <= 0 || n <= 0 || cols <= 0 || blocksPerRow <= 0 {
		return fmt.Errorf("quant batch dp4a dims must be > 0")
	}
	return cudaErr(C.mantleCudaQuantMatVecCachedBatchDp4aWrapper(
		(*C.schar)(q.ptr),
		(*C.float)(scales.ptr),
		(*C.float)(x.ptr),
		(*C.schar)(xq.ptr),
		(*C.float)(xsc.ptr),
		(*C.float)(y.ptr),
		C.int(rows),
		C.int(n),
		C.int(cols),
		C.int(blocksPerRow),
		stream.ptr,
	))
}

// QuantMatVecCachedBatch computes y[n, rows] = x[n, cols] * W^T for the
// cached-int8 quant layout (int8 block data + per-block f32 scales).
func QuantMatVecCachedBatch(q, scales, x, y DeviceBuffer, rows, n, cols, blocksPerRow int, stream Stream) error {
	if q.ptr == nil || scales.ptr == nil || x.ptr == nil || y.ptr == nil {
		return fmt.Errorf("quant batch matvec buffer is nil")
	}
	if rows <= 0 || n <= 0 || cols <= 0 || blocksPerRow <= 0 {
		return fmt.Errorf("quant batch matvec dims must be > 0")
	}
	return cudaErr(C.mantleCudaQuantMatVecCachedBatchWrapper(
		(*C.schar)(q.ptr),
		(*C.float)(scales.ptr),
		(*C.float)(x.ptr),
		(*C.float)(y.ptr),
		C.int(rows),
		C.int(n),
		C.int(cols),
		C.int(blocksPerRow),
		stream.ptr,
	))
}

func QuantMatVecQ4F32(qData, scalesF16, x, y DeviceBuffer, rows, blocksPerRow, cols int, stream Stream) error {
	if qData.ptr == nil || scalesF16.ptr == nil || x.ptr == nil || y.ptr == nil {
		return fmt.Errorf("q4 quant matvec buffer is nil")
	}
	if rows <= 0 || blocksPerRow <= 0 || cols <= 0 {
		return fmt.Errorf("q4 quant matvec rows/blocks/cols must be > 0")
	}
	return cudaErr(C.mantleCudaQuantMatVecQ4F32Wrapper(
		(*C.uchar)(qData.ptr),
		(*C.ushort)(scalesF16.ptr),
		(*C.float)(x.ptr),
		(*C.float)(y.ptr),
		C.int(rows),
		C.int(blocksPerRow),
		C.int(cols),
		stream.ptr,
	))
}

func QuantMatVecK4F32(qData, superScalesF16, subScales, x, y DeviceBuffer, rows, blocksPerRow, cols int, stream Stream) error {
	if qData.ptr == nil || superScalesF16.ptr == nil || subScales.ptr == nil || x.ptr == nil || y.ptr == nil {
		return fmt.Errorf("k4 quant matvec buffer is nil")
	}
	if rows <= 0 || blocksPerRow <= 0 || cols <= 0 {
		return fmt.Errorf("k4 quant matvec rows/blocks/cols must be > 0")
	}
	return cudaErr(C.mantleCudaQuantMatVecK4F32Wrapper(
		(*C.uchar)(qData.ptr),
		(*C.ushort)(superScalesF16.ptr),
		(*C.uchar)(subScales.ptr),
		(*C.float)(x.ptr),
		(*C.float)(y.ptr),
		C.int(rows),
		C.int(blocksPerRow),
		C.int(cols),
		stream.ptr,
	))
}

func DequantizeQ4ToF16(qData, scalesF16, outF16 DeviceBuffer, rows, blocksPerRow, cols int, stream Stream) error {
	if qData.ptr == nil || scalesF16.ptr == nil || outF16.ptr == nil {
		return fmt.Errorf("q4 dequant buffer is nil")
	}
	if rows <= 0 || blocksPerRow <= 0 || cols <= 0 {
		return fmt.Errorf("q4 dequant rows/blocks/cols must be > 0")
	}
	return cudaErr(C.mantleCudaDequantizeQ4ToF16Wrapper(
		(*C.uchar)(qData.ptr),
		(*C.ushort)(scalesF16.ptr),
		(*C.ushort)(outF16.ptr),
		C.int(rows),
		C.int(blocksPerRow),
		C.int(cols),
		stream.ptr,
	))
}

func DequantizeK4ToF16(qData, superScalesF16, subScales, outF16 DeviceBuffer, rows, blocksPerRow, cols int, stream Stream) error {
	if qData.ptr == nil || superScalesF16.ptr == nil || subScales.ptr == nil || outF16.ptr == nil {
		return fmt.Errorf("k4 dequant buffer is nil")
	}
	if rows <= 0 || blocksPerRow <= 0 || cols <= 0 {
		return fmt.Errorf("k4 dequant rows/blocks/cols must be > 0")
	}
	return cudaErr(C.mantleCudaDequantizeK4ToF16Wrapper(
		(*C.uchar)(qData.ptr),
		(*C.ushort)(superScalesF16.ptr),
		(*C.uchar)(subScales.ptr),
		(*C.ushort)(outF16.ptr),
		C.int(rows),
		C.int(blocksPerRow),
		C.int(cols),
		stream.ptr,
	))
}

// FusedRMSNormMatVecBF16 performs fused RMSNorm + MatVec with BF16 weights
func FusedRMSNormMatVecBF16(out, W, x, normWeight DeviceBuffer, eps float32, rows, cols int, stream Stream) error {
	if out.ptr == nil || W.ptr == nil || x.ptr == nil || normWeight.ptr == nil {
		return fmt.Errorf("fused rmsnorm matvec buffer is nil")
	}
	if rows <= 0 || cols <= 0 {
		return fmt.Errorf("fused rmsnorm matvec dimensions must be > 0")
	}
	return cudaErr(C.mantleFusedRMSNormMatVecBF16(
		(*C.float)(out.ptr),
		(*C.ushort)(W.ptr),
		(*C.float)(x.ptr),
		(*C.float)(normWeight.ptr),
		C.float(eps),
		C.int(rows),
		C.int(cols),
		stream.ptr,
	))
}

// FusedRMSNormMatVecF32 performs fused RMSNorm + MatVec with F32 weights
func FusedRMSNormMatVecF32(out, W, x, normWeight DeviceBuffer, eps float32, rows, cols int, stream Stream) error {
	if out.ptr == nil || W.ptr == nil || x.ptr == nil || normWeight.ptr == nil {
		return fmt.Errorf("fused rmsnorm matvec buffer is nil")
	}
	if rows <= 0 || cols <= 0 {
		return fmt.Errorf("fused rmsnorm matvec dimensions must be > 0")
	}
	return cudaErr(C.mantleFusedRMSNormMatVecF32(
		(*C.float)(out.ptr),
		(*C.float)(W.ptr),
		(*C.float)(x.ptr),
		(*C.float)(normWeight.ptr),
		C.float(eps),
		C.int(rows),
		C.int(cols),
		stream.ptr,
	))
}
