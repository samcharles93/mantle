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
extern int mantleCudaQuantMatVecQ4F32(
	const unsigned char* qData,
	const unsigned short* scalesF16,
	const float* x,
	float* y,
	int rows,
	int blocksPerRow,
	int cols,
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
