//go:build cuda

package native

/*
#include <stdint.h>
typedef void* cudaStream_t;
typedef int cudaError_t;

extern const char* cudaGetErrorString(cudaError_t err);

extern int mantleCudaVisionRoPEF32(
	float* x,
	const float* cosT,
	const float* sinT,
	int n,
	int heads,
	int headDim,
	cudaStream_t stream);
extern int mantleCudaVisionAttnSegmentsF32(
	const float* q,
	const float* k,
	const float* v,
	float* out,
	const int* segStart,
	const int* segLen,
	int nSeg,
	int n,
	int heads,
	int kvHeads,
	int headDim,
	float scale,
	cudaStream_t stream);
extern int mantleCudaVisionConv2dGatedF32(
	const float* in,
	const float* convW,
	float* out,
	const int* frameStart,
	const int* frameH,
	const int* frameW,
	int nFrames,
	int n,
	int maxFrameTokens,
	int channels,
	int kSize,
	int merge,
	cudaStream_t stream);
extern int mantleCudaLayerNormRowsF32(
	float* out,
	const float* x,
	const float* weight,
	const float* bias,
	float eps,
	int rows,
	int cols,
	cudaStream_t stream);
extern int mantleCudaGeluExactBiasF32(
	float* x,
	const float* bias,
	int rows,
	int cols,
	cudaStream_t stream);
extern int mantleCudaAddBiasRowsF32(
	float* out,
	const float* bias,
	int rows,
	int cols,
	cudaStream_t stream);

static int mantleCudaVisionRoPEF32Wrapper(
	float* x,
	const float* cosT,
	const float* sinT,
	int n,
	int heads,
	int headDim,
	cudaStream_t stream) {
	return mantleCudaVisionRoPEF32(x, cosT, sinT, n, heads, headDim, stream);
}

static int mantleCudaVisionAttnSegmentsF32Wrapper(
	const float* q,
	const float* k,
	const float* v,
	float* out,
	const int* segStart,
	const int* segLen,
	int nSeg,
	int n,
	int heads,
	int kvHeads,
	int headDim,
	float scale,
	cudaStream_t stream) {
	return mantleCudaVisionAttnSegmentsF32(q, k, v, out, segStart, segLen, nSeg, n, heads, kvHeads, headDim, scale, stream);
}

static int mantleCudaVisionConv2dGatedF32Wrapper(
	const float* in,
	const float* convW,
	float* out,
	const int* frameStart,
	const int* frameH,
	const int* frameW,
	int nFrames,
	int n,
	int maxFrameTokens,
	int channels,
	int kSize,
	int merge,
	cudaStream_t stream) {
	return mantleCudaVisionConv2dGatedF32(in, convW, out, frameStart, frameH, frameW, nFrames, n, maxFrameTokens, channels, kSize, merge, stream);
}

static int mantleCudaLayerNormRowsF32Wrapper(
	float* out,
	const float* x,
	const float* weight,
	const float* bias,
	float eps,
	int rows,
	int cols,
	cudaStream_t stream) {
	return mantleCudaLayerNormRowsF32(out, x, weight, bias, eps, rows, cols, stream);
}

static int mantleCudaGeluExactBiasF32Wrapper(
	float* x,
	const float* bias,
	int rows,
	int cols,
	cudaStream_t stream) {
	return mantleCudaGeluExactBiasF32(x, bias, rows, cols, stream);
}

static int mantleCudaAddBiasRowsF32Wrapper(
	float* out,
	const float* bias,
	int rows,
	int cols,
	cudaStream_t stream) {
	return mantleCudaAddBiasRowsF32(out, bias, rows, cols, stream);
}
*/
import "C"

import (
	"fmt"
	"math"
)

// VisionRoPEF32 applies the axial 2D rotary embedding in place over x laid out
// as [n][heads][headDim], using the per-token cos/sin tables cosT/sinT of
// [n][headDim] (both halves duplicated, as built by the host). For each pair
// (d, d+half): x[d] = x1*cos[d] - x2*sin[d], x[d+half] = x2*cos[d+half] +
// x1*sin[d+half]. Mirrors applyAxialRoPE in internal/backend/simd.
func VisionRoPEF32(x, cosT, sinT DeviceBuffer, n, heads, headDim int, stream Stream) error {
	if x.ptr == nil || cosT.ptr == nil || sinT.ptr == nil {
		return fmt.Errorf("native.VisionRoPEF32: buffer is nil")
	}
	if headDim <= 0 || headDim%2 != 0 {
		return fmt.Errorf("native.VisionRoPEF32: headDim must be positive and even, got %d", headDim)
	}
	nC, err := checkedPositiveCInt("n", n)
	if err != nil {
		return fmt.Errorf("native.VisionRoPEF32: %w", err)
	}
	headsC, err := checkedPositiveCInt("heads", heads)
	if err != nil {
		return fmt.Errorf("native.VisionRoPEF32: %w", err)
	}
	hdC, err := checkedPositiveCInt("headDim", headDim)
	if err != nil {
		return fmt.Errorf("native.VisionRoPEF32: %w", err)
	}
	if err := cudaErr(C.mantleCudaVisionRoPEF32Wrapper(
		(*C.float)(x.ptr),
		(*C.float)(cosT.ptr),
		(*C.float)(sinT.ptr),
		nC,
		headsC,
		hdC,
		stream.ptr,
	)); err != nil {
		return fmt.Errorf("native.VisionRoPEF32(n=%d heads=%d headDim=%d): %w", n, heads, headDim, err)
	}
	return nil
}

// VisionAttnSegmentsF32 runs bidirectional (full, non-causal) scaled-dot-product
// attention with GQA over independent token segments — one attention segment
// per video/image frame. q is [n][heads][headDim], k/v are [n][kvHeads][headDim],
// out is [n][heads][headDim]; segStart/segLen (int32, length nSeg) delimit the
// token ranges. Attention is tiled (online softmax) so no O(L^2) score buffer is
// materialized. scale is applied to the scores before softmax.
func VisionAttnSegmentsF32(q, k, v, out, segStart, segLen DeviceBuffer, n, nSeg, heads, kvHeads, headDim int, scale float32, stream Stream) error {
	if q.ptr == nil || k.ptr == nil || v.ptr == nil || out.ptr == nil || segStart.ptr == nil || segLen.ptr == nil {
		return fmt.Errorf("native.VisionAttnSegmentsF32: buffer is nil")
	}
	if heads <= 0 || kvHeads <= 0 || heads%kvHeads != 0 {
		return fmt.Errorf("native.VisionAttnSegmentsF32: heads (%d) must be a positive multiple of kvHeads (%d)", heads, kvHeads)
	}
	if headDim <= 0 || headDim > 128 {
		return fmt.Errorf("native.VisionAttnSegmentsF32: headDim (%d) must be in [1, 128]", headDim)
	}
	nSegC, err := checkedPositiveCInt("nSeg", nSeg)
	if err != nil {
		return fmt.Errorf("native.VisionAttnSegmentsF32: %w", err)
	}
	nC, err := checkedPositiveCInt("n", n)
	if err != nil {
		return fmt.Errorf("native.VisionAttnSegmentsF32: %w", err)
	}
	headsC, err := checkedPositiveCInt("heads", heads)
	if err != nil {
		return fmt.Errorf("native.VisionAttnSegmentsF32: %w", err)
	}
	kvC, err := checkedPositiveCInt("kvHeads", kvHeads)
	if err != nil {
		return fmt.Errorf("native.VisionAttnSegmentsF32: %w", err)
	}
	hdC, err := checkedPositiveCInt("headDim", headDim)
	if err != nil {
		return fmt.Errorf("native.VisionAttnSegmentsF32: %w", err)
	}
	if err := cudaErr(C.mantleCudaVisionAttnSegmentsF32Wrapper(
		(*C.float)(q.ptr),
		(*C.float)(k.ptr),
		(*C.float)(v.ptr),
		(*C.float)(out.ptr),
		(*C.int)(segStart.ptr),
		(*C.int)(segLen.ptr),
		nSegC,
		nC,
		headsC,
		kvC,
		hdC,
		C.float(scale),
		stream.ptr,
	)); err != nil {
		return fmt.Errorf("native.VisionAttnSegmentsF32(n=%d nSeg=%d heads=%d kvHeads=%d headDim=%d): %w", n, nSeg, heads, kvHeads, headDim, err)
	}
	return nil
}

// VisionConv2dGatedF32 runs one gated depthwise 2D convolution layer over
// merge-block-major tokens. in is [n][3*channels] (B, C, x chunks), convW is
// [channels][kSize][kSize], out is [n][channels]. frameStart/frameH/frameW
// (int32, length nFrames) describe the raster grid each frame's tokens occupy;
// token p in frame f maps to raster position blockMajorToRaster(p-start, H, W,
// merge). The gated input is B[c]*x[c], convolved with zero padding, then
// multiplied by the C chunk. Mirrors convLayer in internal/backend/simd.
func VisionConv2dGatedF32(in, convW, out, frameStart, frameH, frameW DeviceBuffer, nFrames, n, maxFrameTokens, channels, kSize, merge int, stream Stream) error {
	if in.ptr == nil || convW.ptr == nil || out.ptr == nil || frameStart.ptr == nil || frameH.ptr == nil || frameW.ptr == nil {
		return fmt.Errorf("native.VisionConv2dGatedF32: buffer is nil")
	}
	if kSize <= 0 || kSize%2 == 0 {
		return fmt.Errorf("native.VisionConv2dGatedF32: kSize must be positive and odd, got %d", kSize)
	}
	if merge <= 0 {
		return fmt.Errorf("native.VisionConv2dGatedF32: merge must be positive, got %d", merge)
	}
	nFramesC, err := checkedPositiveCInt("nFrames", nFrames)
	if err != nil {
		return fmt.Errorf("native.VisionConv2dGatedF32: %w", err)
	}
	nC, err := checkedPositiveCInt("n", n)
	if err != nil {
		return fmt.Errorf("native.VisionConv2dGatedF32: %w", err)
	}
	mfC, err := checkedPositiveCInt("maxFrameTokens", maxFrameTokens)
	if err != nil {
		return fmt.Errorf("native.VisionConv2dGatedF32: %w", err)
	}
	chC, err := checkedPositiveCInt("channels", channels)
	if err != nil {
		return fmt.Errorf("native.VisionConv2dGatedF32: %w", err)
	}
	kC, err := checkedPositiveCInt("kSize", kSize)
	if err != nil {
		return fmt.Errorf("native.VisionConv2dGatedF32: %w", err)
	}
	mC, err := checkedPositiveCInt("merge", merge)
	if err != nil {
		return fmt.Errorf("native.VisionConv2dGatedF32: %w", err)
	}
	if err := cudaErr(C.mantleCudaVisionConv2dGatedF32Wrapper(
		(*C.float)(in.ptr),
		(*C.float)(convW.ptr),
		(*C.float)(out.ptr),
		(*C.int)(frameStart.ptr),
		(*C.int)(frameH.ptr),
		(*C.int)(frameW.ptr),
		nFramesC,
		nC,
		mfC,
		chC,
		kC,
		mC,
		stream.ptr,
	)); err != nil {
		return fmt.Errorf("native.VisionConv2dGatedF32(nFrames=%d n=%d channels=%d kSize=%d merge=%d): %w", nFrames, n, channels, kSize, merge, err)
	}
	return nil
}

// LayerNormRowsF32 normalizes each row of the [rows][cols] matrix x in place
// (out may alias x): out[i] = (x[i]-mean)*scale*weight[i] + bias[i], with
// scale = 1/sqrt(var+eps). bias may be a zero DeviceBuffer (nil pointer) when
// the model has no bias. Mirrors layerNorm in internal/backend/simd.
func LayerNormRowsF32(out, x, weight, bias DeviceBuffer, eps float32, rows, cols int, stream Stream) error {
	if out.ptr == nil || x.ptr == nil || weight.ptr == nil {
		return fmt.Errorf("native.LayerNormRowsF32: buffer is nil")
	}
	if rows <= 0 || rows > math.MaxInt32 || cols <= 0 || cols > math.MaxInt32 {
		return fmt.Errorf("native.LayerNormRowsF32: rows/cols must be in [1, %d], got %d/%d", math.MaxInt32, rows, cols)
	}
	var biasPtr *C.float
	if bias.ptr != nil {
		biasPtr = (*C.float)(bias.ptr)
	}
	if err := cudaErr(C.mantleCudaLayerNormRowsF32Wrapper(
		(*C.float)(out.ptr),
		(*C.float)(x.ptr),
		(*C.float)(weight.ptr),
		biasPtr,
		C.float(eps),
		C.int(rows),
		C.int(cols),
		stream.ptr,
	)); err != nil {
		return fmt.Errorf("native.LayerNormRowsF32(rows=%d cols=%d): %w", rows, cols, err)
	}
	return nil
}

// GeluExactBiasF32 applies the exact (erf-based) GELU in place over the
// [rows][cols] matrix x, adding the per-column bias first when bias is non-nil:
// x[i] = gelu(x[i] + bias[i%cols]). Mirrors geluExact in internal/backend/simd.
func GeluExactBiasF32(x, bias DeviceBuffer, rows, cols int, stream Stream) error {
	if x.ptr == nil {
		return fmt.Errorf("native.GeluExactBiasF32: buffer is nil")
	}
	if rows <= 0 || cols <= 0 {
		return fmt.Errorf("native.GeluExactBiasF32: rows/cols must be positive, got %d/%d", rows, cols)
	}
	var biasPtr *C.float
	if bias.ptr != nil {
		biasPtr = (*C.float)(bias.ptr)
	}
	if err := cudaErr(C.mantleCudaGeluExactBiasF32Wrapper(
		(*C.float)(x.ptr),
		biasPtr,
		C.int(rows),
		C.int(cols),
		stream.ptr,
	)); err != nil {
		return fmt.Errorf("native.GeluExactBiasF32(rows=%d cols=%d): %w", rows, cols, err)
	}
	return nil
}

// AddBiasRowsF32 adds the per-column bias to every row of the [rows][cols]
// matrix out in place: out[i] += bias[i%cols].
func AddBiasRowsF32(out, bias DeviceBuffer, rows, cols int, stream Stream) error {
	if out.ptr == nil || bias.ptr == nil {
		return fmt.Errorf("native.AddBiasRowsF32: buffer is nil")
	}
	if rows <= 0 || cols <= 0 {
		return fmt.Errorf("native.AddBiasRowsF32: rows/cols must be positive, got %d/%d", rows, cols)
	}
	if err := cudaErr(C.mantleCudaAddBiasRowsF32Wrapper(
		(*C.float)(out.ptr),
		(*C.float)(bias.ptr),
		C.int(rows),
		C.int(cols),
		stream.ptr,
	)); err != nil {
		return fmt.Errorf("native.AddBiasRowsF32(rows=%d cols=%d): %w", rows, cols, err)
	}
	return nil
}
