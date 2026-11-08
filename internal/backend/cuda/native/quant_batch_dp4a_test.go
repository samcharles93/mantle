//go:build cuda

package native

import (
	"math"
	"testing"
	"unsafe"
)

// TestDp4aVsF32Kernel checks the int8 dp4a batch kernel against the f32-x
// kernel at a small shape; they differ only by the per-32-block int8
// quantization of the activations.
func TestDp4aVsF32Kernel(t *testing.T) {
	rows, cols, n := 16, 64, 5
	bpr := cols / 32
	q := make([]int8, rows*bpr*32)
	for i := range q {
		q[i] = int8(i%251 - 125)
	}
	scales := make([]float32, rows*bpr)
	for i := range scales {
		scales[i] = 0.001 + float32(i%13)*0.0001
	}
	// integer-valued activations: int8 quantization is then exact, so the
	// dp4a kernel must match the f32-x kernel bit-for-bit (int8 dots are exact)
	x := make([]float32, n*cols)
	for i := range x {
		x[i] = float32(i%255 - 127)
	}
	y1 := make([]float32, n*rows)
	y2 := make([]float32, n*rows)
	xq := make([]int8, n*cols)
	xsc := make([]float32, n*bpr)
	mk := func(bytes int64) DeviceBuffer {
		buf, err := AllocDevice(bytes)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { buf.Free() })
		return buf
	}
	qBuf, sBuf, xBuf := mk(int64(len(q))), mk(int64(len(scales))*4), mk(int64(len(x))*4)
	y1Buf, y2Buf := mk(int64(len(y1))*4), mk(int64(len(y2))*4)
	xqBuf, xscBuf := mk(int64(len(xq))), mk(int64(len(xsc))*4)
	stream, err := NewStream()
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Destroy()
	MemcpyH2D(qBuf, unsafe.Pointer(&q[0]), int64(len(q)))
	MemcpyH2D(sBuf, unsafe.Pointer(&scales[0]), int64(len(scales))*4)
	MemcpyH2D(xBuf, unsafe.Pointer(&x[0]), int64(len(x))*4)
	if err := QuantMatVecCachedBatch(qBuf, sBuf, xBuf, y1Buf, rows, n, cols, bpr, stream); err != nil {
		t.Fatal(err)
	}
	if err := QuantMatVecCachedBatchDp4a(qBuf, sBuf, xBuf, xqBuf, xscBuf, y2Buf, rows, n, cols, bpr, stream); err != nil {
		t.Fatal(err)
	}
	MemcpyD2H(unsafe.Pointer(&y1[0]), y1Buf, int64(len(y1))*4)
	MemcpyD2H(unsafe.Pointer(&y2[0]), y2Buf, int64(len(y2))*4)
	// dp4a quantizes x to int8 per 32-block, so it differs from the f32-x
	// kernel by the activation quantization noise (~a few % of the output scale)
	maxD, maxY := 0.0, 0.0
	for i := range y1 {
		d := math.Abs(float64(y1[i] - y2[i]))
		if d > maxD {
			maxD = d
		}
		if a := math.Abs(float64(y1[i])); a > maxY {
			maxY = a
		}
	}
	t.Logf("max diff f32 vs dp4a: %g (output scale %g)", maxD, maxY)
	if maxD > 0.05*maxY+1e-6 {
		t.Fatalf("dp4a mismatch beyond activation-quant noise: max diff %g, scale %g", maxD, maxY)
	}
}
