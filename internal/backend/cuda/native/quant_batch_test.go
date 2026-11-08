//go:build cuda

package native

import (
	"testing"
	"unsafe"
)

// BenchmarkQuantMatVecCachedBatch measures the batched quant GEMM kernel
// (cached-int8 layout) at a realistic prefill shape.
func BenchmarkQuantMatVecCachedBatch(b *testing.B) {
	rows, cols, n := 2048, 2048, 610
	bpr := cols / 32
	q := make([]int8, rows*bpr*32)
	for i := range q {
		q[i] = int8(i%251 - 125)
	}
	scales := make([]float32, rows*bpr)
	for i := range scales {
		scales[i] = 0.001
	}
	x := make([]float32, n*cols)
	for i := range x {
		x[i] = float32(i%7) * 0.01
	}
	y := make([]float32, n*rows)
	qBuf, err := AllocDevice(int64(len(q)))
	if err != nil {
		b.Fatal(err)
	}
	defer qBuf.Free()
	sBuf, err := AllocDevice(int64(len(scales)) * 4)
	if err != nil {
		b.Fatal(err)
	}
	defer sBuf.Free()
	xBuf, err := AllocDevice(int64(len(x)) * 4)
	if err != nil {
		b.Fatal(err)
	}
	defer xBuf.Free()
	yBuf, err := AllocDevice(int64(len(y)) * 4)
	if err != nil {
		b.Fatal(err)
	}
	defer yBuf.Free()
	stream, err := NewStream()
	if err != nil {
		b.Fatal(err)
	}
	defer stream.Destroy()
	if err := MemcpyH2D(qBuf, unsafe.Pointer(&q[0]), int64(len(q))); err != nil {
		b.Fatal(err)
	}
	if err := MemcpyH2D(sBuf, unsafe.Pointer(&scales[0]), int64(len(scales))*4); err != nil {
		b.Fatal(err)
	}
	if err := MemcpyH2D(xBuf, unsafe.Pointer(&x[0]), int64(len(x))*4); err != nil {
		b.Fatal(err)
	}
	// warmup
	if err := QuantMatVecCachedBatch(qBuf, sBuf, xBuf, yBuf, rows, n, cols, bpr, stream); err != nil {
		b.Fatal(err)
	}
	if err := stream.Synchronize(); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := QuantMatVecCachedBatch(qBuf, sBuf, xBuf, yBuf, rows, n, cols, bpr, stream); err != nil {
			b.Fatal(err)
		}
	}
	if err := stream.Synchronize(); err != nil {
		b.Fatal(err)
	}
}
