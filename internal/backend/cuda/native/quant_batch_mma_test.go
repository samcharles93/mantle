//go:build cuda

package native

import (
	"fmt"
	"math"
	"testing"
	"unsafe"
)

type quantBatchCase struct {
	rows, cols, n int
	q             DeviceBuffer
	scales        DeviceBuffer
	x             DeviceBuffer
	xq, xsc       DeviceBuffer
	stream        Stream
}

func newQuantBatchCase(tb testing.TB, rows, cols, n int) *quantBatchCase {
	tb.Helper()
	bpr := cols / 32
	q := make([]int8, rows*cols)
	for i := range q {
		q[i] = int8((i*131+7)%255 - 127)
	}
	scales := make([]float32, rows*bpr)
	for i := range scales {
		scales[i] = 0.0005 + float32((i*17)%29)*0.00007
	}
	x := make([]float32, n*cols)
	for i := range x {
		x[i] = float32((i*37)%201-100) * 0.0137
	}
	// A zero block exercises the zero activation scale.
	if len(x) >= 64 {
		clear(x[32:64])
	}
	mk := func(bytes int64) DeviceBuffer {
		buf, err := AllocDevice(bytes)
		if err != nil {
			tb.Fatal(err)
		}
		tb.Cleanup(func() { buf.Free() })
		return buf
	}
	c := &quantBatchCase{rows: rows, cols: cols, n: n}
	c.q, c.scales, c.x = mk(int64(len(q))), mk(int64(len(scales))*4), mk(int64(len(x))*4)
	c.xq, c.xsc = mk(int64(n*cols)), mk(int64(n*bpr)*4)
	stream, err := NewStream()
	if err != nil {
		tb.Fatal(err)
	}
	tb.Cleanup(func() { stream.Destroy() })
	c.stream = stream
	for _, cp := range []struct {
		dst   DeviceBuffer
		src   unsafe.Pointer
		bytes int64
	}{
		{c.q, unsafe.Pointer(&q[0]), int64(len(q))},
		{c.scales, unsafe.Pointer(&scales[0]), int64(len(scales)) * 4},
		{c.x, unsafe.Pointer(&x[0]), int64(len(x)) * 4},
	} {
		if err := MemcpyH2D(cp.dst, cp.src, cp.bytes); err != nil {
			tb.Fatal(err)
		}
	}
	return c
}

func (c *quantBatchCase) run(tb testing.TB, mma bool) []float32 {
	tb.Helper()
	y, err := AllocDevice(int64(c.n*c.rows) * 4)
	if err != nil {
		tb.Fatal(err)
	}
	defer y.Free()
	bpr := c.cols / 32
	if mma {
		err = QuantMatVecCachedBatchMMA(c.q, c.scales, c.x, c.xq, c.xsc, y, c.rows, c.n, c.cols, bpr, c.stream)
	} else {
		err = QuantMatVecCachedBatchDp4a(c.q, c.scales, c.x, c.xq, c.xsc, y, c.rows, c.n, c.cols, bpr, c.stream)
	}
	if err != nil {
		tb.Fatal(err)
	}
	out := make([]float32, c.n*c.rows)
	if err := MemcpyD2H(unsafe.Pointer(&out[0]), y, int64(len(out))*4); err != nil {
		tb.Fatal(err)
	}
	return out
}

// TestQuantBatchMMAMatchesDp4a requires the tensor-core kernel to reproduce the
// dp4a kernel exactly, at shapes covering partial row, token and K tiles.
func TestQuantBatchMMAMatchesDp4a(t *testing.T) {
	if !QuantBatchMMASupported() {
		t.Skip("tensor-core quant GEMM needs compute capability 8.0+")
	}
	for _, s := range []struct{ rows, cols, n int }{
		{64, 128, 64},
		{70, 96, 17},    // three K blocks: a partial K tile
		{130, 2048, 37}, // row and token tails
		{512, 7680, 100},
	} {
		t.Run(fmt.Sprintf("%dx%dx%d", s.rows, s.cols, s.n), func(t *testing.T) {
			c := newQuantBatchCase(t, s.rows, s.cols, s.n)
			want := c.run(t, false)
			got := c.run(t, true)
			for i := range want {
				if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
					t.Fatalf("token %d row %d: mma %v, dp4a %v", i/s.rows, i%s.rows, got[i], want[i])
				}
			}
		})
	}
}

// BenchmarkQuantBatchGEMM times both kernels at the LFM3 FFN up shape over a
// prefill-sized token batch.
func BenchmarkQuantBatchGEMM(b *testing.B) {
	const rows, cols, n = 7680, 2048, 512
	c := newQuantBatchCase(b, rows, cols, n)
	y, err := AllocDevice(int64(n*rows) * 4)
	if err != nil {
		b.Fatal(err)
	}
	defer y.Free()
	bpr := cols / 32
	for _, mode := range []struct {
		name string
		fn   func() error
	}{
		{"dp4a", func() error {
			return QuantMatVecCachedBatchDp4a(c.q, c.scales, c.x, c.xq, c.xsc, y, rows, n, cols, bpr, c.stream)
		}},
		{"mma", func() error {
			return QuantMatVecCachedBatchMMA(c.q, c.scales, c.x, c.xq, c.xsc, y, rows, n, cols, bpr, c.stream)
		}},
	} {
		if mode.name == "mma" && !QuantBatchMMASupported() {
			continue
		}
		b.Run(mode.name, func(b *testing.B) {
			for b.Loop() {
				if err := mode.fn(); err != nil {
					b.Fatal(err)
				}
				if err := c.stream.Synchronize(); err != nil {
					b.Fatal(err)
				}
			}
			macs := float64(rows) * cols * n
			b.ReportMetric(macs/(float64(b.Elapsed().Nanoseconds())/float64(b.N))/1e3, "TMAC/s")
		})
	}
}
