//go:build cuda

package native

import (
	"fmt"
	"math"
	"testing"
	"unsafe"
)

// runHilo runs the two-term tensor-core GEMM and returns y [n, rows] row-major.
func (c *quantBatchCase) runHilo(tb testing.TB) []float32 {
	tb.Helper()
	y, err := AllocDevice(int64(c.n*c.rows) * 4)
	if err != nil {
		tb.Fatal(err)
	}
	defer func() { _ = y.Free() }()
	bpr := c.cols / 32
	if err := QuantMatVecCachedBatchMMAHilo(c.q, c.scales, c.x, c.xq, c.xsc, c.xq2, c.xsc2, y, c.rows, c.n, c.cols, bpr, c.stream); err != nil {
		tb.Fatal(err)
	}
	out := make([]float32, c.n*c.rows)
	if err := MemcpyD2H(unsafe.Pointer(&out[0]), y, int64(len(out))*4); err != nil {
		tb.Fatal(err)
	}
	return out
}

// hiloOracle returns the per-token f32-activation reference: every token runs
// through QuantMatVecInt8BlocksF32, the kernel the sequential prefill uses.
func (c *quantBatchCase) hiloOracle(tb testing.TB) []float32 {
	tb.Helper()
	bpr := c.cols / 32
	y, err := AllocDevice(int64(c.rows) * 4)
	if err != nil {
		tb.Fatal(err)
	}
	defer func() { _ = y.Free() }()
	xBuf, err := AllocDevice(int64(c.cols) * 4)
	if err != nil {
		tb.Fatal(err)
	}
	defer func() { _ = xBuf.Free() }()
	out := make([]float32, c.n*c.rows)
	for tok := 0; tok < c.n; tok++ {
		if err := MemcpyH2D(xBuf, unsafe.Pointer(&c.xh[tok*c.cols]), int64(c.cols)*4); err != nil {
			tb.Fatal(err)
		}
		if err := QuantMatVecInt8BlocksF32(c.q, c.scales, xBuf, y, c.rows, bpr, c.cols, c.stream); err != nil {
			tb.Fatal(err)
		}
		if err := MemcpyD2H(unsafe.Pointer(&out[tok*c.rows]), y, int64(c.rows)*4); err != nil {
			tb.Fatal(err)
		}
	}
	return out
}

// TestQuantBatchMMAHiloMatchesF32Oracle requires the two-term activation split
// to track the f32-activation per-token oracle: the split costs a second
// tensor-core dot per quant block, and in exchange each logit moves by well
// under a thousandth of its magnitude, far below the 0.3 to 0.5 a single int8
// term costs on a real model.
//
// The band is relative to the largest logit in the case, not per element:
// outputs from a cancelled dot are near zero and carry an unbounded relative
// error while their absolute error stays at the activation-split scale.
func TestQuantBatchMMAHiloMatchesF32Oracle(t *testing.T) {
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
			got := c.runHilo(t)
			want := c.hiloOracle(t)
			var worstAbs, scale float32
			for i := range want {
				if diff := float32(math.Abs(float64(got[i] - want[i]))); diff > worstAbs {
					worstAbs = diff
				}
				if mag := float32(math.Abs(float64(want[i]))); mag > scale {
					scale = mag
				}
			}
			rel := worstAbs / scale
			t.Logf("worst abs diff %v, largest logit %v, relative %v", worstAbs, scale, rel)
			const relBand = 1e-4
			if rel > relBand {
				t.Fatalf("hi/lo vs f32 oracle: worst diff %v is %v of the largest logit, over %v", worstAbs, rel, relBand)
			}
		})
	}
}

// BenchmarkQuantMatVecCachedBatchMMA measures the single-term tensor-core GEMM
// at the same shape as BenchmarkQuantMatVecCachedBatch.
func BenchmarkQuantMatVecCachedBatchMMA(b *testing.B) {
	benchmarkQuantBatchMMA(b, false)
}

// BenchmarkQuantMatVecCachedBatchMMAHilo measures the two-term tensor-core GEMM
// at the same shape as BenchmarkQuantMatVecCachedBatch.
func BenchmarkQuantMatVecCachedBatchMMAHilo(b *testing.B) {
	benchmarkQuantBatchMMA(b, true)
}

func benchmarkQuantBatchMMA(b *testing.B, hilo bool) {
	if !QuantBatchMMASupported() {
		b.Skip("tensor-core quant GEMM needs compute capability 8.0+")
	}
	const rows, cols, n = 2048, 2048, 610
	c := newQuantBatchCase(b, rows, cols, n)
	y, err := AllocDevice(int64(n*rows) * 4)
	if err != nil {
		b.Fatal(err)
	}
	defer func() { _ = y.Free() }()
	bpr := cols / 32
	run := func() error {
		if hilo {
			return QuantMatVecCachedBatchMMAHilo(c.q, c.scales, c.x, c.xq, c.xsc, c.xq2, c.xsc2, y, rows, n, cols, bpr, c.stream)
		}
		return QuantMatVecCachedBatchMMA(c.q, c.scales, c.x, c.xq, c.xsc, y, rows, n, cols, bpr, c.stream)
	}
	if err := run(); err != nil {
		b.Fatal(err)
	}
	if err := c.stream.Synchronize(); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for b.Loop() {
		if err := run(); err != nil {
			b.Fatal(err)
		}
		if err := c.stream.Synchronize(); err != nil {
			b.Fatal(err)
		}
	}
	macs := float64(rows) * cols * n
	b.ReportMetric(macs/(float64(b.Elapsed().Nanoseconds())/float64(b.N))/1e3, "TMAC/s")
}
