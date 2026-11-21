package simd

import (
	"fmt"
	"testing"

	instance "github.com/samcharles93/mantle/internal/backend/core"
	"github.com/samcharles93/mantle/pkg/mcf"
)

// cachedQuantMat builds a quant weight with its int8 cache, the form the loader
// produces for q8/q4/k* MCF tensors.
func cachedQuantMat(tb testing.TB, dt mcf.TensorDType, rows, cols int) *Mat {
	tb.Helper()
	var payload []byte
	switch dt {
	case mcf.DTypeQ8:
		payload = buildQPayload(rows, cols, 8, 0.03, makeQVals(rows*cols))
	case mcf.DTypeQ4:
		payload = buildQPayload(rows, cols, 4, 0.05, makeQVals(rows*cols))
	case mcf.DTypeK4:
		payload = buildK4Payload(rows, cols, 0.05, makeKVals(4, rows*cols))
	case mcf.DTypeK6:
		payload = buildKPayload(rows, cols, 6, 0.02, makeKVals(6, rows*cols))
	default:
		tb.Fatalf("unsupported dtype %v", dt)
	}
	w := &Mat{R: rows, C: cols, Stride: cols, DType: dt, Raw: payload}
	qc, err := instance.BuildQuantCache(w)
	if err != nil {
		tb.Fatalf("BuildQuantCache: %v", err)
	}
	w.Quant = qc
	return w
}

func quantTestActivations(n, cols int) []float32 {
	x := make([]float32, n*cols)
	for i := range x {
		x[i] = float32((i*37)%101-50) * 0.013
	}
	// A zero row and a row with a zero block exercise the zero-scale blocks
	// matVecRangeQuantCached skips in its per-block loop.
	if n > 2 {
		clear(x[cols : 2*cols])
		clear(x[2*cols : 2*cols+32])
	}
	return x
}

// TestMatMatQuantMatchesMatVec requires every MatMatQuant output to equal the
// per-row MatVec result bit for bit, which is what lets the batched prefill
// take quantised weights without changing greedy output.
func TestMatMatQuantMatchesMatVec(t *testing.T) {
	if !cpu.HasAVX2 {
		t.Skip("MatMatQuant needs AVX2")
	}
	cases := []struct {
		dt         mcf.TensorDType
		rows, cols int
		n          int
	}{
		{mcf.DTypeQ8, 67, 256, 1},
		{mcf.DTypeQ8, 67, 256, 7},
		{mcf.DTypeQ8, 130, 1312, 16}, // 41 blocks: an 8-block group tail
		{mcf.DTypeQ4, 64, 512, 5},
		{mcf.DTypeK4, 96, 768, 33},
		{mcf.DTypeK6, 40, 512, 4},
	}
	for _, tc := range cases {
		t.Run(fmt.Sprintf("%v_%dx%d_n%d", tc.dt, tc.rows, tc.cols, tc.n), func(t *testing.T) {
			w := cachedQuantMat(t, tc.dt, tc.rows, tc.cols)
			if !canBatchQuant(w) || (tc.n >= 2 && !useBatchQuant(w, tc.n)) {
				t.Fatalf("useBatchQuant refused a cached %v weight", tc.dt)
			}
			x := quantTestActivations(tc.n, tc.cols)
			got := make([]float32, tc.n*tc.rows)
			MatMatQuant(got, w, x, tc.n)
			want := make([]float32, tc.rows)
			for r := range tc.n {
				MatVec(want, w, x[r*tc.cols:(r+1)*tc.cols])
				for i, v := range want {
					if g := got[r*tc.rows+i]; g != v {
						t.Fatalf("row %d col %d: MatMatQuant = %v, MatVec = %v", r, i, g, v)
					}
				}
			}
		})
	}
}

func TestUseBatchQuantRefusals(t *testing.T) {
	w := cachedQuantMat(t, mcf.DTypeQ8, 8, 64)
	uncached := *w
	uncached.Quant = nil
	ragged := cachedQuantMat(t, mcf.DTypeQ8, 8, 80)
	for name, tc := range map[string]struct {
		w *Mat
		n int
	}{
		"no cache":    {&uncached, 4},
		"ragged cols": {ragged, 4},
		"empty batch": {w, 0},
		"single row":  {w, 1},
	} {
		if useBatchQuant(tc.w, tc.n) {
			t.Errorf("%s: useBatchQuant accepted", name)
		}
	}
}

// BenchmarkMatMatQuant compares one batched call against n per-row MatVec
// calls at an LFM3 projection shape and a prefill-sized chunk.
func BenchmarkMatMatQuant(b *testing.B) {
	if !cpu.HasAVX2 {
		b.Skip("needs AVX2")
	}
	const rows, cols, n = 2048, 2048, 64
	w := cachedQuantMat(b, mcf.DTypeQ8, rows, cols)
	x := quantTestActivations(n, cols)
	dst := make([]float32, n*rows)
	b.Run("batched", func(b *testing.B) {
		for b.Loop() {
			MatMatQuant(dst, w, x, n)
		}
	})
	b.Run("perRow", func(b *testing.B) {
		for b.Loop() {
			for r := range n {
				MatVec(dst[r*rows:(r+1)*rows], w, x[r*cols:(r+1)*cols])
			}
		}
	})
}

// TestMatMatQuantPackedMatchesMatVec runs the batched kernel on packed 4-bit
// weights (no int8 cache) and requires every output to equal both the per-row
// packed MatVec and the batched kernel on the cached copy, bit for bit.
func TestMatMatQuantPackedMatchesMatVec(t *testing.T) {
	if !cpu.HasAVX2 {
		t.Skip("MatMatQuant needs AVX2")
	}
	dts := packed4DTypes(t)
	for _, dt := range dts {
		for _, n := range []int{1, 2, 3, 4, 6, 7, 33} {
			const rows, cols = 67, 768
			t.Run(fmt.Sprintf("%#x_n%d", dt, n), func(t *testing.T) {
				w := packed4Mat(t, dt, rows, cols)
				if !canBatchQuant(&w) || (n >= 2 && !useBatchQuant(&w, n)) || !batchWeightsPacked(&w) {
					t.Fatalf("packed %#x weight not taken by the batched kernel", dt)
				}
				x := quantTestActivations(n, cols)
				got := make([]float32, n*rows)
				MatMatQuant(got, &w, x, n)

				cached := w
				qc, err := instance.BuildQuantCache(&cached)
				if err != nil {
					t.Fatal(err)
				}
				cached.Quant = qc
				gotCached := make([]float32, n*rows)
				MatMatQuant(gotCached, &cached, x, n)

				want := make([]float32, rows)
				for r := range n {
					MatVec(want, &w, x[r*cols:(r+1)*cols])
					for i, v := range want {
						if g := got[r*rows+i]; g != v {
							t.Fatalf("row %d col %d: packed MatMatQuant = %v, MatVec = %v", r, i, g, v)
						}
						if g := gotCached[r*rows+i]; g != v {
							t.Fatalf("row %d col %d: cached MatMatQuant = %v, packed MatVec = %v", r, i, g, v)
						}
					}
				}
			})
		}
	}
}

// BenchmarkMatMatQuantK4 compares the batched kernel on packed k4 weights
// with the same kernel on the int8 cache, at a prefill-sized chunk.
func BenchmarkMatMatQuantK4(b *testing.B) {
	if !cpu.HasAVX2 {
		b.Skip("needs AVX2")
	}
	const rows, cols, n = 2048, 2048, 64
	w := packed4Mat(b, mcf.DTypeK4, rows, cols)
	x := quantTestActivations(n, cols)
	dst := make([]float32, n*rows)
	b.Run("packed", func(b *testing.B) {
		for b.Loop() {
			MatMatQuant(dst, &w, x, n)
		}
	})
	cached := w
	qc, err := instance.BuildQuantCache(&cached)
	if err != nil {
		b.Fatal(err)
	}
	cached.Quant = qc
	b.Run("cached", func(b *testing.B) {
		for b.Loop() {
			MatMatQuant(dst, &cached, x, n)
		}
	})
}

// TestSharedQuantRowsMixedMatchesMatVec verifies that prepareBatchQuantRows
// sharing one quantRows across packed 4-bit (k4) and cached int8 (k6) weights
// — the mixed-recipe shape of Wq/Wk alongside Wv — produces bit-identical
// outputs to per-row MatVec across quad, pair and odd-row tails.
func TestSharedQuantRowsMixedMatchesMatVec(t *testing.T) {
	if !cpu.HasAVX2 {
		t.Skip("MatMatQuant needs AVX2")
	}
	const rows, cols = 67, 512
	wPacked := packed4Mat(t, mcf.DTypeK4, rows, cols)
	wCached := cachedQuantMat(t, mcf.DTypeK6, rows, cols)
	for _, n := range []int{1, 2, 3, 4, 6, 7, 17} {
		t.Run(fmt.Sprintf("n=%d", n), func(t *testing.T) {
			x := quantTestActivations(n, cols)
			qr := prepareBatchQuantRows(x, n, cols, &wPacked, wCached)
			defer putQuantRows(qr)
			if n == 1 && qr != nil {
				t.Fatalf("n=1 should skip batched quantRows")
			}
			if n >= 2 && (qr == nil || len(qr.q8) == 0 || len(qr.q8Pair) == 0) {
				t.Fatalf("mixed packed+cached weights should populate both layouts")
			}
			gotPacked := make([]float32, n*rows)
			gotCached := make([]float32, n*rows)
			projectRowsWithQuant(gotPacked, &wPacked, x, n, qr)
			projectRowsWithQuant(gotCached, wCached, x, n, qr)

			want := make([]float32, rows)
			for r := range n {
				MatVec(want, &wPacked, x[r*cols:(r+1)*cols])
				for i, v := range want {
					if g := gotPacked[r*rows+i]; g != v {
						t.Fatalf("packed row %d col %d: got %v want %v", r, i, g, v)
					}
				}
				MatVec(want, wCached, x[r*cols:(r+1)*cols])
				for i, v := range want {
					if g := gotCached[r*rows+i]; g != v {
						t.Fatalf("cached row %d col %d: got %v want %v", r, i, g, v)
					}
				}
			}
		})
	}
}

// TestQuantDotUnsignedWorstCaseInt16Bound drives every lane of the int16
// reduction tree in quantDotUnsigned2x4 and quantDotUnsigned2x2 to its exact
// ±30,480 extreme (every weight code = 15, every activation = ±127) as well as
// the minimum code (0 -> signed -8) and compares against an exact int64
// reference. Random-input tests almost never reach all-max blocks.
func TestQuantDotUnsignedWorstCaseInt16Bound(t *testing.T) {
	if !cpu.HasAVX2 {
		t.Skip("MatMatQuant needs AVX2")
	}
	const blocks = 4
	const n = blocks * 32

	fillU := func(v uint8) []uint8 {
		s := make([]uint8, n)
		for i := range s {
			s[i] = v
		}
		return s
	}
	fillI := func(v int8) []int8 {
		s := make([]int8, n)
		for i := range s {
			s[i] = v
		}
		return s
	}

	// w0 = all 15 (signed +7, hits +30480/-30480 in int16 tree), w1 = all 0 (signed -8).
	w0 := fillU(packed4UnsignedMax)
	w1 := fillU(0)
	// Four activation rows hitting both +127 and -127 extremes.
	xs := [4][]int8{
		fillI(quantActMaxAbs),
		fillI(-quantActMaxAbs),
		fillI(-quantActMaxAbs),
		fillI(quantActMaxAbs),
	}

	// Unit scales so every output is the exact integer dot sum across all blocks.
	wQuads := make([]float32, blocks*8)
	xQuads := make([]float32, blocks*8)
	for i := range wQuads {
		wQuads[i] = 1
		xQuads[i] = 1
	}
	wPairs := make([]float32, blocks*4)
	xPairs := make([]float32, blocks*4)
	for i := range wPairs {
		wPairs[i] = 1
		xPairs[i] = 1
	}

	var bias [4][blocks]int32
	for r := range 4 {
		for b := range blocks {
			var s int64
			for i := range 32 {
				s += int64(xs[r][b*32+i])
			}
			bias[r][b] = int32(s << 3)
		}
	}
	xQuadBias := make([]int32, blocks*8)
	xPairBias := make([]int32, blocks*4)
	for b := range blocks {
		b0, b1, b2, b3 := bias[0][b], bias[1][b], bias[2][b], bias[3][b]
		xQuadBias[8*b], xQuadBias[8*b+1], xQuadBias[8*b+2], xQuadBias[8*b+3] = b0, b1, b0, b1
		xQuadBias[8*b+4], xQuadBias[8*b+5], xQuadBias[8*b+6], xQuadBias[8*b+7] = b2, b3, b2, b3
		xPairBias[4*b], xPairBias[4*b+1], xPairBias[4*b+2], xPairBias[4*b+3] = b0, b1, b0, b1
	}

	refDot := func(u []uint8, x []int8) float32 {
		var total int64
		for i := range n {
			total += (int64(u[i]) - 8) * int64(x[i])
		}
		return float32(total)
	}

	var out2x4 [8]float32
	quantDotUnsigned2x4(w0, w1, xs[0], xs[1], xs[2], xs[3], wQuads, xQuads, xQuadBias, &out2x4)
	want2x4 := [8]float32{
		refDot(w0, xs[0]), refDot(w0, xs[1]), refDot(w1, xs[0]), refDot(w1, xs[1]),
		refDot(w0, xs[2]), refDot(w0, xs[3]), refDot(w1, xs[2]), refDot(w1, xs[3]),
	}
	for i := range out2x4 {
		if out2x4[i] != want2x4[i] {
			t.Fatalf("quantDotUnsigned2x4[%d] = %v, want %v", i, out2x4[i], want2x4[i])
		}
	}

	var out2x2 [4]float32
	quantDotUnsigned2x2(w0, w1, xs[0], xs[1], wPairs, xPairs, xPairBias, &out2x2)
	for i := range out2x2 {
		if out2x2[i] != want2x4[i] {
			t.Fatalf("quantDotUnsigned2x2[%d] = %v, want %v", i, out2x2[i], want2x4[i])
		}
	}
}
