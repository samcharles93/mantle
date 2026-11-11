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
			if !useBatchQuant(w, tc.n) {
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
