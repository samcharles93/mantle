package simd

import (
	"math"
	"math/rand/v2"
	"testing"

	instance "github.com/samcharles93/mantle/internal/backend/core"
	"github.com/samcharles93/mantle/pkg/mcf"
)

func gqTestMat(tb testing.TB, rows, cols, bits, group int) Mat {
	tb.Helper()
	rng := rand.New(rand.NewPCG(21, 22))
	src := make([]float32, rows*cols)
	for i := range src {
		src[i] = float32(rng.NormFloat64() * 0.02)
	}
	dt, payload, err := mcf.EncodeGQTensor(src, rows, cols, bits, group)
	if err != nil {
		tb.Fatal(err)
	}
	return Mat{R: rows, C: cols, Stride: cols, DType: dt, Raw: payload}
}

// TestMatVecGQMatchesDecodedReference checks the cached int8 kernel and the
// uncached row-decode path against an f32 matvec over the decoded weights.
func TestMatVecGQMatchesDecodedReference(t *testing.T) {
	const rows, cols = 64, 320
	for _, tc := range []struct{ bits, group int }{{4, 128}, {3, 64}, {2, 32}, {8, 128}} {
		w := gqTestMat(t, rows, cols, tc.bits, tc.group)
		x := make([]float32, cols)
		for i := range x {
			x[i] = float32((i%13)-6) * 0.11
		}
		want := make([]float32, rows)
		row := make([]float32, cols)
		for r := range rows {
			w.RowTo(row, r)
			var sum float64
			for c, v := range row {
				sum += float64(v) * float64(x[c])
			}
			want[r] = float32(sum)
		}

		uncached := make([]float32, rows)
		MatVec(uncached, &w, x)

		qc, err := instance.BuildQuantCache(&w)
		if err != nil {
			t.Fatal(err)
		}
		w.Quant = qc
		cached := make([]float32, rows)
		MatVec(cached, &w, x)
		qx := PrepareQuantVec(x)
		cachedQx := make([]float32, rows)
		MatVecWithQuant(cachedQx, &w, x, qx)
		ReleaseQuantVec(qx)

		for r := range rows {
			tol := 1e-4 * (1 + math.Abs(float64(want[r])))
			if d := math.Abs(float64(uncached[r] - want[r])); d > tol {
				t.Fatalf("bits=%d group=%d row %d: uncached %v, want %v", tc.bits, tc.group, r, uncached[r], want[r])
			}
			// The cached kernels quantize x to int16, so they get a looser bound.
			loose := 1e-2 * (1 + math.Abs(float64(want[r])))
			if d := math.Abs(float64(cached[r] - want[r])); d > loose {
				t.Fatalf("bits=%d group=%d row %d: cached %v, want %v", tc.bits, tc.group, r, cached[r], want[r])
			}
			if d := math.Abs(float64(cachedQx[r] - want[r])); d > loose {
				t.Fatalf("bits=%d group=%d row %d: int16-activation %v, want %v", tc.bits, tc.group, r, cachedQx[r], want[r])
			}
		}
	}
}

func BenchmarkMatVecGQ4Cached(b *testing.B) {
	const rows, cols = 2048, 1024
	w := gqTestMat(b, rows, cols, 4, 128)
	qc, err := instance.BuildQuantCache(&w)
	if err != nil {
		b.Fatal(err)
	}
	w.Quant = qc
	x := make([]float32, cols)
	for i := range x {
		x[i] = float32((i%13)-6) * 0.11
	}
	dst := make([]float32, rows)
	qx := PrepareQuantVec(x)
	defer ReleaseQuantVec(qx)
	for b.Loop() {
		MatVecWithQuant(dst, &w, x, qx)
	}
}

func BenchmarkMatVecGQ4RowDecode(b *testing.B) {
	const rows, cols = 2048, 1024
	w := gqTestMat(b, rows, cols, 4, 128)
	x := make([]float32, cols)
	for i := range x {
		x[i] = float32((i%13)-6) * 0.11
	}
	dst := make([]float32, rows)
	for b.Loop() {
		MatVec(dst, &w, x)
	}
}
