package simd

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/samcharles93/mantle/pkg/mcf"
)

func TestMatMatBF16MatchesScalar(t *testing.T) {
	if !cpu.HasAVX2 {
		t.Skip("MatMatBF16 requires AVX2")
	}

	tests := []struct {
		rows, cols int
	}{
		{rows: 1, cols: 8},
		{rows: 2, cols: 16},
		{rows: 17, cols: 256},
		{rows: 33, cols: 2051},
		{rows: 64, cols: 6144},
	}
	rng := rand.New(rand.NewSource(7))
	for _, tc := range tests {
		raw := makeBF16MatrixRaw(tc.rows, tc.cols, rng)
		w, err := NewMatFromRaw(tc.rows, tc.cols, mcf.DTypeBF16, raw)
		if err != nil {
			t.Fatalf("NewMatFromRaw: %v", err)
		}
		for _, n := range []int{1, 2, 3, 4, 5, 7, 8, 13, 16} {
			t.Run(fmt.Sprintf("r%d_c%d_n%d", tc.rows, tc.cols, n), func(t *testing.T) {
				x := make([]float32, n*tc.cols)
				for i := range x {
					x[i] = (rng.Float32() - 0.5) * 2
				}
				got := make([]float32, n*tc.rows)
				MatMatBF16(got, &w, x, n)

				want := make([]float32, tc.rows)
				for r := range n {
					matVecRangeBF16Scalar(want, &w, x[r*tc.cols:(r+1)*tc.cols], 0, tc.rows)
					for i, v := range want {
						g := got[r*tc.rows+i]
						if diff := abs32(g - v); diff > 1e-4*float32(tc.cols) {
							t.Fatalf("row %d col %d: got %g want %g (diff %g)", r, i, g, v, diff)
						}
					}
				}
			})
		}
	}
}

// BenchmarkSmallBatchBF16 compares the projection shapes of a 2B model for a
// speculative verify sized batch: MatMatBF16, the packed GEMM, and n matvecs.
func BenchmarkSmallBatchBF16(b *testing.B) {
	if !cpu.HasAVX2 {
		b.Skip("requires AVX2")
	}
	shapes := []struct {
		name       string
		rows, cols int
	}{
		{name: "qproj_2048x2048", rows: 2048, cols: 2048},
		{name: "up_6144x2048", rows: 6144, cols: 2048},
		{name: "down_2048x6144", rows: 2048, cols: 6144},
	}
	rng := rand.New(rand.NewSource(1))
	for _, s := range shapes {
		w, err := NewMatFromRaw(s.rows, s.cols, mcf.DTypeBF16, makeBF16MatrixRaw(s.rows, s.cols, rng))
		if err != nil {
			b.Fatalf("NewMatFromRaw: %v", err)
		}
		for _, n := range []int{1, 4, 8, 16} {
			x := make([]float32, n*s.cols)
			for i := range x {
				x[i] = rng.Float32() - 0.5
			}
			dst := make([]float32, n*s.rows)
			bytes := int64(s.rows * s.cols * 2)

			b.Run(fmt.Sprintf("%s/n%d/matmat", s.name, n), func(b *testing.B) {
				b.SetBytes(bytes)
				for b.Loop() {
					MatMatBF16(dst, &w, x, n)
				}
			})
			b.Run(fmt.Sprintf("%s/n%d/gemm", s.name, n), func(b *testing.B) {
				b.SetBytes(bytes)
				c := Mat{R: n, C: s.rows, Stride: s.rows, DType: mcf.DTypeF32, Data: dst}
				a := Mat{R: n, C: s.cols, Stride: s.cols, DType: mcf.DTypeF32, Data: x}
				cfg := SelectGemmConfig(n, s.cols, s.rows)
				for b.Loop() {
					GemmParWT(cfg, &c, &a, &w, 1, 0, 0)
				}
			})
			b.Run(fmt.Sprintf("%s/n%d/matvecs", s.name, n), func(b *testing.B) {
				b.SetBytes(bytes)
				for b.Loop() {
					for r := range n {
						MatVec(dst[r*s.rows:(r+1)*s.rows], &w, x[r*s.cols:(r+1)*s.cols])
					}
				}
			})
		}
	}
}
