package simd

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/samcharles93/mantle/pkg/mcf"
)

func TestMatVecBF16MatchesScalar(t *testing.T) {
	if !cpu.HasAVX2 && !cpu.HasAVX512 {
		t.Skip("SIMD BF16 matvec path requires AVX2 or AVX512")
	}

	tests := []struct {
		name string
		rows int
		cols int
	}{
		{name: "projection", rows: 17, cols: 256},
		{name: "down_proj", rows: 19, cols: 10240},
	}

	rng := rand.New(rand.NewSource(1))
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			raw := makeBF16MatrixRaw(tc.rows, tc.cols, rng)
			mat, err := NewMatFromRaw(tc.rows, tc.cols, mcf.DTypeBF16, raw)
			if err != nil {
				t.Fatalf("NewMatFromRaw: %v", err)
			}

			x := make([]float32, tc.cols)
			for i := range x {
				x[i] = (rng.Float32() - 0.5) * 2
			}

			got := make([]float32, tc.rows)
			want := make([]float32, tc.rows)

			MatVec(got, &mat, x)
			matVecRangeBF16Scalar(want, &mat, x, 0, tc.rows)

			for i := range got {
				if diff := abs32(got[i] - want[i]); diff > 5e-4 {
					t.Fatalf("row %d diff=%g got=%g want=%g", i, diff, got[i], want[i])
				}
			}
		})
	}
}

// TestMatVecBF16GroupsMatchSingleRows checks that the eight-row kernel matches
// the one-row path bit for bit, over row ranges that leave four-, two- and
// one-row tails, and over widths that are and are not multiples of 8.
func TestMatVecBF16GroupsMatchSingleRows(t *testing.T) {
	if !cpu.HasAVX2 {
		t.Skip("needs AVX2")
	}
	rng := rand.New(rand.NewSource(3))
	for _, tc := range []struct{ rows, cols, rs, re int }{
		{67, 2048, 0, 67}, {40, 2048, 3, 37}, {16, 7680, 0, 16}, {23, 1000, 0, 23}, {13, 12, 0, 13},
	} {
		t.Run(fmt.Sprintf("%dx%d/%d-%d", tc.rows, tc.cols, tc.rs, tc.re), func(t *testing.T) {
			raw := makeBF16MatrixRaw(tc.rows, tc.cols, rng)
			// Row 5 is all zeros, so its products are signed zeros: this checks
			// that the eight-row path gives a zero sum the same sign as the
			// one-row path.
			for i := range tc.cols * 2 {
				raw[5*tc.cols*2+i] = 0
			}
			mat, err := NewMatFromRaw(tc.rows, tc.cols, mcf.DTypeBF16, raw)
			if err != nil {
				t.Fatal(err)
			}
			x := make([]float32, tc.cols)
			for i := range x {
				x[i] = (rng.Float32() - 0.5) * 2
			}
			got := make([]float32, tc.rows)
			want := make([]float32, tc.rows)
			matVecRangeBF16SIMD(got, &mat, x, tc.rs, tc.re)
			for r := tc.rs; r < tc.re; r++ {
				matVecRangeBF16SIMD(want, &mat, x, r, r+1)
			}
			for r := tc.rs; r < tc.re; r++ {
				if math.Float32bits(got[r]) != math.Float32bits(want[r]) {
					t.Fatalf("row %d = %g (%#x), single-row path %g (%#x)", r, got[r], math.Float32bits(got[r]), want[r], math.Float32bits(want[r]))
				}
			}
		})
	}
}

func makeBF16MatrixRaw(rows, cols int, rng *rand.Rand) []byte {
	raw := make([]byte, rows*cols*2)
	for i := 0; i < rows*cols; i++ {
		v := (rng.Float32() - 0.5) * 2
		binary.LittleEndian.PutUint16(raw[i*2:], bf16Bits(v))
	}
	return raw
}

func bf16Bits(v float32) uint16 {
	u := math.Float32bits(v)
	round := uint32(0x7FFF) + ((u >> 16) & 1)
	return uint16((u + round) >> 16)
}

func abs32(v float32) float32 {
	if v < 0 {
		return -v
	}
	return v
}

// BenchmarkMatVecBF16Shapes measures the bf16 decode MatVec at the LFM3-1.7B
// projection shapes: pool is the public entry point at the worker pool's
// width, core is the kernel over every row on the calling goroutine (pin it
// with taskset to pick a core type).
func BenchmarkMatVecBF16Shapes(b *testing.B) {
	if !cpu.HasAVX2 {
		b.Skip("needs AVX2")
	}
	rng := rand.New(rand.NewSource(1))
	for _, shape := range lfm3QuantShapes {
		rows, cols := shape.rows, shape.cols
		mat, err := NewMatFromRaw(rows, cols, mcf.DTypeBF16, makeBF16MatrixRaw(rows, cols, rng))
		if err != nil {
			b.Fatal(err)
		}
		x := make([]float32, cols)
		for i := range x {
			x[i] = rng.Float32() - 0.5
		}
		dst := make([]float32, rows)
		wBytes := rows * cols * 2
		b.Run(shape.name+"/pool", func(b *testing.B) {
			iters := 0
			for b.Loop() {
				MatVec(dst, &mat, x)
				iters++
			}
			reportGEMMRates(b, iters, rows, cols, 1, wBytes, b.Elapsed())
		})
		b.Run(shape.name+"/core", func(b *testing.B) {
			iters := 0
			for b.Loop() {
				matVecRangeBF16(dst, &mat, x, 0, rows)
				iters++
			}
			reportGEMMRates(b, iters, rows, cols, 1, wBytes, b.Elapsed())
		})
	}
}
