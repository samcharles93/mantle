package simd

import (
	"encoding/binary"
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
