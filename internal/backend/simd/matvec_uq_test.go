//go:build goexperiment.simd

package simd

import (
	"math"
	"testing"

	"github.com/samcharles93/mantle/pkg/mcf"
)

// buildUQ4Payload builds a UQ4 payload with a fixed 16-level codebook per
// super-block, per-block scales and deterministic indices.
func buildUQ4Payload(rows, cols int) (payload []byte, codebook []float32, scales []float32, indices []uint8) {
	bits := 4
	cbElems := 1 << bits
	blocksPerRow := (cols + 31) / 32
	totalBlocks := rows * blocksPerRow
	superBlocksPerRow := (blocksPerRow + 7) / 8
	superCount := rows * superBlocksPerRow

	codebook = make([]float32, superCount*cbElems)
	for s := 0; s < superCount; s++ {
		for i := 0; i < cbElems; i++ {
			codebook[s*cbElems+i] = float32(i)*0.125 - 1 // [-1, 0.875]
		}
	}
	scales = make([]float32, totalBlocks)
	for i := range scales {
		scales[i] = 0.02 * float32((i%7)+1)
	}
	indices = make([]uint8, rows*cols)
	for i := range indices {
		indices[i] = uint8(i % cbElems)
	}
	payload, err := mcf.EncodeUQPayload([]uint64{uint64(rows), uint64(cols)}, mcf.DTypeUQ4, codebook, scales, indices)
	if err != nil {
		panic(err)
	}
	return payload, codebook, scales, indices
}

func uqReference(rows, cols int, codebook, scales []float32, indices []uint8, row int) []float32 {
	blocksPerRow := (cols + 31) / 32
	superBlocksPerRow := (blocksPerRow + 7) / 8
	out := make([]float32, cols)
	for c := 0; c < cols; c++ {
		b := row*blocksPerRow + c/32
		s := row*superBlocksPerRow + (c/32)/8
		scale := mcf.FP16ToFloat32(mcf.Float32ToFP16(scales[b]))
		cb := mcf.FP16ToFloat32(mcf.Float32ToFP16(codebook[s*16+int(indices[row*cols+c])]))
		out[c] = scale * cb
	}
	return out
}

func TestRowToQuantUQ4(t *testing.T) {
	const rows, cols = 3, 96
	payload, codebook, scales, indices := buildUQ4Payload(rows, cols)
	verifyPayloadSize(t, payload, rows, cols, mcf.DTypeUQ4)

	w := Mat{R: rows, C: cols, Stride: cols, DType: mcf.DTypeUQ4, Raw: payload}
	row := make([]float32, cols)
	for r := 0; r < rows; r++ {
		w.RowTo(row, r)
		want := uqReference(rows, cols, codebook, scales, indices, r)
		for c := 0; c < cols; c++ {
			if math.Abs(float64(row[c]-want[c])) > 1e-6 {
				t.Fatalf("row %d col %d: got %v want %v", r, c, row[c], want[c])
			}
		}
	}
}

func TestMatVecUQ4(t *testing.T) {
	const rows, cols = 64, 256
	payload, codebook, scales, indices := buildUQ4Payload(rows, cols)
	w := Mat{R: rows, C: cols, Stride: cols, DType: mcf.DTypeUQ4, Raw: payload}
	x := make([]float32, cols)
	for i := range x {
		x[i] = float32((i%13)-6) * 0.11
	}
	dst := make([]float32, rows)
	MatVec(dst, &w, x)

	// Reference: dequantized rows dotted with x.
	for r := 0; r < rows; r++ {
		want := uqReference(rows, cols, codebook, scales, indices, r)
		var sum float32
		for c := 0; c < cols; c++ {
			sum += want[c] * x[c]
		}
		if math.Abs(float64(dst[r]-sum)) > 1e-4*math.Max(1, math.Abs(float64(sum))) {
			t.Fatalf("dst[%d]: got %v want %v", r, dst[r], sum)
		}
	}
}

func BenchmarkMatVecUQ4(b *testing.B) {
	const rows, cols = 512, 512
	payload, _, _, _ := buildUQ4Payload(rows, cols)
	w := Mat{R: rows, C: cols, Stride: cols, DType: mcf.DTypeUQ4, Raw: payload}
	x := make([]float32, cols)
	for i := range x {
		x[i] = float32((i%13)-6) * 0.11
	}
	dst := make([]float32, rows)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MatVec(dst, &w, x)
	}
}
