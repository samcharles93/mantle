//go:build cuda

package cuda

import (
	"math"
	"testing"

	model "github.com/samcharles93/mantle/internal/backend/core"
	"github.com/samcharles93/mantle/pkg/mcf"
)

// TestUQDequantToF16 checks the CUDA dequant-mode decode of a uq* tensor
// (decodeQuantToF16) matches the CPU RowTo reference.
func TestUQDequantToF16(t *testing.T) {
	if !cudaAvailable(t) {
		return
	}
	const rows, cols = 4, 96
	bits := 4
	cbElems := 1 << bits
	blocksPerRow := cols / 32
	totalBlocks := rows * blocksPerRow
	superCount := rows * ((blocksPerRow + 7) / 8)

	codebook := make([]float32, superCount*cbElems)
	for i := range codebook {
		codebook[i] = float32(i%cbElems)*0.125 - 1
	}
	scales := make([]float32, totalBlocks)
	for i := range scales {
		scales[i] = 0.02 * float32((i%7)+1)
	}
	indices := make([]uint8, rows*cols)
	for i := range indices {
		indices[i] = uint8(i % cbElems)
	}
	payload, err := mcf.EncodeUQPayload([]uint64{rows, cols}, mcf.DTypeUQ4, codebook, scales, indices)
	if err != nil {
		t.Fatal(err)
	}
	w := &model.Mat{R: rows, C: cols, Stride: cols, DType: mcf.DTypeUQ4, Raw: payload}

	f16, err := decodeQuantToF16(w)
	if err != nil {
		t.Fatalf("decodeQuantToF16: %v", err)
	}
	if len(f16) != rows*cols {
		t.Fatalf("f16 len %d want %d", len(f16), rows*cols)
	}

	row := make([]float32, cols)
	for r := 0; r < rows; r++ {
		w.RowTo(row, r)
		for c := 0; c < cols; c++ {
			got := mcf.FP16ToFloat32(f16[r*cols+c])
			want := row[c]
			// f16 storage rounds the decoded value; allow 1 ulp-ish slack.
			if math.Abs(float64(got-want)) > 1e-3*math.Max(1, math.Abs(float64(want))) {
				t.Fatalf("row %d col %d: f16 %v vs row %v", r, c, got, want)
			}
		}
	}
}
