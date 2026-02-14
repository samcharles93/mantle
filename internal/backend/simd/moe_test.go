package simd

import "testing"

func TestSelectTopKWeights(t *testing.T) {
	sel := []float32{0.1, 0.9, 0.5}
	raw := []float32{0.2, 0.3, 0.4}
	idx := make([]int, 2)
	weights := make([]float32, 2)

	SelectTopK(sel, raw, 2, 2.0, idx, weights)

	if idx[0] != 1 || idx[1] != 2 {
		t.Fatalf("unexpected topk indices: got %v want [1 2]", idx)
	}
	const tol = 1e-6
	want0 := float32(0.3 / 0.7 * 2.0)
	want1 := float32(0.4 / 0.7 * 2.0)
	if weights[0] < want0-tol || weights[0] > want0+tol {
		t.Fatalf("unexpected weight[0]: got %v want %v±%v", weights[0], want0, tol)
	}
	if weights[1] < want1-tol || weights[1] > want1+tol {
		t.Fatalf("unexpected weight[1]: got %v want %v±%v", weights[1], want1, tol)
	}
}

func TestSelectTopKTieBreaksByIndex(t *testing.T) {
	sel := []float32{1, 1, 0.5}
	raw := []float32{0.2, 0.9, 0.1}
	idx := make([]int, 2)
	weights := make([]float32, 2)

	SelectTopK(sel, raw, 2, 1.0, idx, weights)

	// sel ties at 0 and 1; we prefer the smaller index first.
	if idx[0] != 0 || idx[1] != 1 {
		t.Fatalf("unexpected tie-break ordering: got %v want [0 1]", idx)
	}
}

func BenchmarkSelectTopK(b *testing.B) {
	const n = 128
	sel := make([]float32, n)
	raw := make([]float32, n)
	for i := range n {
		sel[i] = float32(i%7) / 7.0
		raw[i] = float32(i%11) / 11.0
	}
	idx := make([]int, 8)
	weights := make([]float32, 8)

	for i := 0; i < b.N; i++ {
		SelectTopK(sel, raw, 8, 2.0, idx, weights)
	}
}
