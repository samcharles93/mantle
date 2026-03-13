package simd

import "testing"

func TestDeltaNetSplitQKVGroupedLayout(t *testing.T) {
	// Flat contiguous layout: [Q0, Q1 | K0, K1 | V0, V1, V2, V3]
	mixed := []float32{
		1, 3, 2, 4,
		10, 20, 30, 40,
	}
	q := make([]float32, 2)
	k := make([]float32, 2)
	v := make([]float32, 4)

	deltaNetSplitQKV(q, k, v, mixed, 2, 4, 1, 1)

	if got, want := q, []float32{1, 3}; got[0] != want[0] || got[1] != want[1] {
		t.Fatalf("q=%v want %v", got, want)
	}
	if got, want := k, []float32{2, 4}; got[0] != want[0] || got[1] != want[1] {
		t.Fatalf("k=%v want %v", got, want)
	}
	wantV := []float32{10, 20, 30, 40}
	for i := range wantV {
		if v[i] != wantV[i] {
			t.Fatalf("v[%d]=%v want %v", i, v[i], wantV[i])
		}
	}
}
