package simd

import "testing"

func TestDeltaNetSplitQKVGroupedLayout(t *testing.T) {
	mixed := []float32{
		1, 2, 10, 20,
		3, 4, 30, 40,
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
