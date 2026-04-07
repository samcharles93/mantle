package inference

import "testing"

func TestTopGenerationDebugLogits(t *testing.T) {
	got := topGenerationDebugLogits([]float32{1.5, 3.0, 2.0, 3.0, -1.0}, 3)
	want := []generationDebugLogit{
		{id: 1, val: 3.0},
		{id: 3, val: 3.0},
		{id: 2, val: 2.0},
	}
	if len(got) != len(want) {
		t.Fatalf("len=%d want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got[%d]=%v want %v", i, got[i], want[i])
		}
	}
}
