package simd

import "testing"

func TestMambaDepthwiseConv(t *testing.T) {
	kernel := Mat{
		R:      2,
		C:      3,
		Stride: 3,
		Data: []float32{
			1, 2, 3,
			4, 5, 6,
		},
	}
	state := []float32{1, 2, 3, 4}
	in := []float32{5, 6}
	out := make([]float32, 2)

	mambaDepthwiseConv(out, in, &kernel, nil, state)

	if out[0] != 22 || out[1] != 64 {
		t.Fatalf("unexpected conv output: %v", out)
	}
	wantState := []float32{3, 4, 5, 6}
	for i := range state {
		if state[i] != wantState[i] {
			t.Fatalf("state mismatch at %d: got %v want %v", i, state, wantState)
		}
	}
}

func TestMambaScanSingleStep(t *testing.T) {
	ml := &MambaLayer{
		ALog:      []float32{float32(-0.69314718)}, // log(0.5)
		D:         []float32{6},
		HeadCount: 1,
		HeadDim:   1,
		DState:    2,
		Groups:    1,
		GroupSize: 1,
		SSMState:  []float32{1, 2},
	}
	x := []float32{7}
	dt := []float32{1}
	b := []float32{2, 3}
	c := []float32{4, 5}
	out := make([]float32, 1)

	mambaScan(out, ml, x, dt, b, c)

	wantOut := float32(211.4915)
	if diff := out[0] - wantOut; diff < -1e-3 || diff > 1e-3 {
		t.Fatalf("unexpected output: got %f want %f", out[0], wantOut)
	}
	wantState := []float32{14.6065, 22.2131}
	for i := range ml.SSMState {
		if diff := ml.SSMState[i] - wantState[i]; diff < -1e-3 || diff > 1e-3 {
			t.Fatalf("state[%d] = %f want %f", i, ml.SSMState[i], wantState[i])
		}
	}
}

func BenchmarkMambaScanSmall(b *testing.B) {
	ml := &MambaLayer{
		ALog:      make([]float32, 4),
		D:         make([]float32, 4),
		HeadCount: 4,
		HeadDim:   8,
		DState:    8,
		Groups:    1,
		GroupSize: 4,
		SSMState:  make([]float32, 4*8*8),
	}
	x := make([]float32, 32)
	dt := make([]float32, 4)
	bb := make([]float32, 8)
	cc := make([]float32, 8)
	out := make([]float32, 32)

	for i := range x {
		x[i] = float32(i%7) * 0.1
	}
	for i := range dt {
		dt[i] = 0.01
		ml.ALog[i] = -0.5
		ml.D[i] = 0.2
	}
	for i := range bb {
		bb[i] = float32(i%5) * 0.01
		cc[i] = float32(i%3) * 0.02
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mambaScan(out, ml, x, dt, bb, cc)
	}
}
