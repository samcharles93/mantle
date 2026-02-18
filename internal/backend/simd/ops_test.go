package simd

import "testing"

func TestSigmoidValues(t *testing.T) {
	tests := []struct {
		x    float32
		want float32
	}{
		{x: 0, want: 0.5},
		{x: 2, want: 0.8807971},
		{x: -2, want: 0.1192029},
	}
	const tol = 1e-5
	for _, tt := range tests {
		got := Sigmoid(tt.x)
		if got < tt.want-tol || got > tt.want+tol {
			t.Fatalf("sigmoid(%v)=%v want %v±%v", tt.x, got, tt.want, tol)
		}
	}
}

func TestSiluMatchesDefinition(t *testing.T) {
	vals := []float32{-4, -1, 0, 1, 4}
	const tol = 1e-6
	for _, v := range vals {
		got := Silu(v)
		want := v * Sigmoid(v)
		if got < want-tol || got > want+tol {
			t.Fatalf("silu(%v)=%v want %v±%v", v, got, want, tol)
		}
	}
}

func BenchmarkSigmoid(b *testing.B) {
	x := float32(0.5)
	for i := 0; i < b.N; i++ {
		x += Sigmoid(x)
	}
	_ = x
}

func BenchmarkSilu(b *testing.B) {
	x := float32(0.5)
	for i := 0; i < b.N; i++ {
		x += Silu(x)
	}
	_ = x
}
