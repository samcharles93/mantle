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

func TestSiluAndMul(t *testing.T) {
	// Shape mirrors the kernel benchmark idea: last dimension split in half.
	d := 256
	x := make([]float32, d*2)
	for i := range d {
		x[i] = float32(i%7) - 3
		x[d+i] = float32((i%11)+1) / 11
	}
	dst := make([]float32, d)
	SiluAndMul(dst, x)

	const tol = 1e-6
	for i := range d {
		want := Silu(x[i]) * x[d+i]
		got := dst[i]
		if got < want-tol || got > want+tol {
			t.Fatalf("silu_and_mul[%d]=%v want %v±%v", i, got, want, tol)
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

func BenchmarkSiluAndMulSmall(b *testing.B)  { benchSiluAndMul(b, 128, 512) }
func BenchmarkSiluAndMulMedium(b *testing.B) { benchSiluAndMul(b, 512, 1024) }
func BenchmarkSiluAndMulLarge(b *testing.B)  { benchSiluAndMul(b, 1024, 2048) }

func benchSiluAndMul(b *testing.B, rows, width int) {
	if width%2 != 0 {
		b.Fatalf("width must be even")
	}
	x := make([]float32, rows*width)
	for i := range x {
		x[i] = float32((i%23)-11) / 7
	}
	dst := make([]float32, rows*(width/2))

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for r := range rows {
			row := x[r*width : (r+1)*width]
			out := dst[r*(width/2) : (r+1)*(width/2)]
			SiluAndMul(out, row)
		}
	}
}
