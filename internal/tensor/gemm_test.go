package tensor

import (
	"math"
	"testing"
)

func gemmNaive(C, A, B *Mat) {
	for i := 0; i < A.R; i++ {
		for j := 0; j < B.C; j++ {
			var sum float32
			for kk := 0; kk < A.C; kk++ {
				sum += A.Row(i)[kk] * B.Row(kk)[j]
			}
			C.Row(i)[j] = sum
		}
	}
}

func maxAbsDiff(a, b []float32) float64 {
	var maxAbs float64
	for i := range a {
		d := math.Abs(float64(a[i] - b[i]))
		if d > maxAbs {
			maxAbs = d
		}
	}
	return maxAbs
}

func TestGemmParMatchesNaive(t *testing.T) {
	A := NewMat(50, 70)
	B := NewMat(70, 45)
	C0 := NewMat(50, 45)
	C1 := NewMat(50, 45)

	FillRand(&A, 1)
	FillRand(&B, 2)

	gemmNaive(&C0, &A, &B)
	cfg := SelectGemmConfig(A.R, A.C, B.C)
	GemmPar(cfg, &C1, &A, &B, 1, 0, 4)

	if maxAbs := maxAbsDiff(C0.Data, C1.Data); maxAbs > 1e-3 {
		t.Fatalf("max abs diff %g", maxAbs)
	}
}

func TestGemmParNoAllocs(t *testing.T) {
	A := NewMat(16, 16)
	B := NewMat(16, 16)
	C := NewMat(16, 16)

	FillRand(&A, 3)
	FillRand(&B, 4)

	cfg := DefaultGemmConfig()
	allocs := testing.AllocsPerRun(100, func() {
		GemmPar(cfg, &C, &A, &B, 1, 0, 2)
	})

	if allocs != 0 {
		t.Fatalf("unexpected allocs: %v", allocs)
	}
}
