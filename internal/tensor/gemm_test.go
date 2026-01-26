package tensor

import (
	"math"
	"testing"
)

// gemmNaive performs an unoptimised matrix multiplication: C = A * B.  C must
// be sized appropriately (C.R==A.R and C.C==B.C) and is overwritten.
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

// TestGemmParMatchesNaive verifies that the blocked parallel GEMM produces
// results equivalent to a naive implementation within a small tolerance.
func TestGemmParMatchesNaive(t *testing.T) {
	// Use sizes that are not multiples of the tile sizes to exercise all code paths.
	A := NewMat(50, 70)
	B := NewMat(70, 45)
	C0 := NewMat(50, 45)
	C1 := NewMat(50, 45)
	FillRand(&A, 1)
	FillRand(&B, 2)
	gemmNaive(&C0, &A, &B)
	GemmPar(&C1, &A, &B, 1, 0, 4)
	var maxAbs float64
	for i := range C0.Data {
		d := math.Abs(float64(C0.Data[i] - C1.Data[i]))
		if d > maxAbs {
			maxAbs = d
		}
	}
	if maxAbs > 1e-3 {
		t.Fatalf("max absolute difference too large: %g", maxAbs)
	}
}

// TestGemmParNoAllocs uses AllocsPerRun to ensure that GemmPar does not
// allocate heap memory when executed.  A small matrix is used to keep
// execution time reasonable.
func TestGemmParNoAllocs(t *testing.T) {
	A := NewMat(16, 16)
	B := NewMat(16, 16)
	C := NewMat(16, 16)
	FillRand(&A, 3)
	FillRand(&B, 4)
	allocs := testing.AllocsPerRun(100, func() {
		GemmPar(&C, &A, &B, 1, 0, 2)
	})
	if allocs != 0 {
		t.Fatalf("GemmPar allocated %v times (expected 0)", allocs)
	}
}

func BenchmarkGemmPar(b *testing.B) {
	// A moderately sized GEMM intended to be stable enough for iteration.
	// Keep it large enough to amortise goroutine overhead.
	const (
		m = 256
		k = 256
		n = 256
	)
	A := NewMat(m, k)
	B := NewMat(k, n)
	C := NewMat(m, n)
	FillRand(&A, 7)
	FillRand(&B, 8)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		GemmPar(&C, &A, &B, 1, 0, 0)
	}
}
