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

func gemmNaiveAlphaBeta(C, A, B *Mat, alpha, beta float32) {
	for i := 0; i < A.R; i++ {
		for j := 0; j < B.C; j++ {
			var sum float32
			for kk := 0; kk < A.C; kk++ {
				sum += A.Row(i)[kk] * B.Row(kk)[j]
			}
			C.Row(i)[j] = alpha*sum + beta*C.Row(i)[j]
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
	if maxAbs := maxAbsDiff(C0.Data, C1.Data); maxAbs > 1e-3 {
		t.Fatalf("max absolute difference too large: %g", maxAbs)
	}
}

func TestGemmParAlphaBetaNonSquare(t *testing.T) {
	A := NewMat(37, 53)
	B := NewMat(53, 61)
	C0 := NewMat(37, 61)
	C1 := NewMat(37, 61)
	FillRand(&A, 5)
	FillRand(&B, 6)
	FillRand(&C0, 7)
	copy(C1.Data, C0.Data)

	alpha := float32(0.75)
	beta := float32(0.6)
	gemmNaiveAlphaBeta(&C0, &A, &B, alpha, beta)
	GemmPar(&C1, &A, &B, alpha, beta, 3)

	if maxAbs := maxAbsDiff(C0.Data, C1.Data); maxAbs > 1e-3 {
		t.Fatalf("max absolute difference too large: %g", maxAbs)
	}
}

func TestGemmParTileBoundaries(t *testing.T) {
	cases := []struct {
		m, k, n int
	}{
		{tileM - 1, tileK - 1, tileN - 1},
		{tileM, tileK, tileN},
		{tileM + 1, tileK + 1, tileN + 1},
		{tileM - 1, tileK + 1, tileN + 1},
		{tileM + 1, tileK - 1, tileN - 1},
	}

	for _, tc := range cases {
		if tc.m <= 0 || tc.k <= 0 || tc.n <= 0 {
			continue
		}
		A := NewMat(tc.m, tc.k)
		B := NewMat(tc.k, tc.n)
		C0 := NewMat(tc.m, tc.n)
		C1 := NewMat(tc.m, tc.n)
		FillRand(&A, 11)
		FillRand(&B, 12)

		gemmNaiveAlphaBeta(&C0, &A, &B, 1, 0)
		GemmPar(&C1, &A, &B, 1, 0, 2)

		if maxAbs := maxAbsDiff(C0.Data, C1.Data); maxAbs > 1e-3 {
			t.Fatalf("tile case m=%d k=%d n=%d max diff %g", tc.m, tc.k, tc.n, maxAbs)
		}
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

func BenchmarkGemmParTileKSweep(b *testing.B) {
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

	origK := tileK
	b.Cleanup(func() {
		tileK = origK
	})

	for _, tk := range []int{4, 8, 12, 16, 24, 32, 48, 64} {
		if tk > maxTileK {
			continue
		}
		b.Run("K"+itoa(tk), func(b *testing.B) {
			tileK = tk
			b.ReportAllocs()
			b.ResetTimer()
			for b.Loop() {
				GemmPar(&C, &A, &B, 1, 0, 0)
			}
		})
	}
}

func itoa(v int) string {
	if v == 0 {
		return "0"
	}
	var buf [20]byte
	i := len(buf)
	n := v
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	return string(buf[i:])
}
