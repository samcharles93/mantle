package simd

import (
	"fmt"
	"testing"
)

func matVecNaive(dst []float32, w *Mat, x []float32) {
	for i := 0; i < w.R; i++ {
		row := w.Data[i*w.Stride : i*w.Stride+w.C]
		var sum float32
		for j := 0; j < w.C; j++ {
			sum += row[j] * x[j]
		}
		dst[i] = sum
	}
}

func BenchmarkMatVecNaive(b *testing.B) {
	r, c := 2048, 2048
	w := NewMat(r, c)
	x := make([]float32, c)
	dst := make([]float32, r)

	for b.Loop() {
		matVecNaive(dst, &w, x)
	}
}

func BenchmarkMatVecParWG(b *testing.B) {
	r, c := 2048, 2048
	w := NewMat(r, c)
	x := make([]float32, c)
	dst := make([]float32, r)

	for b.Loop() {
		MatVec(dst, &w, x)
	}
}

func BenchmarkMatVecPool(b *testing.B) {
	r, c := 2048, 2048
	w := NewMat(r, c)
	x := make([]float32, c)
	dst := make([]float32, r)

	for b.Loop() {
		MatVec(dst, &w, x)
	}
}

// SIMD vs Scalar comparison benchmarks

func BenchmarkMatVecPoolSIMD(b *testing.B) {
	r, c := 2048, 2048
	w := NewMat(r, c)
	x := make([]float32, c)
	dst := make([]float32, r)

	b.ResetTimer()
	for b.Loop() {
		matVecRangeF32SIMD(dst, &w, x, 0, r)
	}
}

func BenchmarkMatVecPoolScalar(b *testing.B) {
	r, c := 2048, 2048
	w := NewMat(r, c)
	x := make([]float32, c)
	dst := make([]float32, r)

	b.ResetTimer()
	for b.Loop() {
		matVecRangeF32Scalar(dst, &w, x, 0, r)
	}
}

func BenchmarkMatVecScalable(b *testing.B) {
	sizes := []struct{ r, c int }{
		{128, 128},
		{512, 512},
		{2048, 2048},
		{4096, 1024},
	}
	for _, size := range sizes {
		b.Run(fmt.Sprintf("%dx%d", size.r, size.c), func(b *testing.B) {
			w := NewMat(size.r, size.c)
			x := make([]float32, size.c)
			dst := make([]float32, size.r)

			b.ResetTimer()
			for b.Loop() {
				MatVec(dst, &w, x)
			}
		})
	}
}

func TestEnsureWorkerBuffers(t *testing.T) {
	w := &matVecWorker{}
	qbuf, scales := w.ensureWorkerBuffers(64, 16)
	if len(qbuf) != 64 || len(scales) != 16 {
		t.Errorf("expected buffers of size 64 and 16, got %d %d", len(qbuf), len(scales))
	}
	// Reuse
	qbuf2, scales2 := w.ensureWorkerBuffers(32, 8)
	if len(qbuf2) != 32 || len(scales2) != 8 {
		t.Errorf("expected buffers of size 32 and 8, got %d %d", len(qbuf2), len(scales2))
	}
	if cap(qbuf2) < 64 || cap(scales2) < 16 {
		t.Errorf("expected reuse, but cap too small: %d %d", cap(qbuf2), cap(scales2))
	}
	// Expand
	qbuf3, scales3 := w.ensureWorkerBuffers(128, 32)
	if len(qbuf3) != 128 || len(scales3) != 32 {
		t.Errorf("expected buffers of size 128 and 32, got %d %d", len(qbuf3), len(scales3))
	}
}

func BenchmarkGemmParSIMD(b *testing.B) {
	const m, k, n = 256, 256, 256
	A := NewMat(m, k)
	B := NewMat(k, n)
	C := NewMat(m, n)

	b.ResetTimer()
	for b.Loop() {
		blockUpdateAlpha1SIMD(C.Data, A.Data, B.Data, C.Stride, A.Stride, B.Stride, 0, m, 0, n, 0, k)
	}
}

func BenchmarkGemmParScalar(b *testing.B) {
	const m, k, n = 256, 256, 256
	A := NewMat(m, k)
	B := NewMat(k, n)
	C := NewMat(m, n)

	b.ResetTimer()
	for b.Loop() {
		blockUpdateAlpha1Scalar(C.Data, A.Data, B.Data, C.Stride, A.Stride, B.Stride, 0, m, 0, n, 0, k)
	}
}
