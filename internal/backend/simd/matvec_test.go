package simd

import (
	"fmt"
	"math"
	"testing"

	"github.com/samcharles93/mantle/pkg/mcf"
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
	FillRand(&w, 1)

	for b.Loop() {
		matVecNaive(dst, &w, x)
	}
}

func BenchmarkMatVecParWG(b *testing.B) {
	r, c := 2048, 2048
	w := NewMat(r, c)
	x := make([]float32, c)
	dst := make([]float32, r)
	FillRand(&w, 1)

	for b.Loop() {
		MatVec(dst, &w, x)
	}
}

func BenchmarkMatVecPool(b *testing.B) {
	r, c := 2048, 2048
	w := NewMat(r, c)
	x := make([]float32, c)
	dst := make([]float32, r)
	FillRand(&w, 1)

	for b.Loop() {
		MatVec(dst, &w, x)
	}
}

func TestMatVecRawBF16(t *testing.T) {
	r, c := 128, 192
	w := NewMat(r, c)
	x := make([]float32, c)
	dstF32 := make([]float32, r)
	dstRaw := make([]float32, r)
	FillRand(&w, 42)

	raw := encodeBF16Raw(w.Data)
	wRaw, err := NewMatFromRaw(r, c, mcf.DTypeBF16, raw)
	if err != nil {
		t.Fatalf("NewMatFromRaw bf16: %v", err)
	}

	MatVec(dstF32, &w, x)
	MatVec(dstRaw, &wRaw, x)

	// bf16 is coarse; allow small relative error.
	for i := range dstF32 {
		a := dstF32[i]
		b := dstRaw[i]
		if !closeEnough(a, b, 5e-2) {
			t.Fatalf("bf16 mismatch at %d: f32=%g raw=%g", i, a, b)
		}
	}
}

func TestMatVecRawF16(t *testing.T) {
	r, c := 128, 192
	w := NewMat(r, c)
	x := make([]float32, c)
	dstF32 := make([]float32, r)
	dstRaw := make([]float32, r)
	FillRand(&w, 7)

	raw := encodeFP16Raw(w.Data)
	wRaw, err := NewMatFromRaw(r, c, mcf.DTypeF16, raw)
	if err != nil {
		t.Fatalf("NewMatFromRaw f16: %v", err)
	}

	MatVec(dstF32, &w, x)
	MatVec(dstRaw, &wRaw, x)

	for i := range dstF32 {
		a := dstF32[i]
		b := dstRaw[i]
		if !closeEnough(a, b, 2e-2) {
			t.Fatalf("f16 mismatch at %d: f32=%g raw=%g", i, a, b)
		}
	}
}

func BenchmarkMatVecPoolBF16(b *testing.B) {
	benchMatVecPoolBF16(b, 2048, 2048)
}

func BenchmarkMatVecPoolF16(b *testing.B) {
	r, c := 2048, 2048
	w := NewMat(r, c)
	x := make([]float32, c)
	dst := make([]float32, r)
	FillRand(&w, 1)

	raw := encodeFP16Raw(w.Data)
	wRaw, err := NewMatFromRaw(r, c, mcf.DTypeF16, raw)
	if err != nil {
		b.Fatalf("NewMatFromRaw f16: %v", err)
	}

	b.ResetTimer()
	for b.Loop() {
		MatVec(dst, &wRaw, x)
	}
}

func BenchmarkMatVecPoolBF16_2560x2560(b *testing.B) {
	benchMatVecPoolBF16(b, 2560, 2560)
}

func BenchmarkMatVecPoolBF16_9728x2560(b *testing.B) {
	benchMatVecPoolBF16(b, 9728, 2560)
}

func benchMatVecPoolBF16(b *testing.B, r, c int) {
	data := make([]float32, r*c)
	w := Mat{
		R:      r,
		C:      c,
		Stride: c,
		DType:  mcf.DTypeF32,
		Data:   data,
	}
	FillRand(&w, 1)

	raw := encodeBF16Raw(data)
	wRaw, err := NewMatFromRaw(r, c, mcf.DTypeBF16, raw)
	if err != nil {
		b.Fatalf("NewMatFromRaw bf16: %v", err)
	}

	x := make([]float32, c)
	dst := make([]float32, r)
	b.ResetTimer()
	for b.Loop() {
		MatVec(dst, &wRaw, x)
	}
}

func closeEnough(a, b float32, rel float64) bool {
	da := float64(a)
	db := float64(b)
	diff := math.Abs(da - db)
	scale := math.Max(1.0, math.Max(math.Abs(da), math.Abs(db)))
	return diff <= rel*scale
}

// SIMD vs Scalar comparison benchmarks

func BenchmarkMatVecPoolSIMD(b *testing.B) {
	r, c := 2048, 2048
	w := NewMat(r, c)
	x := make([]float32, c)
	dst := make([]float32, r)
	FillRand(&w, 1)

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
	FillRand(&w, 1)

	b.ResetTimer()
	for b.Loop() {
		matVecRangeF32Scalar(dst, &w, x, 0, r)
	}
}

func BenchmarkMatVecPoolBF16SIMD(b *testing.B) {
	r, c := 2048, 2048
	data := make([]float32, r*c)
	w := Mat{
		R:      r,
		C:      c,
		Stride: c,
		DType:  mcf.DTypeF32,
		Data:   data,
	}
	FillRand(&w, 1)

	raw := encodeBF16Raw(data)
	wRaw, err := NewMatFromRaw(r, c, mcf.DTypeBF16, raw)
	if err != nil {
		b.Fatalf("NewMatFromRaw bf16: %v", err)
	}

	x := make([]float32, c)
	dst := make([]float32, r)
	b.ResetTimer()
	for b.Loop() {
		matVecRangeBF16SIMD(dst, &wRaw, x, 0, r)
	}
}

func BenchmarkMatVecPoolBF16Scalar(b *testing.B) {
	benchMatVecPoolBF16Scalar(b, 2048, 2048)
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
			FillRand(&w, 1)

			b.ResetTimer()
			for b.Loop() {
				MatVec(dst, &w, x)
			}
		})
	}
}

func BenchmarkMatVecMixedPrecision(b *testing.B) {
	r, c := 2048, 2048
	// FP32 matvec
	wF32 := NewMat(r, c)
	xF32 := make([]float32, c)
	dstF32 := make([]float32, r)
	FillRand(&wF32, 1)
	// BF16 matvec
	raw := encodeBF16Raw(wF32.Data)
	wBF16, _ := NewMatFromRaw(r, c, mcf.DTypeBF16, raw)
	dstBF16 := make([]float32, r)

	b.ResetTimer()
	for b.Loop() {
		MatVec(dstF32, &wF32, xF32)
		MatVec(dstBF16, &wBF16, xF32)
	}
}

func benchMatVecPoolBF16Scalar(b *testing.B, r, c int) {
	data := make([]float32, r*c)
	w := Mat{
		R:      r,
		C:      c,
		Stride: c,
		DType:  mcf.DTypeF32,
		Data:   data,
	}
	FillRand(&w, 1)

	raw := encodeBF16Raw(data)
	wRaw, err := NewMatFromRaw(r, c, mcf.DTypeBF16, raw)
	if err != nil {
		b.Fatalf("NewMatFromRaw bf16: %v", err)
	}

	x := make([]float32, c)
	dst := make([]float32, r)
	b.ResetTimer()
	for b.Loop() {
		matVecRangeBF16Scalar(dst, &wRaw, x, 0, r)
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
	FillRand(&A, 7)
	FillRand(&B, 8)

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
	FillRand(&A, 7)
	FillRand(&B, 8)

	b.ResetTimer()
	for b.Loop() {
		blockUpdateAlpha1Scalar(C.Data, A.Data, B.Data, C.Stride, A.Stride, B.Stride, 0, m, 0, n, 0, k)
	}
}
