package tensor

import (
	"math"
	"runtime"
	"sync"
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

func matVecParWaitGroup(dst []float32, w *Mat, x []float32) {
	workers := min(runtime.GOMAXPROCS(0), w.R)
	var wg sync.WaitGroup
	chunk := (w.R + workers - 1) / workers
	for i := range workers {
		rs := i * chunk
		re := min(rs+chunk, w.R)
		if rs >= re {
			break
		}
		wg.Add(1)
		go func(rs, re int) {
			defer wg.Done()
			for r := rs; r < re; r++ {
				row := w.Data[r*w.Stride : r*w.Stride+w.C]
				var sum float32
				for j := 0; j < w.C; j++ {
					sum += row[j] * x[j]
				}
				dst[r] = sum
			}
		}(rs, re)
	}
	wg.Wait()
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
		matVecParWaitGroup(dst, &w, x)
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
