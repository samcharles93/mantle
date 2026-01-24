package tensor

import (
	"runtime"
	"sync"
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
