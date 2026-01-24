package tensor

import "runtime"

type matVecTask struct {
	dst    []float32
	w      *Mat
	x      []float32
	rs, re int
	done   chan struct{}
}

type matVecPool struct {
	size      int
	tasks     chan matVecTask
	doneSlots chan chan struct{}
}

var matVecWorkPool *matVecPool

func init() {
	matVecWorkPool = newMatVecPool()
}

func newMatVecPool() *matVecPool {
	size := runtime.GOMAXPROCS(0)
	if size < 1 {
		size = 1
	}
	p := &matVecPool{
		size:      size,
		tasks:     make(chan matVecTask, size*2),
		doneSlots: make(chan chan struct{}, size),
	}
	for i := 0; i < size; i++ {
		p.doneSlots <- make(chan struct{}, 1)
	}
	for i := 0; i < size; i++ {
		go func() {
			for task := range p.tasks {
				matVecRange(task.dst, task.w, task.x, task.rs, task.re)
				task.done <- struct{}{}
			}
		}()
	}
	return p
}

// MatVec computes dst = w * x where w is a matrix and x is a vector.
// It runs in parallel using a worker pool.
func MatVec(dst []float32, w *Mat, x []float32) {
	if w.R == 0 || w.C == 0 {
		return
	}
	
	workers := matVecWorkPool.size
	if workers > w.R {
		workers = w.R
	}

	if workers <= 1 {
		matVecRange(dst, w, x, 0, w.R)
		return
	}

	chunk := (w.R + workers - 1) / workers
	done := <-matVecWorkPool.doneSlots

	activeWorkers := 0
	for i := 0; i < workers; i++ {
		rs := i * chunk
		re := rs + chunk
		if re > w.R {
			re = w.R
		}
		if rs >= re {
			break
		}
		activeWorkers++
		matVecWorkPool.tasks <- matVecTask{
			dst:  dst,
			w:    w,
			x:    x,
			rs:   rs,
			re:   re,
			done: done,
		}
	}

	for i := 0; i < activeWorkers; i++ {
		<-done
	}
	matVecWorkPool.doneSlots <- done
}

func matVecRange(dst []float32, w *Mat, x []float32, rs, re int) {
	for i := rs; i < re; i++ {
		row := w.Data[i*w.Stride : i*w.Stride+w.C]
		var sum float32
		j := 0
		for ; j+3 < w.C; j += 4 {
			sum += row[j]*x[j] + row[j+1]*x[j+1] + row[j+2]*x[j+2] + row[j+3]*x[j+3]
		}
		for ; j < w.C; j++ {
			sum += row[j] * x[j]
		}
		dst[i] = sum
	}
}
