package simd

import (
	"runtime"
)

type gemmTask struct {
	C, A, B     *Mat
	alpha, beta float32
	rs, re      int
	cfg         GemmConfig
	done        chan struct{}
}

type gemmPool struct {
	size      int
	tasks     chan gemmTask
	doneSlots chan chan struct{}
}

func newGemmPool() *gemmPool {
	size := max(runtime.GOMAXPROCS(0), 1)
	p := &gemmPool{
		size:      size,
		tasks:     make(chan gemmTask, size*2),
		doneSlots: make(chan chan struct{}, size),
	}
	for range size {
		p.doneSlots <- make(chan struct{}, 1)
	}
	for range size {
		packB := make([]float32, maxTileK*maxTileN)
		go func(packB []float32) {
			for task := range p.tasks {
				gemmRangeRows(task.cfg, task.C, task.A, task.B, task.alpha, task.beta, task.rs, task.re, packB)
				task.done <- struct{}{}
			}
		}(packB)
	}
	return p
}

var gemmWorkPool = newGemmPool()

// GemmPar computes the matrix product C = alpha*A*B + beta*C using a
// blocked algorithm and parallelising across ranges of output rows.
func GemmPar(cfg GemmConfig, C, A, B *Mat, alpha, beta float32, workers int) {
	if A.C != B.R || C.R != A.R || C.C != B.C {
		panic("gemm: dimension mismatch")
	}

	if C.R == 0 || C.C == 0 {
		return
	}

	if workers <= 0 {
		workers = runtime.GOMAXPROCS(0)
	}
	if workers > C.R {
		workers = C.R
	}
	if workers <= 1 {
		gemmRangeRows(cfg, C, A, B, alpha, beta, 0, C.R, nil)
		return
	}
	if workers > gemmWorkPool.size {
		workers = gemmWorkPool.size
	}

	chunk := (C.R + workers - 1) / workers
	done := <-gemmWorkPool.doneSlots

	for w := 0; w < workers; w++ {
		rs := w * chunk
		re := min(rs+chunk, C.R)

		gemmWorkPool.tasks <- gemmTask{
			C:     C,
			A:     A,
			B:     B,
			alpha: alpha,
			beta:  beta,
			rs:    rs,
			re:    re,
			cfg:   cfg,
			done:  done,
		}
	}

	for range workers {
		<-done
	}
	gemmWorkPool.doneSlots <- done
}
