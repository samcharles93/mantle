package tensor

import "runtime"

// Tuned for the benchmark shape (256^3) and typical L1/L2 sizes.
// tileN is kept moderate to reduce register pressure in the inner loop.
const (
	tileM = 64
	tileN = 64
	tileK = 32
)

type gemmTask struct {
	C, A, B     *Mat
	alpha, beta float32
	rs, re      int
	done        chan struct{}
}

type gemmPool struct {
	size      int
	tasks     chan gemmTask
	doneSlots chan chan struct{}
}

func newGemmPool() *gemmPool {
	size := runtime.GOMAXPROCS(0)
	if size < 1 {
		size = 1
	}
	p := &gemmPool{
		size: size,
		// Buffered to reduce per-call synchronization on task dispatch.
		tasks:     make(chan gemmTask, size*2),
		doneSlots: make(chan chan struct{}, size),
	}
	for i := 0; i < size; i++ {
		// Buffered so workers can signal completion without blocking.
		p.doneSlots <- make(chan struct{}, size)
	}
	for w := 0; w < size; w++ {
		go func() {
			for task := range p.tasks {
				gemmRangeRows(task.C, task.A, task.B, task.alpha, task.beta, task.rs, task.re)
				task.done <- struct{}{}
			}
		}()
	}
	return p
}

var gemmWorkPool = newGemmPool()

// GemmPar computes the matrix product C = alpha*A*B + beta*C using a
// blocked algorithm and parallelising across ranges of output rows.
//
// Panics if dimensions do not conform (A.C==B.R, C.R==A.R, C.C==B.C).
func GemmPar(C, A, B *Mat, alpha, beta float32, workers int) {
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
		gemmRangeRows(C, A, B, alpha, beta, 0, C.R)
		return
	}
	if workers > gemmWorkPool.size {
		workers = gemmWorkPool.size
	}

	// Even row partitioning.
	chunk := (C.R + workers - 1) / workers

	done := <-gemmWorkPool.doneSlots
	for w := 0; w < workers; w++ {
		rs := w * chunk
		re := rs + chunk
		if re > C.R {
			re = C.R
		}
		gemmWorkPool.tasks <- gemmTask{
			C:     C,
			A:     A,
			B:     B,
			alpha: alpha,
			beta:  beta,
			rs:    rs,
			re:    re,
			done:  done,
		}
	}
	for i := 0; i < workers; i++ {
		<-done
	}
	gemmWorkPool.doneSlots <- done
}

// gemmRangeRows performs a blocked GEMM on a contiguous range of rows of C.
// It updates rows [rs, re) of C using the corresponding rows of A.
func gemmRangeRows(C, A, B *Mat, alpha, beta float32, rs, re int) {
	// Fast path: common inference case alpha=1, beta=0.
	if alpha == 1 && beta == 0 {
		gemmRangeRowsAlpha1Beta0(C, A, B, rs, re)
		return
	}

	// Scale/zero C range according to beta.
	if beta == 0 {
		cStride := C.Stride
		n := C.C
		for i := rs; i < re; i++ {
			base := i * cStride
			row := C.Data[base : base+n]
			for j := range row {
				row[j] = 0
			}
		}
	} else if beta != 1 {
		cStride := C.Stride
		n := C.C
		for i := rs; i < re; i++ {
			base := i * cStride
			row := C.Data[base : base+n]
			for j := range row {
				row[j] *= beta
			}
		}
	}

	// m := A.R
	// _ = m
	n := B.C
	k := A.C

	aStride := A.Stride
	bStride := B.Stride
	cStride := C.Stride

	// Blocked loops.
	for i0 := rs; i0 < re; i0 += tileM {
		iMax := min(i0+tileM, re)
		for k0 := 0; k0 < k; k0 += tileK {
			kMax := min(k0+tileK, k)
			for j0 := 0; j0 < n; j0 += tileN {
				jMax := min(j0+tileN, n)
				blockUpdateGeneric(C.Data, A.Data, B.Data, cStride, aStride, bStride, alpha, i0, iMax, j0, jMax, k0, kMax)
			}
		}
	}
}

func gemmRangeRowsAlpha1Beta0(C, A, B *Mat, rs, re int) {
	// beta==0: clear C range once, then accumulate.
	cStride := C.Stride
	n := C.C
	cData := C.Data

	// Zero contiguous segments with the built-in clear for better codegen.
	for i := rs; i < re; i++ {
		base := i * cStride
		clear(cData[base : base+n])
	}

	k := A.C
	aStride := A.Stride
	bStride := B.Stride
	aData := A.Data
	bData := B.Data

	// Blocked loops.
	for i0 := rs; i0 < re; i0 += tileM {
		iMax := min(i0+tileM, re)
		for k0 := 0; k0 < k; k0 += tileK {
			kMax := min(k0+tileK, k)
			for j0 := 0; j0 < n; j0 += tileN {
				jMax := min(j0+tileN, n)
				blockUpdateAlpha1(cData, aData, bData, cStride, aStride, bStride, i0, iMax, j0, jMax, k0, kMax)
			}
		}
	}
}

func blockUpdateGeneric(cData, aData, bData []float32, cStride, aStride, bStride int, alpha float32, i0, iMax, j0, jMax, k0, kMax int) {
	width := jMax - j0
	for i := i0; i < iMax; i++ {
		aRow := aData[i*aStride:]
		cOff := i*cStride + j0
		cRow := cData[cOff : cOff+width]

		for kk := k0; kk < kMax; kk++ {
			aik := aRow[kk] * alpha
			bOff := kk*bStride + j0
			bRow := bData[bOff : bOff+width]

			j := 0
			for ; j+7 < width; j += 8 {
				cRow[j+0] += aik * bRow[j+0]
				cRow[j+1] += aik * bRow[j+1]
				cRow[j+2] += aik * bRow[j+2]
				cRow[j+3] += aik * bRow[j+3]
				cRow[j+4] += aik * bRow[j+4]
				cRow[j+5] += aik * bRow[j+5]
				cRow[j+6] += aik * bRow[j+6]
				cRow[j+7] += aik * bRow[j+7]
			}
			for ; j < width; j++ {
				cRow[j] += aik * bRow[j]
			}
		}
	}
}

func blockUpdateAlpha1(cData, aData, bData []float32, cStride, aStride, bStride int, i0, iMax, j0, jMax, k0, kMax int) {
	width := jMax - j0
	for i := i0; i < iMax; i++ {
		aRow := aData[i*aStride:]
		cOff := i*cStride + j0
		cRow := cData[cOff : cOff+width]

		for kk := k0; kk < kMax; kk++ {
			aik := aRow[kk]
			bOff := kk*bStride + j0
			bRow := bData[bOff : bOff+width]

			// Unroll by 4: reduces loop overhead while keeping register pressure lower than x8.
			j := 0
			for ; j+3 < width; j += 4 {
				cRow[j+0] += aik * bRow[j+0]
				cRow[j+1] += aik * bRow[j+1]
				cRow[j+2] += aik * bRow[j+2]
				cRow[j+3] += aik * bRow[j+3]
			}
			for ; j < width; j++ {
				cRow[j] += aik * bRow[j]
			}
		}
	}
}
