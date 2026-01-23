package tensor

import (
    "runtime"
)

// Tile sizes for blocked matrix multiplication.  These values were chosen
// empirically to provide a reasonable starting point for cache utilisation on
// modern CPUs.  They can be tuned per architecture for higher performance.
const (
    tileM = 64
    tileN = 64
    tileK = 32
)

type gemmTask struct {
    C, A, B       *Mat
    alpha, beta   float32
    rs, re        int
    done          chan struct{}
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
        size:      size,
        tasks:     make(chan gemmTask),
        doneSlots: make(chan chan struct{}, size),
    }
    for i := 0; i < size; i++ {
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
// blocked algorithm and parallelising across ranges of output rows.  All
// matrices must be stored in row‑major order.  If workers <= 0 then the
// number of logical CPUs (GOMAXPROCS) is used.  When beta == 0 the
// destination matrix C is overwritten; when beta == 1 the result is added
// to the existing contents of C; other values scale the original C.
//
// This routine panics if the matrix dimensions do not conform to the
// multiplication (A.C == B.R, C.R == A.R, C.C == B.C).
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

    // Determine chunk size per worker; distribute rows as evenly as possible.
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
// It updates rows [rs, re) of C using the corresponding rows of A.  The
// blocking parameters tileM, tileN and tileK divide the loops into cache‑
// friendly chunks.
func gemmRangeRows(C, A, B *Mat, alpha, beta float32, rs, re int) {
    // Scale or zero the portion of C according to beta.  This ensures we do
    // not accumulate stale data when beta==0.
    if beta == 0 {
        for i := rs; i < re; i++ {
            row := C.Row(i)
            for j := range row {
                row[j] = 0
            }
        }
    } else if beta != 1 {
        for i := rs; i < re; i++ {
            row := C.Row(i)
            for j := range row {
                row[j] *= beta
            }
        }
    }

    m, n, k := A.R, B.C, A.C
    _ = m
    // Blocked loops over i (rows), k (inner dimension) and j (columns).
    for i0 := rs; i0 < re; i0 += tileM {
        iMax := i0 + tileM
        if iMax > re {
            iMax = re
        }
        for k0 := 0; k0 < k; k0 += tileK {
            kMax := k0 + tileK
            if kMax > k {
                kMax = k
            }
            for j0 := 0; j0 < n; j0 += tileN {
                jMax := j0 + tileN
                if jMax > n {
                    jMax = n
                }
                blockUpdate(C, A, B, alpha, i0, iMax, j0, jMax, k0, kMax)
            }
        }
    }
}

// blockUpdate multiplies a block of A and B and accumulates into C.  It
// iterates over rows i in [i0, iMax), the shared dimension k in [k0, kMax)
// and columns j in [j0, jMax).  The loop order is chosen to minimise
// redundant loads and leverage cache locality.
func blockUpdate(C, A, B *Mat, alpha float32, i0, iMax, j0, jMax, k0, kMax int) {
    for i := i0; i < iMax; i++ {
        cRow := C.Row(i)
        aRow := A.Row(i)
        for kk := k0; kk < kMax; kk++ {
            aik := aRow[kk] * alpha
            bRow := B.Row(kk)
            // Unroll the inner j loop in multiples of 4 for marginal speed up
            j := j0
            for ; j+3 < jMax; j += 4 {
                cRow[j+0] += aik * bRow[j+0]
                cRow[j+1] += aik * bRow[j+1]
                cRow[j+2] += aik * bRow[j+2]
                cRow[j+3] += aik * bRow[j+3]
            }
            for ; j < jMax; j++ {
                cRow[j] += aik * bRow[j]
            }
        }
    }
}
