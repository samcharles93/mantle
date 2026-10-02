package simd

import (
	"runtime"

	core "github.com/samcharles93/mantle/internal/backend/core"
	"github.com/samcharles93/mantle/pkg/mcf"
)

// GemmParWT computes C = alpha*A*Wᵀ + beta*C.
//
// A is [m,k] row-major with f32 elements in A.Data. W is [n,k] row-major — the
// standard Mantle weight layout — holding either f32 in W.Data or f16/bf16 in
// W.Raw. C is [m,n] row-major f32 in C.Data.
//
// Unlike GemmPar, which computes A*B, this transposes the weight operand. That
// is the shape every projection in the runtime needs: X @ Wᵀ with W stored
// [out,in].
//
// Parallelism is over output columns, not output rows. Each worker packs a
// weight tile once and applies it to every row block; partitioning by rows
// instead would make every worker re-decode the whole weight matrix, which for
// quantised weights costs more than the multiply itself.
func GemmParWT(cfg GemmConfig, C, A, W *Mat, alpha, beta float32, workers int) {
	if A.C != W.C || C.R != A.R || C.C != W.R {
		panic("gemm: dimension mismatch (want A[m,k] * W[n,k]^T -> C[m,n])")
	}
	if C.R == 0 || C.C == 0 {
		return
	}
	cfg.BTransposed = true

	tk, tn := cfg.TileK, cfg.TileN
	if tk <= 0 || tn <= 0 || !cpu.HasAVX2 || alpha != 1 {
		gemmSerialWT(cfg, C, A, W, alpha, beta)
		return
	}

	// The column-parallel path folds beta in up front, then only accumulates.
	scaleRowsBeta(C, 0, C.R, beta)

	n := C.C
	nTiles := (n + tn - 1) / tn
	workers = normalizeWorkers(workers, nTiles, gemmWorkPool.size)

	perWorker := (nTiles + workers - 1) / workers
	done := <-gemmWorkPool.doneSlots
	dispatched := 0
	for range workers {
		cs := dispatched * perWorker * tn
		if cs >= n {
			break
		}
		ce := min(cs+perWorker*tn, n)
		gemmWorkPool.tasks <- gemmTask{
			C:     C,
			A:     A,
			B:     W,
			alpha: 1,
			beta:  1,
			rs:    cs,
			re:    ce,
			cfg:   cfg,
			done:  done,
		}
		dispatched++
	}
	for range dispatched {
		<-done
	}
	gemmWorkPool.doneSlots <- done
}

func normalizeWorkers(workers, nTiles, poolSize int) int {
	if workers <= 0 {
		workers = runtime.GOMAXPROCS(0)
	}
	if workers > nTiles {
		workers = nTiles
	}
	if workers > poolSize {
		workers = poolSize
	}
	return max(workers, 1)
}

// gemmRangeColsWT accumulates A*Wᵀ into C for the column strip [cs,ce). C must
// already carry its beta term. Each weight tile in the strip is packed once and
// reused for every output-row block.
func gemmRangeColsWT(cfg GemmConfig, C, A, W *Mat, cs, ce int, packB []float32) {
	tm, tn, tk := cfg.TileM, cfg.TileN, cfg.TileK
	cStride, aStride := C.Stride, A.Stride
	cData, aData := C.Data, A.Data
	k := A.C

	for j0 := cs; j0 < ce; j0 += tn {
		jMax := min(j0+tn, ce)
		width := jMax - j0
		for k0 := 0; k0 < k; k0 += tk {
			kMax := min(k0+tk, k)
			kInner := kMax - k0
			packBTileWT(packB, W, k0, kMax, j0, jMax)
			for i0 := 0; i0 < C.R; i0 += tm {
				iMax := min(i0+tm, C.R)
				blockUpdateAlpha1SIMDPacked(cData, aData, packB, cStride, aStride, i0, iMax, j0, width, k0, kInner)
			}
		}
	}
}

// gemmSerialWT is the correct-for-any-alpha/beta fallback. It reads the weights
// directly per element, so it is slow, but it needs no staging buffer and no
// SIMD support.
func gemmSerialWT(cfg GemmConfig, C, A, W *Mat, alpha, beta float32) {
	scaleRowsBeta(C, 0, C.R, beta)
	if alpha == 0 {
		return
	}
	tm, tn, tk := cfg.TileM, cfg.TileN, cfg.TileK
	cStride, aStride := C.Stride, A.Stride
	k := A.C
	for i0 := 0; i0 < C.R; i0 += tm {
		iMax := min(i0+tm, C.R)
		for k0 := 0; k0 < k; k0 += tk {
			kMax := min(k0+tk, k)
			for j0 := 0; j0 < C.C; j0 += tn {
				jMax := min(j0+tn, C.C)
				blockUpdateWT(C.Data, A.Data, W, cStride, aStride, alpha, i0, iMax, j0, jMax, k0, kMax)
			}
		}
	}
}

// --- weight decoding -------------------------------------------------------

// matElem returns W[row,col] as f32, decoding f16/bf16 storage when needed.
// It must stay equivalent to Mat.RowTo for the same element.
func matElem(w *Mat, row, col int) float32 {
	if w.Data != nil {
		return w.Data[row*w.Stride+col]
	}
	off := (row*w.Stride + col) * 2
	u := uint16(w.Raw[off]) | uint16(w.Raw[off+1])<<8
	if w.DType == mcf.DTypeBF16 {
		return core.BF16ToFloat32(u)
	}
	return core.FP16ToFloat32(u)
}

func scaleRowsBeta(C *Mat, rs, re int, beta float32) {
	if beta == 1 {
		return
	}
	for i := rs; i < re; i++ {
		row := C.Data[i*C.Stride : i*C.Stride+C.C]
		if beta == 0 {
			clear(row)
			continue
		}
		for j := range row {
			row[j] *= beta
		}
	}
}

// packBTileWT stages the transposed, decoded weight tile that the packed inner
// loop consumes: dst[kk*width+jj] = decode(W[j0+jj, k0+kk]).
//
// The jj loop is the outer one so each source row of W is read contiguously,
// and the dtype dispatch happens once rather than per element.
func packBTileWT(dst []float32, w *Mat, k0, kMax, j0, jMax int) {
	width := jMax - j0
	kInner := kMax - k0
	if width <= 0 || kInner <= 0 {
		return
	}
	if width > maxTileN || kInner > maxTileK {
		panic("packBTileWT exceeds max tile size")
	}

	if w.Data != nil {
		for jj := 0; jj < width; jj++ {
			src := (j0+jj)*w.Stride + k0
			for kk := 0; kk < kInner; kk++ {
				dst[kk*width+jj] = w.Data[src+kk]
			}
		}
		return
	}

	decode := core.BF16ToFloat32
	if w.DType != mcf.DTypeBF16 {
		decode = core.FP16ToFloat32
	}
	for jj := 0; jj < width; jj++ {
		src := ((j0+jj)*w.Stride + k0) * 2
		for kk := 0; kk < kInner; kk++ {
			off := src + kk*2
			u := uint16(w.Raw[off]) | uint16(w.Raw[off+1])<<8
			dst[kk*width+jj] = decode(u)
		}
	}
}

// blockUpdateWT accumulates alpha*A*Wᵀ into C for a single tile. C must
// already carry its beta term.
func blockUpdateWT(cData, aData []float32, w *Mat, cStride, aStride int, alpha float32, i0, iMax, j0, jMax, k0, kMax int) {
	for i := i0; i < iMax; i++ {
		aBase := i * aStride
		cBase := i * cStride
		for j := j0; j < jMax; j++ {
			var sum float32
			for k := k0; k < kMax; k++ {
				sum += aData[aBase+k] * matElem(w, j, k)
			}
			cData[cBase+j] += alpha * sum
		}
	}
}
