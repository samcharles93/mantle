package simd

import (
	"simd/archsimd"
	"sync"
	"sync/atomic"

	"github.com/samcharles93/mantle/pkg/mcf"
)

// quantMaxBlocksPerRow is the fused-row limit in matVecRangeQuantCached, the
// path MatMatQuant must reproduce; wider rows take its per-block loop instead.
const quantMaxBlocksPerRow = 256

// quantRows holds n activation rows quantised exactly as matVecWithQuant
// quantises a single input vector (int8 values and one scale per 32-element
// block), plus the scales of each row pair interleaved for the 2x2 kernel.
type quantRows struct {
	q8     []int8    // [n, blocks*32]
	q16    []int16   // one row of quantizeVecBlocksInto's widened output, unused
	scales []float32 // [n, blocks]
	// pairScales[(r/2)*blocks*4 + b*4 : +4] is {xs[r][b], xs[r+1][b],
	// xs[r][b], xs[r+1][b]} for each even r with r+1 < n.
	pairScales []float32
	blocks     int
}

var quantRowsPool = sync.Pool{New: func() any { return new(quantRows) }}

func getQuantRows(x []float32, n, c int) *quantRows {
	qr := quantRowsPool.Get().(*quantRows)
	blocks := (c + 31) / 32
	qr.blocks = blocks
	qr.q8 = ensureInt8Slice(qr.q8, n*blocks*32)
	qr.q16 = ensureInt16Slice(qr.q16, blocks*32)
	qr.scales = ensureFloat32Slice(qr.scales, n*blocks)
	for r := range n {
		quantizeVecBlocksInto(x[r*c:(r+1)*c], blocks, qr.q8[r*blocks*32:(r+1)*blocks*32],
			qr.q16, qr.scales[r*blocks:(r+1)*blocks])
	}
	pairs := n / 2
	qr.pairScales = ensureFloat32Slice(qr.pairScales, pairs*blocks*4)
	for p := range pairs {
		xs0 := qr.scales[2*p*blocks : (2*p+1)*blocks]
		xs1 := qr.scales[(2*p+1)*blocks : (2*p+2)*blocks]
		out := qr.pairScales[p*blocks*4 : (p+1)*blocks*4]
		for b := range blocks {
			out[4*b], out[4*b+1], out[4*b+2], out[4*b+3] = xs0[b], xs1[b], xs0[b], xs1[b]
		}
	}
	return qr
}

func putQuantRows(qr *quantRows) { quantRowsPool.Put(qr) }

// useBatchQuant reports whether dst = x * wᵀ for n activation rows can run
// through MatMatQuant. It needs the int8 cache matVecRangeQuantCached uses on
// AVX2 hosts and the shape that path's fused-row kernel takes, so the two
// produce identical bits.
func useBatchQuant(w *Mat, n int) bool {
	return n >= 1 && cpu.HasAVX2 && w.Raw != nil && mcf.DTypeRequiresAligned64(w.DType) &&
		!mcf.IsUQDType(w.DType) && w.Quant != nil && w.Quant.ValidFor(w) &&
		w.C%32 == 0 && w.Quant.BlocksPerRow <= quantMaxBlocksPerRow
}

// MatMatQuant computes dst[r*w.R+i] = dot(w row i, x[r*w.C:(r+1)*w.C]) for
// every activation row r in [0,n), for a quantised w with a valid int8 cache.
//
// Every output equals what MatVec produces for that activation row alone: the
// rows are quantised the same way, block dots are exact integers, and each
// output's scale combine performs the same float32 operations in the same
// block order (one SIMD lane per output). What changes is that each weight row
// is read once and reused for all n rows while it sits in L1.
func MatMatQuant(dst []float32, w *Mat, x []float32, n int) {
	if w.R == 0 || w.C == 0 || n == 0 {
		return
	}
	if !useBatchQuant(w, n) {
		panic("matmat: MatMatQuant needs a cached int8 quant weight on an AVX2 host")
	}
	if len(dst) < n*w.R || len(x) < n*w.C {
		panic("matmat shape mismatch")
	}

	qr := getQuantRows(x, n, w.C)
	defer putQuantRows(qr)

	job := &quantJob{dst: dst, w: w, qr: qr, n: n, tile: quantTileRows}
	tiles := (w.R + job.tile - 1) / job.tile
	pool := getMatVecPool()
	workers := min(pool.size, tiles)
	if workers <= 1 {
		matMatRangeQuant(dst, w, qr, n, 0, w.R, nil)
		return
	}
	done := <-pool.doneSlots
	for range workers {
		pool.tasks <- matVecTask{qj: job, done: done}
	}
	for range workers {
		<-done
	}
	pool.doneSlots <- done
}

// quantTileRows is the number of weight rows a worker claims at a time. Small
// even tiles let the faster cores of a hybrid CPU take more of the matrix than
// an equal static split would give them.
const quantTileRows = 16

// quantJob is one MatMatQuant call shared by every worker task: each worker
// claims tiles of weight rows from next until none remain.
type quantJob struct {
	dst  []float32
	w    *Mat
	qr   *quantRows
	n    int
	tile int
	next atomic.Int64
}

func (j *quantJob) run(worker *matVecWorker) {
	for {
		rs := int(j.next.Add(int64(j.tile))) - j.tile
		if rs >= j.w.R {
			return
		}
		matMatRangeQuant(j.dst, j.w, j.qr, j.n, rs, min(rs+j.tile, j.w.R), worker)
	}
}

// quantKernelScratch is the per-worker state of matMatRangeQuant: the block
// scales of the current weight row pair, interleaved for quantDot2x2.
type quantKernelScratch struct {
	wPairs []float32 // [blocks, 4]: {s0[b], s0[b], s1[b], s1[b]}
}

// matMatRangeQuant computes output columns [rs,re) for all n activation rows.
// worker, when non-nil, supplies the scratch buffers.
func matMatRangeQuant(dst []float32, w *Mat, qr *quantRows, n, rs, re int, worker *matVecWorker) {
	qc := w.Quant
	bpr := qc.BlocksPerRow
	c := bpr * 32
	var scratch *quantKernelScratch
	if worker != nil {
		worker.mu.Lock()
		scratch = &worker.quant
		scratch.wPairs = ensureFloat32Slice(scratch.wPairs, 4*bpr)
		worker.mu.Unlock()
	} else {
		scratch = &quantKernelScratch{wPairs: make([]float32, 4*bpr)}
	}
	wPairs := scratch.wPairs[:4*bpr]
	ld := w.R

	i := rs
	for ; i+1 < re; i += 2 {
		w0 := qc.Q[i*c : (i+1)*c]
		w1 := qc.Q[(i+1)*c : (i+2)*c]
		s0 := qc.Scales[i*bpr : (i+1)*bpr]
		s1 := qc.Scales[(i+1)*bpr : (i+2)*bpr]
		for b := range bpr {
			wPairs[4*b], wPairs[4*b+1], wPairs[4*b+2], wPairs[4*b+3] = s0[b], s0[b], s1[b], s1[b]
		}
		r := 0
		for ; r+3 < n; r += 4 {
			x0 := qr.q8[r*c : (r+1)*c]
			x1 := qr.q8[(r+1)*c : (r+2)*c]
			x2 := qr.q8[(r+2)*c : (r+3)*c]
			x3 := qr.q8[(r+3)*c : (r+4)*c]
			xPairs01 := qr.pairScales[(r/2)*bpr*4 : (r/2+1)*bpr*4]
			xPairs23 := qr.pairScales[(r/2+1)*bpr*4 : (r/2+2)*bpr*4]
			var out01, out23 [4]float32
			sum01, sum23 := quantDot2x4(w0, w1, x0, x1, x2, x3, wPairs, xPairs01, xPairs23)
			sum01.StoreArray(&out01)
			sum23.StoreArray(&out23)
			dst[r*ld+i], dst[(r+1)*ld+i] = out01[0], out01[1]
			dst[r*ld+i+1], dst[(r+1)*ld+i+1] = out01[2], out01[3]
			dst[(r+2)*ld+i], dst[(r+3)*ld+i] = out23[0], out23[1]
			dst[(r+2)*ld+i+1], dst[(r+3)*ld+i+1] = out23[2], out23[3]
		}
		if r+1 < n {
			x0 := qr.q8[r*c : (r+1)*c]
			x1 := qr.q8[(r+1)*c : (r+2)*c]
			xPairs := qr.pairScales[(r/2)*bpr*4 : (r/2+1)*bpr*4]
			var out [4]float32
			quantDot2x2(w0, w1, x0, x1, wPairs, xPairs).StoreArray(&out)
			dst[r*ld+i], dst[(r+1)*ld+i] = out[0], out[1]
			dst[r*ld+i+1], dst[(r+1)*ld+i+1] = out[2], out[3]
			r += 2
		}
		if r < n {
			dst[r*ld+i] = quantDotRow(w0, s0, qr, r)
			dst[r*ld+i+1] = quantDotRow(w1, s1, qr, r)
		}
	}
	if i < re {
		w0 := qc.Q[i*c : (i+1)*c]
		s0 := qc.Scales[i*bpr : (i+1)*bpr]
		for r := range n {
			dst[r*ld+i] = quantDotRow(w0, s0, qr, r)
		}
	}
}

// quantDotRow is the single-output form of quantDot2x2: weight row wrow with
// block scales ws against activation row r, in the scalar order of
// matVecRangeQuantCached.
func quantDotRow(wrow []int8, ws []float32, qr *quantRows, r int) float32 {
	bpr := len(ws)
	c := bpr * 32
	x := qr.q8[r*c : (r+1)*c]
	xs := qr.scales[r*bpr : (r+1)*bpr]
	ones := archsimd.BroadcastInt16x16(1)
	var sum float32
	for b := range bpr {
		w := archsimd.LoadInt8x32Array((*[32]int8)(wrow[b*32 : b*32+32]))
		v := archsimd.LoadInt8x32Array((*[32]int8)(x[b*32 : b*32+32]))
		var tmp [8]int32
		blockDotInt8(w.Abs().ToBits(), v.MulSign(w), ones).StoreArray(&tmp)
		dot := tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7]
		sum += float32(dot) * (ws[b] * xs[b])
	}
	return sum
}

// blockDotInt8 returns the eight int32 partial sums of one 32-element int8
// block dot, given |w| and x*sign(w). It is exact: every |w| <= 128 and every
// quantised |x| <= 127, so no VPMADDUBSW pair sum (at most 2*128*127) reaches
// the int16 saturation bound.
func blockDotInt8(absW archsimd.Uint8x32, signedX archsimd.Int8x32, ones archsimd.Int16x16) archsimd.Int32x8 {
	return absW.DotProductPairsSaturated(signedX).DotProductPairs(ones)
}

// quantDot2x2 returns the four outputs {w0·x0, w0·x1, w1·x0, w1·x1} of two
// int8 weight rows against two quantised activation rows. Per block, each
// loaded vector feeds two products and a horizontal-add tree across the four
// pairs yields one vector of exact block dots, which a Float32x4 lane per
// output folds in with sum += float32(dot) * (wScale * xScale): the same
// rounded operations, in the same block order, as matVecRangeQuantCached.
//
// wPairs and xPairs hold the matching block scales interleaved four wide.
func quantDot2x2(w0, w1, x0, x1 []int8, wPairs, xPairs []float32) archsimd.Float32x4 {
	blocks := len(wPairs) / 4
	ones := archsimd.BroadcastInt16x16(1)
	var sum archsimd.Float32x4
	for b := range blocks {
		off := b * 32
		wa := archsimd.LoadInt8x32Array((*[32]int8)(w0[off : off+32]))
		we := archsimd.LoadInt8x32Array((*[32]int8)(w1[off : off+32]))
		xu := archsimd.LoadInt8x32Array((*[32]int8)(x0[off : off+32]))
		xv := archsimd.LoadInt8x32Array((*[32]int8)(x1[off : off+32]))
		aa := wa.Abs().ToBits()
		ae := we.Abs().ToBits()
		c00 := blockDotInt8(aa, xu.MulSign(wa), ones)
		c01 := blockDotInt8(aa, xv.MulSign(wa), ones)
		c10 := blockDotInt8(ae, xu.MulSign(we), ones)
		c11 := blockDotInt8(ae, xv.MulSign(we), ones)

		h := c00.ConcatAddPairsGrouped(c01).ConcatAddPairsGrouped(c10.ConcatAddPairsGrouped(c11))
		dots := h.GetLo().Add(h.GetHi()).ConvertToFloat32()
		ws := archsimd.LoadFloat32x4Array((*[4]float32)(wPairs[4*b : 4*b+4]))
		xs := archsimd.LoadFloat32x4Array((*[4]float32)(xPairs[4*b : 4*b+4]))
		sum = sum.Add(dots.Mul(ws.Mul(xs)))
	}
	return sum
}

// quantDot2x4 is quantDot2x2 for activation rows x0..x3 at once: each weight
// block is loaded and its absolute value taken once for four rows. The two
// results hold {w0·x0, w0·x1, w1·x0, w1·x1} and the same for x2, x3; the two
// trees run one after the other so the live vectors fit in 16 registers.
func quantDot2x4(w0, w1, x0, x1, x2, x3 []int8, wPairs, xPairs01, xPairs23 []float32) (archsimd.Float32x4, archsimd.Float32x4) {
	blocks := len(wPairs) / 4
	ones := archsimd.BroadcastInt16x16(1)
	var sum01, sum23 archsimd.Float32x4
	for b := range blocks {
		off := b * 32
		wa := archsimd.LoadInt8x32Array((*[32]int8)(w0[off : off+32]))
		we := archsimd.LoadInt8x32Array((*[32]int8)(w1[off : off+32]))
		aa := wa.Abs().ToBits()
		ae := we.Abs().ToBits()
		ws := archsimd.LoadFloat32x4Array((*[4]float32)(wPairs[4*b : 4*b+4]))

		xu := archsimd.LoadInt8x32Array((*[32]int8)(x0[off : off+32]))
		xv := archsimd.LoadInt8x32Array((*[32]int8)(x1[off : off+32]))
		c00 := blockDotInt8(aa, xu.MulSign(wa), ones)
		c01 := blockDotInt8(aa, xv.MulSign(wa), ones)
		c10 := blockDotInt8(ae, xu.MulSign(we), ones)
		c11 := blockDotInt8(ae, xv.MulSign(we), ones)
		h := c00.ConcatAddPairsGrouped(c01).ConcatAddPairsGrouped(c10.ConcatAddPairsGrouped(c11))
		xs := archsimd.LoadFloat32x4Array((*[4]float32)(xPairs01[4*b : 4*b+4]))
		sum01 = sum01.Add(h.GetLo().Add(h.GetHi()).ConvertToFloat32().Mul(ws.Mul(xs)))

		xu = archsimd.LoadInt8x32Array((*[32]int8)(x2[off : off+32]))
		xv = archsimd.LoadInt8x32Array((*[32]int8)(x3[off : off+32]))
		c00 = blockDotInt8(aa, xu.MulSign(wa), ones)
		c01 = blockDotInt8(aa, xv.MulSign(wa), ones)
		c10 = blockDotInt8(ae, xu.MulSign(we), ones)
		c11 = blockDotInt8(ae, xv.MulSign(we), ones)
		h = c00.ConcatAddPairsGrouped(c01).ConcatAddPairsGrouped(c10.ConcatAddPairsGrouped(c11))
		xs = archsimd.LoadFloat32x4Array((*[4]float32)(xPairs23[4*b : 4*b+4]))
		sum23 = sum23.Add(h.GetLo().Add(h.GetHi()).ConvertToFloat32().Mul(ws.Mul(xs)))
	}
	return sum01, sum23
}
