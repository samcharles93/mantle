package simd

import (
	"simd/archsimd"
	"sync"
	"sync/atomic"

	instance "github.com/samcharles93/mantle/internal/backend/core"
	"github.com/samcharles93/mantle/pkg/mcf"
)

// quantMaxBlocksPerRow is the fused-row limit in matVecRangeQuantCached, the
// path MatMatQuant must reproduce; wider rows take its per-block loop instead.
const quantMaxBlocksPerRow = 256

// quantRows holds n activation rows quantised exactly as matVecWithQuant
// quantises a single input vector (int8 values and one scale per 32-element
// block), plus the scales of each row pair interleaved for the 2x2 kernel.
type quantRows struct {
	q8 []int8 // [n, blocks*32]
	// q8Pair is q8 with each block in pair order (16 even elements, then 16
	// odd), matching packed 4-bit weights decoded by decodePacked4Row. It is
	// filled only for packed weights.
	q8Pair []int8
	q16    []int16   // one row of quantizeVecBlocksInto's widened output, unused
	scales []float32 // [n, blocks]
	// pairScales[(r/2)*blocks*4 + b*4 : +4] is {xs[r][b], xs[r+1][b],
	// xs[r][b], xs[r+1][b]} for each even r with r+1 < n.
	pairScales []float32
	blocks     int
}

var quantRowsPool = sync.Pool{New: func() any { return new(quantRows) }}

// quantRowsTask is a range of activation rows to quantise on a pool worker.
type quantRowsTask struct {
	x      []float32
	c      int
	blocks int
	q8     []int8
	scales []float32
	rs, re int
}

// quantizeRowsInto fills q8 and scales for n activation rows, spreading the rows
// over the worker pool so the caller does not serialise the whole chunk before
// the matmul dispatch. Falls back to the serial loop when a pool is
// unavailable, and produces the same bytes either way: each row is quantised by
// the same function with a private int16 scratch.
func quantizeRowsInto(x []float32, n, c, blocks int, q8 []int8, q16 []int16, scales []float32) {
	rowLen := blocks * 32
	pool := getMatVecPool()
	workers := 0
	if pool != nil && pool.size > 1 && n > 1 {
		workers = min(pool.size, n)
	}
	if workers < 2 {
		for r := range n {
			quantizeVecBlocksInto(x[r*c:(r+1)*c], blocks,
				q8[r*rowLen:(r+1)*rowLen], q16, scales[r*blocks:(r+1)*blocks])
		}
		return
	}
	done := <-pool.doneSlots
	for i := range workers {
		rs := i * n / workers
		re := (i + 1) * n / workers
		pool.tasks <- matVecTask{qrows: &quantRowsTask{
			x: x, c: c, blocks: blocks, q8: q8, scales: scales, rs: rs, re: re,
		}, done: done}
	}
	for range workers {
		<-done
	}
	pool.doneSlots <- done
}

func getQuantRows(x []float32, n, c int, pair bool) *quantRows {
	qr := quantRowsPool.Get().(*quantRows)
	blocks := (c + 31) / 32
	qr.blocks = blocks
	qr.q8 = ensureInt8Slice(qr.q8, n*blocks*32)
	qr.q16 = ensureInt16Slice(qr.q16, blocks*32)
	qr.scales = ensureFloat32Slice(qr.scales, n*blocks)
	quantizeRowsInto(x, n, c, blocks, qr.q8, qr.q16, qr.scales)
	if pair {
		qr.q8Pair = ensureInt8Slice(qr.q8Pair, n*blocks*32)
		for blk := range n * blocks {
			src := qr.q8[blk*32 : blk*32+32]
			dst := qr.q8Pair[blk*32 : blk*32+32]
			for i := range 16 {
				dst[i] = src[2*i]
				dst[16+i] = src[2*i+1]
			}
		}
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

// zeroFloat32s seeds the quantDot accumulators with VEX loads. The zero value
// of a Float32x4 compiles to a legacy-SSE move of X15, which takes an assist on
// Golden Cove once the kernel's 256-bit constants are live.
var zeroFloat32s [8]float32

// useBatchQuant reports whether dst = x * wᵀ for n activation rows can run
// through MatMatQuant. It needs, on an AVX2 host, either the int8 cache
// matVecRangeQuantCached uses or a packed 4-bit payload (which
// matVecRangePacked4 matches bit for bit), and the shape the fused-row
// kernel takes, so batched and per-row products produce identical bits.
func useBatchQuant(w *Mat, n int) bool {
	if n < 1 || !cpu.HasAVX2 || w.Raw == nil || !mcf.DTypeRequiresAligned64(w.DType) ||
		mcf.IsUQDType(w.DType) || w.C%32 != 0 {
		return false
	}
	if w.Quant != nil && w.Quant.ValidFor(w) {
		return w.Quant.BlocksPerRow <= quantMaxBlocksPerRow
	}
	return isPacked4DType(w.DType) && w.C/32 <= quantMaxBlocksPerRow
}

// batchWeightsPacked reports whether MatMatQuant reads w from its packed
// payload rather than the int8 cache.
func batchWeightsPacked(w *Mat) bool {
	return (w.Quant == nil || !w.Quant.ValidFor(w)) && isPacked4DType(w.DType)
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
		panic("matmat: MatMatQuant needs a cached int8 or packed 4-bit quant weight on an AVX2 host")
	}
	if len(dst) < n*w.R || len(x) < n*w.C {
		panic("matmat shape mismatch")
	}

	qr := getQuantRows(x, n, w.C, batchWeightsPacked(w))
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
// scales of the current weight row pair, interleaved for quantDot2x2, and for
// packed weights the decoded row pair and its block scales.
type quantKernelScratch struct {
	wPairs []float32 // [blocks, 4]: {s0[b], s0[b], s1[b], s1[b]}
	rows   []int8    // [2, blocks*32] decoded packed rows, pair order
	scales []float32 // [2, blocks] their block scales
}

// matMatRangeQuant computes output columns [rs,re) for all n activation rows.
// worker, when non-nil, supplies the scratch buffers.
func matMatRangeQuant(dst []float32, w *Mat, qr *quantRows, n, rs, re int, worker *matVecWorker) {
	bpr := w.C / 32
	c := bpr * 32
	packed := batchWeightsPacked(w)
	ensure := func(s *quantKernelScratch) {
		s.wPairs = ensureFloat32Slice(s.wPairs, 4*bpr)
		if packed {
			s.rows = ensureInt8Slice(s.rows, 2*c)
			s.scales = ensureFloat32Slice(s.scales, 2*bpr)
		}
	}
	var scratch *quantKernelScratch
	if worker != nil {
		worker.mu.Lock()
		scratch = &worker.quant
		ensure(scratch)
		worker.mu.Unlock()
	} else {
		scratch = &quantKernelScratch{}
		ensure(scratch)
	}
	wPairs := scratch.wPairs[:4*bpr]
	ld := w.R

	// rowsAt returns weight rows i and i+1 (or just i when two is false) with
	// their block scales, from the int8 cache or decoded from the packed
	// payload into scratch.
	act := qr.q8
	var layout instance.QuantLayout
	if packed {
		var err error
		if layout, err = instance.QuantLayoutForMat(w.R, w.C, w.DType, len(w.Raw)); err != nil {
			panic(err)
		}
		act = qr.q8Pair
	}
	rowsAt := func(i int, two bool) (w0, w1 []int8, s0, s1 []float32) {
		if !packed {
			qc := w.Quant
			w0, s0 = qc.Q[i*c:(i+1)*c], qc.Scales[i*bpr:(i+1)*bpr]
			if two {
				w1, s1 = qc.Q[(i+1)*c:(i+2)*c], qc.Scales[(i+1)*bpr:(i+2)*bpr]
			}
			return
		}
		data := w.Raw[layout.DataOff : layout.DataOff+layout.DataBytes]
		w0, s0 = scratch.rows[:c], scratch.scales[:bpr]
		decodePacked4Row(w0, data[i*bpr*16:(i+1)*bpr*16], bpr)
		packed4RowScales(s0, layout, w.Raw, i)
		if two {
			w1, s1 = scratch.rows[c:2*c], scratch.scales[bpr:2*bpr]
			decodePacked4Row(w1, data[(i+1)*bpr*16:(i+2)*bpr*16], bpr)
			packed4RowScales(s1, layout, w.Raw, i+1)
		}
		return
	}

	i := rs
	for ; i+1 < re; i += 2 {
		w0, w1, s0, s1 := rowsAt(i, true)
		for b := range bpr {
			wPairs[4*b], wPairs[4*b+1], wPairs[4*b+2], wPairs[4*b+3] = s0[b], s0[b], s1[b], s1[b]
		}
		r := 0
		for ; r+3 < n; r += 4 {
			x0 := act[r*c : (r+1)*c]
			x1 := act[(r+1)*c : (r+2)*c]
			x2 := act[(r+2)*c : (r+3)*c]
			x3 := act[(r+3)*c : (r+4)*c]
			xPairs01 := qr.pairScales[(r/2)*bpr*4 : (r/2+1)*bpr*4]
			xPairs23 := qr.pairScales[(r/2+1)*bpr*4 : (r/2+2)*bpr*4]
			var out [8]float32
			quantDot2x4(w0, w1, x0, x1, x2, x3, wPairs, xPairs01, xPairs23, &out)
			dst[r*ld+i], dst[(r+1)*ld+i] = out[0], out[1]
			dst[r*ld+i+1], dst[(r+1)*ld+i+1] = out[2], out[3]
			dst[(r+2)*ld+i], dst[(r+3)*ld+i] = out[4], out[5]
			dst[(r+2)*ld+i+1], dst[(r+3)*ld+i+1] = out[6], out[7]
		}
		if r+1 < n {
			x0 := act[r*c : (r+1)*c]
			x1 := act[(r+1)*c : (r+2)*c]
			xPairs := qr.pairScales[(r/2)*bpr*4 : (r/2+1)*bpr*4]
			var out [4]float32
			quantDot2x2(w0, w1, x0, x1, wPairs, xPairs, &out)
			dst[r*ld+i], dst[(r+1)*ld+i] = out[0], out[1]
			dst[r*ld+i+1], dst[(r+1)*ld+i+1] = out[2], out[3]
			r += 2
		}
		if r < n {
			dst[r*ld+i] = quantDotRow(w0, s0, act, qr, r)
			dst[r*ld+i+1] = quantDotRow(w1, s1, act, qr, r)
		}
	}
	if i < re {
		w0, _, s0, _ := rowsAt(i, false)
		for r := range n {
			dst[r*ld+i] = quantDotRow(w0, s0, act, qr, r)
		}
	}
}

// quantDotRow is the single-output form of quantDot2x2: weight row wrow with
// block scales ws against activation row r, in the scalar order of
// matVecRangeQuantCached. Every block dot is computed before the scalar fold,
// so the 256-bit work ends once per row (see endAVX) rather than once per block.
func quantDotRow(wrow []int8, ws []float32, act []int8, qr *quantRows, r int) float32 {
	bpr := len(ws)
	c := bpr * 32
	x := act[r*c : (r+1)*c]
	xs := qr.scales[r*bpr : (r+1)*bpr]
	ones := archsimd.BroadcastInt16x16(1)
	var dots [quantMaxBlocksPerRow]int32
	for b := range bpr {
		w := archsimd.LoadInt8x32Array((*[32]int8)(wrow[b*32 : b*32+32]))
		v := archsimd.LoadInt8x32Array((*[32]int8)(x[b*32 : b*32+32]))
		var tmp [8]int32
		blockDotInt8(w.Abs().ToBits(), v.MulSign(w), ones).StoreArray(&tmp)
		dots[b] = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7]
	}
	endAVX()
	var sum float32
	for b, dot := range dots[:bpr] {
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

// quantDot2x2 stores in out the four outputs {w0·x0, w0·x1, w1·x0, w1·x1} of two
// int8 weight rows against two quantised activation rows. Per block, each
// loaded vector feeds two products and a horizontal-add tree across the four
// pairs yields one vector of exact block dots, which a Float32x4 lane per
// output folds in with sum += float32(dot) * (wScale * xScale): the same
// rounded operations, in the same block order, as matVecRangeQuantCached.
//
// wPairs and xPairs hold the matching block scales interleaved four wide.
//
// No Float32x4 crosses the function boundary: the sum starts from a vector
// load and leaves through a vector store, then endAVX runs. Go zeroes, spills
// and returns 128-bit values with legacy-SSE moves, and each one issued after
// the 256-bit loop would take an assist on Golden Cove, as would the caller's
// scalar stores of the outputs.
func quantDot2x2(w0, w1, x0, x1 []int8, wPairs, xPairs []float32, out *[4]float32) {
	// Every operand is resliced to one length up front so that the loop
	// condition proves each per-block slice in bounds (see quantDot2x4).
	np := len(wPairs) / 4 * 4
	n := np * 8
	w0, w1, x0, x1 = w0[:n], w1[:n], x0[:n], x1[:n]
	wPairs, xPairs = wPairs[:np], xPairs[:np]
	ones := archsimd.BroadcastInt16x16(1)
	sum := archsimd.LoadFloat32x4Array((*[4]float32)(zeroFloat32s[:4]))
	for off, p := 0, 0; off+32 <= n && p+4 <= np; off, p = off+32, p+4 {
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
		ws := archsimd.LoadFloat32x4Array((*[4]float32)(wPairs[p : p+4]))
		xs := archsimd.LoadFloat32x4Array((*[4]float32)(xPairs[p : p+4]))
		sum = sum.Add(dots.Mul(ws.Mul(xs)))
	}
	sum.StoreArray(out)
	endAVX()
}

// quantDot2x4 is quantDot2x2 for activation rows x0..x3 at once: each weight
// block is loaded and its absolute value taken once for four rows. out gets
// {w0·x0, w0·x1, w1·x0, w1·x1} and then the same for x2, x3. The two
// horizontal-add trees run one after the other so the live vectors fit in 16
// registers, and their block dots are folded together in one Float32x8 lane
// per output: the same operations, in the same order, as quantDot2x2's
// Float32x4, so every output keeps its bits.
//
// The accumulator is 256 bits wide for a codegen reason as well as a count
// one: Go spills a Float32x4 with a legacy-SSE move, which costs an assist per
// block on Golden Cove whenever register pressure forces one (see endAVX),
// while a Float32x8 spills with a VEX move.
//
// The loop carries no bounds checks beyond one per operand family: each
// operand is resliced once to n (or np) elements, so its length is the very
// value the loop condition tests and the compiler proves the remaining
// [off:off+32] and [p:p+4] slices in range. Indexing by b*32 left nine checks,
// and their address arithmetic, per block.
func quantDot2x4(w0, w1, x0, x1, x2, x3 []int8, wPairs, xPairs01, xPairs23 []float32, out *[8]float32) {
	np := len(wPairs) / 4 * 4
	n := np * 8
	w0, w1 = w0[:n], w1[:n]
	x0, x1, x2, x3 = x0[:n], x1[:n], x2[:n], x3[:n]
	wPairs, xPairs01, xPairs23 = wPairs[:np], xPairs01[:np], xPairs23[:np]
	ones := archsimd.BroadcastInt16x16(1)
	sum := archsimd.LoadFloat32x8Array(&zeroFloat32s)
	for off, p := 0, 0; off+32 <= n && p+4 <= np; off, p = off+32, p+4 {
		wa := archsimd.LoadInt8x32Array((*[32]int8)(w0[off : off+32]))
		we := archsimd.LoadInt8x32Array((*[32]int8)(w1[off : off+32]))
		aa := wa.Abs().ToBits()
		ae := we.Abs().ToBits()

		xu := archsimd.LoadInt8x32Array((*[32]int8)(x0[off : off+32]))
		xv := archsimd.LoadInt8x32Array((*[32]int8)(x1[off : off+32]))
		c00 := blockDotInt8(aa, xu.MulSign(wa), ones)
		c01 := blockDotInt8(aa, xv.MulSign(wa), ones)
		c10 := blockDotInt8(ae, xu.MulSign(we), ones)
		c11 := blockDotInt8(ae, xv.MulSign(we), ones)
		h01 := c00.ConcatAddPairsGrouped(c01).ConcatAddPairsGrouped(c10.ConcatAddPairsGrouped(c11))

		xu = archsimd.LoadInt8x32Array((*[32]int8)(x2[off : off+32]))
		xv = archsimd.LoadInt8x32Array((*[32]int8)(x3[off : off+32]))
		c00 = blockDotInt8(aa, xu.MulSign(wa), ones)
		c01 = blockDotInt8(aa, xv.MulSign(wa), ones)
		c10 = blockDotInt8(ae, xu.MulSign(we), ones)
		c11 = blockDotInt8(ae, xv.MulSign(we), ones)
		h23 := c00.ConcatAddPairsGrouped(c01).ConcatAddPairsGrouped(c10.ConcatAddPairsGrouped(c11))

		// Each tree's dots are its low half plus its high half; pairing the
		// halves across the two trees yields all eight dots in one add.
		dots := h01.SetHi(h23.GetLo()).Add(h23.SetLo(h01.GetHi())).ConvertToFloat32()
		ws := archsimd.LoadFloat32x4Array((*[4]float32)(wPairs[p : p+4]))
		xs01 := archsimd.LoadFloat32x4Array((*[4]float32)(xPairs01[p : p+4]))
		xs23 := archsimd.LoadFloat32x4Array((*[4]float32)(xPairs23[p : p+4]))
		scales := dots.SetLo(ws.Mul(xs01)).SetHi(ws.Mul(xs23))
		sum = sum.Add(dots.Mul(scales))
	}
	sum.StoreArray(out)
	endAVX()
}
