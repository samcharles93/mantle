package simd

import (
	"math"
	"simd/archsimd"
	"sync"
	"sync/atomic"

	instance "github.com/samcharles93/mantle/internal/backend/core"
	"github.com/samcharles93/mantle/pkg/mcf"
)

// quantMaxBlocksPerRow is the fused-row limit in matVecRangeQuantCached, the
// path MatMatQuant must reproduce; wider rows take its per-block loop instead.
const quantMaxBlocksPerRow = 256

// packed4UnsignedMax is the maximum biased 4-bit weight code (7 + 8 = 15)
// produced by decodePacked4RowUnsigned.
const packed4UnsignedMax = 15

// Compile-time guard for the int16 reduction tree in quantDotUnsigned2x2 and
// quantDotUnsigned2x4: eight VPMADDUBSW pair sums (16 products) are added in
// signed int16 before widening via VPMADDWD. That requires every quantised
// activation to stay in [-quantActMaxAbs, +quantActMaxAbs] (never -128), so
// max |sum| = 8 * 2 * 15 * 127 = 30480 <= 32767. Any widening of either bound
// that could overflow int16 fails compilation here.
const _ = uint16(math.MaxInt16 - 8*2*packed4UnsignedMax*quantActMaxAbs)

// quantRows holds n activation rows quantised exactly as matVecWithQuant
// quantises a single input vector (int8 values and one scale per 32-element
// block), plus the interleaved activation, scale and sign-folding bias tables
// for the repacked 8x4, packed 2x4 and cached 2x4 kernels.
type quantRows struct {
	// q8 holds element-order int8 activations for cached-int8 weights; empty
	// when no cached-int8 weight will read this quantRows.
	q8 []int8 // [n, blocks*32]
	// q8Pair is q8 with each block in pair order (16 even elements, then 16
	// odd), matching packed 4-bit weights decoded by decodePacked4RowUnsigned.
	// Filled only when an un-repacked 4-bit weight (or R%8 tail) will read it.
	q8Pair []int8
	// q32 holds element-order int8 activations packed 4 bytes per little-endian
	// int32 for repacked 8-row 4-bit kernels. Filled only when repacked weights
	// will read this quantRows.
	q32    []int32   // [n, blocks*8]
	scales []float32 // [n, blocks]
	// bias[r*blocks+b] is 8*sum(q8[r][b*32:+32]) for packed 4-bit sign
	// folding: dot(w+8, x) - bias == dot(w, x) in exact int32. Filled whenever
	// q8Pair or q32 is filled.
	bias []int32
	// quadScales[(r/4)*blocks*8 + b*8 : +8] is
	// {xs[r][b], xs[r+1][b], xs[r][b], xs[r+1][b],
	//  xs[r+2][b], xs[r+3][b], xs[r+2][b], xs[r+3][b]}
	// for each 4-row group with r+3 < n (used by 2x4 kernels).
	quadScales []float32
	// quadBias mirrors quadScales for bias; filled only when q8Pair is filled.
	quadBias []int32
	// tailPairScales and tailPairBias hold the 4-wide interleave for the single
	// trailing 2-row pair when n%4 >= 2 (used by 2x2 kernels).
	tailPairScales []float32 // [blocks*4]
	tailPairBias   []int32   // [blocks*4]

	// Repacked 8x4 / 8x2 / 8x1 activation tables (filled only when q32 is filled).
	// quadAct[(r/4)*blocks*32 + b*32 + k*4 : +4] holds the 4 consecutive int8
	// slice k (0..7) of rows r..r+3 as one contiguous stream.
	// quadMeta[(r/4)*blocks*8 + b*8 : +8] holds Float32bits(xs[r..r+3][b]) in
	// lanes 0..3 and uint32(bias[r..r+3][b]) in lanes 4..7 so one 256-bit load
	// feeds both scale and bias broadcasts without creating 128-bit XMM temps.
	quadAct  []int32  // [(n/4), blocks*32]
	quadMeta []uint32 // [(n/4), blocks*8]
	// Trailing 2-row pair (when n%4 >= 2) and trailing odd row (when n%2 == 1).
	tailPairAct  []int32  // [blocks*16]
	tailPairMeta []uint32 // [blocks*8]
	oddRowMeta   []uint32 // [blocks*8]

	blocks int
}

var quantRowsPool = sync.Pool{New: func() any { return new(quantRows) }}

// quantRowsTask is a range of activation rows to quantise, rs a multiple of 4
// so every 4-row group (and any trailing 2-row pair or odd row) is whole inside
// one task.
type quantRowsTask struct {
	x      []float32
	c      int
	qr     *quantRows
	rs, re int
}

// run quantises rows [rs, re) and fills the interleaved activation/scale/bias
// tables for every 4-row group (and any trailing pair/odd row this task owns).
func (t *quantRowsTask) run() {
	qr, blocks := t.qr, t.qr.blocks
	rowLen := blocks * 32
	rowLen32 := blocks * 8
	wantNormal := len(qr.q8) > 0
	wantPacked := len(qr.q8Pair) > 0
	wantRepacked := len(qr.q32) > 0
	wantBias := wantPacked || wantRepacked
	want2x4Scales := len(qr.quadScales) > 0
	for r := t.rs; r < t.re; r++ {
		var q8, q8Pair []int8
		var q32, bias []int32
		if wantNormal {
			q8 = qr.q8[r*rowLen : (r+1)*rowLen]
		}
		if wantPacked {
			q8Pair = qr.q8Pair[r*rowLen : (r+1)*rowLen]
		}
		if wantRepacked {
			q32 = qr.q32[r*rowLen32 : (r+1)*rowLen32]
		}
		if wantBias {
			bias = qr.bias[r*blocks : (r+1)*blocks]
		}
		quantizeActivationBlocks(t.x[r*t.c:(r+1)*t.c], blocks, qr.scales[r*blocks:(r+1)*blocks],
			q8, q32, nil, nil, q8Pair, bias)
	}
	r := t.rs
	for ; r+3 < t.re; r += 4 {
		qIdx := r / 4
		xs0 := qr.scales[r*blocks : (r+1)*blocks]
		xs1 := qr.scales[(r+1)*blocks : (r+2)*blocks]
		xs2 := qr.scales[(r+2)*blocks : (r+3)*blocks]
		xs3 := qr.scales[(r+3)*blocks : (r+4)*blocks]
		if want2x4Scales {
			out := qr.quadScales[qIdx*blocks*8 : (qIdx+1)*blocks*8]
			for b := range blocks {
				s0, s1, s2, s3 := xs0[b], xs1[b], xs2[b], xs3[b]
				out[8*b], out[8*b+1], out[8*b+2], out[8*b+3] = s0, s1, s0, s1
				out[8*b+4], out[8*b+5], out[8*b+6], out[8*b+7] = s2, s3, s2, s3
			}
		}
		if wantPacked {
			b0 := qr.bias[r*blocks : (r+1)*blocks]
			b1 := qr.bias[(r+1)*blocks : (r+2)*blocks]
			b2 := qr.bias[(r+2)*blocks : (r+3)*blocks]
			b3 := qr.bias[(r+3)*blocks : (r+4)*blocks]
			bout := qr.quadBias[qIdx*blocks*8 : (qIdx+1)*blocks*8]
			for b := range blocks {
				v0, v1, v2, v3 := b0[b], b1[b], b2[b], b3[b]
				bout[8*b], bout[8*b+1], bout[8*b+2], bout[8*b+3] = v0, v1, v0, v1
				bout[8*b+4], bout[8*b+5], bout[8*b+6], bout[8*b+7] = v2, v3, v2, v3
			}
		}
		if wantRepacked {
			r0 := qr.q32[r*rowLen32 : (r+1)*rowLen32]
			r1 := qr.q32[(r+1)*rowLen32 : (r+2)*rowLen32]
			r2 := qr.q32[(r+2)*rowLen32 : (r+3)*rowLen32]
			r3 := qr.q32[(r+3)*rowLen32 : (r+4)*rowLen32]
			aout := qr.quadAct[qIdx*blocks*32 : (qIdx+1)*blocks*32]
			mout := qr.quadMeta[qIdx*blocks*8 : (qIdx+1)*blocks*8]
			b0 := qr.bias[r*blocks : (r+1)*blocks]
			b1 := qr.bias[(r+1)*blocks : (r+2)*blocks]
			b2 := qr.bias[(r+2)*blocks : (r+3)*blocks]
			b3 := qr.bias[(r+3)*blocks : (r+4)*blocks]
			for b := range blocks {
				base := b * 8
				ao := aout[b*32 : (b+1)*32]
				for k := range 8 {
					ao[4*k], ao[4*k+1], ao[4*k+2], ao[4*k+3] = r0[base+k], r1[base+k], r2[base+k], r3[base+k]
				}
				mout[8*b] = math.Float32bits(xs0[b])
				mout[8*b+1] = math.Float32bits(xs1[b])
				mout[8*b+2] = math.Float32bits(xs2[b])
				mout[8*b+3] = math.Float32bits(xs3[b])
				mout[8*b+4] = uint32(b0[b])
				mout[8*b+5] = uint32(b1[b])
				mout[8*b+6] = uint32(b2[b])
				mout[8*b+7] = uint32(b3[b])
			}
		}
	}
	if r+1 < t.re {
		xs0 := qr.scales[r*blocks : (r+1)*blocks]
		xs1 := qr.scales[(r+1)*blocks : (r+2)*blocks]
		if len(qr.tailPairScales) > 0 {
			out := qr.tailPairScales[:blocks*4]
			for b := range blocks {
				out[4*b], out[4*b+1], out[4*b+2], out[4*b+3] = xs0[b], xs1[b], xs0[b], xs1[b]
			}
		}
		if wantPacked {
			b0 := qr.bias[r*blocks : (r+1)*blocks]
			b1 := qr.bias[(r+1)*blocks : (r+2)*blocks]
			bout := qr.tailPairBias[:blocks*4]
			for b := range blocks {
				bout[4*b], bout[4*b+1], bout[4*b+2], bout[4*b+3] = b0[b], b1[b], b0[b], b1[b]
			}
		}
		if wantRepacked {
			r0 := qr.q32[r*rowLen32 : (r+1)*rowLen32]
			r1 := qr.q32[(r+1)*rowLen32 : (r+2)*rowLen32]
			aout := qr.tailPairAct[:blocks*16]
			mout := qr.tailPairMeta[:blocks*8]
			b0 := qr.bias[r*blocks : (r+1)*blocks]
			b1 := qr.bias[(r+1)*blocks : (r+2)*blocks]
			for b := range blocks {
				base := b * 8
				ao := aout[b*16 : (b+1)*16]
				for k := range 8 {
					ao[2*k], ao[2*k+1] = r0[base+k], r1[base+k]
				}
				mout[8*b] = math.Float32bits(xs0[b])
				mout[8*b+1] = math.Float32bits(xs1[b])
				mout[8*b+2], mout[8*b+3] = 0, 0
				mout[8*b+4] = uint32(b0[b])
				mout[8*b+5] = uint32(b1[b])
				mout[8*b+6], mout[8*b+7] = 0, 0
			}
		}
		r += 2
	}
	if r < t.re && wantRepacked {
		xs := qr.scales[r*blocks : (r+1)*blocks]
		bs := qr.bias[r*blocks : (r+1)*blocks]
		mout := qr.oddRowMeta[:blocks*8]
		for b := range blocks {
			mout[8*b] = math.Float32bits(xs[b])
			mout[8*b+1], mout[8*b+2], mout[8*b+3] = 0, 0, 0
			mout[8*b+4] = uint32(bs[b])
			mout[8*b+5], mout[8*b+6], mout[8*b+7] = 0, 0, 0
		}
	}
}

// quantizeRows fills qr for n activation rows, spreading 4-row groups over the
// worker pool so the caller does not serialise the whole chunk before the
// matmul dispatch. Each row is quantised by the same function whichever task
// takes it, so the bytes do not depend on the split.
func quantizeRows(qr *quantRows, x []float32, n, c int) {
	quads := (n + 3) / 4
	pool := getMatVecPool()
	workers := 0
	if pool != nil && pool.size > 1 {
		workers = min(pool.size, quads)
	}
	if workers < 2 {
		t := quantRowsTask{x: x, c: c, qr: qr, rs: 0, re: n}
		t.run()
		return
	}
	tasks := make([]quantRowsTask, workers)
	done := <-pool.doneSlots
	for i := range workers {
		tasks[i] = quantRowsTask{
			x: x, c: c, qr: qr,
			rs: 4 * (i * quads / workers),
			re: min(n, 4*((i+1)*quads/workers)),
		}
		pool.tasks <- matVecTask{qrows: &tasks[i], done: done}
	}
	for range workers {
		<-done
	}
	pool.doneSlots <- done
}

func getQuantRows(x []float32, n, c int, packed bool) *quantRows {
	return getQuantRowsLayout(x, n, c, !packed, packed, false)
}

func getQuantRowsForWeight(x []float32, n int, w *Mat) *quantRows {
	var needNormal, needPacked, needRepacked bool
	if batchWeightsRepacked4x8(w) {
		needRepacked = true
		if w.R%8 != 0 {
			needPacked = true
		}
	} else if batchWeightsPacked(w) {
		needPacked = true
	} else {
		needNormal = true
	}
	return getQuantRowsLayout(x, n, w.C, needNormal, needPacked, needRepacked)
}

func getQuantRowsLayout(x []float32, n, c int, needNormal, needPacked, needRepacked bool) *quantRows {
	qr := quantRowsPool.Get().(*quantRows)
	blocks := (c + 31) / 32
	qr.blocks = blocks
	qr.scales = ensureFloat32Slice(qr.scales, n*blocks)
	need2x4Scales := needNormal || needPacked
	if need2x4Scales {
		qr.quadScales = ensureFloat32Slice(qr.quadScales, (n/4)*blocks*8)
		if n%4 >= 2 {
			qr.tailPairScales = ensureFloat32Slice(qr.tailPairScales, blocks*4)
		} else {
			qr.tailPairScales = qr.tailPairScales[:0]
		}
	} else {
		qr.quadScales = qr.quadScales[:0]
		qr.tailPairScales = qr.tailPairScales[:0]
	}
	if needNormal {
		qr.q8 = ensureInt8Slice(qr.q8, n*blocks*32)
	} else {
		qr.q8 = qr.q8[:0]
	}
	if needPacked || needRepacked {
		qr.bias = ensureInt32Slice(qr.bias, n*blocks)
	} else {
		qr.bias = qr.bias[:0]
	}
	if needPacked {
		qr.q8Pair = ensureInt8Slice(qr.q8Pair, n*blocks*32)
		qr.quadBias = ensureInt32Slice(qr.quadBias, (n/4)*blocks*8)
		if n%4 >= 2 {
			qr.tailPairBias = ensureInt32Slice(qr.tailPairBias, blocks*4)
		} else {
			qr.tailPairBias = qr.tailPairBias[:0]
		}
	} else {
		qr.q8Pair = qr.q8Pair[:0]
		qr.quadBias = qr.quadBias[:0]
		qr.tailPairBias = qr.tailPairBias[:0]
	}
	if needRepacked {
		qr.q32 = ensureInt32Slice(qr.q32, n*blocks*8)
		qr.quadAct = ensureInt32Slice(qr.quadAct, (n/4)*blocks*32)
		qr.quadMeta = ensureUint32Slice(qr.quadMeta, (n/4)*blocks*8)
		if n%4 >= 2 {
			qr.tailPairAct = ensureInt32Slice(qr.tailPairAct, blocks*16)
			qr.tailPairMeta = ensureUint32Slice(qr.tailPairMeta, blocks*8)
		} else {
			qr.tailPairAct = qr.tailPairAct[:0]
			qr.tailPairMeta = qr.tailPairMeta[:0]
		}
		if n%2 == 1 {
			qr.oddRowMeta = ensureUint32Slice(qr.oddRowMeta, blocks*8)
		} else {
			qr.oddRowMeta = qr.oddRowMeta[:0]
		}
	} else {
		qr.q32 = qr.q32[:0]
		qr.quadAct = qr.quadAct[:0]
		qr.quadMeta = qr.quadMeta[:0]
		qr.tailPairAct = qr.tailPairAct[:0]
		qr.tailPairMeta = qr.tailPairMeta[:0]
		qr.oddRowMeta = qr.oddRowMeta[:0]
	}
	quantizeRows(qr, x, n, c)
	return qr
}

func putQuantRows(qr *quantRows) {
	if qr != nil {
		quantRowsPool.Put(qr)
	}
}

// prepareBatchQuantRows quantises x once for a group of projections sharing
// the same input block (such as Q/K/V/AttnGate or FfnUp/FfnGate). It returns
// nil when none of weights will take MatMatQuant at batch size n. Mixed
// weight groups (for instance repacked k4 alongside cached k6) populate all
// required layouts in one pass.
func prepareBatchQuantRows(x []float32, n, c int, weights ...*Mat) *quantRows {
	var needNormal, needPacked, needRepacked bool
	for _, w := range weights {
		if w == nil || !useBatchQuant(w, n) {
			continue
		}
		if batchWeightsRepacked4x8(w) {
			needRepacked = true
			if w.R%8 != 0 {
				needPacked = true
			}
		} else if batchWeightsPacked(w) {
			needPacked = true
		} else {
			needNormal = true
		}
	}
	if !needNormal && !needPacked && !needRepacked {
		return nil
	}
	return getQuantRowsLayout(x, n, c, needNormal, needPacked, needRepacked)
}

// zeroFloat32s seeds the quantDot accumulators with VEX loads. The zero value
// of a Float32x4 compiles to a legacy-SSE move of X15, which takes an assist on
// Golden Cove once the kernel's 256-bit constants are live.
var zeroFloat32s [8]float32

// canBatchQuant reports whether w's layout and the host ISA are supported by
// MatMatQuant. It needs, on an AVX2 host, either the int8 cache
// matVecRangeQuantCached uses or a packed 4-bit payload (which
// matVecRangePacked4 matches bit for bit), and the shape the fused-row
// kernel takes, so batched and per-row products produce identical bits.
func canBatchQuant(w *Mat) bool {
	if !cpu.HasAVX2 || w.Raw == nil || !mcf.DTypeRequiresAligned64(w.DType) ||
		mcf.IsUQDType(w.DType) || w.C%32 != 0 {
		return false
	}
	if w.Quant != nil && w.Quant.ValidFor(w) {
		return w.Quant.BlocksPerRow <= quantMaxBlocksPerRow
	}
	return isPacked4DType(w.DType) && w.C/32 <= quantMaxBlocksPerRow
}

// useBatchQuant reports whether dst = x * wᵀ for n activation rows should run
// through MatMatQuant rather than per-row MatVec. At n == 1, MatVec's 8-row
// grouped packed kernel avoids per-call weight expansion and runs ~1.6-2x
// faster; from n >= 2 upward MatMatQuant amortises the weight read and wins.
func useBatchQuant(w *Mat, n int) bool {
	return n >= 2 && canBatchQuant(w)
}

// batchWeightsPacked reports whether MatMatQuant reads w from its packed
// payload rather than the int8 cache.
func batchWeightsPacked(w *Mat) bool {
	return (w.Quant == nil || !w.Quant.ValidFor(w)) && isPacked4DType(w.DType)
}

// MatMatQuant computes dst[r*w.R+i] = dot(w row i, x[r*w.C:(r+1)*w.C]) for
// every activation row r in [0,n), for a quantised w with a valid int8 cache
// or packed 4-bit payload.
//
// Every output equals what MatVec produces for that activation row alone: the
// rows are quantised the same way, block dots are exact integers, and each
// output's scale combine performs the same float32 operations in the same
// block order (one SIMD lane per output).
func MatMatQuant(dst []float32, w *Mat, x []float32, n int) {
	if w.R == 0 || w.C == 0 || n == 0 {
		return
	}
	if n < 1 || !canBatchQuant(w) {
		panic("matmat: MatMatQuant needs a cached int8 or packed 4-bit quant weight on an AVX2 host")
	}
	if len(dst) < n*w.R || len(x) < n*w.C {
		panic("matmat shape mismatch")
	}

	qr := getQuantRowsForWeight(x, n, w)
	defer putQuantRows(qr)
	matMatQuantWithRows(dst, w, qr, n)
}

// matMatQuantWithRows dispatches MatMatQuant using already-quantised activation
// rows qr. Callers must ensure qr matches w's layout (packed vs cached) and
// column count.
func matMatQuantWithRows(dst []float32, w *Mat, qr *quantRows, n int) {
	if w.R == 0 || w.C == 0 || n == 0 {
		return
	}
	if len(dst) < n*w.R {
		panic("matmat shape mismatch")
	}
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
// scales of the current weight row pair or 8-row repacked group, and for
// un-repacked 4-bit weights the decoded unsigned [0,15] row pair and scales.
type quantKernelScratch struct {
	wQuads       []float32 // [blocks, 8]: {s0, s0, s1, s1, s0, s0, s1, s1}
	wPairs       []float32 // [blocks, 4]: {s0, s0, s1, s1} for trailing 2x2
	wGroupScales []float32 // [blocks, 8]: block-major scales for 8-row repacked group
	urows        []uint8   // [2, blocks*32] decoded packed rows in [0,15], pair order
	scales       []float32 // [2, blocks] their block scales
}

// matMatRangeQuant computes output columns [rs,re) for all n activation rows.
// worker, when non-nil, supplies the scratch buffers.
func matMatRangeQuant(dst []float32, w *Mat, qr *quantRows, n, rs, re int, worker *matVecWorker) {
	bpr := w.C / 32
	c := bpr * 32
	packed := batchWeightsPacked(w)
	repacked := packed && w.Packed4x8 != nil && w.Packed4x8.ValidFor(w) && rs%8 == 0 && len(qr.q32) > 0
	needTailPacked := packed && (!repacked || re%8 != 0)
	needQuads := n >= 4 && (!repacked || needTailPacked)
	needTailPair := n%4 >= 2 && (!repacked || needTailPacked)
	ensure := func(s *quantKernelScratch) {
		if repacked {
			s.wGroupScales = ensureFloat32Slice(s.wGroupScales, 8*bpr)
			s.scales = ensureFloat32Slice(s.scales, bpr)
		}
		if needQuads {
			s.wQuads = ensureFloat32Slice(s.wQuads, 8*bpr)
		}
		if needTailPair {
			s.wPairs = ensureFloat32Slice(s.wPairs, 4*bpr)
		}
		if needTailPacked {
			s.urows = ensureUint8Slice(s.urows, 2*c)
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
	var wQuads, wPairs []float32
	if needQuads {
		wQuads = scratch.wQuads[:8*bpr]
	}
	if needTailPair {
		wPairs = scratch.wPairs[:4*bpr]
	}
	ld := w.R

	fillWeightScales := func(s0, s1 []float32) {
		if needQuads {
			for b := range bpr {
				a, d := s0[b], s1[b]
				wQuads[8*b], wQuads[8*b+1], wQuads[8*b+2], wQuads[8*b+3] = a, a, d, d
				wQuads[8*b+4], wQuads[8*b+5], wQuads[8*b+6], wQuads[8*b+7] = a, a, d, d
			}
		}
		if needTailPair {
			for b := range bpr {
				wPairs[4*b], wPairs[4*b+1], wPairs[4*b+2], wPairs[4*b+3] = s0[b], s0[b], s1[b], s1[b]
			}
		}
	}

	if packed {
		layout, err := instance.QuantLayoutForMat(w.R, w.C, w.DType, len(w.Raw))
		if err != nil {
			panic(err)
		}
		i := rs
		if repacked {
			p4 := w.Packed4x8
			wGroupScales := scratch.wGroupScales[:8*bpr]
			for ; i+7 < re; i += 8 {
				packed4GroupWeightScales(wGroupScales, layout, w.Raw, i, scratch.scales)
				g := i / 8
				wData := p4.Data[g*bpr*128 : (g+1)*bpr*128]
				r := 0
				for ; r+3 < n; r += 4 {
					qIdx := r / 4
					var out [32]float32
					quantDotRepacked8x4(
						wData, wGroupScales,
						qr.quadAct[qIdx*bpr*32:(qIdx+1)*bpr*32],
						qr.quadMeta[qIdx*bpr*8:(qIdx+1)*bpr*8],
						&out,
					)
					copy(dst[r*ld+i:r*ld+i+8], out[0:8])
					copy(dst[(r+1)*ld+i:(r+1)*ld+i+8], out[8:16])
					copy(dst[(r+2)*ld+i:(r+2)*ld+i+8], out[16:24])
					copy(dst[(r+3)*ld+i:(r+3)*ld+i+8], out[24:32])
				}
				if r+1 < n {
					var out [16]float32
					quantDotRepacked8x2(
						wData, wGroupScales,
						qr.tailPairAct, qr.tailPairMeta,
						&out,
					)
					copy(dst[r*ld+i:r*ld+i+8], out[0:8])
					copy(dst[(r+1)*ld+i:(r+1)*ld+i+8], out[8:16])
					r += 2
				}
				if r < n {
					var out [8]float32
					quantDotRepacked8x1(
						wData, wGroupScales,
						qr.q32[r*bpr*8:(r+1)*bpr*8],
						qr.oddRowMeta,
						&out,
					)
					copy(dst[r*ld+i:r*ld+i+8], out[0:8])
				}
			}
			if i >= re {
				return
			}
		}

		data := w.Raw[layout.DataOff : layout.DataOff+layout.DataBytes]
		act := qr.q8Pair
		u0 := scratch.urows[:c]
		u1 := scratch.urows[c : 2*c]
		s0 := scratch.scales[:bpr]
		s1 := scratch.scales[bpr : 2*bpr]

		for ; i+1 < re; i += 2 {
			decodePacked4RowUnsigned(u0, data[i*bpr*16:(i+1)*bpr*16], bpr)
			packed4RowScales(s0, layout, w.Raw, i)
			decodePacked4RowUnsigned(u1, data[(i+1)*bpr*16:(i+2)*bpr*16], bpr)
			packed4RowScales(s1, layout, w.Raw, i+1)
			fillWeightScales(s0, s1)
			r := 0
			for ; r+3 < n; r += 4 {
				x0 := act[r*c : (r+1)*c]
				x1 := act[(r+1)*c : (r+2)*c]
				x2 := act[(r+2)*c : (r+3)*c]
				x3 := act[(r+3)*c : (r+4)*c]
				xQuads := qr.quadScales[(r/4)*bpr*8 : (r/4+1)*bpr*8]
				xBias := qr.quadBias[(r/4)*bpr*8 : (r/4+1)*bpr*8]
				var out [8]float32
				quantDotUnsigned2x4(u0, u1, x0, x1, x2, x3, wQuads, xQuads, xBias, &out)
				dst[r*ld+i], dst[(r+1)*ld+i] = out[0], out[1]
				dst[r*ld+i+1], dst[(r+1)*ld+i+1] = out[2], out[3]
				dst[(r+2)*ld+i], dst[(r+3)*ld+i] = out[4], out[5]
				dst[(r+2)*ld+i+1], dst[(r+3)*ld+i+1] = out[6], out[7]
			}
			if r+1 < n {
				x0 := act[r*c : (r+1)*c]
				x1 := act[(r+1)*c : (r+2)*c]
				var out [4]float32
				quantDotUnsigned2x2(u0, u1, x0, x1, wPairs, qr.tailPairScales, qr.tailPairBias, &out)
				dst[r*ld+i], dst[(r+1)*ld+i] = out[0], out[1]
				dst[r*ld+i+1], dst[(r+1)*ld+i+1] = out[2], out[3]
				r += 2
			}
			if r < n {
				dst[r*ld+i] = quantDotUnsignedRow(u0, s0, act, qr, r)
				dst[r*ld+i+1] = quantDotUnsignedRow(u1, s1, act, qr, r)
			}
		}
		if i < re {
			decodePacked4RowUnsigned(u0, data[i*bpr*16:(i+1)*bpr*16], bpr)
			packed4RowScales(s0, layout, w.Raw, i)
			for r := range n {
				dst[r*ld+i] = quantDotUnsignedRow(u0, s0, act, qr, r)
			}
		}
		return
	}

	qc := w.Quant
	act := qr.q8
	i := rs
	for ; i+1 < re; i += 2 {
		w0, s0 := qc.Q[i*c:(i+1)*c], qc.Scales[i*bpr:(i+1)*bpr]
		w1, s1 := qc.Q[(i+1)*c:(i+2)*c], qc.Scales[(i+1)*bpr:(i+2)*bpr]
		fillWeightScales(s0, s1)
		r := 0
		for ; r+3 < n; r += 4 {
			x0 := act[r*c : (r+1)*c]
			x1 := act[(r+1)*c : (r+2)*c]
			x2 := act[(r+2)*c : (r+3)*c]
			x3 := act[(r+3)*c : (r+4)*c]
			xQuads := qr.quadScales[(r/4)*bpr*8 : (r/4+1)*bpr*8]
			var out [8]float32
			quantDot2x4(w0, w1, x0, x1, x2, x3, wQuads, xQuads, &out)
			dst[r*ld+i], dst[(r+1)*ld+i] = out[0], out[1]
			dst[r*ld+i+1], dst[(r+1)*ld+i+1] = out[2], out[3]
			dst[(r+2)*ld+i], dst[(r+3)*ld+i] = out[4], out[5]
			dst[(r+2)*ld+i+1], dst[(r+3)*ld+i+1] = out[6], out[7]
		}
		if r+1 < n {
			x0 := act[r*c : (r+1)*c]
			x1 := act[(r+1)*c : (r+2)*c]
			var out [4]float32
			quantDot2x2(w0, w1, x0, x1, wPairs, qr.tailPairScales, &out)
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
		w0, s0 := qc.Q[i*c:(i+1)*c], qc.Scales[i*bpr:(i+1)*bpr]
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

// quantDotUnsignedRow is quantDotRow for unsigned [0,15] packed 4-bit weights:
// dot(w+8, x) - 8*sum(x) == dot(w, x) in exact int32.
func quantDotUnsignedRow(wrow []uint8, ws []float32, act []int8, qr *quantRows, r int) float32 {
	bpr := len(ws)
	c := bpr * 32
	x := act[r*c : (r+1)*c]
	xs := qr.scales[r*bpr : (r+1)*bpr]
	xb := qr.bias[r*bpr : (r+1)*bpr]
	ones := archsimd.BroadcastInt16x16(1)
	var dots [quantMaxBlocksPerRow]int32
	for b := range bpr {
		w := archsimd.LoadUint8x32Array((*[32]uint8)(wrow[b*32 : b*32+32]))
		v := archsimd.LoadInt8x32Array((*[32]int8)(x[b*32 : b*32+32]))
		var tmp [8]int32
		blockDotInt8(w, v, ones).StoreArray(&tmp)
		dots[b] = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7]
	}
	endAVX()
	var sum float32
	for b, dot := range dots[:bpr] {
		sum += float32(dot-xb[b]) * (ws[b] * xs[b])
	}
	return sum
}

// blockDotInt8 returns the eight int32 partial sums of one 32-element block
// dot, given unsigned uint8 weights and signed int8 activations. It is exact
// whenever every VPMADDUBSW pair sum stays inside int16: both |w|<=128,
// |x|<=127 (max 2*128*127 = 32512) and unsigned 4-bit u in [0,15], |x|<=127
// (max 2*15*127 = 3810) satisfy that with zero saturation.
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

// quantDotUnsigned2x2 is quantDot2x2 for unsigned [0,15] packed 4-bit weights
// (w + 8), dropping Abs and MulSign, reducing all four products in int16,
// widening with one VPMADDWD, and subtracting the exact per-block int32
// correction 8*sum(x) before converting to float32.
//
// Invariant: callers must supply activations clamped to [-127, +127] (never
// -128, enforced by quantActMaxAbs in quant_act.go) so each VPMADDUBSW pair
// sum is at most 2*15*127 = 3810 and summing eight pairs stays inside signed
// int16 (30480 <= 32767) with zero overflow.
func quantDotUnsigned2x2(w0, w1 []uint8, x0, x1 []int8, wPairs, xPairs []float32, xBias []int32, out *[4]float32) {
	np := len(wPairs) / 4 * 4
	n := np * 8
	w0, w1 = w0[:n], w1[:n]
	x0, x1 = x0[:n], x1[:n]
	wPairs, xPairs, xBias = wPairs[:np], xPairs[:np], xBias[:np]
	ones := archsimd.BroadcastInt16x8(1)
	sum := archsimd.LoadFloat32x4Array((*[4]float32)(zeroFloat32s[:4]))
	for off, p := 0, 0; off+32 <= n && p+4 <= np; off, p = off+32, p+4 {
		wa := archsimd.LoadUint8x32Array((*[32]uint8)(w0[off : off+32]))
		we := archsimd.LoadUint8x32Array((*[32]uint8)(w1[off : off+32]))
		xu := archsimd.LoadInt8x32Array((*[32]int8)(x0[off : off+32]))
		xv := archsimd.LoadInt8x32Array((*[32]int8)(x1[off : off+32]))
		c00 := wa.DotProductPairsSaturated(xu)
		c01 := wa.DotProductPairsSaturated(xv)
		c10 := we.DotProductPairsSaturated(xu)
		c11 := we.DotProductPairsSaturated(xv)

		h := c00.ConcatAddPairsGrouped(c01).ConcatAddPairsGrouped(c10.ConcatAddPairsGrouped(c11))
		pb := archsimd.LoadInt32x4Array((*[4]int32)(xBias[p : p+4]))
		dots := h.GetLo().Add(h.GetHi()).DotProductPairs(ones).Sub(pb).ConvertToFloat32()
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
// All vectors inside the loop are 256 bits wide for a codegen reason as well
// as a count one: Go spills a Float32x4 with a legacy-SSE move, which costs an
// assist per block on Golden Cove whenever register pressure forces one (see
// endAVX), while a Float32x8 spills with a VEX move.
//
// The loop carries no bounds checks beyond one per operand family: each
// operand is resliced once to n (or nq) elements, so its length is the very
// value the loop condition tests and the compiler proves the remaining
// [off:off+32] and [q:q+8] slices in range.
func quantDot2x4(w0, w1, x0, x1, x2, x3 []int8, wQuads, xQuads []float32, out *[8]float32) {
	nq := len(wQuads) / 8 * 8
	n := nq * 4
	w0, w1 = w0[:n], w1[:n]
	x0, x1, x2, x3 = x0[:n], x1[:n], x2[:n], x3[:n]
	wQuads, xQuads = wQuads[:nq], xQuads[:nq]
	ones := archsimd.BroadcastInt16x16(1)
	sum := archsimd.LoadFloat32x8Array(&zeroFloat32s)
	for off, q := 0, 0; off+32 <= n && q+8 <= nq; off, q = off+32, q+8 {
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
		dots := h01.ConcatPermute128Scalars(0, 2, h23).Add(h01.ConcatPermute128Scalars(1, 3, h23)).ConvertToFloat32()
		ws := archsimd.LoadFloat32x8Array((*[8]float32)(wQuads[q : q+8]))
		xs := archsimd.LoadFloat32x8Array((*[8]float32)(xQuads[q : q+8]))
		sum = sum.Add(dots.Mul(ws.Mul(xs)))
	}
	sum.StoreArray(out)
	endAVX()
}

// quantDotUnsigned2x4 is quantDot2x4 for unsigned [0,15] packed 4-bit weights
// (w + 8): no Abs and no per-product MulSign. Subtracting the precomputed
// 8*sum(x) bias recovers the exact signed block dots before the float32 scale
// fold, keeping every output bit-identical to quantDot2x4.
//
// Invariant: callers must supply activations clamped to [-127, +127] (never
// -128, enforced by quantActMaxAbs in quant_act.go). Under that bound every
// VPMADDUBSW pair sum has magnitude at most 2*15*127 = 3810, so the two-level
// VPHADDW tree and cross-lane half sum reach at most 8*3810 = 30480 <= 32767
// (exact in signed int16), letting one VPMADDWD widen all eight 32-element
// block dots to exact int32 instead of eight VPMADDWD per block.
func quantDotUnsigned2x4(w0, w1 []uint8, x0, x1, x2, x3 []int8, wQuads, xQuads []float32, xBias []int32, out *[8]float32) {
	nq := len(wQuads) / 8 * 8
	n := nq * 4
	w0, w1 = w0[:n], w1[:n]
	x0, x1, x2, x3 = x0[:n], x1[:n], x2[:n], x3[:n]
	wQuads, xQuads, xBias = wQuads[:nq], xQuads[:nq], xBias[:nq]
	ones := archsimd.BroadcastInt16x16(1)
	sum := archsimd.LoadFloat32x8Array(&zeroFloat32s)
	for off, q := 0, 0; off+32 <= n && q+8 <= nq; off, q = off+32, q+8 {
		wa := archsimd.LoadUint8x32Array((*[32]uint8)(w0[off : off+32]))
		we := archsimd.LoadUint8x32Array((*[32]uint8)(w1[off : off+32]))

		xu := archsimd.LoadInt8x32Array((*[32]int8)(x0[off : off+32]))
		xv := archsimd.LoadInt8x32Array((*[32]int8)(x1[off : off+32]))
		c00 := wa.DotProductPairsSaturated(xu)
		c01 := wa.DotProductPairsSaturated(xv)
		c10 := we.DotProductPairsSaturated(xu)
		c11 := we.DotProductPairsSaturated(xv)
		h01 := c00.ConcatAddPairsGrouped(c01).ConcatAddPairsGrouped(c10.ConcatAddPairsGrouped(c11))

		xu = archsimd.LoadInt8x32Array((*[32]int8)(x2[off : off+32]))
		xv = archsimd.LoadInt8x32Array((*[32]int8)(x3[off : off+32]))
		c00 = wa.DotProductPairsSaturated(xu)
		c01 = wa.DotProductPairsSaturated(xv)
		c10 = we.DotProductPairsSaturated(xu)
		c11 = we.DotProductPairsSaturated(xv)
		h23 := c00.ConcatAddPairsGrouped(c01).ConcatAddPairsGrouped(c10.ConcatAddPairsGrouped(c11))

		s16 := h01.ConcatPermute128Scalars(0, 2, h23).Add(h01.ConcatPermute128Scalars(1, 3, h23))
		qb := archsimd.LoadInt32x8Array((*[8]int32)(xBias[q : q+8]))
		dots := s16.DotProductPairs(ones).Sub(qb).ConvertToFloat32()
		ws := archsimd.LoadFloat32x8Array((*[8]float32)(wQuads[q : q+8]))
		xs := archsimd.LoadFloat32x8Array((*[8]float32)(xQuads[q : q+8]))
		sum = sum.Add(dots.Mul(ws.Mul(xs)))
	}
	sum.StoreArray(out)
	endAVX()
}
