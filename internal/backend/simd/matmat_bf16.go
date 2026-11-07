package simd

import (
	"simd/archsimd"
	"sync"

	instance "github.com/samcharles93/mantle/internal/backend/core"
	"github.com/samcharles93/mantle/pkg/mcf"
)

// smallBatchMaxRows is the largest activation batch MatMatBF16 accepts. Above
// it the packed GEMM amortises its weight packing and wins.
const smallBatchMaxRows = 16

// useSmallBatchBF16 reports whether dst = x * wᵀ for n activation rows should
// run through MatMatBF16 rather than the packed GEMM.
func useSmallBatchBF16(w *Mat, n int) bool {
	return n >= 1 && n <= smallBatchMaxRows && cpu.HasAVX2 &&
		w.DType == mcf.DTypeBF16 && w.Raw != nil
}

// MatMatBF16 computes dst[r*w.R+i] = dot(w row i, x[r*w.C:(r+1)*w.C]) for every
// activation row r in [0,n). w is a BF16 [R,C] weight in w.Raw; x and dst are
// contiguous row-major f32 blocks.
//
// Decode is memory bound, so the point of this kernel is that each weight row
// is read from memory and widened to f32 once per batch, then reused from cache
// for every activation row. The packed GEMM instead copies every weight tile
// through a scalar conversion first, which dominates for small batches.
func MatMatBF16(dst []float32, w *Mat, x []float32, n int) {
	if w.R == 0 || w.C == 0 || n == 0 {
		return
	}
	if n < 0 || n > smallBatchMaxRows {
		panic("matmat: batch size out of range")
	}
	if w.DType != mcf.DTypeBF16 || w.Raw == nil {
		panic("matmat: MatMatBF16 needs BF16 raw weights")
	}
	if len(dst) < n*w.R || len(x) < n*w.C {
		panic("matmat shape mismatch")
	}

	xp := packActivationGroups(x, n, w.C)
	defer putPackedActivations(xp)

	pool := getMatVecPool()
	workers := min(pool.size, (w.R+1)/2)
	if workers <= 1 {
		matMatRangeBF16(dst, w, x, *xp, n, 0, w.R)
		return
	}

	// Keep chunk boundaries even so every worker runs the two-row kernel.
	chunk := (w.R + workers - 1) / workers
	chunk += chunk & 1
	done := <-pool.doneSlots
	active := 0
	for rs := 0; rs < w.R; rs += chunk {
		pool.tasks <- matVecTask{
			dst:  dst,
			w:    w,
			x:    x,
			xp:   *xp,
			rs:   rs,
			re:   min(rs+chunk, w.R),
			n:    n,
			done: done,
		}
		active++
	}
	for range active {
		<-done
	}
	pool.doneSlots <- done
}

var packedActivationPool = sync.Pool{New: func() any { return new([]float32) }}

// packActivationGroups interleaves activation rows in groups of four so the
// 2x4 kernel walks one buffer: for group g and 8-wide chunk j, the 32 values
// at (g*c8+j)*32 are rows 4g..4g+3 of columns [8j,8j+8) in row order. A final
// partial group is zero padded. Columns past the last full chunk stay in x.
func packActivationGroups(x []float32, n, c int) *[]float32 {
	groups, c8 := (n+3)/4, c/8
	need := groups * c8 * 32
	bp := packedActivationPool.Get().(*[]float32)
	if cap(*bp) < need {
		*bp = make([]float32, need)
	}
	buf := (*bp)[:need]
	for g := range groups {
		for j := range c8 {
			out := buf[(g*c8+j)*32 : (g*c8+j+1)*32]
			for q := range 4 {
				dst := out[q*8 : (q+1)*8]
				if row := 4*g + q; row < n {
					copy(dst, x[row*c+8*j:row*c+8*j+8])
				} else {
					clear(dst)
				}
			}
		}
	}
	*bp = buf
	return bp
}

func putPackedActivations(bp *[]float32) { packedActivationPool.Put(bp) }

// matMatRangeBF16 computes output columns [rs,re) for all n activation rows.
// xp is x packed by packActivationGroups.
func matMatRangeBF16(dst []float32, w *Mat, x, xp []float32, n, rs, re int) {
	u16raw, ok := instance.RawUint16LE(w.Raw)
	if !ok || !cpu.HasAVX2 {
		matMatRangeBF16Fallback(dst, w, x, n, rs, re)
		return
	}
	c := w.C
	c8 := c / 8
	i := rs
	for ; i+1 < re; i += 2 {
		w0 := u16raw[i*w.Stride : i*w.Stride+c]
		w1 := u16raw[(i+1)*w.Stride : (i+1)*w.Stride+c]
		for r := 0; r < n; r += 4 {
			g := r / 4
			matMat2x4BF16(dst, w0, w1, xp[g*c8*32:(g+1)*c8*32], x, c, w.R, i, r, min(4, n-r))
		}
	}
	if i < re {
		w0 := u16raw[i*w.Stride : i*w.Stride+c]
		for r := range n {
			dst[r*w.R+i] = dotBF16Row(w0, x[r*c:(r+1)*c])
		}
	}
}

// matMat2x4BF16 accumulates weight rows i, i+1 against activation rows
// r..r+3: eight independent accumulators, each weight chunk widened once. xg
// is this row group's packed activations. Loads go through fixed-size array
// pointers so the loop carries three slices and plain bounds checks. Only the
// first valid rows of the group exist in x and dst; the rest are padding.
func matMat2x4BF16(dst []float32, w0, w1 []uint16, xg, x []float32, c, ldd, i, r, valid int) {
	c8 := c / 8
	var a00, a01, a02, a03, a10, a11, a12, a13 archsimd.Float32x8
	for j := range c8 {
		f0 := archsimd.LoadUint16x8Array((*[8]uint16)(w0[8*j : 8*j+8])).ExtendToUint32().ShiftAllLeft(16).BitsToFloat32()
		f1 := archsimd.LoadUint16x8Array((*[8]uint16)(w1[8*j : 8*j+8])).ExtendToUint32().ShiftAllLeft(16).BitsToFloat32()
		blk := (*[32]float32)(xg[32*j : 32*j+32])

		v := archsimd.LoadFloat32x8Array((*[8]float32)(blk[0:8]))
		a00 = v.MulAdd(f0, a00)
		a10 = v.MulAdd(f1, a10)
		v = archsimd.LoadFloat32x8Array((*[8]float32)(blk[8:16]))
		a01 = v.MulAdd(f0, a01)
		a11 = v.MulAdd(f1, a11)
		v = archsimd.LoadFloat32x8Array((*[8]float32)(blk[16:24]))
		a02 = v.MulAdd(f0, a02)
		a12 = v.MulAdd(f1, a12)
		v = archsimd.LoadFloat32x8Array((*[8]float32)(blk[24:32]))
		a03 = v.MulAdd(f0, a03)
		a13 = v.MulAdd(f1, a13)
	}

	s0 := [4]float32{hsum8(a00), hsum8(a01), hsum8(a02), hsum8(a03)}
	s1 := [4]float32{hsum8(a10), hsum8(a11), hsum8(a12), hsum8(a13)}
	for q := range valid {
		row := x[(r+q)*c : (r+q+1)*c]
		for j := 8 * c8; j < c; j++ {
			s0[q] += bf16ToF32Table(w0[j]) * row[j]
			s1[q] += bf16ToF32Table(w1[j]) * row[j]
		}
		dst[(r+q)*ldd+i], dst[(r+q)*ldd+i+1] = s0[q], s1[q]
	}
}

// dotBF16Row is a single BF16 row dot product with two accumulators.
func dotBF16Row(w []uint16, x []float32) float32 {
	c := len(x)
	if c > 0 {
		_ = w[c-1]
	}
	var a0, a1 archsimd.Float32x8
	j := 0
	for ; j+16 <= c; j += 16 {
		f0 := archsimd.LoadUint16x8(w[j:]).ExtendToUint32().ShiftAllLeft(16).BitsToFloat32()
		f1 := archsimd.LoadUint16x8(w[j+8:]).ExtendToUint32().ShiftAllLeft(16).BitsToFloat32()
		a0 = f0.MulAdd(archsimd.LoadFloat32x8(x[j:]), a0)
		a1 = f1.MulAdd(archsimd.LoadFloat32x8(x[j+8:]), a1)
	}
	for ; j+8 <= c; j += 8 {
		f0 := archsimd.LoadUint16x8(w[j:]).ExtendToUint32().ShiftAllLeft(16).BitsToFloat32()
		a0 = f0.MulAdd(archsimd.LoadFloat32x8(x[j:]), a0)
	}
	s := hsum8(a0.Add(a1))
	for ; j < c; j++ {
		s += bf16ToF32Table(w[j]) * x[j]
	}
	return s
}

func hsum8(v archsimd.Float32x8) float32 {
	var t [8]float32
	v.StoreArray(&t)
	return t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7]
}

// matMatRangeBF16Fallback runs one matvec per activation row when the raw bytes
// cannot be viewed as uint16 or AVX2 is unavailable.
func matMatRangeBF16Fallback(dst []float32, w *Mat, x []float32, n, rs, re int) {
	tmp := make([]float32, w.R)
	for r := range n {
		matVecRangeBF16(tmp, w, x[r*w.C:(r+1)*w.C], rs, re)
		copy(dst[r*w.R+rs:r*w.R+re], tmp[rs:re])
	}
}
