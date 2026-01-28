package tensor

import (
	"runtime"
	"sync"

	"github.com/samcharles93/mantle/pkg/mcf"
	"simd/archsimd"
)

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

var matVecPoolOnce sync.Once

func getMatVecPool() *matVecPool {
	matVecPoolOnce.Do(func() {
		matVecWorkPool = newMatVecPool()
	})
	return matVecWorkPool
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
	if len(dst) < w.R || len(x) < w.C {
		panic("matvec shape mismatch")
	}

	pool := getMatVecPool()
	workers := pool.size
	if workers > w.R {
		workers = w.R
	}

	if workers <= 1 {
		matVecRange(dst, w, x, 0, w.R)
		return
	}

	chunk := (w.R + workers - 1) / workers
	done := <-pool.doneSlots

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
		pool.tasks <- matVecTask{
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
	pool.doneSlots <- done
}

func matVecRange(dst []float32, w *Mat, x []float32, rs, re int) {
	if w.Raw != nil && w.DType != mcf.DTypeF32 {
		switch w.DType {
		case mcf.DTypeBF16:
			matVecRangeBF16(dst, w, x, rs, re)
			return
		case mcf.DTypeF16:
			matVecRangeF16(dst, w, x, rs, re)
			return
		default:
			panic("unsupported dtype for matvec")
		}
	}

	if cpu.HasAVX2 {
		matVecRangeF32SIMD(dst, w, x, rs, re)
		return
	}
	matVecRangeF32Scalar(dst, w, x, rs, re)
}

// matVecRangeF32Scalar computes matvec for F32 using scalar operations.
func matVecRangeF32Scalar(dst []float32, w *Mat, x []float32, rs, re int) {
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

// matVecRangeF32SIMD computes matvec for F32 using AVX2 SIMD.
// Uses a single accumulator to minimize register pressure.
func matVecRangeF32SIMD(dst []float32, w *Mat, x []float32, rs, re int) {
	c := w.C
	for i := rs; i < re; i++ {
		row := w.Data[i*w.Stride : i*w.Stride+w.C]

		// Single accumulator - reduces register pressure
		var acc archsimd.Float32x8
		j := 0
		// Process 8 elements at a time
		for ; j+8 <= c; j += 8 {
			vrow := archsimd.LoadFloat32x8Slice(row[j:])
			vx := archsimd.LoadFloat32x8Slice(x[j:])
			acc = acc.Add(vrow.Mul(vx))
		}

		// Horizontal reduction: store to array and sum scalarly
		var tmp [8]float32
		acc.Store(&tmp)
		var sum float32
		sum += tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7]

		// Handle remaining elements
		for ; j < c; j++ {
			sum += row[j] * x[j]
		}
		dst[i] = sum
	}
}

func matVecRangeBF16(dst []float32, w *Mat, x []float32, rs, re int) {
	if cpu.HasAVX2 {
		matVecRangeBF16SIMD(dst, w, x, rs, re)
		return
	}
	matVecRangeBF16Scalar(dst, w, x, rs, re)
}

// matVecRangeBF16Scalar computes matvec for BF16 using scalar operations.
func matVecRangeBF16Scalar(dst []float32, w *Mat, x []float32, rs, re int) {
	raw := w.Raw
	if u16raw, ok := rawUint16LE(raw); ok {
		for i := rs; i < re; i++ {
			rowBase := i * w.Stride
			row := u16raw[rowBase : rowBase+w.Stride]
			if w.C > 0 {
				// Help bounds-check elimination for the hot inner loop.
				_ = row[w.C-1]
			}
			var sum float32
			j := 0
			for ; j+7 < w.C; j += 8 {
				sum += bf16ToF32Table(row[j+0])*x[j+0] +
					bf16ToF32Table(row[j+1])*x[j+1] +
					bf16ToF32Table(row[j+2])*x[j+2] +
					bf16ToF32Table(row[j+3])*x[j+3] +
					bf16ToF32Table(row[j+4])*x[j+4] +
					bf16ToF32Table(row[j+5])*x[j+5] +
					bf16ToF32Table(row[j+6])*x[j+6] +
					bf16ToF32Table(row[j+7])*x[j+7]
			}
			for ; j < w.C; j++ {
				sum += bf16ToF32Table(row[j]) * x[j]
			}
			dst[i] = sum
		}
		return
	}

	rowBytes := w.Stride * 2
	for i := rs; i < re; i++ {
		off := i * rowBytes
		if w.C > 0 {
			// Help bounds-check elimination for the hot inner loop.
			_ = raw[off+(w.C-1)*2+1]
		}
		var sum float32
		j := 0
		offj := off
		for ; j+7 < w.C; j += 8 {
			u0 := u16le(raw, offj+0)
			u1 := u16le(raw, offj+2)
			u2 := u16le(raw, offj+4)
			u3 := u16le(raw, offj+6)
			u4 := u16le(raw, offj+8)
			u5 := u16le(raw, offj+10)
			u6 := u16le(raw, offj+12)
			u7 := u16le(raw, offj+14)
			sum += bf16ToF32Table(u0)*x[j+0] +
				bf16ToF32Table(u1)*x[j+1] +
				bf16ToF32Table(u2)*x[j+2] +
				bf16ToF32Table(u3)*x[j+3] +
				bf16ToF32Table(u4)*x[j+4] +
				bf16ToF32Table(u5)*x[j+5] +
				bf16ToF32Table(u6)*x[j+6] +
				bf16ToF32Table(u7)*x[j+7]
			offj += 16
		}
		for ; j < w.C; j++ {
			u := u16le(raw, offj)
			sum += bf16ToF32Table(u) * x[j]
			offj += 2
		}
		dst[i] = sum
	}
}

// matVecRangeBF16SIMD computes matvec for BF16 using AVX2 SIMD.
// Uses a single accumulator to minimize register pressure.
func matVecRangeBF16SIMD(dst []float32, w *Mat, x []float32, rs, re int) {
	raw := w.Raw
	if u16raw, ok := rawUint16LE(raw); ok {
		c := w.C
		for i := rs; i < re; i++ {
			rowBase := i * w.Stride
			row := u16raw[rowBase : rowBase+w.Stride]
			if c > 0 {
				_ = row[c-1]
			}

			// Single accumulator
			var acc archsimd.Float32x8
			j := 0
			// Process 8 elements at a time
			for ; j+8 <= c; j += 8 {
				// Load 8 BF16 values as uint16, convert to F32
				vu := archsimd.LoadUint16x8Slice(row[j:])
				vf := vu.ExtendToUint32().ShiftAllLeft(16).AsFloat32x8()
				vx := archsimd.LoadFloat32x8Slice(x[j:])
				acc = acc.Add(vf.Mul(vx))
			}

			// Horizontal reduction using AddPairsGrouped
			zero := archsimd.BroadcastFloat32x8(0)
			pairs := acc.AddPairsGrouped(zero) // [(a+b), (c+d), (e+f), (g+h), 0, 0, 0, 0]
			lo := pairs.GetLo()                // [(a+b), (c+d), (e+f), (g+h)]
			var sum float32
			sum += lo.GetElem(0) + lo.GetElem(1) + lo.GetElem(2) + lo.GetElem(3)

			// Handle remaining elements
			for ; j < c; j++ {
				sum += bf16ToF32Table(row[j]) * x[j]
			}
			dst[i] = sum
		}
		return
	}

	// Slow path: need to load from bytes with u16le
	rowBytes := w.Stride * 2
	c := w.C
	for i := rs; i < re; i++ {
		off := i * rowBytes
		if c > 0 {
			_ = raw[off+(c-1)*2+1]
		}

		// For the byte path, use scalar for now
		var sum float32
		j := 0
		offj := off
		for ; j < c; j++ {
			u := u16le(raw, offj)
			sum += bf16ToF32Table(u) * x[j]
			offj += 2
		}
		dst[i] = sum
	}
}

func matVecRangeF16(dst []float32, w *Mat, x []float32, rs, re int) {
	raw := w.Raw
	if u16raw, ok := rawUint16LE(raw); ok {
		for i := rs; i < re; i++ {
			rowBase := i * w.Stride
			row := u16raw[rowBase : rowBase+w.Stride]
			if w.C > 0 {
				_ = row[w.C-1]
			}
			var sum float32
			j := 0
			for ; j+7 < w.C; j += 8 {
				sum += fp16ToF32Table(row[j+0])*x[j+0] +
					fp16ToF32Table(row[j+1])*x[j+1] +
					fp16ToF32Table(row[j+2])*x[j+2] +
					fp16ToF32Table(row[j+3])*x[j+3] +
					fp16ToF32Table(row[j+4])*x[j+4] +
					fp16ToF32Table(row[j+5])*x[j+5] +
					fp16ToF32Table(row[j+6])*x[j+6] +
					fp16ToF32Table(row[j+7])*x[j+7]
			}
			for ; j < w.C; j++ {
				sum += fp16ToF32Table(row[j]) * x[j]
			}
			dst[i] = sum
		}
		return
	}

	rowBytes := w.Stride * 2
	for i := rs; i < re; i++ {
		off := i * rowBytes
		if w.C > 0 {
			_ = raw[off+(w.C-1)*2+1]
		}
		var sum float32
		j := 0
		offj := off
		for ; j+7 < w.C; j += 8 {
			u0 := u16le(raw, offj+0)
			u1 := u16le(raw, offj+2)
			u2 := u16le(raw, offj+4)
			u3 := u16le(raw, offj+6)
			u4 := u16le(raw, offj+8)
			u5 := u16le(raw, offj+10)
			u6 := u16le(raw, offj+12)
			u7 := u16le(raw, offj+14)
			sum += fp16ToF32Table(u0)*x[j+0] +
				fp16ToF32Table(u1)*x[j+1] +
				fp16ToF32Table(u2)*x[j+2] +
				fp16ToF32Table(u3)*x[j+3] +
				fp16ToF32Table(u4)*x[j+4] +
				fp16ToF32Table(u5)*x[j+5] +
				fp16ToF32Table(u6)*x[j+6] +
				fp16ToF32Table(u7)*x[j+7]
			offj += 16
		}
		for ; j < w.C; j++ {
			u := u16le(raw, offj)
			sum += fp16ToF32Table(u) * x[j]
			offj += 2
		}
		dst[i] = sum
	}
}
