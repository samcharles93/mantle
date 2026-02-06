package simd

import (
	"runtime"
	"sync"

	"simd/archsimd"

	"github.com/samcharles93/mantle/pkg/mcf"
)

type matVecTask struct {
	dst    []float32
	w      *Mat
	x      []float32
	rs, re int
	done   chan struct{}
	qx     *QuantVec
}

type matVecWorker struct {
	tasks  chan matVecTask
	qbuf   []int8
	scales []float32
	mu     sync.Mutex
}

const (
	l1CacheSize = 32 * 1024
)

// computeOptimalBatchSize calculates the optimal batch size for quantized matvec
// to maximize cache utilization while minimizing allocations
func computeOptimalBatchSize(blocksPerRow int, totalRows int) int {
	// Each row requires: blocksPerRow*32 bytes for qbuf + blocksPerRow*4 bytes for scales
	bytesPerRow := blocksPerRow*32 + blocksPerRow*4

	const l1Reserved = 8 * 1024
	availableL1 := l1CacheSize - l1Reserved
	if availableL1 <= 0 {
		availableL1 = 8 * 1024
	}

	maxRowsL1 := availableL1 / bytesPerRow
	if maxRowsL1 <= 0 {
		maxRowsL1 = 1
	}

	// Limit batch size to reasonable values
	const minBatch = 1
	const maxBatch = 16

	// Ensure we don't exceed total rows
	batch := maxRowsL1
	if batch < minBatch {
		batch = minBatch
	}
	if batch > maxBatch {
		batch = maxBatch
	}
	if batch > totalRows {
		batch = totalRows
	}

	return batch
}

// ensureWorkerBuffers ensures the worker has buffers of at least required size
func (w *matVecWorker) ensureWorkerBuffers(requiredQbuf, requiredScales int) ([]int8, []float32) {
	if w == nil {
		return nil, nil
	}
	w.mu.Lock()
	defer w.mu.Unlock()

	if len(w.qbuf) < requiredQbuf {
		w.qbuf = make([]int8, requiredQbuf)
	}
	if len(w.scales) < requiredScales {
		w.scales = make([]float32, requiredScales)
	}
	return w.qbuf[:requiredQbuf], w.scales[:requiredScales]
}

type matVecPool struct {
	size      int
	tasks     chan matVecTask
	doneSlots chan chan struct{}
	workers   []*matVecWorker
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
		workers:   make([]*matVecWorker, size),
	}
	for i := 0; i < size; i++ {
		p.doneSlots <- make(chan struct{}, 1)
		worker := &matVecWorker{
			tasks: make(chan matVecTask, 1),
		}
		p.workers[i] = worker
		go func(w *matVecWorker) {
			for task := range w.tasks {
				matVecRangeWithWorker(task.dst, task.w, task.x, task.rs, task.re, task.qx, w)
				task.done <- struct{}{}
			}
		}(worker)
	}

	// Start dispatcher
	go func() {
		workerIdx := 0
		for task := range p.tasks {
			worker := p.workers[workerIdx]
			worker.tasks <- task
			workerIdx = (workerIdx + 1) % size
		}
		// Close worker channels when main task channel closes
		for _, worker := range p.workers {
			close(worker.tasks)
		}
	}()

	return p
}

// MatVec computes dst = w * x where w is a matrix and x is a vector.
// It runs in parallel using a worker pool.
func MatVec(dst []float32, w *Mat, x []float32) {
	matVecWithQuant(dst, w, x, nil)
}

// MatVecWithQuant computes dst = w * x using a pre-quantized input vector when available.
func MatVecWithQuant(dst []float32, w *Mat, x []float32, qx *QuantVec) {
	matVecWithQuant(dst, w, x, qx)
}

func matVecWithQuant(dst []float32, w *Mat, x []float32, qx *QuantVec) {
	if w.R == 0 || w.C == 0 {
		return
	}
	if len(dst) < w.R || len(x) < w.C {
		panic("matvec shape mismatch")
	}

	useQuant := w.Raw != nil && mcf.DTypeRequiresAligned64(w.DType) && cpu.HasAVX2
	var localQx *QuantVec
	if useQuant {
		blocks := (w.C + 31) / 32
		if qx != nil && qx.Blocks == blocks && len(qx.q) >= blocks*32 && len(qx.q16) >= blocks*32 && len(qx.scales) >= blocks {
			localQx = qx
		} else {
			localQx = getQuantVec(blocks)
			quantizeVecBlocksInto(x, blocks, localQx.q, localQx.q16, localQx.scales)
			defer putQuantVec(localQx)
		}
	}

	pool := getMatVecPool()
	workers := pool.size
	if workers > w.R {
		workers = w.R
	}

	if workers <= 1 {
		matVecRange(dst, w, x, 0, w.R, localQx)
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
			qx:   localQx,
		}
	}

	for i := 0; i < activeWorkers; i++ {
		<-done
	}
	pool.doneSlots <- done
}

func matVecRange(dst []float32, w *Mat, x []float32, rs, re int, qx *QuantVec) {
	matVecRangeWithWorker(dst, w, x, rs, re, qx, nil)
}

func matVecRangeWithWorker(dst []float32, w *Mat, x []float32, rs, re int, qx *QuantVec, worker *matVecWorker) {
	if w.Raw != nil && w.DType != mcf.DTypeF32 {
		if matVecRangeQuantWithWorker(dst, w, x, rs, re, qx, worker) {
			return
		}
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
		i := rs
		for ; i+1 < re; i += 2 {
			row0Base := i * w.Stride
			row1Base := (i + 1) * w.Stride
			row0 := u16raw[row0Base : row0Base+w.Stride]
			row1 := u16raw[row1Base : row1Base+w.Stride]
			if c > 0 {
				_ = row0[c-1]
				_ = row1[c-1]
			}

			var acc0 archsimd.Float32x8
			var acc1 archsimd.Float32x8
			j := 0
			for ; j+8 <= c; j += 8 {
				vx := archsimd.LoadFloat32x8Slice(x[j:])

				vu0 := archsimd.LoadUint16x8Slice(row0[j:])
				vf0 := vu0.ExtendToUint32().ShiftAllLeft(16).AsFloat32x8()
				acc0 = acc0.Add(vf0.Mul(vx))

				vu1 := archsimd.LoadUint16x8Slice(row1[j:])
				vf1 := vu1.ExtendToUint32().ShiftAllLeft(16).AsFloat32x8()
				acc1 = acc1.Add(vf1.Mul(vx))
			}

			var tmp0 [8]float32
			var tmp1 [8]float32
			acc0.Store(&tmp0)
			acc1.Store(&tmp1)
			var sum0 float32
			var sum1 float32
			sum0 += tmp0[0] + tmp0[1] + tmp0[2] + tmp0[3] + tmp0[4] + tmp0[5] + tmp0[6] + tmp0[7]
			sum1 += tmp1[0] + tmp1[1] + tmp1[2] + tmp1[3] + tmp1[4] + tmp1[5] + tmp1[6] + tmp1[7]

			for ; j < c; j++ {
				xv := x[j]
				sum0 += bf16ToF32Table(row0[j]) * xv
				sum1 += bf16ToF32Table(row1[j]) * xv
			}
			dst[i] = sum0
			dst[i+1] = sum1
		}
		if i < re {
			rowBase := i * w.Stride
			row := u16raw[rowBase : rowBase+w.Stride]
			if c > 0 {
				_ = row[c-1]
			}

			var acc archsimd.Float32x8
			j := 0
			for ; j+8 <= c; j += 8 {
				vu := archsimd.LoadUint16x8Slice(row[j:])
				vf := vu.ExtendToUint32().ShiftAllLeft(16).AsFloat32x8()
				vx := archsimd.LoadFloat32x8Slice(x[j:])
				acc = acc.Add(vf.Mul(vx))
			}

			var tmp [8]float32
			acc.Store(&tmp)
			var sum float32
			sum += tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7]

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
