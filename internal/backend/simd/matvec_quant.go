package simd

import (
	"math"
	"simd/archsimd"
	"sync"

	instance "github.com/samcharles93/mantle/internal/backend/core"
	"github.com/samcharles93/mantle/pkg/mcf"
)

type QuantVec = instance.QuantVec

type quantDecodeScratch struct {
	qbuf   []int8
	scales []float32
}

var quantVecPool = sync.Pool{
	New: func() any {
		return &QuantVec{}
	},
}

var quantDecodeScratchPool = sync.Pool{
	New: func() any {
		return &quantDecodeScratch{}
	},
}

func getQuantVec(blocks int) *QuantVec {
	qx := quantVecPool.Get().(*QuantVec)
	qx.Blocks = blocks
	qx.Q = ensureInt8Slice(qx.Q, blocks*32)
	qx.Q16 = ensureInt16Slice(qx.Q16, blocks*32)
	qx.Scales = ensureFloat32Slice(qx.Scales, blocks)
	return qx
}

func putQuantVec(qx *QuantVec) {
	if qx == nil {
		return
	}
	quantVecPool.Put(qx)
}

func getQuantDecodeScratch(qbufLen, scalesLen int) *quantDecodeScratch {
	scratch := quantDecodeScratchPool.Get().(*quantDecodeScratch)
	scratch.qbuf = ensureInt8Slice(scratch.qbuf, qbufLen)
	scratch.scales = ensureFloat32Slice(scratch.scales, scalesLen)
	return scratch
}

func putQuantDecodeScratch(scratch *quantDecodeScratch) {
	if scratch == nil {
		return
	}
	quantDecodeScratchPool.Put(scratch)
}

func ensureInt8Slice(dst []int8, n int) []int8 {
	if n <= 0 {
		return dst[:0]
	}
	if cap(dst) < n {
		return make([]int8, n)
	}
	return dst[:n]
}

func ensureInt16Slice(dst []int16, n int) []int16 {
	if n <= 0 {
		return dst[:0]
	}
	if cap(dst) < n {
		return make([]int16, n)
	}
	return dst[:n]
}

func ensureFloat32Slice(dst []float32, n int) []float32 {
	if n <= 0 {
		return dst[:0]
	}
	if cap(dst) < n {
		return make([]float32, n)
	}
	return dst[:n]
}

var q4SignTable = [16]int8{
	0, 1, 2, 3, 4, 5, 6, 7,
	-8, -7, -6, -5, -4, -3, -2, -1,
}

func matVecRangeQuantWithWorker(dst []float32, w *Mat, x []float32, rs, re int, qx *QuantVec, worker *matVecWorker) bool {
	if mcf.IsUQDType(w.DType) {
		// Non-uniform codebooks have no int8 dot path; use the core LUT path.
		instance.MatVecRangeUQ(dst, w, x, rs, re)
		return true
	}
	if w.Quant != nil && w.Quant.ValidFor(w) {
		matVecRangeQuantCached(dst, w, x, rs, re, qx)
		return true
	}
	switch w.DType {
	case mcf.DTypeQ8:
		matVecRangeQWithWorker(dst, w, x, rs, re, 8, qx, worker)
		return true
	case mcf.DTypeQ4:
		matVecRangeQWithWorker(dst, w, x, rs, re, 4, qx, worker)
		return true
	case mcf.DTypeK6:
		matVecRangeKWithWorker(dst, w, x, rs, re, 6, qx, worker)
		return true
	case mcf.DTypeK4:
		matVecRangeKWithWorker(dst, w, x, rs, re, 4, qx, worker)
		return true
	case mcf.DTypeK3:
		matVecRangeKWithWorker(dst, w, x, rs, re, 3, qx, worker)
		return true
	case mcf.DTypeK2:
		matVecRangeKWithWorker(dst, w, x, rs, re, 2, qx, worker)
		return true
	default:
		return false
	}
}

func matVecRangeQuantCached(dst []float32, w *Mat, x []float32, rs, re int, qx *QuantVec) {
	qc := w.Quant
	blocksPerRow := qc.BlocksPerRow
	useInt8 := qx != nil && len(qx.Q) >= blocksPerRow*32 && len(qx.Q16) >= blocksPerRow*32 && len(qx.Scales) >= blocksPerRow

	// dotBuf holds one int32 dot per 32-element block for the fused row
	// kernel; rows with more blocks than this fall back to the per-block loop.
	var dotBuf [256]int32
	for r := rs; r < re; r++ {
		blockBase := r * blocksPerRow
		rowOff := blockBase * 32
		qRow := qc.Q[rowOff : rowOff+blocksPerRow*32]
		scales := qc.Scales[blockBase : blockBase+blocksPerRow]

		if useInt8 && blocksPerRow <= len(dotBuf) && w.C%32 == 0 {
			// One tight pass computes every block's int32 dot (no per-block
			// call overhead); the scales combine afterwards in the same
			// order as the per-block loop, so results are bit-identical.
			dots := dotBuf[:blocksPerRow]
			dotBlocksInt8Int16(qRow, qx.Q16[:blocksPerRow*32], dots, blocksPerRow)
			var sum float32
			for b := range blocksPerRow {
				sum += float32(dots[b]) * (scales[b] * qx.Scales[b])
			}
			dst[r] = sum
			continue
		}

		var sum float32
		for b := range blocksPerRow {
			col := b * 32
			n := w.C - col
			if n <= 0 {
				break
			}
			if n > 32 {
				n = 32
			}
			scale := scales[b]
			if scale == 0 {
				continue
			}
			off := b * 32
			qBlock := qRow[off : off+32]
			if useInt8 {
				xScale := qx.Scales[b]
				if xScale == 0 {
					continue
				}
				xBlock := qx.Q16[off : off+32]
				dot := dotInt8Int16(qBlock, xBlock, 32)
				sum += float32(dot) * (scale * xScale)
			} else {
				sum += scale * dotInt8Float32(qBlock, x[col:], n)
			}
		}
		dst[r] = sum
	}
}

// dotBlocksInt8Int16 computes out[b] = dot(q[b*32:(b+1)*32], x[b*32:(b+1)*32])
// for every block in one pass, with no per-block function call: each block is
// two 16-element DotProductPairs reduced to one int32. q and x must hold
// exactly blocks*32 elements and out exactly blocks elements.
func dotBlocksInt8Int16(q []int8, x []int16, out []int32, blocks int) {
	if !cpu.HasAVX2 {
		for b := 0; b < blocks; b++ {
			base := b * 32
			out[b] = dotInt8Int16Scalar(q[base:base+32], x[base:base+32], 32)
		}
		return
	}
	for b := 0; b < blocks; b++ {
		base := b * 32
		vq0 := archsimd.LoadInt8x16(q[base:]).ExtendToInt16()
		vq1 := archsimd.LoadInt8x16(q[base+16:]).ExtendToInt16()
		ix0 := archsimd.LoadInt16x16(x[base:])
		ix1 := archsimd.LoadInt16x16(x[base+16:])
		d := vq0.DotProductPairs(ix0).Add(vq1.DotProductPairs(ix1))
		var tmp [8]int32
		d.StoreArray(&tmp)
		out[b] = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7]
	}
}

func matVecRangeQWithWorker(dst []float32, w *Mat, x []float32, rs, re, bits int, qx *QuantVec, worker *matVecWorker) {
	layout, err := instance.QuantLayoutForMat(w.R, w.C, w.DType, len(w.Raw))
	if err != nil {
		panic(err)
	}

	useInt8 := qx != nil && len(qx.Q) >= layout.BlocksPerRow*32 && len(qx.Q16) >= layout.BlocksPerRow*32 && len(qx.Scales) >= layout.BlocksPerRow

	scalesRaw := w.Raw[layout.ScaleOff : layout.ScaleOff+layout.ScaleCount*2]
	data := w.Raw[layout.DataOff : layout.DataOff+layout.DataBytes]

	// Use cache-aware batching for optimal performance
	// Calculate optimal batch size based on L1 cache size and blocks per row
	totalRows := re - rs
	batchSize := computeOptimalBatchSize(layout.BlocksPerRow, totalRows)
	if batchSize <= 0 {
		batchSize = 1
	}
	if batchSize > totalRows {
		batchSize = totalRows
	}

	var scratch *quantDecodeScratch
	if worker == nil {
		scratch = getQuantDecodeScratch(batchSize*layout.BlocksPerRow*32, batchSize*layout.BlocksPerRow)
		defer putQuantDecodeScratch(scratch)
	}

	for batchStart := rs; batchStart < re; batchStart += batchSize {
		batchEnd := min(batchStart+batchSize, re)
		actualBatch := batchEnd - batchStart

		// Use worker buffers if available, otherwise allocate
		var qbuf []int8
		var scales []float32
		if worker != nil {
			qbuf, scales = worker.ensureWorkerBuffers(actualBatch*layout.BlocksPerRow*32, actualBatch*layout.BlocksPerRow)
		} else {
			requiredQbuf := actualBatch * layout.BlocksPerRow * 32
			requiredScales := actualBatch * layout.BlocksPerRow
			qbuf = scratch.qbuf[:requiredQbuf]
			scales = scratch.scales[:requiredScales]
		}

		// Decode all blocks for this batch
		for r := batchStart; r < batchEnd; r++ {
			rowIdx := r - batchStart
			blockBase := r * layout.BlocksPerRow
			for b := 0; b < layout.BlocksPerRow; b++ {
				blockIdx := blockBase + b
				scale := instance.ScaleAtRawLE(scalesRaw, blockIdx)
				scales[rowIdx*layout.BlocksPerRow+b] = scale
				if scale == 0 {
					continue
				}
				dataOff := blockIdx * layout.BlockDataBytes
				off := (rowIdx*layout.BlocksPerRow + b) * 32
				decodeBlock((*[32]int8)(qbuf[off:off+32]), data[dataOff:dataOff+layout.BlockDataBytes], bits)
			}
		}

		// Compute matvec using pre-dequantized data
		for r := batchStart; r < batchEnd; r++ {
			rowIdx := r - batchStart
			var sum float32
			for b := 0; b < layout.BlocksPerRow; b++ {
				col := b * 32
				n := w.C - col
				if n <= 0 {
					break
				}
				if n > 32 {
					n = 32
				}
				scale := scales[rowIdx*layout.BlocksPerRow+b]
				if scale == 0 {
					continue
				}
				off := (rowIdx*layout.BlocksPerRow + b) * 32
				if useInt8 {
					xScale := qx.Scales[b]
					if xScale == 0 {
						continue
					}
					xBlock := qx.Q16[b*32 : b*32+32]
					dot := dotInt8Int16(qbuf[off:off+32], xBlock, 32)
					sum += float32(dot) * (scale * xScale)
				} else {
					sum += scale * dotInt8Float32(qbuf[off:off+32], x[col:], n)
				}
			}
			dst[r] = sum
		}
	}
}

func matVecRangeKWithWorker(dst []float32, w *Mat, x []float32, rs, re, bits int, qx *QuantVec, worker *matVecWorker) {
	layout, err := instance.QuantLayoutForMat(w.R, w.C, w.DType, len(w.Raw))
	if err != nil {
		panic(err)
	}

	useInt8 := qx != nil && len(qx.Q) >= layout.BlocksPerRow*32 && len(qx.Q16) >= layout.BlocksPerRow*32 && len(qx.Scales) >= layout.BlocksPerRow

	superRaw := w.Raw[layout.ScaleOff : layout.ScaleOff+layout.ScaleCount*2]
	subRaw := w.Raw[layout.SubScaleOff : layout.SubScaleOff+layout.SubScaleCount]
	data := w.Raw[layout.DataOff : layout.DataOff+layout.DataBytes]

	// Use cache-aware batching for optimal performance
	// Calculate optimal batch size based on L1 cache size and blocks per row
	totalRows := re - rs
	batchSize := computeOptimalBatchSize(layout.BlocksPerRow, totalRows)
	if batchSize <= 0 {
		batchSize = 1
	}
	if batchSize > totalRows {
		batchSize = totalRows
	}

	var scratch *quantDecodeScratch
	if worker == nil {
		scratch = getQuantDecodeScratch(batchSize*layout.BlocksPerRow*32, batchSize*layout.BlocksPerRow)
		defer putQuantDecodeScratch(scratch)
	}

	for batchStart := rs; batchStart < re; batchStart += batchSize {
		batchEnd := min(batchStart+batchSize, re)
		actualBatch := batchEnd - batchStart

		// Use worker buffers if available, otherwise allocate
		var qbuf []int8
		var scales []float32
		if worker != nil {
			qbuf, scales = worker.ensureWorkerBuffers(actualBatch*layout.BlocksPerRow*32, actualBatch*layout.BlocksPerRow)
		} else {
			requiredQbuf := actualBatch * layout.BlocksPerRow * 32
			requiredScales := actualBatch * layout.BlocksPerRow
			qbuf = scratch.qbuf[:requiredQbuf]
			scales = scratch.scales[:requiredScales]
		}

		// Decode all blocks for this batch
		for r := batchStart; r < batchEnd; r++ {
			rowIdx := r - batchStart
			blockBase := r * layout.BlocksPerRow
			superBase := r * layout.SuperBlocksPerRow
			for b := 0; b < layout.BlocksPerRow; b++ {
				blockIdx := blockBase + b
				superIdx := superBase + (b / 8)
				superScale := instance.ScaleAtRawLE(superRaw, superIdx)
				if superScale == 0 {
					scales[rowIdx*layout.BlocksPerRow+b] = 0
					continue
				}

				u6 := subRaw[blockIdx] & 0x3F
				if u6 == 0 {
					scales[rowIdx*layout.BlocksPerRow+b] = 0
					continue
				}
				scale := superScale * (float32(u6) / 32.0)
				scales[rowIdx*layout.BlocksPerRow+b] = scale

				dataOff := blockIdx * layout.BlockDataBytes
				off := (rowIdx*layout.BlocksPerRow + b) * 32
				decodeBlock((*[32]int8)(qbuf[off:off+32]), data[dataOff:dataOff+layout.BlockDataBytes], bits)
			}
		}

		// Compute matvec using pre-dequantized data
		for r := batchStart; r < batchEnd; r++ {
			rowIdx := r - batchStart
			var sum float32
			for b := 0; b < layout.BlocksPerRow; b++ {
				col := b * 32
				n := w.C - col
				if n <= 0 {
					break
				}
				if n > 32 {
					n = 32
				}
				scale := scales[rowIdx*layout.BlocksPerRow+b]
				if scale == 0 {
					continue
				}
				off := (rowIdx*layout.BlocksPerRow + b) * 32
				if useInt8 {
					xScale := qx.Scales[b]
					if xScale == 0 {
						continue
					}
					xBlock := qx.Q16[b*32 : b*32+32]
					dot := dotInt8Int16(qbuf[off:off+32], xBlock, 32)
					sum += float32(dot) * (scale * xScale)
				} else {
					sum += scale * dotInt8Float32(qbuf[off:off+32], x[col:], n)
				}
			}
			dst[r] = sum
		}
	}
}

func decodeBlock(dst *[32]int8, src []byte, bits int) {
	switch bits {
	case 8:
		// Fast path: 8-bit is just a byte copy
		for i := range 32 {
			dst[i] = int8(src[i])
		}
	case 4:
		decodeBlock4(dst, src)
	default:
		decodeBlockBits(dst, src, bits)
	}
}

// decodeBlock4 decodes 4-bit quantized block.
// Optimized with loop unrolling for better performance.
func decodeBlock4(dst *[32]int8, src []byte) {
	// Unrolled loop: process 4 bytes (16 values) at a time
	for i := 0; i < 16; i += 4 {
		b0 := src[i]
		b1 := src[i+1]
		b2 := src[i+2]
		b3 := src[i+3]

		base := i * 2
		dst[base] = q4SignTable[b0&0x0F]
		dst[base+1] = q4SignTable[b0>>4&0x0F]
		dst[base+2] = q4SignTable[b1&0x0F]
		dst[base+3] = q4SignTable[b1>>4&0x0F]
		dst[base+4] = q4SignTable[b2&0x0F]
		dst[base+5] = q4SignTable[b2>>4&0x0F]
		dst[base+6] = q4SignTable[b3&0x0F]
		dst[base+7] = q4SignTable[b3>>4&0x0F]
	}
}

func decodeBlockBits(dst *[32]int8, src []byte, bits int) {
	if cpu.HasAVX2 && (bits == 2 || bits == 3 || bits == 6) {
		decodeBlockBitsSIMD(dst, src, bits)
		return
	}
	decodeBlockBitsScalar(dst, src, bits)
}

func decodeBlockBitsScalar(dst *[32]int8, src []byte, bits int) {
	bitPos := 0
	for i := range 32 {
		var v uint8
		for b := range bits {
			byteIdx := bitPos >> 3
			bitIdx := uint(bitPos & 7)
			if (src[byteIdx]>>bitIdx)&1 == 1 {
				v |= 1 << uint(b)
			}
			bitPos++
		}
		dst[i] = instance.SignExtend(v, bits)
	}
}

func decodeBlockBitsSIMD(dst *[32]int8, src []byte, bits int) {
	switch bits {
	case 2:
		decodeBlock2BitsSIMD(dst, src)
	case 3:
		decodeBlock3BitsSIMD(dst, src)
	case 6:
		decodeBlock6BitsSIMD(dst, src)
	default:
		decodeBlockBitsScalar(dst, src, bits)
	}
}

func decodeBlock2BitsSIMD(dst *[32]int8, src []byte) {
	// For 2 bits: 32 values = 64 bits = 8 bytes
	// Each byte contains 4 values (2 bits each)
	// Optimized scalar extraction with unrolled loop

	// Process 4 bytes (16 values) at a time
	for i := 0; i < 8; i += 4 {
		b0 := src[i]
		b1 := src[i+1]
		b2 := src[i+2]
		b3 := src[i+3]

		base := i * 4
		// Byte 0
		dst[base] = instance.SignExtend(b0&0x03, 2)
		dst[base+1] = instance.SignExtend((b0>>2)&0x03, 2)
		dst[base+2] = instance.SignExtend((b0>>4)&0x03, 2)
		dst[base+3] = instance.SignExtend(b0>>6, 2)

		// Byte 1
		dst[base+4] = instance.SignExtend(b1&0x03, 2)
		dst[base+5] = instance.SignExtend((b1>>2)&0x03, 2)
		dst[base+6] = instance.SignExtend((b1>>4)&0x03, 2)
		dst[base+7] = instance.SignExtend(b1>>6, 2)

		// Byte 2
		dst[base+8] = instance.SignExtend(b2&0x03, 2)
		dst[base+9] = instance.SignExtend((b2>>2)&0x03, 2)
		dst[base+10] = instance.SignExtend((b2>>4)&0x03, 2)
		dst[base+11] = instance.SignExtend(b2>>6, 2)

		// Byte 3
		dst[base+12] = instance.SignExtend(b3&0x03, 2)
		dst[base+13] = instance.SignExtend((b3>>2)&0x03, 2)
		dst[base+14] = instance.SignExtend((b3>>4)&0x03, 2)
		dst[base+15] = instance.SignExtend(b3>>6, 2)
	}
}

func decodeBlock3BitsSIMD(dst *[32]int8, src []byte) {
	// For 3 bits: 32 values = 96 bits = 12 bytes.
	decodeBitsBuffer(dst, src, 3)
}

func decodeBlock6BitsSIMD(dst *[32]int8, src []byte) {
	// For 6 bits: 32 values = 192 bits = 24 bytes.
	decodeBitsBuffer(dst, src, 6)
}

func decodeBitsBuffer(dst *[32]int8, src []byte, bits int) {
	if bits <= 0 || bits >= 8 {
		decodeBlockBitsScalar(dst, src, bits)
		return
	}

	mask := uint64((1 << bits) - 1)
	var bitBuf uint64
	var bitCount uint
	srcIdx := 0
	for i := range 32 {
		for bitCount < uint(bits) {
			var next uint64
			if srcIdx < len(src) {
				next = uint64(src[srcIdx])
			}
			bitBuf |= next << bitCount
			bitCount += 8
			srcIdx++
		}
		val := uint8(bitBuf & mask)
		dst[i] = instance.SignExtend(val, bits)
		bitBuf >>= bits
		if bitCount < uint(bits) {
			bitCount = 0
		} else {
			bitCount -= uint(bits)
		}
	}
}

func dotInt8Float32(q []int8, x []float32, n int) float32 {
	if cpu.HasAVX2 && n >= 16 {
		return dotInt8Float32SIMD(q, x, n)
	}
	return dotInt8Float32Scalar(q, x, n)
}

func dotInt8Int16(q []int8, x []int16, n int) int32 {
	if cpu.HasAVX2 && n >= 16 {
		return dotInt8Int16SIMD(q, x, n)
	}
	return dotInt8Int16Scalar(q, x, n)
}

func dotInt8Float32Scalar(q []int8, x []float32, n int) float32 {
	var sum float32
	for i := range n {
		sum += float32(q[i]) * x[i]
	}
	return sum
}

func dotInt8Float32SIMD(q []int8, x []float32, n int) float32 {
	var acc archsimd.Float32x8
	i := 0
	for ; i+16 <= n; i += 16 {
		vq := archsimd.LoadInt8x16(q[i:])
		v16 := vq.ExtendToInt16()

		lo := v16.GetLo().ExtendToInt32().ConvertToFloat32()
		hi := v16.GetHi().ExtendToInt32().ConvertToFloat32()

		vxLo := archsimd.LoadFloat32x8(x[i:])
		vxHi := archsimd.LoadFloat32x8(x[i+8:])

		acc = lo.MulAdd(vxLo, acc)
		acc = hi.MulAdd(vxHi, acc)
	}

	var tmp [8]float32
	acc.StoreArray(&tmp)
	sum := tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7]

	for ; i < n; i++ {
		sum += float32(q[i]) * x[i]
	}
	return sum
}

func dotInt8Int16Scalar(q []int8, x []int16, n int) int32 {
	var sum int32
	for i := range n {
		sum += int32(q[i]) * int32(x[i])
	}
	return sum
}

func dotInt8Int16SIMD(q []int8, x []int16, n int) int32 {
	var acc archsimd.Int32x8
	i := 0
	for ; i+16 <= n; i += 16 {
		vq := archsimd.LoadInt8x16(q[i:])
		iq := vq.ExtendToInt16()
		ix := archsimd.LoadInt16x16(x[i:])
		acc = acc.Add(iq.DotProductPairs(ix))
	}

	var tmp [8]int32
	acc.StoreArray(&tmp)
	sum := tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7]
	for ; i < n; i++ {
		sum += int32(q[i]) * int32(x[i])
	}
	return sum
}

func quantizeVecBlocksInto(x []float32, blocks int, qx []int8, qx16 []int16, scales []float32) {
	if blocks <= 0 {
		return
	}
	for i := range qx {
		qx[i] = 0
	}
	for i := range qx16 {
		qx16[i] = 0
	}
	for i := range scales {
		scales[i] = 0
	}
	for b := range blocks {
		base := b * 32
		maxAbs := float32(0)
		for i := range 32 {
			idx := base + i
			if idx >= len(x) {
				continue
			}
			v := x[idx]
			if v < 0 {
				v = -v
			}
			if v > maxAbs {
				maxAbs = v
			}
		}
		if maxAbs == 0 {
			continue
		}
		scale := maxAbs / 127.0
		scales[b] = scale
		inv := float32(1.0) / scale
		for i := range 32 {
			idx := base + i
			if idx >= len(x) {
				continue
			}
			v := x[idx]
			q := int32(math.Round(float64(v * inv)))
			if q > 127 {
				q = 127
			} else if q < -127 {
				q = -127
			}
			qx[base+i] = int8(q)
			qx16[base+i] = int16(int8(q))
		}
	}
}
