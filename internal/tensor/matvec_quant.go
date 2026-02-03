package tensor

import (
	"errors"
	"math"
	"sync"

	"simd/archsimd"

	"github.com/samcharles93/mantle/pkg/mcf"
)

type quantLayout struct {
	family            byte
	bits              int
	blocksPerRow      int
	superBlocksPerRow int
	blockDataBytes    int

	scaleOff   int
	scaleCount int

	subScaleOff   int
	subScaleCount int

	dataOff   int
	dataBytes int
}

type QuantVec struct {
	q      []int8
	q16    []int16
	scales []float32
	Blocks int
}

var quantVecPool = sync.Pool{
	New: func() any {
		return &QuantVec{}
	},
}

func getQuantVec(blocks int) *QuantVec {
	qx := quantVecPool.Get().(*QuantVec)
	qx.Blocks = blocks
	qx.q = ensureInt8Slice(qx.q, blocks*32)
	qx.q16 = ensureInt16Slice(qx.q16, blocks*32)
	qx.scales = ensureFloat32Slice(qx.scales, blocks)
	return qx
}

func putQuantVec(qx *QuantVec) {
	if qx == nil {
		return
	}
	quantVecPool.Put(qx)
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

func quantLayoutForMat(rows, cols int, dt mcf.TensorDType, rawLen int) (quantLayout, error) {
	if rows <= 0 || cols <= 0 {
		return quantLayout{}, errors.New("invalid quant mat shape")
	}
	if !mcf.DTypeRequiresAligned64(dt) {
		return quantLayout{}, errors.New("dtype is not quantized")
	}

	bits := 0
	family := byte(0)
	switch dt {
	case mcf.DTypeQ8:
		bits = 8
		family = 'q'
	case mcf.DTypeQ4:
		bits = 4
		family = 'q'
	case mcf.DTypeK6:
		bits = 6
		family = 'k'
	case mcf.DTypeK4:
		bits = 4
		family = 'k'
	case mcf.DTypeK3:
		bits = 3
		family = 'k'
	case mcf.DTypeK2:
		bits = 2
		family = 'k'
	default:
		return quantLayout{}, errors.New("unsupported quant dtype")
	}

	wantU64, err := mcf.QuantPayloadSize([]uint64{uint64(rows), uint64(cols)}, dt)
	if err != nil {
		return quantLayout{}, err
	}
	if wantU64 > uint64(int(^uint(0)>>1)) {
		return quantLayout{}, errors.New("quant payload too large")
	}
	if int(wantU64) != rawLen {
		return quantLayout{}, errors.New("quant payload size mismatch")
	}

	blocksPerRow := (cols + 31) / 32
	totalBlocks, ok := mulInt(rows, blocksPerRow)
	if !ok {
		return quantLayout{}, errors.New("quant payload too large")
	}
	blockDataBytes := (32 * bits) / 8
	dataBytes, ok := mulInt(totalBlocks, blockDataBytes)
	if !ok {
		return quantLayout{}, errors.New("quant payload too large")
	}

	scaleBytes, ok := mulInt(totalBlocks, 2)
	if !ok {
		return quantLayout{}, errors.New("quant payload too large")
	}

	layout := quantLayout{
		family:         family,
		bits:           bits,
		blocksPerRow:   blocksPerRow,
		blockDataBytes: blockDataBytes,
		scaleOff:       0,
	}

	if family == 'q' {
		layout.scaleCount = totalBlocks
		off, ok := align64Int(scaleBytes)
		if !ok {
			return quantLayout{}, errors.New("quant payload too large")
		}
		layout.dataOff = off
		layout.dataBytes = dataBytes
		return layout, nil
	}

	superBlocksPerRow := (blocksPerRow + 7) / 8
	superCount, ok := mulInt(rows, superBlocksPerRow)
	if !ok {
		return quantLayout{}, errors.New("quant payload too large")
	}
	superBytes, ok := mulInt(superCount, 2)
	if !ok {
		return quantLayout{}, errors.New("quant payload too large")
	}

	layout.scaleCount = superCount
	layout.superBlocksPerRow = superBlocksPerRow

	subBytes := totalBlocks
	off, ok := align64Int(superBytes)
	if !ok {
		return quantLayout{}, errors.New("quant payload too large")
	}
	layout.subScaleOff = off
	layout.subScaleCount = totalBlocks

	off, ok = addInt(off, subBytes)
	if !ok {
		return quantLayout{}, errors.New("quant payload too large")
	}
	off, ok = align64Int(off)
	if !ok {
		return quantLayout{}, errors.New("quant payload too large")
	}
	layout.dataOff = off
	layout.dataBytes = dataBytes
	return layout, nil
}

func matVecRangeQuantWithWorker(dst []float32, w *Mat, x []float32, rs, re int, qx *QuantVec, worker *matVecWorker) bool {
	if w.Quant != nil && w.Quant.validFor(w) {
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
	useInt8 := qx != nil && len(qx.q) >= blocksPerRow*32 && len(qx.q16) >= blocksPerRow*32 && len(qx.scales) >= blocksPerRow

	for r := rs; r < re; r++ {
		blockBase := r * blocksPerRow
		rowOff := blockBase * 32
		qRow := qc.Q[rowOff : rowOff+blocksPerRow*32]
		scales := qc.Scales[blockBase : blockBase+blocksPerRow]

		var sum float32
		for b := 0; b < blocksPerRow; b++ {
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
				xScale := qx.scales[b]
				if xScale == 0 {
					continue
				}
				xBlock := qx.q16[off : off+32]
				dot := dotInt8Int16(qBlock, xBlock, 32)
				sum += float32(dot) * (scale * xScale)
			} else {
				sum += scale * dotInt8Float32(qBlock, x[col:], n)
			}
		}
		dst[r] = sum
	}
}

func matVecRangeQWithWorker(dst []float32, w *Mat, x []float32, rs, re, bits int, qx *QuantVec, worker *matVecWorker) {
	layout, err := quantLayoutForMat(w.R, w.C, w.DType, len(w.Raw))
	if err != nil {
		panic(err)
	}

	useInt8 := qx != nil && len(qx.q) >= layout.blocksPerRow*32 && len(qx.q16) >= layout.blocksPerRow*32 && len(qx.scales) >= layout.blocksPerRow

	scalesRaw := w.Raw[layout.scaleOff : layout.scaleOff+layout.scaleCount*2]
	data := w.Raw[layout.dataOff : layout.dataOff+layout.dataBytes]

	// Use cache-aware batching for optimal performance
	// Calculate optimal batch size based on L1 cache size and blocks per row
	totalRows := re - rs
	batchSize := computeOptimalBatchSize(layout.blocksPerRow, totalRows)
	if batchSize <= 0 {
		batchSize = 1
	}
	if batchSize > totalRows {
		batchSize = totalRows
	}

	for batchStart := rs; batchStart < re; batchStart += batchSize {
		batchEnd := batchStart + batchSize
		if batchEnd > re {
			batchEnd = re
		}
		actualBatch := batchEnd - batchStart

		// Use worker buffers if available, otherwise allocate
		var qbuf []int8
		var scales []float32
		if worker != nil {
			qbuf, scales = worker.ensureWorkerBuffers(actualBatch*layout.blocksPerRow*32, actualBatch*layout.blocksPerRow)
		} else {
			qbuf = make([]int8, actualBatch*layout.blocksPerRow*32)
			scales = make([]float32, actualBatch*layout.blocksPerRow)
		}

		// Decode all blocks for this batch
		for r := batchStart; r < batchEnd; r++ {
			rowIdx := r - batchStart
			blockBase := r * layout.blocksPerRow
			for b := 0; b < layout.blocksPerRow; b++ {
				blockIdx := blockBase + b
				scale := scaleAtRawLE(scalesRaw, blockIdx)
				scales[rowIdx*layout.blocksPerRow+b] = scale
				if scale == 0 {
					continue
				}
				dataOff := blockIdx * layout.blockDataBytes
				off := (rowIdx*layout.blocksPerRow + b) * 32
				decodeBlock((*[32]int8)(qbuf[off:off+32]), data[dataOff:dataOff+layout.blockDataBytes], bits)
			}
		}

		// Compute matvec using pre-dequantized data
		for r := batchStart; r < batchEnd; r++ {
			rowIdx := r - batchStart
			var sum float32
			for b := 0; b < layout.blocksPerRow; b++ {
				col := b * 32
				n := w.C - col
				if n <= 0 {
					break
				}
				if n > 32 {
					n = 32
				}
				scale := scales[rowIdx*layout.blocksPerRow+b]
				if scale == 0 {
					continue
				}
				off := (rowIdx*layout.blocksPerRow + b) * 32
				if useInt8 {
					xScale := qx.scales[b]
					if xScale == 0 {
						continue
					}
					xBlock := qx.q16[b*32 : b*32+32]
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
	layout, err := quantLayoutForMat(w.R, w.C, w.DType, len(w.Raw))
	if err != nil {
		panic(err)
	}

	useInt8 := qx != nil && len(qx.q) >= layout.blocksPerRow*32 && len(qx.q16) >= layout.blocksPerRow*32 && len(qx.scales) >= layout.blocksPerRow

	superRaw := w.Raw[layout.scaleOff : layout.scaleOff+layout.scaleCount*2]
	subRaw := w.Raw[layout.subScaleOff : layout.subScaleOff+layout.subScaleCount]
	data := w.Raw[layout.dataOff : layout.dataOff+layout.dataBytes]

	// Use cache-aware batching for optimal performance
	// Calculate optimal batch size based on L1 cache size and blocks per row
	totalRows := re - rs
	batchSize := computeOptimalBatchSize(layout.blocksPerRow, totalRows)
	if batchSize <= 0 {
		batchSize = 1
	}
	if batchSize > totalRows {
		batchSize = totalRows
	}

	for batchStart := rs; batchStart < re; batchStart += batchSize {
		batchEnd := batchStart + batchSize
		if batchEnd > re {
			batchEnd = re
		}
		actualBatch := batchEnd - batchStart

		// Use worker buffers if available, otherwise allocate
		var qbuf []int8
		var scales []float32
		if worker != nil {
			qbuf, scales = worker.ensureWorkerBuffers(actualBatch*layout.blocksPerRow*32, actualBatch*layout.blocksPerRow)
		} else {
			qbuf = make([]int8, actualBatch*layout.blocksPerRow*32)
			scales = make([]float32, actualBatch*layout.blocksPerRow)
		}

		// Decode all blocks for this batch
		for r := batchStart; r < batchEnd; r++ {
			rowIdx := r - batchStart
			blockBase := r * layout.blocksPerRow
			superBase := r * layout.superBlocksPerRow
			for b := 0; b < layout.blocksPerRow; b++ {
				blockIdx := blockBase + b
				superIdx := superBase + (b / 8)
				superScale := scaleAtRawLE(superRaw, superIdx)
				if superScale == 0 {
					scales[rowIdx*layout.blocksPerRow+b] = 0
					continue
				}

				u6 := subRaw[blockIdx] & 0x3F
				if u6 == 0 {
					scales[rowIdx*layout.blocksPerRow+b] = 0
					continue
				}
				scale := superScale * (float32(u6) / 32.0)
				scales[rowIdx*layout.blocksPerRow+b] = scale

				dataOff := blockIdx * layout.blockDataBytes
				off := (rowIdx*layout.blocksPerRow + b) * 32
				decodeBlock((*[32]int8)(qbuf[off:off+32]), data[dataOff:dataOff+layout.blockDataBytes], bits)
			}
		}

		// Compute matvec using pre-dequantized data
		for r := batchStart; r < batchEnd; r++ {
			rowIdx := r - batchStart
			var sum float32
			for b := 0; b < layout.blocksPerRow; b++ {
				col := b * 32
				n := w.C - col
				if n <= 0 {
					break
				}
				if n > 32 {
					n = 32
				}
				scale := scales[rowIdx*layout.blocksPerRow+b]
				if scale == 0 {
					continue
				}
				off := (rowIdx*layout.blocksPerRow + b) * 32
				if useInt8 {
					xScale := qx.scales[b]
					if xScale == 0 {
						continue
					}
					xBlock := qx.q16[b*32 : b*32+32]
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

func rowToQuant(dst []float32, m *Mat, row int) error {
	layout, err := quantLayoutForMat(m.R, m.C, m.DType, len(m.Raw))
	if err != nil {
		return err
	}
	if row < 0 || row >= m.R {
		return errors.New("row out of range")
	}

	if layout.family == 'q' {
		scalesRaw := m.Raw[layout.scaleOff : layout.scaleOff+layout.scaleCount*2]
		data := m.Raw[layout.dataOff : layout.dataOff+layout.dataBytes]
		var qbuf [32]int8

		blockBase := row * layout.blocksPerRow
		for b := 0; b < layout.blocksPerRow; b++ {
			col := b * 32
			n := m.C - col
			if n <= 0 {
				break
			}
			if n > 32 {
				n = 32
			}
			blockIdx := blockBase + b
			scale := scaleAtRawLE(scalesRaw, blockIdx)
			dataOff := blockIdx * layout.blockDataBytes
			block := data[dataOff : dataOff+layout.blockDataBytes]
			if scale == 0 {
				for i := 0; i < n; i++ {
					dst[col+i] = 0
				}
				continue
			}
			decodeBlock(&qbuf, block, layout.bits)
			for i := 0; i < n; i++ {
				dst[col+i] = float32(qbuf[i]) * scale
			}
		}
		return nil
	}

	superRaw := m.Raw[layout.scaleOff : layout.scaleOff+layout.scaleCount*2]
	subRaw := m.Raw[layout.subScaleOff : layout.subScaleOff+layout.subScaleCount]
	data := m.Raw[layout.dataOff : layout.dataOff+layout.dataBytes]
	var qbuf [32]int8

	blockBase := row * layout.blocksPerRow
	superBase := row * layout.superBlocksPerRow
	for b := 0; b < layout.blocksPerRow; b++ {
		col := b * 32
		n := m.C - col
		if n <= 0 {
			break
		}
		if n > 32 {
			n = 32
		}
		blockIdx := blockBase + b
		superIdx := superBase + (b / 8)
		superScale := scaleAtRawLE(superRaw, superIdx)
		u6 := subRaw[blockIdx] & 0x3F
		if superScale == 0 || u6 == 0 {
			for i := 0; i < n; i++ {
				dst[col+i] = 0
			}
			continue
		}
		scale := superScale * (float32(u6) / 32.0)

		dataOff := blockIdx * layout.blockDataBytes
		block := data[dataOff : dataOff+layout.blockDataBytes]
		decodeBlock(&qbuf, block, layout.bits)
		for i := 0; i < n; i++ {
			dst[col+i] = float32(qbuf[i]) * scale
		}
	}
	return nil
}

func scaleAtRawLE(raw []byte, idx int) float32 {
	off := idx * 2
	if off < 0 || off+1 >= len(raw) {
		return 0
	}
	u := uint16(raw[off]) | uint16(raw[off+1])<<8
	return Float16ToFloat32(u)
}

func decodeBlock(dst *[32]int8, src []byte, bits int) {
	switch bits {
	case 8:
		// Fast path: 8-bit is just a byte copy
		for i := 0; i < 32; i++ {
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
	for i := 0; i < 32; i++ {
		var v uint8
		for b := 0; b < bits; b++ {
			byteIdx := bitPos >> 3
			bitIdx := uint(bitPos & 7)
			if (src[byteIdx]>>bitIdx)&1 == 1 {
				v |= 1 << uint(b)
			}
			bitPos++
		}
		dst[i] = signExtend(v, bits)
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
		dst[base] = signExtend(b0&0x03, 2)
		dst[base+1] = signExtend((b0>>2)&0x03, 2)
		dst[base+2] = signExtend((b0>>4)&0x03, 2)
		dst[base+3] = signExtend(b0>>6, 2)

		// Byte 1
		dst[base+4] = signExtend(b1&0x03, 2)
		dst[base+5] = signExtend((b1>>2)&0x03, 2)
		dst[base+6] = signExtend((b1>>4)&0x03, 2)
		dst[base+7] = signExtend(b1>>6, 2)

		// Byte 2
		dst[base+8] = signExtend(b2&0x03, 2)
		dst[base+9] = signExtend((b2>>2)&0x03, 2)
		dst[base+10] = signExtend((b2>>4)&0x03, 2)
		dst[base+11] = signExtend(b2>>6, 2)

		// Byte 3
		dst[base+12] = signExtend(b3&0x03, 2)
		dst[base+13] = signExtend((b3>>2)&0x03, 2)
		dst[base+14] = signExtend((b3>>4)&0x03, 2)
		dst[base+15] = signExtend(b3>>6, 2)
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
	for i := 0; i < 32; i++ {
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
		dst[i] = signExtend(val, bits)
		bitBuf >>= bits
		if bitCount < uint(bits) {
			bitCount = 0
		} else {
			bitCount -= uint(bits)
		}
	}
}

func signExtend(v uint8, bits int) int8 {
	shift := uint(8 - bits)
	return int8(v<<shift) >> shift
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
	for i := 0; i < n; i++ {
		sum += float32(q[i]) * x[i]
	}
	return sum
}

func dotInt8Float32SIMD(q []int8, x []float32, n int) float32 {
	var acc archsimd.Float32x8
	i := 0
	for ; i+16 <= n; i += 16 {
		vq := archsimd.LoadInt8x16Slice(q[i:])
		v16 := vq.ExtendToInt16()

		lo := v16.GetLo().ExtendToInt32().ConvertToFloat32()
		hi := v16.GetHi().ExtendToInt32().ConvertToFloat32()

		vxLo := archsimd.LoadFloat32x8Slice(x[i:])
		vxHi := archsimd.LoadFloat32x8Slice(x[i+8:])

		acc = acc.Add(lo.Mul(vxLo))
		acc = acc.Add(hi.Mul(vxHi))
	}

	var tmp [8]float32
	acc.Store(&tmp)
	sum := tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7]

	for ; i < n; i++ {
		sum += float32(q[i]) * x[i]
	}
	return sum
}

func dotInt8Int16Scalar(q []int8, x []int16, n int) int32 {
	var sum int32
	for i := 0; i < n; i++ {
		sum += int32(q[i]) * int32(x[i])
	}
	return sum
}

func dotInt8Int16SIMD(q []int8, x []int16, n int) int32 {
	var acc archsimd.Int32x8
	i := 0
	for ; i+16 <= n; i += 16 {
		vq := archsimd.LoadInt8x16Slice(q[i:])
		iq := vq.ExtendToInt16()
		ix := archsimd.LoadInt16x16Slice(x[i:])
		acc = acc.Add(iq.DotProductPairs(ix))
	}

	var tmp [8]int32
	acc.Store(&tmp)
	sum := tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7]
	for ; i < n; i++ {
		sum += int32(q[i]) * int32(x[i])
	}
	return sum
}

func quantizeVecBlocks(x []float32, blocks int) ([]int8, []int16, []float32) {
	if blocks <= 0 {
		return nil, nil, nil
	}
	qx := make([]int8, blocks*32)
	qx16 := make([]int16, blocks*32)
	scales := make([]float32, blocks)
	quantizeVecBlocksInto(x, blocks, qx, qx16, scales)
	return qx, qx16, scales
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
	for b := 0; b < blocks; b++ {
		base := b * 32
		maxAbs := float32(0)
		for i := 0; i < 32; i++ {
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
		for i := 0; i < 32; i++ {
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

func mulInt(a, b int) (int, bool) {
	if a == 0 || b == 0 {
		return 0, true
	}
	if a > int(^uint(0)>>1)/b {
		return 0, false
	}
	return a * b, true
}

func addInt(a, b int) (int, bool) {
	if a > int(^uint(0)>>1)-b {
		return 0, false
	}
	return a + b, true
}

func align64Int(n int) (int, bool) {
	if n > int(^uint(0)>>1)-63 {
		return 0, false
	}
	return (n + 63) &^ 63, true
}
