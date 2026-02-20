package core

import (
	"errors"

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

var q4SignTable = [16]int8{
	0, 1, 2, 3, 4, 5, 6, 7,
	-8, -7, -6, -5, -4, -3, -2, -1,
}

// BuildQuantCache pre-unpacks a quantized matrix into int8 blocks and per-block scales.
func BuildQuantCache(m *Mat) (*QuantCache, error) {
	if m == nil {
		return nil, errors.New("quant cache: nil matrix")
	}
	if m.Raw == nil {
		return nil, errors.New("quant cache: missing raw payload")
	}
	if !mcf.DTypeRequiresAligned64(m.DType) {
		return nil, errors.New("quant cache: dtype is not quantized")
	}

	layout, err := quantLayoutForMat(m.R, m.C, m.DType, len(m.Raw))
	if err != nil {
		return nil, err
	}

	totalBlocks, ok := mulInt(m.R, layout.blocksPerRow)
	if !ok {
		return nil, errors.New("quant cache: payload too large")
	}
	qLen, ok := mulInt(totalBlocks, 32)
	if !ok {
		return nil, errors.New("quant cache: payload too large")
	}

	q := make([]int8, qLen)
	scales := make([]float32, totalBlocks)

	if layout.family == 'q' {
		scalesRaw := m.Raw[layout.scaleOff : layout.scaleOff+layout.scaleCount*2]
		data := m.Raw[layout.dataOff : layout.dataOff+layout.dataBytes]

		for blockIdx := range totalBlocks {
			scale := scaleAtRawLE(scalesRaw, blockIdx)
			scales[blockIdx] = scale
			if scale == 0 {
				continue
			}
			dataOff := blockIdx * layout.blockDataBytes
			off := blockIdx * 32
			decodeBlock((*[32]int8)(q[off:off+32]), data[dataOff:dataOff+layout.blockDataBytes], layout.bits)
		}
		return &QuantCache{Q: q, Scales: scales, BlocksPerRow: layout.blocksPerRow}, nil
	}

	superRaw := m.Raw[layout.scaleOff : layout.scaleOff+layout.scaleCount*2]
	subRaw := m.Raw[layout.subScaleOff : layout.subScaleOff+layout.subScaleCount]
	data := m.Raw[layout.dataOff : layout.dataOff+layout.dataBytes]

	for r := 0; r < m.R; r++ {
		blockBase := r * layout.blocksPerRow
		superBase := r * layout.superBlocksPerRow
		for b := 0; b < layout.blocksPerRow; b++ {
			blockIdx := blockBase + b
			superIdx := superBase + (b / 8)
			superScale := scaleAtRawLE(superRaw, superIdx)
			u6 := subRaw[blockIdx] & 0x3F
			if superScale == 0 || u6 == 0 {
				continue
			}
			scale := superScale * (float32(u6) / 32.0)
			scales[blockIdx] = scale
			dataOff := blockIdx * layout.blockDataBytes
			off := blockIdx * 32
			decodeBlock((*[32]int8)(q[off:off+32]), data[dataOff:dataOff+layout.blockDataBytes], layout.bits)
		}
	}

	return &QuantCache{Q: q, Scales: scales, BlocksPerRow: layout.blocksPerRow}, nil
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
		for i := range 32 {
			dst[i] = int8(src[i])
		}
	case 4:
		decodeBlock4(dst, src)
	default:
		decodeBlockBits(dst, src, bits)
	}
}

func decodeBlock4(dst *[32]int8, src []byte) {
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
		dst[i] = signExtend(v, bits)
	}
}

func signExtend(v uint8, bits int) int8 {
	shift := 8 - bits
	return int8(int8(v<<uint(shift)) >> uint(shift))
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
