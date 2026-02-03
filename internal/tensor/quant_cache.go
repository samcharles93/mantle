package tensor

import (
	"errors"

	"github.com/samcharles93/mantle/pkg/mcf"
)

// QuantCache stores pre-unpacked quantized weights for faster matvec.
// Q holds int8 values for every block (32 values per block).
// Scales holds the float32 scale per block.
type QuantCache struct {
	Q            []int8
	Scales       []float32
	BlocksPerRow int
}

func (qc *QuantCache) validFor(m *Mat) bool {
	if qc == nil || m == nil {
		return false
	}
	if qc.BlocksPerRow <= 0 || m.R <= 0 {
		return false
	}
	blocksPerRow := (m.C + 31) / 32
	if blocksPerRow != qc.BlocksPerRow {
		return false
	}
	totalBlocks, ok := mulInt(m.R, blocksPerRow)
	if !ok {
		return false
	}
	qLen, ok := mulInt(totalBlocks, 32)
	if !ok {
		return false
	}
	if len(qc.Q) < qLen || len(qc.Scales) < totalBlocks {
		return false
	}
	return true
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
		scalesU16, scalesOK := rawUint16LE(scalesRaw)
		data := m.Raw[layout.dataOff : layout.dataOff+layout.dataBytes]

		for blockIdx := 0; blockIdx < totalBlocks; blockIdx++ {
			scale := scaleAt(scalesU16, scalesRaw, blockIdx, scalesOK)
			scales[blockIdx] = scale
			if scale == 0 {
				continue
			}
			dataOff := blockIdx * layout.blockDataBytes
			off := blockIdx * 32
			decodeBlock((*[32]int8)(q[off:off+32]), data[dataOff:dataOff+layout.blockDataBytes], layout.bits)
		}
		return &QuantCache{
			Q:            q,
			Scales:       scales,
			BlocksPerRow: layout.blocksPerRow,
		}, nil
	}

	superRaw := m.Raw[layout.scaleOff : layout.scaleOff+layout.scaleCount*2]
	superU16, superOK := rawUint16LE(superRaw)
	subRaw := m.Raw[layout.subScaleOff : layout.subScaleOff+layout.subScaleCount]
	data := m.Raw[layout.dataOff : layout.dataOff+layout.dataBytes]

	for r := 0; r < m.R; r++ {
		blockBase := r * layout.blocksPerRow
		superBase := r * layout.superBlocksPerRow
		for b := 0; b < layout.blocksPerRow; b++ {
			blockIdx := blockBase + b
			superIdx := superBase + (b / 8)
			superScale := scaleAt(superU16, superRaw, superIdx, superOK)
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

	return &QuantCache{
		Q:            q,
		Scales:       scales,
		BlocksPerRow: layout.blocksPerRow,
	}, nil
}
