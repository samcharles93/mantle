package core

import "errors"

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
