package core

import "fmt"

func errRowOutOfRange(row int) error {
	return fmt.Errorf("row out of range: %d", row)
}

func rowToQuant(dst []float32, m *Mat, row int) error {
	layout, err := QuantLayoutForMat(m.R, m.C, m.DType, len(m.Raw))
	if err != nil {
		return err
	}
	if row < 0 || row >= m.R {
		return errRowOutOfRange(row)
	}

	if layout.Family == 'q' {
		scalesRaw := m.Raw[layout.ScaleOff : layout.ScaleOff+layout.ScaleCount*2]
		data := m.Raw[layout.DataOff : layout.DataOff+layout.DataBytes]
		var qbuf [32]int8

		blockBase := row * layout.BlocksPerRow
		for b := 0; b < layout.BlocksPerRow; b++ {
			col := b * 32
			n := m.C - col
			if n <= 0 {
				break
			}
			if n > 32 {
				n = 32
			}
			blockIdx := blockBase + b
			scale := ScaleAtRawLE(scalesRaw, blockIdx)
			dataOff := blockIdx * layout.BlockDataBytes
			block := data[dataOff : dataOff+layout.BlockDataBytes]
			if scale == 0 {
				for i := 0; i < n; i++ {
					dst[col+i] = 0
				}
				continue
			}
			decodeBlock(&qbuf, block, layout.Bits)
			for i := 0; i < n; i++ {
				dst[col+i] = float32(qbuf[i]) * scale
			}
		}
		return nil
	}

	superRaw := m.Raw[layout.ScaleOff : layout.ScaleOff+layout.ScaleCount*2]
	subRaw := m.Raw[layout.SubScaleOff : layout.SubScaleOff+layout.SubScaleCount]
	data := m.Raw[layout.DataOff : layout.DataOff+layout.DataBytes]
	var qbuf [32]int8

	blockBase := row * layout.BlocksPerRow
	superBase := row * layout.SuperBlocksPerRow
	for b := 0; b < layout.BlocksPerRow; b++ {
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
		superScale := ScaleAtRawLE(superRaw, superIdx)
		u6 := subRaw[blockIdx] & 0x3F
		if superScale == 0 || u6 == 0 {
			for i := 0; i < n; i++ {
				dst[col+i] = 0
			}
			continue
		}
		scale := superScale * (float32(u6) / 32.0)

		dataOff := blockIdx * layout.BlockDataBytes
		block := data[dataOff : dataOff+layout.BlockDataBytes]
		decodeBlock(&qbuf, block, layout.Bits)
		for i := 0; i < n; i++ {
			dst[col+i] = float32(qbuf[i]) * scale
		}
	}
	return nil
}
