package mcf

import "errors"

const (
	quantBlockSize   uint64 = 32
	quantSuperBlocks uint64 = 8
)

// QuantPayloadSize returns the exact payload size for a quantized tensor.
// Shape must be rank-2 (rows, cols). The size includes internal 64-byte
// alignment between sub-regions but does not pad the final payload.
func QuantPayloadSize(shape []uint64, dt TensorDType) (uint64, error) {
	if !DTypeRequiresAligned64(dt) {
		return 0, errors.New("mcf: dtype is not a quantized payload")
	}
	if len(shape) != 2 {
		return 0, errors.New("mcf: quant tensors must be rank-2")
	}
	rows := shape[0]
	cols := shape[1]
	if rows == 0 || cols == 0 {
		return 0, errors.New("mcf: invalid quant tensor shape")
	}

	bits := 0
	family := byte(0)
	switch dt {
	case DTypeQ8:
		bits = 8
		family = 'q'
	case DTypeQ4:
		bits = 4
		family = 'q'
	case DTypeK6:
		bits = 6
		family = 'k'
	case DTypeK4:
		bits = 4
		family = 'k'
	case DTypeK3:
		bits = 3
		family = 'k'
	case DTypeK2:
		bits = 2
		family = 'k'
	default:
		return 0, errors.New("mcf: unsupported quant dtype")
	}

	blocksPerRow := (cols + quantBlockSize - 1) / quantBlockSize
	totalBlocks, ok := mulUint64(rows, blocksPerRow)
	if !ok {
		return 0, errors.New("mcf: quant tensor too large")
	}

	blockBits, ok := mulUint64(quantBlockSize, uint64(bits))
	if !ok || blockBits%8 != 0 {
		return 0, errors.New("mcf: invalid quant block size")
	}
	blockDataBytes := blockBits / 8
	dataBytes, ok := mulUint64(totalBlocks, blockDataBytes)
	if !ok {
		return 0, errors.New("mcf: quant tensor too large")
	}

	scaleBytes, ok := mulUint64(totalBlocks, 2)
	if !ok {
		return 0, errors.New("mcf: quant tensor too large")
	}

	if family == 'q' {
		off, ok := align64(scaleBytes)
		if !ok {
			return 0, errors.New("mcf: quant tensor too large")
		}
		size, ok := addUint64(off, dataBytes)
		if !ok {
			return 0, errors.New("mcf: quant tensor too large")
		}
		return size, nil
	}

	superBlocksPerRow := (blocksPerRow + quantSuperBlocks - 1) / quantSuperBlocks
	superCount, ok := mulUint64(rows, superBlocksPerRow)
	if !ok {
		return 0, errors.New("mcf: quant tensor too large")
	}
	superBytes, ok := mulUint64(superCount, 2)
	if !ok {
		return 0, errors.New("mcf: quant tensor too large")
	}

	off, ok := align64(superBytes)
	if !ok {
		return 0, errors.New("mcf: quant tensor too large")
	}
	off, ok = addUint64(off, totalBlocks)
	if !ok {
		return 0, errors.New("mcf: quant tensor too large")
	}
	off, ok = align64(off)
	if !ok {
		return 0, errors.New("mcf: quant tensor too large")
	}
	size, ok := addUint64(off, dataBytes)
	if !ok {
		return 0, errors.New("mcf: quant tensor too large")
	}
	return size, nil
}

func align64(n uint64) (uint64, bool) {
	if n > ^uint64(0)-63 {
		return 0, false
	}
	return (n + 63) & ^uint64(63), true
}

func addUint64(a, b uint64) (uint64, bool) {
	if a > ^uint64(0)-b {
		return 0, false
	}
	return a + b, true
}
