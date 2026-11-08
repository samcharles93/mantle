package mcf

import (
	"errors"
	"fmt"
)

// UQ (non-uniform codebook) quantization payload codec. Introduced in MCF
// v1.1 (dtypes 0x0040-0x0046). See model-container-file.md "UQ family".
//
// Payload layout for a rank-2 tensor (rows, cols), little-endian, every
// region 64-byte aligned:
//
//	[ Region 1: Codebooks ]   f16 array, superCount * 2^b values
//	  -> align64
//	[ Region 2: BlockScales ] f16 array, totalBlocks values (block = 32 elems)
//	  -> align64
//	[ Region 3: Indices ]     packed unsigned b-bit indices, LSB first,
//	                           totalBlocks * (32*b/8) bytes
//
// A super-block is 256 elements = 8 blocks and carries one codebook of 2^b
// f16 levels. Reconstruction: w = blockScale * codebook[super][idx].

// IsUQDType reports whether dt is in the uq* family.
func IsUQDType(dt TensorDType) bool {
	return dt >= DTypeUQ2 && dt <= DTypeUQ8
}

// UQBits returns the index bit width of a uq* dtype.
func UQBits(dt TensorDType) (int, bool) {
	if !IsUQDType(dt) {
		return 0, false
	}
	return int(dt-DTypeUQ2) + 2, true
}

// UQPayloadLayout describes the region offsets of a uq* payload.
type UQPayloadLayout struct {
	Bits              int
	BlocksPerRow      uint64
	SuperBlocksPerRow uint64
	TotalBlocks       uint64
	SuperCount        uint64
	CodebookElems     uint64

	CodebookOff   uint64
	CodebookBytes uint64
	ScaleOff      uint64
	ScaleBytes    uint64
	DataOff       uint64
	DataBytes     uint64
	TotalBytes    uint64
}

// UQPayloadLayoutFor computes the region layout for a rank-2 uq* tensor.
func UQPayloadLayoutFor(shape []uint64, dt TensorDType) (UQPayloadLayout, error) {
	bits, ok := UQBits(dt)
	if !ok {
		return UQPayloadLayout{}, fmt.Errorf("mcf: %v is not a uq* dtype", dt)
	}
	if len(shape) != 2 {
		return UQPayloadLayout{}, errors.New("mcf: quant tensors must be rank-2")
	}
	rows, cols := shape[0], shape[1]
	if rows == 0 || cols == 0 {
		return UQPayloadLayout{}, errors.New("mcf: invalid quant tensor shape")
	}
	blocksPerRow := (cols + QuantBlockSize - 1) / QuantBlockSize
	totalBlocks, ok := mulUint64(rows, blocksPerRow)
	if !ok {
		return UQPayloadLayout{}, errors.New("mcf: quant tensor too large")
	}
	superBlocksPerRow := (blocksPerRow + QuantSuperBlocks - 1) / QuantSuperBlocks
	superCount, ok := mulUint64(rows, superBlocksPerRow)
	if !ok {
		return UQPayloadLayout{}, errors.New("mcf: quant tensor too large")
	}
	codebookElems := uint64(1) << uint(bits)
	codebookBytes, ok := mulUint64(superCount, codebookElems*2)
	if !ok {
		return UQPayloadLayout{}, errors.New("mcf: quant tensor too large")
	}
	scaleBytes, ok := mulUint64(totalBlocks, 2)
	if !ok {
		return UQPayloadLayout{}, errors.New("mcf: quant tensor too large")
	}
	dataBytes, ok := mulUint64(totalBlocks, uint64(32*bits)/8)
	if !ok || 32*uint64(bits)%8 != 0 {
		return UQPayloadLayout{}, errors.New("mcf: invalid uq block size")
	}
	scaleOff, ok := align64(codebookBytes)
	if !ok {
		return UQPayloadLayout{}, errors.New("mcf: quant tensor too large")
	}
	dataOff, ok := align64(scaleOff + scaleBytes)
	if !ok {
		return UQPayloadLayout{}, errors.New("mcf: quant tensor too large")
	}
	total, ok := addUint64(dataOff, dataBytes)
	if !ok {
		return UQPayloadLayout{}, errors.New("mcf: quant tensor too large")
	}
	return UQPayloadLayout{
		Bits:              bits,
		BlocksPerRow:      blocksPerRow,
		SuperBlocksPerRow: superBlocksPerRow,
		TotalBlocks:       totalBlocks,
		SuperCount:        superCount,
		CodebookElems:     codebookElems,
		CodebookOff:       0,
		CodebookBytes:     codebookBytes,
		ScaleOff:          scaleOff,
		ScaleBytes:        scaleBytes,
		DataOff:           dataOff,
		DataBytes:         dataBytes,
		TotalBytes:        total,
	}, nil
}

// EncodeUQPayload packs codebooks, per-block scales and per-element indices
// into a uq* payload. codebooks must hold superCount*2^b float32 values
// (f16-rounded on encode), scales totalBlocks values, indices rows*cols
// values each < 2^b.
func EncodeUQPayload(shape []uint64, dt TensorDType, codebooks, scales []float32, indices []uint8) ([]byte, error) {
	layout, err := UQPayloadLayoutFor(shape, dt)
	if err != nil {
		return nil, err
	}
	rows, cols := shape[0], shape[1]
	if uint64(len(codebooks)) != layout.SuperCount*layout.CodebookElems {
		return nil, fmt.Errorf("mcf: uq codebooks length %d want %d", len(codebooks), layout.SuperCount*layout.CodebookElems)
	}
	if uint64(len(scales)) != layout.TotalBlocks {
		return nil, fmt.Errorf("mcf: uq scales length %d want %d", len(scales), layout.TotalBlocks)
	}
	if uint64(len(indices)) != rows*cols {
		return nil, fmt.Errorf("mcf: uq indices length %d want %d", len(indices), rows*cols)
	}
	out := make([]byte, layout.TotalBytes)
	for i, cb := range codebooks {
		putF16LE(out, layout.CodebookOff+uint64(i)*2, float32ToFP16Bits(cb))
	}
	for i, s := range scales {
		putF16LE(out, layout.ScaleOff+uint64(i)*2, float32ToFP16Bits(s))
	}
	bitPos := uint64(0)
	for _, idx := range indices {
		if uint64(idx) >= layout.CodebookElems {
			return nil, fmt.Errorf("mcf: uq index %d out of range for %d bits", idx, layout.Bits)
		}
		for b := uint64(0); b < uint64(layout.Bits); b++ {
			if idx&(1<<b) != 0 {
				out[layout.DataOff+bitPos/8] |= 1 << (bitPos % 8)
			}
			bitPos++
		}
	}
	return out, nil
}

// DecodeUQRow decodes one row of a uq* payload into dst (len >= cols) using
// the reference reconstruction w = blockScale * codebook[idx].
func DecodeUQRow(payload []byte, layout UQPayloadLayout, row int, cols int, dst []float32) error {
	if row < 0 {
		return errors.New("mcf: negative row")
	}
	if uint64(len(payload)) < layout.TotalBytes {
		return errors.New("mcf: uq payload truncated")
	}
	if len(dst) < cols {
		return errors.New("mcf: dst too small")
	}
	rowU := uint64(row)
	blockBase := rowU * layout.BlocksPerRow
	superBase := rowU * layout.SuperBlocksPerRow
	for b := uint64(0); b < layout.BlocksPerRow; b++ {
		col := b * QuantBlockSize
		n := cols - int(col)
		if n <= 0 {
			break
		}
		if n > 32 {
			n = 32
		}
		blockIdx := blockBase + b
		superIdx := superBase + b/QuantSuperBlocks
		scale := f16LEAt(payload, layout.ScaleOff+blockIdx*2)
		cbOff := layout.CodebookOff + superIdx*layout.CodebookElems*2
		dataOff := layout.DataOff + blockIdx*uint64(32*layout.Bits)/8
		bitPos := uint64(0)
		for i := 0; i < n; i++ {
			var idx uint64
			for bb := uint64(0); bb < uint64(layout.Bits); bb++ {
				if payload[dataOff+bitPos/8]&(1<<(bitPos%8)) != 0 {
					idx |= 1 << bb
				}
				bitPos++
			}
			if cbOff+idx*2+1 >= uint64(len(payload)) {
				return errors.New("mcf: uq codebook out of bounds")
			}
			dst[int(col)+i] = scale * f16LEAt(payload, cbOff+idx*2)
		}
	}
	return nil
}

func putF16LE(b []byte, off uint64, u uint16) {
	b[off] = byte(u)
	b[off+1] = byte(u >> 8)
}

func f16LEAt(b []byte, off uint64) float32 {
	u := uint16(b[off]) | uint16(b[off+1])<<8
	return fp16ToFloat32(u)
}

// Float32ToFP16 converts a float32 to IEEE half bits with round-to-nearest-even.
func Float32ToFP16(f float32) uint16 { return float32ToFP16Bits(f) }

// FP16ToFloat32 converts IEEE half bits to float32.
func FP16ToFloat32(h uint16) float32 { return fp16ToFloat32(h) }
