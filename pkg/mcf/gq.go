package mcf

import (
	"errors"
	"fmt"
	"math"
)

// GQ (symmetric group quantization) payload codec. Introduced in MCF v1.2
// (dtypes 0x0050-0x0056, 0x0060-0x0066, 0x0070-0x0076). It stores the output
// of symmetric scalar quantizers such as GSQ, and compressed-tensors
// "pack-quantized" symmetric checkpoints, exactly.
//
// A dtype fixes the code width b (2-8) and the group size G (32, 64 or 128).
// Payload layout for a rank-2 tensor (rows, cols), little-endian, every
// region 64-byte aligned:
//
//	[ Region 1: GroupScales ] bf16 array, rows * ceil(cols/G) values
//	  -> align64
//	[ Region 2: Codes ]       per row, ceil(cols/32) blocks of 32 codes, each
//	                          b-bit two's complement, LSB first, 4*b bytes per
//	                          block; codes past cols are zero
//
// Codes span [-2^(b-1), 2^(b-1)-1]. Reconstruction: w = scale[group] * code,
// where group = col / G.

const (
	dtypeGQBase32  TensorDType = 0x0050
	dtypeGQBase64  TensorDType = 0x0060
	dtypeGQBase128 TensorDType = 0x0070
	gqMinBits                  = 2
	gqMaxBits                  = 8
)

// GQDType returns the gq* dtype for a code width and group size.
func GQDType(bits, group int) (TensorDType, error) {
	if bits < gqMinBits || bits > gqMaxBits {
		return DTypeUnknown, fmt.Errorf("mcf: gq* bit width %d outside [%d, %d]", bits, gqMinBits, gqMaxBits)
	}
	var base TensorDType
	switch group {
	case 32:
		base = dtypeGQBase32
	case 64:
		base = dtypeGQBase64
	case 128:
		base = dtypeGQBase128
	default:
		return DTypeUnknown, fmt.Errorf("mcf: gq* group size %d unsupported (use 32, 64 or 128)", group)
	}
	return base + TensorDType(bits-gqMinBits), nil
}

// GQParams returns the code width and group size of a gq* dtype.
func GQParams(dt TensorDType) (bits, group int, ok bool) {
	var base TensorDType
	switch {
	case dt >= dtypeGQBase32 && dt <= dtypeGQBase32+gqMaxBits-gqMinBits:
		base, group = dtypeGQBase32, 32
	case dt >= dtypeGQBase64 && dt <= dtypeGQBase64+gqMaxBits-gqMinBits:
		base, group = dtypeGQBase64, 64
	case dt >= dtypeGQBase128 && dt <= dtypeGQBase128+gqMaxBits-gqMinBits:
		base, group = dtypeGQBase128, 128
	default:
		return 0, 0, false
	}
	return int(dt-base) + gqMinBits, group, true
}

// IsGQDType reports whether dt is in the gq* family.
func IsGQDType(dt TensorDType) bool {
	_, _, ok := GQParams(dt)
	return ok
}

// GQName returns the method name of a gq* dtype, e.g. "gq4-128".
func GQName(dt TensorDType) string {
	bits, group, ok := GQParams(dt)
	if !ok {
		return ""
	}
	return fmt.Sprintf("gq%d-%d", bits, group)
}

// GQPayloadLayout describes the region offsets of a gq* payload.
type GQPayloadLayout struct {
	Bits           int
	Group          int
	GroupsPerRow   uint64
	BlocksPerRow   uint64
	BlockDataBytes uint64
	ScaleOff       uint64
	ScaleBytes     uint64
	DataOff        uint64
	DataBytes      uint64
	TotalBytes     uint64
}

// GQPayloadLayoutFor computes the region layout for a rank-2 gq* tensor.
func GQPayloadLayoutFor(shape []uint64, dt TensorDType) (GQPayloadLayout, error) {
	bits, group, ok := GQParams(dt)
	if !ok {
		return GQPayloadLayout{}, fmt.Errorf("mcf: %v is not a gq* dtype", dt)
	}
	if len(shape) != 2 {
		return GQPayloadLayout{}, errors.New("mcf: quant tensors must be rank-2")
	}
	rows, cols := shape[0], shape[1]
	if rows == 0 || cols == 0 {
		return GQPayloadLayout{}, errors.New("mcf: invalid quant tensor shape")
	}
	tooLarge := errors.New("mcf: quant tensor too large")
	groupsPerRow := (cols + uint64(group) - 1) / uint64(group)
	blocksPerRow := (cols + QuantBlockSize - 1) / QuantBlockSize
	scaleCount, ok := mulUint64(rows, groupsPerRow)
	if !ok {
		return GQPayloadLayout{}, tooLarge
	}
	scaleBytes, ok := mulUint64(scaleCount, 2)
	if !ok {
		return GQPayloadLayout{}, tooLarge
	}
	totalBlocks, ok := mulUint64(rows, blocksPerRow)
	if !ok {
		return GQPayloadLayout{}, tooLarge
	}
	blockBytes := QuantBlockSize * uint64(bits) / 8
	dataBytes, ok := mulUint64(totalBlocks, blockBytes)
	if !ok {
		return GQPayloadLayout{}, tooLarge
	}
	dataOff, ok := align64(scaleBytes)
	if !ok {
		return GQPayloadLayout{}, tooLarge
	}
	total, ok := addUint64(dataOff, dataBytes)
	if !ok {
		return GQPayloadLayout{}, tooLarge
	}
	return GQPayloadLayout{
		Bits:           bits,
		Group:          group,
		GroupsPerRow:   groupsPerRow,
		BlocksPerRow:   blocksPerRow,
		BlockDataBytes: blockBytes,
		ScaleOff:       0,
		ScaleBytes:     scaleBytes,
		DataOff:        dataOff,
		DataBytes:      dataBytes,
		TotalBytes:     total,
	}, nil
}

// BF16FromFloat32 rounds f to bfloat16 bits (nearest even).
func BF16FromFloat32(f float32) uint16 { return bf16FromF32Bits(math.Float32bits(f)) }

// BF16ToFloat32 widens bfloat16 bits to float32.
func BF16ToFloat32(u uint16) float32 { return bf16ToF32(u) }

// EncodeGQPayload builds a gq* payload from per-group bf16 scale bits
// (rows*groupsPerRow, row-major) and signed codes (rows*cols, row-major).
// Every code must lie in the dtype's range. Importers use it to store
// externally produced codes and scales without re-quantizing.
func EncodeGQPayload(shape []uint64, dt TensorDType, scales []uint16, codes []int8) ([]byte, error) {
	l, err := GQPayloadLayoutFor(shape, dt)
	if err != nil {
		return nil, err
	}
	rows, cols := int(shape[0]), int(shape[1])
	if uint64(len(scales)) != l.ScaleBytes/2 {
		return nil, fmt.Errorf("mcf: gq* payload has %d scales, want %d", len(scales), l.ScaleBytes/2)
	}
	if len(codes) != rows*cols {
		return nil, fmt.Errorf("mcf: gq* payload has %d codes, want %d", len(codes), rows*cols)
	}
	lo, hi := -(1 << (l.Bits - 1)), (1<<(l.Bits-1))-1
	payload := make([]byte, l.TotalBytes)
	for i, s := range scales {
		putF16LE(payload, uint64(i)*2, s)
	}
	blocksPerRow := int(l.BlocksPerRow)
	blockBytes := int(l.BlockDataBytes)
	data := payload[l.DataOff:]
	var q [32]int8
	for r := range rows {
		row := codes[r*cols : (r+1)*cols]
		for b := range blocksPerRow {
			clear(q[:])
			for i, c := range row[b*32 : min((b+1)*32, cols)] {
				if int(c) < lo || int(c) > hi {
					return nil, fmt.Errorf("mcf: gq* code %d at row %d outside [%d, %d]", c, r, lo, hi)
				}
				q[i] = c
			}
			off := (r*blocksPerRow + b) * blockBytes
			packCodes(data[off:off+blockBytes], &q, l.Bits)
		}
	}
	return payload, nil
}

// EncodeGQTensor quantizes a row-major rank-2 tensor to a gq* payload. Each
// group takes the x^2-weighted error-minimising scale (see bestScale), rounded
// to bf16, and the codes nearest to it.
func EncodeGQTensor(src []float32, rows, cols, bits, group int) (TensorDType, []byte, error) {
	dt, err := GQDType(bits, group)
	if err != nil {
		return DTypeUnknown, nil, err
	}
	if err := checkEncodeShape(src, rows, cols); err != nil {
		return DTypeUnknown, nil, err
	}
	l, err := GQPayloadLayoutFor([]uint64{uint64(rows), uint64(cols)}, dt)
	if err != nil {
		return DTypeUnknown, nil, err
	}
	payload := make([]byte, l.TotalBytes)
	scales := payload[:l.ScaleBytes]
	data := payload[l.DataOff:]
	groupsPerRow := int(l.GroupsPerRow)
	blocksPerRow := int(l.BlocksPerRow)
	blockBytes := int(l.BlockDataBytes)
	lo, hi := -(1 << (bits - 1)), (1<<(bits-1))-1
	grid := gridForBits(bits)

	parallelRows(rows, func(r0, r1 int) {
		codes := make([]int8, blocksPerRow*32)
		var q [32]int8
		for r := r0; r < r1; r++ {
			row := src[r*cols : (r+1)*cols]
			clear(codes)
			for g := range groupsPerRow {
				c0, c1 := g*group, min((g+1)*group, cols)
				x := row[c0:c1]
				h := nearestBF16Scale(x, codes[c0:c1], bestScale(x, lo, hi, true, grid), lo, hi)
				putF16LE(scales, uint64(r*groupsPerRow+g)*2, h)
			}
			for b := range blocksPerRow {
				copy(q[:], codes[b*32:(b+1)*32])
				off := (r*blocksPerRow + b) * blockBytes
				packCodes(data[off:off+blockBytes], &q, bits)
			}
		}
	})
	return dt, payload, nil
}

// nearestBF16Scale rounds s to bf16, also tries the bf16 values either side
// of it, and keeps whichever gives the lowest weighted error for x, leaving
// its codes in q. bf16 keeps 8 significant bits, coarse enough at high code
// widths for the neighbours to matter.
func nearestBF16Scale(x []float32, q []int8, s float32, lo, hi int) uint16 {
	h := BF16FromFloat32(s)
	if s == 0 {
		quantizeWithScale(x, q, 0, lo, hi)
		return h
	}
	tmp := make([]int8, len(x))
	best := h
	bestErr := math.Inf(1)
	for _, c := range []uint16{h - 1, h, h + 1} {
		if c&0x7F80 == 0x7F80 || (c^h)&0x8000 != 0 {
			continue
		}
		if e := quantizeWithScale(x, tmp, bf16ToF32(c), lo, hi); e < bestErr {
			best, bestErr = c, e
			copy(q, tmp)
		}
	}
	return best
}

// DecodeGQRow reconstructs one row of a gq* payload into dst (len >= cols).
func DecodeGQRow(payload []byte, l GQPayloadLayout, row, cols int, dst []float32) error {
	if uint64(len(payload)) != l.TotalBytes {
		return errors.New("mcf: gq* payload size mismatch")
	}
	if row < 0 || uint64(row) >= l.ScaleBytes/2/l.GroupsPerRow || len(dst) < cols {
		return errors.New("mcf: gq* row out of range")
	}
	blockBytes := int(l.BlockDataBytes)
	groupsPerRow := int(l.GroupsPerRow)
	shift := 32 - l.Bits
	for c := range cols {
		off := int(l.DataOff) + (row*int(l.BlocksPerRow)+c/32)*blockBytes
		bitPos := (c % 32) * l.Bits
		var u uint32
		for b := range l.Bits {
			p := bitPos + b
			u |= uint32(payload[off+p>>3]>>uint(p&7)&1) << uint(b)
		}
		code := int32(u<<uint(shift)) >> uint(shift)
		scale := bf16ToF32(uint16(payload[(row*groupsPerRow+c/l.Group)*2]) | uint16(payload[(row*groupsPerRow+c/l.Group)*2+1])<<8)
		dst[c] = scale * float32(code)
	}
	return nil
}
