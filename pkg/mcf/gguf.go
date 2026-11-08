package mcf

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
)

// Minimal read-only support for the GGUF container format, sufficient to
// import quantized tensor payloads (e.g. GSQ-refined checkpoints exported as
// GGUF K-quant/Q8_0) into MCF tensors. See quantization-design.md section 2.1.
//
// All parsing is bounds-checked and overflow-safe; unknown sections or value
// types are rejected rather than guessed.

const ggufMagic = "GGUF"

// GGUF metadata value types.
const (
	ggufValUint8   uint32 = 0
	ggufValInt8    uint32 = 1
	ggufValUint16  uint32 = 2
	ggufValInt16   uint32 = 3
	ggufValUint32  uint32 = 4
	ggufValInt32   uint32 = 5
	ggufValFloat32 uint32 = 6
	ggufValBool    uint32 = 7
	ggufValString  uint32 = 8
	ggufValArray   uint32 = 9
	ggufValUint64  uint32 = 10
	ggufValInt64   uint32 = 11
	ggufValFloat64 uint32 = 12
)

// ggml_type identifiers (subset; see ggml.h GGML_TYPE_*).
const (
	ggmlTypeF32  uint32 = 0
	ggmlTypeF16  uint32 = 1
	ggmlTypeQ4_0 uint32 = 2
	ggmlTypeQ4_1 uint32 = 3
	ggmlTypeQ5_0 uint32 = 6
	ggmlTypeQ5_1 uint32 = 7
	ggmlTypeQ8_0 uint32 = 8
	ggmlTypeQ8_1 uint32 = 9
	ggmlTypeQ2_K uint32 = 10
	ggmlTypeQ3_K uint32 = 11
	ggmlTypeQ4_K uint32 = 12
	ggmlTypeQ5_K uint32 = 13
	ggmlTypeQ6_K uint32 = 14
	ggmlTypeQ8_K uint32 = 15
	ggmlTypeBF16 uint32 = 30
)

// ggmlTypeBlock describes the block layout of a ggml quantized type:
// BlockBytes is the on-disk size of one block holding BlockElems elements.
type ggmlTypeBlock struct {
	BlockBytes uint64
	BlockElems uint64
}

var ggmlTypeBlocks = map[uint32]ggmlTypeBlock{
	ggmlTypeF32:  {4, 1},
	ggmlTypeF16:  {2, 1},
	ggmlTypeBF16: {2, 1},
	ggmlTypeQ4_0: {18, 32},
	ggmlTypeQ4_1: {20, 32},
	ggmlTypeQ5_0: {22, 32},
	ggmlTypeQ5_1: {24, 32},
	ggmlTypeQ8_0: {34, 32},
	ggmlTypeQ8_1: {36, 32},
	ggmlTypeQ2_K: {84, 256},
	ggmlTypeQ3_K: {110, 256},
	ggmlTypeQ4_K: {144, 256},
	ggmlTypeQ5_K: {176, 256},
	ggmlTypeQ6_K: {210, 256},
	ggmlTypeQ8_K: {260, 256},
}

// GGMLTypeName returns the canonical name of a ggml_type, or "type<N>".
func GGMLTypeName(t uint32) string {
	switch t {
	case ggmlTypeF32:
		return "F32"
	case ggmlTypeF16:
		return "F16"
	case ggmlTypeBF16:
		return "BF16"
	case ggmlTypeQ4_0:
		return "Q4_0"
	case ggmlTypeQ4_1:
		return "Q4_1"
	case ggmlTypeQ5_0:
		return "Q5_0"
	case ggmlTypeQ5_1:
		return "Q5_1"
	case ggmlTypeQ8_0:
		return "Q8_0"
	case ggmlTypeQ8_1:
		return "Q8_1"
	case ggmlTypeQ2_K:
		return "Q2_K"
	case ggmlTypeQ3_K:
		return "Q3_K"
	case ggmlTypeQ4_K:
		return "Q4_K"
	case ggmlTypeQ5_K:
		return "Q5_K"
	case ggmlTypeQ6_K:
		return "Q6_K"
	case ggmlTypeQ8_K:
		return "Q8_K"
	default:
		return fmt.Sprintf("type<%d>", t)
	}
}

// GGUFTensor describes one tensor info record in a GGUF file.
type GGUFTensor struct {
	Name string
	// Dims in GGUF order: Dims[0] is the innermost dimension (elements per
	// row, i.e. columns for a rank-2 weight matrix).
	Dims []uint64
	// Type is the ggml_type identifier.
	Type uint32
	// Offset of the tensor payload relative to the start of the tensor data
	// section (see GGUFFile.DataStart).
	Offset uint64
}

// NumElems returns the total element count of the tensor.
func (t GGUFTensor) NumElems() (uint64, bool) {
	n := uint64(1)
	for _, d := range t.Dims {
		var ok bool
		n, ok = mulUint64(n, d)
		if !ok {
			return 0, false
		}
	}
	return n, true
}

// GGUFFile is a parsed view over a GGUF container.
type GGUFFile struct {
	r         io.ReaderAt
	size      int64
	Version   uint32
	Alignment uint64
	// Metadata holds scalar/string metadata values (arrays are skipped and
	// omitted).
	Metadata  map[string]any
	tensors   []GGUFTensor
	dataStart int64
}

// ErrGGUF is returned for malformed GGUF input.
var ErrGGUF = errors.New("mcf: malformed GGUF")

type ggufReader struct {
	r    io.ReaderAt
	size int64
	off  int64
}

func (g *ggufReader) readAt(off int64, p []byte) error {
	if off < 0 || off > g.size || int64(len(p)) > g.size-off {
		return fmt.Errorf("%w: read out of bounds (off=%d len=%d size=%d)", ErrGGUF, off, len(p), g.size)
	}
	_, err := g.r.ReadAt(p, off)
	return err
}

func (g *ggufReader) u32() (uint32, error) {
	var b [4]byte
	if err := g.readAt(g.off, b[:]); err != nil {
		return 0, err
	}
	g.off += 4
	return binary.LittleEndian.Uint32(b[:]), nil
}

func (g *ggufReader) u64() (uint64, error) {
	var b [8]byte
	if err := g.readAt(g.off, b[:]); err != nil {
		return 0, err
	}
	g.off += 8
	return binary.LittleEndian.Uint64(b[:]), nil
}

func (g *ggufReader) str() (string, error) {
	n, err := g.u64()
	if err != nil {
		return "", err
	}
	if n > uint64(g.size-g.off) {
		return "", fmt.Errorf("%w: string length out of bounds", ErrGGUF)
	}
	buf := make([]byte, n)
	if err := g.readAt(g.off, buf); err != nil {
		return "", err
	}
	g.off += int64(n)
	return string(buf), nil
}

func ggufValueSize(vt uint32) (int64, bool) {
	switch vt {
	case ggufValUint8, ggufValInt8, ggufValBool:
		return 1, true
	case ggufValUint16, ggufValInt16:
		return 2, true
	case ggufValUint32, ggufValInt32, ggufValFloat32:
		return 4, true
	case ggufValUint64, ggufValInt64, ggufValFloat64:
		return 8, true
	}
	return 0, false
}

func (g *ggufReader) skipValue(vt uint32) error {
	if vt == ggufValString {
		_, err := g.str()
		return err
	}
	if vt == ggufValArray {
		et, err := g.u32()
		if err != nil {
			return err
		}
		n, err := g.u64()
		if err != nil {
			return err
		}
		if et == ggufValString {
			for i := uint64(0); i < n; i++ {
				if _, err := g.str(); err != nil {
					return err
				}
			}
			return nil
		}
		sz, ok := ggufValueSize(et)
		if !ok {
			return fmt.Errorf("%w: unknown array element type %d", ErrGGUF, et)
		}
		if n > uint64(g.size-g.off)/uint64(sz) {
			return fmt.Errorf("%w: array length out of bounds", ErrGGUF)
		}
		g.off += int64(n) * sz
		return nil
	}
	sz, ok := ggufValueSize(vt)
	if !ok {
		return fmt.Errorf("%w: unknown value type %d", ErrGGUF, vt)
	}
	if g.off+sz > g.size {
		return fmt.Errorf("%w: value out of bounds", ErrGGUF)
	}
	g.off += sz
	return nil
}

func (g *ggufReader) readScalar(vt uint32) (any, error) {
	var b [8]byte
	sz, ok := ggufValueSize(vt)
	if !ok {
		return nil, fmt.Errorf("%w: unknown value type %d", ErrGGUF, vt)
	}
	if err := g.readAt(g.off, b[:sz]); err != nil {
		return nil, err
	}
	g.off += sz
	switch vt {
	case ggufValUint8:
		return b[0], nil
	case ggufValInt8:
		return int8(b[0]), nil
	case ggufValUint16:
		return binary.LittleEndian.Uint16(b[:]), nil
	case ggufValInt16:
		return int16(binary.LittleEndian.Uint16(b[:])), nil
	case ggufValUint32:
		return binary.LittleEndian.Uint32(b[:]), nil
	case ggufValInt32:
		return int32(binary.LittleEndian.Uint32(b[:])), nil
	case ggufValFloat32:
		return math.Float32frombits(binary.LittleEndian.Uint32(b[:])), nil
	case ggufValBool:
		return b[0] != 0, nil
	case ggufValUint64:
		return binary.LittleEndian.Uint64(b[:]), nil
	case ggufValInt64:
		return int64(binary.LittleEndian.Uint64(b[:])), nil
	case ggufValFloat64:
		return math.Float64frombits(binary.LittleEndian.Uint64(b[:])), nil
	}
	return nil, fmt.Errorf("%w: not a scalar type %d", ErrGGUF, vt)
}

// ParseGGUF parses the header, metadata and tensor index of a GGUF container
// backed by r (size = total bytes readable from r). Tensor payloads are read
// lazily via TensorPayload.
func ParseGGUF(r io.ReaderAt, size int64) (*GGUFFile, error) {
	if size < 32 {
		return nil, fmt.Errorf("%w: file too small (%d bytes)", ErrGGUF, size)
	}
	g := &ggufReader{r: r, size: size}
	var magic [4]byte
	if err := g.readAt(0, magic[:]); err != nil {
		return nil, err
	}
	if string(magic[:]) != ggufMagic {
		return nil, fmt.Errorf("%w: bad magic %q", ErrGGUF, string(magic[:]))
	}
	g.off = 4
	ver, err := g.u32()
	if err != nil {
		return nil, err
	}
	if ver < 2 || ver > 3 {
		return nil, fmt.Errorf("%w: unsupported version %d", ErrGGUF, ver)
	}
	nTensors, err := g.u64()
	if err != nil {
		return nil, err
	}
	nKV, err := g.u64()
	if err != nil {
		return nil, err
	}
	if nKV > uint64(size) {
		return nil, fmt.Errorf("%w: metadata count out of bounds", ErrGGUF)
	}
	meta := make(map[string]any, nKV)
	alignment := uint64(32)
	for i := uint64(0); i < nKV; i++ {
		key, err := g.str()
		if err != nil {
			return nil, err
		}
		vt, err := g.u32()
		if err != nil {
			return nil, err
		}
		switch vt {
		case ggufValString:
			v, err := g.str()
			if err != nil {
				return nil, err
			}
			meta[key] = v
		case ggufValArray:
			if err := g.skipValue(vt); err != nil {
				return nil, err
			}
		default:
			v, err := g.readScalar(vt)
			if err != nil {
				return nil, err
			}
			meta[key] = v
		}
	}
	if a, ok := meta["general.alignment"]; ok {
		switch v := a.(type) {
		case uint32:
			alignment = uint64(v)
		case uint64:
			alignment = v
		}
	}
	if alignment == 0 || alignment&(alignment-1) != 0 {
		return nil, fmt.Errorf("%w: invalid alignment %d", ErrGGUF, alignment)
	}
	if nTensors > uint64(size) {
		return nil, fmt.Errorf("%w: tensor count out of bounds", ErrGGUF)
	}
	tensors := make([]GGUFTensor, 0, nTensors)
	for i := uint64(0); i < nTensors; i++ {
		name, err := g.str()
		if err != nil {
			return nil, err
		}
		nDims, err := g.u32()
		if err != nil {
			return nil, err
		}
		if nDims > 8 || uint64(nDims) > uint64(size-g.off)/8 {
			return nil, fmt.Errorf("%w: bad dims count %d", ErrGGUF, nDims)
		}
		dims := make([]uint64, nDims)
		for d := range dims {
			v, err := g.u64()
			if err != nil {
				return nil, err
			}
			dims[d] = v
		}
		gt, err := g.u32()
		if err != nil {
			return nil, err
		}
		off, err := g.u64()
		if err != nil {
			return nil, err
		}
		tensors = append(tensors, GGUFTensor{Name: name, Dims: dims, Type: gt, Offset: off})
	}
	dataStart := g.off
	rem := dataStart % int64(alignment)
	if rem != 0 {
		dataStart += int64(alignment) - rem
	}
	if dataStart > size {
		return nil, fmt.Errorf("%w: data section out of bounds", ErrGGUF)
	}
	return &GGUFFile{
		r:         r,
		size:      size,
		Version:   ver,
		Alignment: alignment,
		Metadata:  meta,
		tensors:   tensors,
		dataStart: dataStart,
	}, nil
}

// TensorCount returns the number of tensor info records.
func (f *GGUFFile) TensorCount() int { return len(f.tensors) }

// Tensor returns the i-th tensor info.
func (f *GGUFFile) Tensor(i int) (GGUFTensor, error) {
	if i < 0 || i >= len(f.tensors) {
		return GGUFTensor{}, fmt.Errorf("%w: tensor index %d out of range", ErrGGUF, i)
	}
	return f.tensors[i], nil
}

// FindTensor returns the tensor info with the given name.
func (f *GGUFFile) FindTensor(name string) (GGUFTensor, bool) {
	for _, t := range f.tensors {
		if t.Name == name {
			return t, true
		}
	}
	return GGUFTensor{}, false
}

// DataStart returns the absolute file offset of the tensor data section.
func (f *GGUFFile) DataStart() int64 { return f.dataStart }

// TensorPayload reads the raw payload bytes of a tensor. The payload size is
// derived from the ggml type block layout; types without a known block size
// are rejected.
func (f *GGUFFile) TensorPayload(t GGUFTensor) ([]byte, error) {
	blk, ok := ggmlTypeBlocks[t.Type]
	if !ok {
		return nil, fmt.Errorf("%w: unknown block layout for ggml type %s", ErrGGUF, GGMLTypeName(t.Type))
	}
	numElems, ok := t.NumElems()
	if !ok {
		return nil, fmt.Errorf("%w: tensor %q element count overflow", ErrGGUF, t.Name)
	}
	if numElems%blk.BlockElems != 0 {
		return nil, fmt.Errorf("%w: tensor %q has %d elems, not a multiple of block %d for %s",
			ErrGGUF, t.Name, numElems, blk.BlockElems, GGMLTypeName(t.Type))
	}
	blocks, ok := mulUint64(numElems/blk.BlockElems, 1)
	if !ok {
		return nil, fmt.Errorf("%w: overflow", ErrGGUF)
	}
	payloadBytes, ok := mulUint64(blocks, blk.BlockBytes)
	if !ok {
		return nil, fmt.Errorf("%w: payload size overflow", ErrGGUF)
	}
	abs := f.dataStart + int64(t.Offset)
	if t.Offset > uint64(f.size) || payloadBytes > uint64(f.size) || abs > f.size || payloadBytes > uint64(f.size-abs) {
		return nil, fmt.Errorf("%w: tensor %q payload out of bounds (off=%d size=%d file=%d)",
			ErrGGUF, t.Name, abs, payloadBytes, f.size)
	}
	buf := make([]byte, payloadBytes)
	if _, err := f.r.ReadAt(buf, abs); err != nil {
		return nil, err
	}
	return buf, nil
}

// TranscodeGGUFQuant converts a GGUF quantized tensor payload into an MCF
// quant tensor payload, returning the MCF dtype, shape (rows, cols) and
// payload. The conversion is value-exact: decoding the MCF tensor yields the
// same row values as GGUF dequantization of the source payload.
//
// Supported today: Q8_0 -> DTypeQ8 (block 32, f16 scale, int8 data — the MCF
// q-family layout is the same values with scales and data in separate
// regions). K-quants (Q2_K..Q8_K) carry a per-superblock min term (or int8
// sub-scales) that MCF's symmetric k* family cannot represent byte-exactly;
// they must be dequantized and re-encoded (see quantization-design.md).
func TranscodeGGUFQuant(t GGUFTensor, payload []byte) (TensorDType, []uint64, []byte, error) {
	if t.Type != ggmlTypeQ8_0 {
		return 0, nil, nil, fmt.Errorf("%w: transcode supports Q8_0 only, got %s", ErrGGUF, GGMLTypeName(t.Type))
	}
	if len(t.Dims) != 2 {
		return 0, nil, nil, fmt.Errorf("%w: quant transcode requires rank-2 tensor, got rank %d", ErrGGUF, len(t.Dims))
	}
	cols := t.Dims[0]
	rows := t.Dims[1]
	if cols == 0 || rows == 0 || cols%QuantBlockSize != 0 {
		return 0, nil, nil, fmt.Errorf("%w: Q8_0 requires cols %% %d == 0, got cols=%d", ErrGGUF, QuantBlockSize, cols)
	}
	blocksPerRow := cols / QuantBlockSize
	totalBlocks, ok := mulUint64(rows, blocksPerRow)
	if !ok {
		return 0, nil, nil, fmt.Errorf("%w: overflow", ErrGGUF)
	}
	const q8BlockBytes = 2 + QuantBlockSize // f16 scale + 32 int8
	wantPayload, ok := mulUint64(totalBlocks, q8BlockBytes)
	if !ok || uint64(len(payload)) != wantPayload {
		return 0, nil, nil, fmt.Errorf("%w: Q8_0 payload size mismatch: have %d want %d", ErrGGUF, len(payload), wantPayload)
	}

	wantMCF, err := QuantPayloadSize([]uint64{rows, cols}, DTypeQ8)
	if err != nil {
		return 0, nil, nil, err
	}
	out := make([]byte, wantMCF)
	// Region 1: BlockScales (f16 per block), copied verbatim.
	scaleBytes, ok := mulUint64(totalBlocks, 2)
	if !ok {
		return 0, nil, nil, fmt.Errorf("%w: overflow", ErrGGUF)
	}
	dataOff, ok := align64(scaleBytes)
	if !ok {
		return 0, nil, nil, fmt.Errorf("%w: overflow", ErrGGUF)
	}
	for b := uint64(0); b < totalBlocks; b++ {
		src := b * q8BlockBytes
		copy(out[b*2:], payload[src:src+2])                                   // f16 scale, unchanged
		copy(out[dataOff+b*QuantBlockSize:], payload[src+2:src+q8BlockBytes]) // int8 data
	}
	return DTypeQ8, []uint64{rows, cols}, out, nil
}
