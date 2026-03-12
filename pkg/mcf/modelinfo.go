package mcf

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"sort"
)

// ModelInfo payload format (v1), little-endian.
//
// Layout:
//   [0]   ModelInfoHeader
//   [8]   modelInfoFixedV1
//   [...] string/data blobs (length-prefixed), aligned to 8 bytes
//   [...] kv table (ModelInfoKV entries), aligned to 8 bytes
//
// String blob encoding:
//   u32 byte_len
//   []byte (byte_len bytes, no NUL terminator)
//   (then 8-byte alignment as needed)
//
// KV encoding:
//   Each entry stores:
//     KeyOff   -> string blob
//     Type     -> KVUint32 / KVFloat32 / KVString
//     ValueOff -> for KVString: string blob; for KVUint32: u32 stored at ValueOff;
//                for KVFloat32: f32 stored at ValueOff.

const modelInfoVersionV1 uint32 = 1

type ModelInfoHeader struct {
	Version uint32 // = 1
	Flags   uint32 // reserved, must be zero
}

type Arch uint32

const (
	ArchUnknown Arch = iota
	ArchLLaMA
	ArchMistral
	ArchQwen
	ArchGemma
	ArchGranite
	ArchSmollM3
)

type RopeType uint32

const (
	RopeNone RopeType = iota
	RopeStandard
	RopeLinear
	RopeYarn
)

const (
	KVUint32  = 1
	KVFloat32 = 2
	KVString  = 3
)

type ModelInfoKV struct {
	KeyOff   uint64
	Type     uint32
	_        uint32 // padding
	ValueOff uint64
}

type ModelInfo struct {
	Arch      Arch
	ModelName string
	BaseModel string

	VocabSize   uint32
	HiddenSize  uint32
	LayerCount  uint32
	HeadCount   uint32
	HeadCountKV uint32

	ContextLength uint32
	TrainContext  uint32

	RopeType     RopeType
	RopeFreqBase float32
	RopeScale    float32

	Extras map[string]any
}

type modelInfoFixedV1 struct {
	Arch          uint32
	_             uint32 // padding
	VocabSize     uint32
	HiddenSize    uint32
	LayerCount    uint32
	HeadCount     uint32
	HeadCountKV   uint32
	ContextLength uint32
	TrainContext  uint32
	RopeType      uint32
	RopeFreqBase  float32
	RopeScale     float32

	ModelNameOff uint64
	BaseModelOff uint64

	KVCount uint32
	_       uint32 // padding
	KVOff   uint64
}

func EncodeModelInfo(mi *ModelInfo) ([]byte, error) {
	if mi == nil {
		return nil, errors.New("modelinfo: nil ModelInfo")
	}

	hdr := ModelInfoHeader{
		Version: modelInfoVersionV1,
		Flags:   0,
	}

	var fixed modelInfoFixedV1
	fixed.Arch = uint32(mi.Arch)

	fixed.VocabSize = mi.VocabSize
	fixed.HiddenSize = mi.HiddenSize
	fixed.LayerCount = mi.LayerCount
	fixed.HeadCount = mi.HeadCount
	fixed.HeadCountKV = mi.HeadCountKV
	fixed.ContextLength = mi.ContextLength
	fixed.TrainContext = mi.TrainContext

	fixed.RopeType = uint32(mi.RopeType)
	fixed.RopeFreqBase = mi.RopeFreqBase
	fixed.RopeScale = mi.RopeScale

	b := newBlobBuilder()

	// Reserve header + fixed up front.
	{
		tmp := make([]byte, binary.Size(hdr)+binary.Size(fixed))
		_ = b.addRaw(tmp) // offsets start after this placeholder
	}

	// Core strings.
	if mi.ModelName != "" {
		off, err := b.addString(mi.ModelName)
		if err != nil {
			return nil, err
		}
		fixed.ModelNameOff = off
	}
	if mi.BaseModel != "" {
		off, err := b.addString(mi.BaseModel)
		if err != nil {
			return nil, err
		}
		fixed.BaseModelOff = off
	}

	// Extras KV table.
	kvs, err := encodeExtrasKV(b, mi.Extras)
	if err != nil {
		return nil, err
	}

	// Write KV table after blobs, aligned.
	b.align(8)
	kvOff := b.offset()
	if len(kvs) > 0 {
		for i := range kvs {
			if err := b.writeStruct(&kvs[i]); err != nil {
				return nil, err
			}
		}
	}
	fixed.KVCount = uint32(len(kvs))
	fixed.KVOff = kvOff

	// Patch header+fixed.
	out := b.bytes()
	if len(out) < binary.Size(hdr)+binary.Size(fixed) {
		return nil, errors.New("modelinfo: internal size invariant failed")
	}

	// Write hdr at start.
	{
		var buf bytes.Buffer
		if err := binary.Write(&buf, binary.LittleEndian, &hdr); err != nil {
			return nil, err
		}
		copy(out[0:binary.Size(hdr)], buf.Bytes())
	}

	// Write fixed immediately after hdr.
	{
		var buf bytes.Buffer
		if err := binary.Write(&buf, binary.LittleEndian, &fixed); err != nil {
			return nil, err
		}
		start := binary.Size(hdr)
		end := start + binary.Size(fixed)
		copy(out[start:end], buf.Bytes())
	}

	return out, nil
}

func encodeExtrasKV(b *blobBuilder, extras map[string]any) ([]ModelInfoKV, error) {
	if len(extras) == 0 {
		return nil, nil
	}

	keys := make([]string, 0, len(extras))
	for k := range extras {
		if k == "" {
			return nil, errors.New("modelinfo: extras contains empty key")
		}
		keys = append(keys, k)
	}
	sort.Strings(keys)

	kvs := make([]ModelInfoKV, 0, len(keys))
	for _, k := range keys {
		v := extras[k]

		keyOff, err := b.addString(k)
		if err != nil {
			return nil, err
		}

		var kv ModelInfoKV
		kv.KeyOff = keyOff

		switch vv := v.(type) {
		case string:
			valOff, err := b.addString(vv)
			if err != nil {
				return nil, err
			}
			kv.Type = KVString
			kv.ValueOff = valOff

		case []byte:
			// Store bytes as string-blob (length + raw), preserve exact bytes.
			valOff, err := b.addBytesAsBlob(vv)
			if err != nil {
				return nil, err
			}
			kv.Type = KVString
			kv.ValueOff = valOff

		case uint32:
			valOff, err := b.addU32(vv)
			if err != nil {
				return nil, err
			}
			kv.Type = KVUint32
			kv.ValueOff = valOff

		case uint64:
			if vv > math.MaxUint32 {
				return nil, fmt.Errorf("modelinfo: extras[%q] uint64 overflows uint32 (%d)", k, vv)
			}
			valOff, err := b.addU32(uint32(vv))
			if err != nil {
				return nil, err
			}
			kv.Type = KVUint32
			kv.ValueOff = valOff

		case int:
			if vv < 0 || vv > math.MaxUint32 {
				return nil, fmt.Errorf("modelinfo: extras[%q] int out of uint32 range (%d)", k, vv)
			}
			valOff, err := b.addU32(uint32(vv))
			if err != nil {
				return nil, err
			}
			kv.Type = KVUint32
			kv.ValueOff = valOff

		case int32:
			if vv < 0 {
				return nil, fmt.Errorf("modelinfo: extras[%q] int32 negative (%d)", k, vv)
			}
			valOff, err := b.addU32(uint32(vv))
			if err != nil {
				return nil, err
			}
			kv.Type = KVUint32
			kv.ValueOff = valOff

		case float32:
			valOff, err := b.addF32(vv)
			if err != nil {
				return nil, err
			}
			kv.Type = KVFloat32
			kv.ValueOff = valOff

		case float64:
			if math.IsNaN(vv) || math.IsInf(vv, 0) {
				return nil, fmt.Errorf("modelinfo: extras[%q] invalid float64 (%v)", k, vv)
			}
			if vv < -math.MaxFloat32 || vv > math.MaxFloat32 {
				return nil, fmt.Errorf("modelinfo: extras[%q] float64 out of float32 range (%v)", k, vv)
			}
			valOff, err := b.addF32(float32(vv))
			if err != nil {
				return nil, err
			}
			kv.Type = KVFloat32
			kv.ValueOff = valOff

		case nil:
			// Skip nils silently so callers can merge maps easily.
			continue

		default:
			return nil, fmt.Errorf("modelinfo: extras[%q] unsupported type %T", k, v)
		}

		kvs = append(kvs, kv)
	}

	return kvs, nil
}

type blobBuilder struct {
	buf bytes.Buffer
}

func newBlobBuilder() *blobBuilder {
	return &blobBuilder{}
}

func (b *blobBuilder) bytes() []byte {
	return b.buf.Bytes()
}

func (b *blobBuilder) offset() uint64 {
	return uint64(b.buf.Len())
}

func (b *blobBuilder) align(n int) {
	if n <= 1 {
		return
	}
	pad := (n - (b.buf.Len() % n)) % n
	if pad == 0 {
		return
	}
	_, _ = b.buf.Write(make([]byte, pad))
}

func (b *blobBuilder) addRaw(p []byte) uint64 {
	off := b.offset()
	_, _ = b.buf.Write(p)
	return off
}

func (b *blobBuilder) writeStruct(v any) error {
	offBefore := b.offset()
	if err := binary.Write(&b.buf, binary.LittleEndian, v); err != nil {
		return err
	}
	_ = offBefore
	return nil
}

func (b *blobBuilder) addBytesAsBlob(p []byte) (uint64, error) {
	if len(p) > math.MaxUint32 {
		return 0, errors.New("modelinfo: blob too large")
	}
	b.align(8)
	off := b.offset()
	if err := binary.Write(&b.buf, binary.LittleEndian, uint32(len(p))); err != nil {
		return 0, err
	}
	_, _ = b.buf.Write(p)
	b.align(8)
	return off, nil
}

func (b *blobBuilder) addString(s string) (uint64, error) {
	if len(s) == 0 {
		return 0, nil
	}
	return b.addBytesAsBlob([]byte(s))
}

func (b *blobBuilder) addU32(v uint32) (uint64, error) {
	b.align(8)
	off := b.offset()
	if err := binary.Write(&b.buf, binary.LittleEndian, v); err != nil {
		return 0, err
	}
	b.align(8)
	return off, nil
}

func (b *blobBuilder) addF32(v float32) (uint64, error) {
	if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
		return 0, fmt.Errorf("modelinfo: invalid float32 %v", v)
	}
	b.align(8)
	off := b.offset()
	if err := binary.Write(&b.buf, binary.LittleEndian, v); err != nil {
		return 0, err
	}
	b.align(8)
	return off, nil
}
