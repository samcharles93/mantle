package gguf

import (
	"bytes"
	"fmt"
	"os"
	"syscall"
)

const (
	magicGGUF = "GGUF"
)

type ValueType uint32

const (
	TypeUint8   ValueType = 0
	TypeInt8    ValueType = 1
	TypeUint16  ValueType = 2
	TypeInt16   ValueType = 3
	TypeUint32  ValueType = 4
	TypeInt32   ValueType = 5
	TypeFloat32 ValueType = 6
	TypeBool    ValueType = 7
	TypeString  ValueType = 8
	TypeArray   ValueType = 9
	TypeUint64  ValueType = 10
	TypeInt64   ValueType = 11
	TypeFloat64 ValueType = 12
)

func (t ValueType) String() string {
	switch t {
	case TypeUint8:
		return "u8"
	case TypeInt8:
		return "i8"
	case TypeUint16:
		return "u16"
	case TypeInt16:
		return "i16"
	case TypeUint32:
		return "u32"
	case TypeInt32:
		return "i32"
	case TypeUint64:
		return "u64"
	case TypeInt64:
		return "i64"
	case TypeFloat32:
		return "f32"
	case TypeFloat64:
		return "f64"
	case TypeBool:
		return "bool"
	case TypeString:
		return "string"
	case TypeArray:
		return "array"
	default:
		return fmt.Sprintf("type(%d)", uint32(t))
	}
}

type ArrayValue struct {
	ElemType ValueType
	Values   []any
}

type Value struct {
	Type  ValueType
	Value any
}

type Header struct {
	Version     uint32
	TensorCount uint64
	KVCount     uint64
}

type TensorType uint32

const (
	GGMLTypeF32  TensorType = 0
	GGMLTypeF16  TensorType = 1
	GGMLTypeQ4_0 TensorType = 2
	GGMLTypeQ4_1 TensorType = 3
	GGMLTypeQ4_2 TensorType = 4
	GGMLTypeQ4_3 TensorType = 5
	GGMLTypeQ5_0 TensorType = 6
	GGMLTypeQ5_1 TensorType = 7
	GGMLTypeQ8_0 TensorType = 8
	GGMLTypeQ8_1 TensorType = 9
	GGMLTypeQ2_K TensorType = 10
	GGMLTypeQ3_K TensorType = 11
	GGMLTypeQ4_K TensorType = 12
	GGMLTypeQ5_K TensorType = 13
	GGMLTypeQ6_K TensorType = 14
	GGMLTypeQ8_K TensorType = 15
	GGMLTypeI8   TensorType = 16
	GGMLTypeI16  TensorType = 17
	GGMLTypeI32  TensorType = 18
	GGMLTypeI64  TensorType = 19
	GGMLTypeF64  TensorType = 20
)

func (t TensorType) String() string {
	switch t {
	case GGMLTypeF32:
		return "F32"
	case GGMLTypeF16:
		return "F16"
	case GGMLTypeQ4_0:
		return "Q4_0"
	case GGMLTypeQ4_1:
		return "Q4_1"
	case GGMLTypeQ4_2:
		return "Q4_2"
	case GGMLTypeQ4_3:
		return "Q4_3"
	case GGMLTypeQ5_0:
		return "Q5_0"
	case GGMLTypeQ5_1:
		return "Q5_1"
	case GGMLTypeQ8_0:
		return "Q8_0"
	case GGMLTypeQ8_1:
		return "Q8_1"
	case GGMLTypeQ2_K:
		return "Q2_K"
	case GGMLTypeQ3_K:
		return "Q3_K"
	case GGMLTypeQ4_K:
		return "Q4_K"
	case GGMLTypeQ5_K:
		return "Q5_K"
	case GGMLTypeQ6_K:
		return "Q6_K"
	case GGMLTypeQ8_K:
		return "Q8_K"
	case GGMLTypeI8:
		return "I8"
	case GGMLTypeI16:
		return "I16"
	case GGMLTypeI32:
		return "I32"
	case GGMLTypeI64:
		return "I64"
	case GGMLTypeF64:
		return "F64"
	default:
		return fmt.Sprintf("type(%d)", uint32(t))
	}
}

type TensorInfo struct {
	Name   string
	NDim   uint32
	Dims   []uint64
	Type   TensorType
	Offset uint64
}

type File struct {
	Path       string
	Header     Header
	KV         map[string]Value
	Tensors    []TensorInfo
	Alignment  uint64
	DataOffset uint64
	Data       []byte // mmap data
}

func Open(path string) (*File, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}

	st, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, err
	}
	size := st.Size()

	// Try mmap - need platform validation.
	var data []byte
	if size > 0 {
		b, _ := syscall.Mmap(int(f.Fd()), 0, int(size), syscall.PROT_READ, syscall.MAP_SHARED)
		data = b
	}

	var r *reader

	if data != nil {
		f.Close()
		r = newReader(bytes.NewReader(data), size)
	} else {
		r = newReader(f, size)
	}

	// Helper to ensure proper cleanup on error
	cleanup := func() {
		if data != nil {
			syscall.Munmap(data)
		} else {
			f.Close()
		}
	}

	magic, err := r.readN(4)
	if err != nil {
		cleanup()
		return nil, err
	}
	if string(magic) != magicGGUF {
		cleanup()
		return nil, fmt.Errorf("invalid magic: %q", string(magic))
	}

	version, err := r.readU32()
	if err != nil {
		cleanup()
		return nil, err
	}
	tensorCount, err := r.readU64()
	if err != nil {
		cleanup()
		return nil, err
	}
	kvCount, err := r.readU64()
	if err != nil {
		cleanup()
		return nil, err
	}

	kv := make(map[string]Value, kvCount)
	for i := range kvCount {
		key, err := r.readString()
		if err != nil {
			cleanup()
			return nil, fmt.Errorf("read key %d: %w", i, err)
		}
		vtypeU32, err := r.readU32()
		if err != nil {
			cleanup()
			return nil, fmt.Errorf("read value type for %s: %w", key, err)
		}
		vtype := ValueType(vtypeU32)
		val, err := readValue(r, vtype)
		if err != nil {
			cleanup()
			return nil, fmt.Errorf("read value for %s: %w", key, err)
		}
		kv[key] = Value{Type: vtype, Value: val}
	}

	tensors := make([]TensorInfo, 0, tensorCount)
	for i := range tensorCount {
		name, err := r.readString()
		if err != nil {
			cleanup()
			return nil, fmt.Errorf("read tensor name %d: %w", i, err)
		}
		nDim, err := r.readU32()
		if err != nil {
			cleanup()
			return nil, fmt.Errorf("read tensor dims %s: %w", name, err)
		}
		dims := make([]uint64, nDim)
		for d := range nDim {
			v, err := r.readU64()
			if err != nil {
				cleanup()
				return nil, fmt.Errorf("read tensor dim %s[%d]: %w", name, d, err)
			}
			dims[d] = v
		}
		ttypeU32, err := r.readU32()
		if err != nil {
			cleanup()
			return nil, fmt.Errorf("read tensor type %s: %w", name, err)
		}
		offset, err := r.readU64()
		if err != nil {
			cleanup()
			return nil, fmt.Errorf("read tensor offset %s: %w", name, err)
		}
		tensors = append(tensors, TensorInfo{
			Name:   name,
			NDim:   nDim,
			Dims:   dims,
			Type:   TensorType(ttypeU32),
			Offset: offset,
		})
	}

	if data == nil {
		f.Close()
	}

	alignment := uint64(32)
	if v, ok := kv["general.alignment"]; ok {
		if u, ok := asUint64(v.Value); ok && u > 0 {
			alignment = u
		}
	}

	dataOffset := align(uint64(r.off), alignment)

	return &File{
		Path:       path,
		Header:     Header{Version: version, TensorCount: tensorCount, KVCount: kvCount},
		KV:         kv,
		Tensors:    tensors,
		Alignment:  alignment,
		DataOffset: dataOffset,
		Data:       data,
	}, nil
}

func (f *File) Close() error {
	if f.Data != nil {
		return syscall.Munmap(f.Data)
	}
	return nil
}

func readValue(r *reader, vtype ValueType) (any, error) {
	switch vtype {
	case TypeUint8:
		return r.readU8()
	case TypeInt8:
		return r.readI8()
	case TypeUint16:
		return r.readU16()
	case TypeInt16:
		return r.readI16()
	case TypeUint32:
		return r.readU32()
	case TypeInt32:
		return r.readI32()
	case TypeUint64:
		return r.readU64()
	case TypeInt64:
		return r.readI64()
	case TypeFloat32:
		return r.readF32()
	case TypeFloat64:
		return r.readF64()
	case TypeBool:
		v, err := r.readU8()
		if err != nil {
			return false, err
		}
		return v != 0, nil
	case TypeString:
		return r.readString()
	case TypeArray:
		elemTypeU32, err := r.readU32()
		if err != nil {
			return nil, err
		}
		elemType := ValueType(elemTypeU32)
		count, err := r.readU64()
		if err != nil {
			return nil, err
		}
		values := make([]any, 0, count)
		for range count {
			v, err := readValue(r, elemType)
			if err != nil {
				return nil, err
			}
			values = append(values, v)
		}
		return ArrayValue{ElemType: elemType, Values: values}, nil
	default:
		return nil, fmt.Errorf("unsupported value type %d", uint32(vtype))
	}
}

func align(offset, alignment uint64) uint64 {
	if alignment == 0 {
		return offset
	}
	rem := offset % alignment
	if rem == 0 {
		return offset
	}
	return offset + (alignment - rem)
}

func asUint64(v any) (uint64, bool) {
	switch t := v.(type) {
	case uint8:
		return uint64(t), true
	case uint16:
		return uint64(t), true
	case uint32:
		return uint64(t), true
	case uint64:
		return t, true
	case int8:
		if t < 0 {
			return 0, false
		}
		return uint64(t), true
	case int16:
		if t < 0 {
			return 0, false
		}
		return uint64(t), true
	case int32:
		if t < 0 {
			return 0, false
		}
		return uint64(t), true
	case int64:
		if t < 0 {
			return 0, false
		}
		return uint64(t), true
	default:
		return 0, false
	}
}
