package mcfstore

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"

	"github.com/samcharles93/mantle/pkg/mcf"
)

var ErrTensorNotFound = errors.New("mcfstore: tensor not found")

type File struct {
	file     *mcf.File
	index    *mcf.TensorIndex
	dataSect *mcf.MCFSection
}

type TensorInfo struct {
	DType    mcf.TensorDType
	Shape    []int
	DataOff  uint64
	DataSize uint64
}

func Open(path string) (*File, error) {
	mf, err := mcf.Open(path)
	if err != nil {
		return nil, err
	}

	cleanup := func(err error) (*File, error) {
		_ = mf.Close()
		return nil, err
	}

	indexSec := mf.Section(mcf.SectionTensorIndex)
	if indexSec == nil {
		return cleanup(errors.New("mcf: missing tensor index section"))
	}
	indexData := mf.SectionData(indexSec)
	if len(indexData) == 0 {
		return cleanup(errors.New("mcf: empty tensor index section"))
	}
	index, err := mcf.ParseTensorIndexSection(indexData)
	if err != nil {
		return cleanup(err)
	}

	dataSec := mf.Section(mcf.SectionTensorData)
	if dataSec == nil {
		return cleanup(errors.New("mcf: missing tensor data section"))
	}

	return &File{file: mf, index: index, dataSect: dataSec}, nil
}

func (f *File) Close() error {
	if f == nil || f.file == nil {
		return nil
	}
	err := f.file.Close()
	f.file = nil
	f.index = nil
	f.dataSect = nil
	return err
}

func (f *File) SectionData(t mcf.SectionType) []byte {
	if f == nil || f.file == nil {
		return nil
	}
	sec := f.file.Section(t)
	return f.file.SectionData(sec)
}

func (f *File) Tensor(name string) (TensorInfo, error) {
	if f == nil || f.index == nil {
		return TensorInfo{}, ErrTensorNotFound
	}
	idx, ok := f.index.Find(name)
	if !ok {
		return TensorInfo{}, ErrTensorNotFound
	}
	entry, err := f.index.Entry(idx)
	if err != nil {
		return TensorInfo{}, err
	}
	shapeU64, err := f.index.Shape(idx)
	if err != nil {
		return TensorInfo{}, err
	}
	shape, err := shapeToInt(shapeU64)
	if err != nil {
		return TensorInfo{}, err
	}
	return TensorInfo{
		DType:    entry.DType,
		Shape:    shape,
		DataOff:  entry.DataOff,
		DataSize: entry.DataSize,
	}, nil
}

func (f *File) ReadTensorF32(name string) ([]float32, TensorInfo, error) {
	info, err := f.Tensor(name)
	if err != nil {
		return nil, TensorInfo{}, err
	}
	idx, ok := f.index.Find(name)
	if !ok {
		return nil, TensorInfo{}, ErrTensorNotFound
	}
	raw, err := f.index.TensorData(f.file, idx)
	if err != nil {
		return nil, TensorInfo{}, err
	}
	if err := f.validateTensorRange(info.DataOff, info.DataSize); err != nil {
		return nil, TensorInfo{}, err
	}

	n, err := numElements(info.Shape)
	if err != nil {
		return nil, TensorInfo{}, fmt.Errorf("tensor %s: %w", name, err)
	}

	switch info.DType {
	case mcf.DTypeF32:
		if len(raw) != n*4 {
			return nil, TensorInfo{}, fmt.Errorf("tensor %s: invalid f32 data size", name)
		}
		out := make([]float32, n)
		for i := 0; i < n; i++ {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		}
		return out, info, nil
	case mcf.DTypeBF16:
		if len(raw) != n*2 {
			return nil, TensorInfo{}, fmt.Errorf("tensor %s: invalid bf16 data size", name)
		}
		out := make([]float32, n)
		for i := 0; i < n; i++ {
			u := binary.LittleEndian.Uint16(raw[i*2:])
			out[i] = bf16ToF32(u)
		}
		return out, info, nil
	case mcf.DTypeF16:
		if len(raw) != n*2 {
			return nil, TensorInfo{}, fmt.Errorf("tensor %s: invalid f16 data size", name)
		}
		out := make([]float32, n)
		for i := 0; i < n; i++ {
			u := binary.LittleEndian.Uint16(raw[i*2:])
			out[i] = fp16ToF32(u)
		}
		return out, info, nil
	default:
		return nil, TensorInfo{}, fmt.Errorf("tensor %s: unsupported dtype %d", name, info.DType)
	}
}

func (f *File) validateTensorRange(off, size uint64) error {
	if f == nil || f.dataSect == nil {
		return errors.New("mcf: missing tensor data section")
	}
	end := off + size
	if end < off {
		return errors.New("mcf: tensor data offset overflow")
	}
	if off < f.dataSect.Offset || end > f.dataSect.Offset+f.dataSect.Size {
		return errors.New("mcf: tensor data out of bounds")
	}
	return nil
}

func shapeToInt(shape []uint64) ([]int, error) {
	if len(shape) == 0 {
		return nil, errors.New("empty shape")
	}
	out := make([]int, len(shape))
	for i, v := range shape {
		if v == 0 {
			return nil, errors.New("invalid dim 0")
		}
		if v > uint64(int(^uint(0)>>1)) {
			return nil, errors.New("dimension too large")
		}
		out[i] = int(v)
	}
	return out, nil
}

func numElements(shape []int) (int, error) {
	if len(shape) == 0 {
		return 0, errors.New("empty shape")
	}
	n := 1
	maxInt := int(^uint(0) >> 1)
	for _, d := range shape {
		if d <= 0 {
			return 0, fmt.Errorf("invalid dim %d", d)
		}
		if n > maxInt/d {
			return 0, errors.New("tensor too large")
		}
		n *= d
	}
	return n, nil
}

func bf16ToF32(u uint16) float32 {
	return math.Float32frombits(uint32(u) << 16)
}

func fp16ToF32(h uint16) float32 {
	sign := uint32(h>>15) & 0x1
	exp := uint32(h>>10) & 0x1F
	frac := uint32(h & 0x3FF)
	var f uint32
	switch exp {
	case 0:
		if frac == 0 {
			f = sign << 31
		} else {
			e := uint32(127 - 15 + 1)
			for (frac & 0x400) == 0 {
				frac <<= 1
				e--
			}
			frac &= 0x3FF
			f = (sign << 31) | (e << 23) | (frac << 13)
		}
	case 0x1F:
		f = (sign << 31) | 0x7F800000 | (frac << 13)
	default:
		e := exp + (127 - 15)
		f = (sign << 31) | (e << 23) | (frac << 13)
	}
	return math.Float32frombits(f)
}
