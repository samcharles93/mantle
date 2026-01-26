package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
)

type TensorInfo struct {
	DType string
	Shape []int
	Start int64
	End   int64
}

type File struct {
	Path      string
	DataStart int64
	Tensors   map[string]TensorInfo
}

type tensorHeader struct {
	DType       string  `json:"dtype"`
	Shape       []int   `json:"shape"`
	DataOffsets []int64 `json:"data_offsets"`
}

func Open(path string) (*File, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() { _ = f.Close() }()

	headerLen, err := readU64(f)
	if err != nil {
		return nil, err
	}
	headerBytes := make([]byte, headerLen)
	if _, err := io.ReadFull(f, headerBytes); err != nil {
		return nil, err
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(headerBytes, &raw); err != nil {
		return nil, err
	}
	delete(raw, "__metadata__")

	tensors := make(map[string]TensorInfo, len(raw))
	for name, msg := range raw {
		var th tensorHeader
		if err := json.Unmarshal(msg, &th); err != nil {
			return nil, fmt.Errorf("parse tensor %s: %w", name, err)
		}
		if len(th.DataOffsets) != 2 {
			return nil, fmt.Errorf("tensor %s: invalid data_offsets", name)
		}
		tensors[name] = TensorInfo{
			DType: th.DType,
			Shape: th.Shape,
			Start: th.DataOffsets[0],
			End:   th.DataOffsets[1],
		}
	}
	return &File{
		Path:      path,
		DataStart: int64(8 + headerLen),
		Tensors:   tensors,
	}, nil
}

func (f *File) Tensor(name string) (TensorInfo, bool) {
	t, ok := f.Tensors[name]
	return t, ok
}

func (f *File) ReadTensor(name string) ([]byte, TensorInfo, error) {
	t, ok := f.Tensors[name]
	if !ok {
		return nil, TensorInfo{}, fmt.Errorf("tensor not found: %s", name)
	}
	if t.End < t.Start {
		return nil, TensorInfo{}, fmt.Errorf("tensor %s: invalid offsets", name)
	}
	n := t.End - t.Start
	buf := make([]byte, n)

	file, err := os.Open(f.Path)
	if err != nil {
		return nil, TensorInfo{}, err
	}
	defer func() { _ = file.Close() }()

	off := f.DataStart + t.Start
	if _, err := file.ReadAt(buf, off); err != nil {
		return nil, TensorInfo{}, fmt.Errorf("read tensor %s: %w", name, err)
	}
	return buf, t, nil
}

func (f *File) ReadTensorF32(name string) ([]float32, TensorInfo, error) {
	raw, info, err := f.ReadTensor(name)
	if err != nil {
		return nil, TensorInfo{}, err
	}
	n, err := numElements(info.Shape)
	if err != nil {
		return nil, TensorInfo{}, fmt.Errorf("tensor %s: %w", name, err)
	}
	switch info.DType {
	case "F32":
		if len(raw) != n*4 {
			return nil, TensorInfo{}, fmt.Errorf("tensor %s: invalid f32 data size", name)
		}
		out := make([]float32, n)
		for i := 0; i < n; i++ {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		}
		return out, info, nil
	case "BF16":
		if len(raw) != n*2 {
			return nil, TensorInfo{}, fmt.Errorf("tensor %s: invalid bf16 data size", name)
		}
		out := make([]float32, n)
		for i := 0; i < n; i++ {
			u := binary.LittleEndian.Uint16(raw[i*2:])
			out[i] = bf16ToF32(u)
		}
		return out, info, nil
	case "F16":
		if len(raw) != n*2 {
			return nil, TensorInfo{}, fmt.Errorf("tensor %s: invalid f16 data size", name)
		}
		out := make([]float32, n)
		for i := 0; i < n; i++ {
			u := binary.LittleEndian.Uint16(raw[i*2:])
			out[i] = fp16ToFloat32(u)
		}
		return out, info, nil
	default:
		return nil, TensorInfo{}, fmt.Errorf("unsupported dtype %s", info.DType)
	}
}

func numElements(shape []int) (int, error) {
	if len(shape) == 0 {
		return 0, fmt.Errorf("empty shape")
	}
	n := 1
	for _, d := range shape {
		if d <= 0 {
			return 0, fmt.Errorf("invalid dim %d", d)
		}
		if n > (int(^uint(0)>>1))/d {
			return 0, fmt.Errorf("tensor too large")
		}
		n *= d
	}
	return n, nil
}

func readU64(r io.Reader) (uint64, error) {
	var buf [8]byte
	if _, err := io.ReadFull(r, buf[:]); err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint64(buf[:]), nil
}

func bf16ToF32(u uint16) float32 {
	return math.Float32frombits(uint32(u) << 16)
}

func fp16ToFloat32(h uint16) float32 {
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
