package gguf

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"unsafe"
)

// TensorByName returns the tensor info for the given name.
func (f *File) TensorByName(name string) (TensorInfo, bool) {
	for _, t := range f.Tensors {
		if t.Name == name {
			return t, true
		}
	}
	return TensorInfo{}, false
}

// ReadTensorF32 loads a tensor by name and returns its data as float32 along with its dims.
// Supported types: F32, Q4_K, Q6_K.
func ReadTensorF32(f *File, name string) ([]float32, []uint64, error) {
	info, ok := f.TensorByName(name)
	if !ok {
		return nil, nil, fmt.Errorf("tensor not found: %s", name)
	}
	n, err := tensorElements(info.Dims)
	if err != nil {
		return nil, nil, fmt.Errorf("tensor %s: %w", name, err)
	}
	byteSize, err := tensorByteSize(info.Type, n)
	if err != nil {
		return nil, nil, fmt.Errorf("tensor %s: %w", name, err)
	}

	var buf []byte
	off := int64(f.DataOffset + info.Offset)

	if f.Data != nil {
		if int64(len(f.Data)) < off+int64(byteSize) {
			return nil, nil, fmt.Errorf("tensor %s: unexpected EOF (mmap)", name)
		}
		buf = f.Data[off : off+int64(byteSize)]
	} else {
		buf = make([]byte, byteSize)
		file, err := os.Open(f.Path)
		if err != nil {
			return nil, nil, err
		}
		defer func() { _ = file.Close() }()

		if _, err := file.ReadAt(buf, off); err != nil {
			return nil, nil, fmt.Errorf("read tensor %s: %w", name, err)
		}
	}

	switch info.Type {
	case GGMLTypeF16:
		out := make([]float32, n)
		for i := range n {
			out[i] = fp16ToFloat32(buf[i*2 : i*2+2])
		}
		return out, info.Dims, nil
	case GGMLTypeF32:
		out := make([]float32, n)
		if len(buf) >= n*4 {
			src := unsafe.Slice((*float32)(unsafe.Pointer(&buf[0])), n)
			copy(out, src)
		} else {
			for i := range n {
				b := buf[i*4 : i*4+4]
				out[i] = float32FromBytes(b)
			}
		}
		return out, info.Dims, nil
	case GGMLTypeQ4_K:
		out, err := DequantizeQ4K(buf, n)
		if err != nil {
			return nil, nil, err
		}
		return out, info.Dims, nil
	case GGMLTypeQ6_K:
		out, err := DequantizeQ6K(buf, n)
		if err != nil {
			return nil, nil, err
		}
		return out, info.Dims, nil
	default:
		return nil, nil, ErrUnsupportedType
	}
}

// ReadTensorRaw loads a tensor by name and returns its raw bytes, dims, and type.
func ReadTensorRaw(f *File, name string) ([]byte, []uint64, TensorType, error) {
	info, ok := f.TensorByName(name)
	if !ok {
		return nil, nil, 0, fmt.Errorf("tensor not found: %s", name)
	}
	n, err := tensorElements(info.Dims)
	if err != nil {
		return nil, nil, 0, fmt.Errorf("tensor %s: %w", name, err)
	}
	byteSize, err := tensorByteSize(info.Type, n)
	if err != nil {
		return nil, nil, 0, fmt.Errorf("tensor %s: %w", name, err)
	}

	off := int64(f.DataOffset + info.Offset)
	if f.Data != nil {
		if int64(len(f.Data)) < off+int64(byteSize) {
			return nil, nil, 0, fmt.Errorf("tensor %s: unexpected EOF (mmap)", name)
		}
		// Return a copy or slice? The original allocated a buffer.
		// If we slice mmap, we are safe as long as mmap is valid.
		// However, callers might modify the buffer? Usually no.
		// But let's return a slice to be zero-copy efficient.
		return f.Data[off : off+int64(byteSize)], info.Dims, info.Type, nil
	}

	buf := make([]byte, byteSize)
	file, err := os.Open(f.Path)
	if err != nil {
		return nil, nil, 0, err
	}
	defer func() { _ = file.Close() }()

	if _, err := file.ReadAt(buf, off); err != nil {
		return nil, nil, 0, fmt.Errorf("read tensor %s: %w", name, err)
	}
	return buf, info.Dims, info.Type, nil
}

func tensorElements(dims []uint64) (int, error) {
	if len(dims) == 0 {
		return 0, fmt.Errorf("empty dims")
	}
	var n uint64 = 1
	for _, d := range dims {
		if d == 0 {
			return 0, fmt.Errorf("zero dimension")
		}
		n *= d
	}
	if n > uint64(^uint(0)>>1) {
		return 0, fmt.Errorf("tensor too large")
	}
	return int(n), nil
}

func tensorByteSize(t TensorType, n int) (int, error) {
	switch t {
	case GGMLTypeF16:
		return n * 2, nil
	case GGMLTypeF32:
		return n * 4, nil
	case GGMLTypeQ4_K:
		if n%QK_K != 0 {
			return 0, fmt.Errorf("q4_k: n must be multiple of %d", QK_K)
		}
		return (n / QK_K) * q4kBlockSize, nil
	case GGMLTypeQ6_K:
		if n%QK_K != 0 {
			return 0, fmt.Errorf("q6_k: n must be multiple of %d", QK_K)
		}
		return (n / QK_K) * q6kBlockSize, nil
	default:
		return 0, ErrUnsupportedType
	}
}

func float32FromBytes(b []byte) float32 {
	u := binary.LittleEndian.Uint32(b)
	return math.Float32frombits(u)
}
