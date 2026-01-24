package gguf

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"math"
)

type reader struct {
	r    *bufio.Reader
	off  int64
	size int64
}

func newReader(rd io.Reader, size int64) *reader {
	return &reader{
		r:    bufio.NewReader(rd),
		size: size,
	}
}

func (r *reader) readN(n int) ([]byte, error) {
	if n < 0 {
		return nil, fmt.Errorf("invalid read length %d", n)
	}
	if r.size > 0 && r.off+int64(n) > r.size {
		return nil, io.ErrUnexpectedEOF
	}
	buf := make([]byte, n)
	if _, err := io.ReadFull(r.r, buf); err != nil {
		return nil, err
	}
	r.off += int64(n)
	return buf, nil
}

func (r *reader) readU8() (uint8, error) {
	b, err := r.readN(1)
	if err != nil {
		return 0, err
	}
	return b[0], nil
}

func (r *reader) readI8() (int8, error) {
	v, err := r.readU8()
	return int8(v), err
}

func (r *reader) readU16() (uint16, error) {
	b, err := r.readN(2)
	if err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint16(b), nil
}

func (r *reader) readI16() (int16, error) {
	v, err := r.readU16()
	return int16(v), err
}

func (r *reader) readU32() (uint32, error) {
	b, err := r.readN(4)
	if err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint32(b), nil
}

func (r *reader) readI32() (int32, error) {
	v, err := r.readU32()
	return int32(v), err
}

func (r *reader) readU64() (uint64, error) {
	b, err := r.readN(8)
	if err != nil {
		return 0, err
	}
	return binary.LittleEndian.Uint64(b), nil
}

func (r *reader) readI64() (int64, error) {
	v, err := r.readU64()
	return int64(v), err
}

func (r *reader) readF32() (float32, error) {
	u, err := r.readU32()
	if err != nil {
		return 0, err
	}
	return math.Float32frombits(u), nil
}

func (r *reader) readF64() (float64, error) {
	u, err := r.readU64()
	if err != nil {
		return 0, err
	}
	return math.Float64frombits(u), nil
}

func (r *reader) readString() (string, error) {
	n, err := r.readU64()
	if err != nil {
		return "", err
	}
	if n == 0 {
		return "", nil
	}
	if n > uint64(r.size) {
		return "", fmt.Errorf("string length too large: %d", n)
	}
	b, err := r.readN(int(n))
	if err != nil {
		return "", err
	}
	return string(b), nil
}
