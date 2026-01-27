package mcf

import (
	"math"
	"os"
	"unsafe"
)

const (
	mcfAlign = 8
)

func rangesOverlap(a0, a1, b0, b1 uint64) bool {
	// half-open ranges [a0,a1) and [b0,b1)
	return a0 < b1 && b0 < a1
}

func structBytes[T any](p *T) []byte {
	n := int(unsafe.Sizeof(*p))
	return unsafe.Slice((*byte)(unsafe.Pointer(p)), n)
}

func structSliceBytes[T any](s []T) []byte {
	if len(s) == 0 {
		return nil
	}
	elem := int(unsafe.Sizeof(s[0]))
	total := elem * len(s)
	if total < 0 || total > math.MaxInt {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&s[0])), total)
}

func writeFull(f *os.File, p []byte) error {
	for len(p) > 0 {
		n, err := f.Write(p)
		if err != nil {
			return err
		}
		p = p[n:]
	}
	return nil
}
