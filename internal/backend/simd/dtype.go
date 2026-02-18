package simd

import (
	"math"
	"unsafe"

	"github.com/samcharles93/mantle/pkg/mcf"
)

var nativeLittleEndian = func() bool {
	var x uint16 = 1
	b := (*[2]byte)(unsafe.Pointer(&x))
	return b[0] == 1
}()

// bf16Table maps every possible BF16 bit-pattern to float32.
var bf16Table = func() [1 << 16]float32 {
	var tbl [1 << 16]float32
	for i := range tbl {
		tbl[i] = math.Float32frombits(uint32(i) << 16)
	}
	return tbl
}()

// fp16Table maps every possible FP16 bit-pattern to float32.
var fp16Table = func() [1 << 16]float32 {
	var tbl [1 << 16]float32
	for i := range tbl {
		tbl[i] = fp16ToF32(uint16(i))
	}
	return tbl
}()

func dtypeElemSize(dt mcf.TensorDType) (int, bool) {
	switch dt {
	case mcf.DTypeF32:
		return 4, true
	case mcf.DTypeF16, mcf.DTypeBF16:
		return 2, true
	default:
		return 0, false
	}
}

func u16le(b []byte, off int) uint16 {
	_ = b[off+1]
	return uint16(b[off]) | uint16(b[off+1])<<8
}

// rawUint16LE provides a fast unsafe view when the host is little-endian and
// the backing storage is suitably aligned. Callers must still bounds-check.
func rawUint16LE(raw []byte) ([]uint16, bool) {
	if !nativeLittleEndian || len(raw) == 0 || len(raw)%2 != 0 {
		return nil, false
	}
	if uintptr(unsafe.Pointer(&raw[0]))%2 != 0 {
		return nil, false
	}
	return unsafe.Slice((*uint16)(unsafe.Pointer(&raw[0])), len(raw)/2), true
}

func bf16ToF32(u uint16) float32 {
	return math.Float32frombits(uint32(u) << 16)
}

func bf16ToF32Table(u uint16) float32 {
	return bf16Table[u]
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

func fp16ToF32Table(u uint16) float32 {
	return fp16Table[u]
}
