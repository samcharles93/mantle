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

func bf16FromF32Bits(u uint32) uint16 {
	// Round-to-nearest-even on the truncated 16 bits.
	rnd := uint32(0x7FFF + ((u >> 16) & 1))
	return uint16((u + rnd) >> 16)
}

// float32ToFP16Bits implements IEEE 754 binary16 rounding (nearest-even).
func float32ToFP16Bits(f float32) uint16 {
	u := math.Float32bits(f)
	sign := (u >> 31) & 0x1
	exp := int((u >> 23) & 0xFF)
	frac := u & 0x7FFFFF

	if exp == 0xFF {
		// Inf/NaN
		if frac != 0 {
			return uint16((sign << 15) | 0x7C00 | (frac >> 13) | 1)
		}
		return uint16((sign << 15) | 0x7C00)
	}

	// unbiased exponent
	e := exp - 127
	if e > 15 {
		// overflow -> inf
		return uint16((sign << 15) | 0x7C00)
	}
	if e < -14 {
		// subnormal or zero
		if e < -24 {
			return uint16(sign << 15)
		}
		// add implicit leading 1 then shift into subnormal range
		frac |= 0x800000
		shift := uint32(-14 - e)
		// round-to-nearest-even
		rnd := uint32(1<<(shift-1)) - 1 + ((frac >> shift) & 1)
		frac = (frac + rnd) >> shift
		return uint16((sign << 15) | (frac >> 13))
	}

	// normal
	exp16 := uint32(e + 15)
	// round-to-nearest-even on frac>>13
	rnd := uint32(0xFFF + ((frac >> 13) & 1))
	frac = frac + rnd
	if (frac & 0x800000) != 0 {
		// carry into exponent
		exp16++
		frac = 0
		if exp16 >= 0x1F {
			return uint16((sign << 15) | 0x7C00)
		}
	}
	return uint16((sign << 15) | (exp16 << 10) | (frac >> 13))
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

func encodeBF16Raw(data []float32) []byte {
	raw := make([]byte, len(data)*2)
	for i, v := range data {
		u := bf16FromF32Bits(math.Float32bits(v))
		raw[i*2] = byte(u)
		raw[i*2+1] = byte(u >> 8)
	}
	return raw
}

func encodeFP16Raw(data []float32) []byte {
	raw := make([]byte, len(data)*2)
	for i, v := range data {
		u := float32ToFP16Bits(v)
		raw[i*2] = byte(u)
		raw[i*2+1] = byte(u >> 8)
	}
	return raw
}
