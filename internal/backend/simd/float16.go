package simd

import "math"

// Float32ToFloat16 converts a float32 to a float16 (uint16)
// This is a simplified implementation.
func Float32ToFloat16(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := uint16((bits >> 31) & 0x1)
	exp := int16((bits >> 23) & 0xFF)
	mant := bits & 0x7FFFFF

	var outExp uint16
	var outMant uint16

	switch exp {
	case 0:
		// Subnormal or zero
		// We treat subnormals as zero for speed/simplicity in this context if acceptable,
		// or implement proper handling. Llama.cpp usually flushes subnormals.
		// For zero:
		if mant == 0 {
			return sign << 15
		}
		// Denormal float32 to... zero? Let's just return 0 for now to keep it simple and safe.
		// Proper conversion is complex. By standard:
		return sign << 15
	case 0xFF:
		// Inf or NaN
		outExp = 0x1F
		if mant != 0 {
			outMant = 0x200 // Some non-zero mantissa
		}
	default:
		// Normalized
		newExp := exp - 127 + 15
		if newExp >= 31 {
			// Overflow to Inf
			outExp = 0x1F
		} else if newExp <= 0 {
			// Underflow to zero (simplification)
			outExp = 0
		} else {
			outExp = uint16(newExp)
			outMant = uint16(mant >> 13)
		}
	}

	return (sign << 15) | (outExp << 10) | outMant
}

// Float16ToFloat32 converts a float16 (uint16) to float32
func Float16ToFloat32(h uint16) float32 {
	sign := uint32((h >> 15) & 0x1)
	exp := uint32((h >> 10) & 0x1F)
	mant := uint32(h & 0x3FF)

	var outBits uint32

	switch exp {
	case 0:
		// Subnormal or zero -> zero
		outBits = sign << 31
	case 0x1F:
		// Inf or NaN
		outBits = (sign << 31) | 0x7F800000
		if mant != 0 {
			outBits |= (mant << 13) // Map NaN payload
		}
	default:
		// Normalized
		newExp := exp + 127 - 15
		newMant := mant << 13
		outBits = (sign << 31) | (newExp << 23) | newMant
	}

	return math.Float32frombits(outBits)
}

func Float32ToFloat16Slice(src []float32, dst []uint16) {
	for i, v := range src {
		dst[i] = Float32ToFloat16(v)
	}
}

func Float16ToFloat32Slice(src []uint16, dst []float32) {
	for i, v := range src {
		dst[i] = Float16ToFloat32(v)
	}
}
