package simd

import (
	"simd/archsimd"

	instance "github.com/samcharles93/mantle/internal/backend/core"
)

// f16Consts holds the masks f16x8ToF32 needs. archsimd broadcasts are
// emulated, so callers build them once outside their loops.
type f16Consts struct {
	expMant, exp, sign, bias, zero archsimd.Uint32x8
}

func newF16Consts() f16Consts {
	return f16Consts{
		expMant: archsimd.BroadcastUint32x8(0x7fff),
		exp:     archsimd.BroadcastUint32x8(0x7c00),
		sign:    archsimd.BroadcastUint32x8(0x8000),
		bias:    archsimd.BroadcastUint32x8(112 << 23), // (127-15) << 23
		zero:    archsimd.BroadcastUint32x8(0),
	}
}

// f16x8ToF32 widens eight IEEE half values to float32 with exactly the results
// of instance.Float16ToFloat32: normals rebias, Inf and NaN keep their
// mantissa, and subnormals flush to a signed zero.
func f16x8ToF32(h *[8]uint16, k *f16Consts) archsimd.Float32x8 {
	u := archsimd.LoadUint16x8Array(h).ExtendToUint32()
	em := u.And(k.expMant)
	e := em.And(k.exp)
	bits := em.ShiftAllLeft(13).Add(k.bias)
	// Inf/NaN: a second bias moves exponent 31+112 up to 255.
	bits = bits.Add(k.bias.Masked(e.Equal(k.exp)))
	bits = bits.Masked(e.NotEqual(k.zero))
	return bits.Or(u.And(k.sign).ShiftAllLeft(16)).BitsToFloat32()
}

// dotF16SIMD is DotF16 for AVX2 hosts. Like dotSIMD it keeps vector partial
// sums, so its rounding differs from the scalar loop.
func dotF16SIMD(a []float32, b []uint16) float32 {
	n := len(a)
	if n > len(b) {
		panic("dotF16: length mismatch")
	}
	k := newF16Consts()
	var acc0, acc1 archsimd.Float32x8
	i := 0
	for ; i+16 <= n; i += 16 {
		acc0 = archsimd.LoadFloat32x8Array((*[8]float32)(a[i:i+8])).MulAdd(f16x8ToF32((*[8]uint16)(b[i:i+8]), &k), acc0)
		acc1 = archsimd.LoadFloat32x8Array((*[8]float32)(a[i+8:i+16])).MulAdd(f16x8ToF32((*[8]uint16)(b[i+8:i+16]), &k), acc1)
	}
	for ; i+8 <= n; i += 8 {
		acc0 = archsimd.LoadFloat32x8Array((*[8]float32)(a[i:i+8])).MulAdd(f16x8ToF32((*[8]uint16)(b[i:i+8]), &k), acc0)
	}
	sum := hsum8(acc0.Add(acc1))
	for ; i < n; i++ {
		sum += a[i] * instance.Float16ToFloat32(b[i])
	}
	return sum
}

// accumulateScaledF16 performs out[d] = out[d] + s*v[d] for every d, with a
// separate multiply and add per element, so a caller looping over positions
// reproduces the scalar sum += s*v order bit for bit. len(out) must be a
// multiple of 8 and v at least as long.
func accumulateScaledF16(out []float32, s float32, v []uint16, k *f16Consts) {
	sv := archsimd.BroadcastFloat32x8(s)
	for d := 0; d+8 <= len(out); d += 8 {
		o := (*[8]float32)(out[d : d+8])
		archsimd.LoadFloat32x8Array(o).Add(sv.Mul(f16x8ToF32((*[8]uint16)(v[d:d+8]), k))).StoreArray(o)
	}
}

// accumulateScaledF32 is accumulateScaledF16 for an f32 cache row.
func accumulateScaledF32(out []float32, s float32, v []float32) {
	sv := archsimd.BroadcastFloat32x8(s)
	for d := 0; d+8 <= len(out); d += 8 {
		o := (*[8]float32)(out[d : d+8])
		archsimd.LoadFloat32x8Array(o).Add(sv.Mul(archsimd.LoadFloat32x8Array((*[8]float32)(v[d : d+8])))).StoreArray(o)
	}
}
