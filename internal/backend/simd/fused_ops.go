//go:build goexperiment.simd

package simd

import (
	"simd/archsimd"
)

// FusedSiluAct applies SiLU activation and stores result.
// Computes: dst = silu(gate) * up
func FusedSiluAct(dst, gate, up []float32) {
	if cpu.HasAVX2 {
		fusedSiluActAVX2(dst, gate, up)
		return
	}
	fusedSiluActScalar(dst, gate, up)
}

// fusedSiluActScalar applies SiLU activation using scalar operations.
func fusedSiluActScalar(dst, gate, up []float32) {
	for i := range gate {
		g := gate[i]
		silu := g / (1.0 + fastExp(-g))
		dst[i] = silu * up[i]
	}
}

// fusedSiluActAVX2 applies SiLU activation using AVX2 SIMD.
// Uses a piecewise rational approximation for sigmoid to avoid exp().
// SiLU(x) = x * sigmoid(x), approximated as:
//
//	sigmoid(x) ≈ 0.5 + 0.5 * x / (1 + |x|)
//
// This provides reasonable accuracy for inference while being fast.
func fusedSiluActAVX2(dst, gate, up []float32) {
	i := 0
	vone := archsimd.BroadcastFloat32x8(1.0)
	vhalf := archsimd.BroadcastFloat32x8(0.5)
	// Sign bit mask for float32: 0x7FFFFFFF clears sign bit for abs
	signMask := archsimd.BroadcastInt32x8(0x7FFFFFFF)

	for ; i+8 <= len(gate); i += 8 {
		vgate := archsimd.LoadFloat32x8Slice(gate[i:])
		vup := archsimd.LoadFloat32x8Slice(up[i:])

		// Compute |x| by clearing sign bit
		vgateInt := vgate.AsInt32x8()
		vabs := vgateInt.And(signMask).AsFloat32x8()

		// sigmoid ≈ 0.5 + 0.5 * x / (1 + |x|)
		vdenom := vone.Add(vabs)
		vfrac := vgate.Div(vdenom)
		vsig := vhalf.Add(vhalf.Mul(vfrac))

		// SiLU = gate * sigmoid * up
		vsilu := vgate.Mul(vsig).Mul(vup)
		vsilu.StoreSlice(dst[i:])
	}

	// Handle remaining elements with same approximation
	for ; i < len(gate); i++ {
		g := gate[i]
		absG := g
		if absG < 0 {
			absG = -absG
		}
		// sigmoid ≈ 0.5 + 0.5 * x / (1 + |x|)
		sig := 0.5 + 0.5*g/(1.0+absG)
		dst[i] = g * sig * up[i]
	}
}

// FusedGeluAct applies GELU activation and stores result.
// Computes: dst = gelu(gate) * up
func FusedGeluAct(dst, gate, up []float32) {
	if cpu.HasAVX2 {
		fusedGeluActAVX2(dst, gate, up)
		return
	}
	for i := range gate {
		dst[i] = Gelu(gate[i]) * up[i]
	}
}

func fusedGeluActAVX2(dst, gate, up []float32) {
	i := 0
	for ; i+8 <= len(gate); i += 8 {
		vgate := archsimd.LoadFloat32x8Slice(gate[i:])
		vup := archsimd.LoadFloat32x8Slice(up[i:])
		vact := fastGeluVec(vgate).Mul(vup)
		vact.StoreSlice(dst[i:])
	}
	for ; i < len(gate); i++ {
		dst[i] = Gelu(gate[i]) * up[i]
	}
}
