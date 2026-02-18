//go:build goexperiment.simd

package simd

import (
	"math"

	"simd/archsimd"
)

// FusedRMSNormMatVecCPU performs RMSNorm followed by MatVec in a single optimized pass (CPU-only).
// This is useful for output head computation where we normalize then project.
func FusedRMSNormMatVecCPU(dst []float32, w *Mat, x, weight []float32, eps float32) {
	if cpu.HasAVX2 {
		fusedRMSNormMatVecAVX2(dst, w, x, weight, eps)
		return
	}
	fusedRMSNormMatVecScalar(dst, w, x, weight, eps)
}

// fusedRMSNormMatVecScalar performs RMSNorm + MatVec using scalar operations.
func fusedRMSNormMatVecScalar(dst []float32, w *Mat, x, weight []float32, eps float32) {
	// Compute RMS
	var sum float32
	for _, v := range x {
		sum += v * v
	}
	rms := float32(1.0) / float32(math.Sqrt(float64(sum/float32(len(x))+eps)))

	// Normalize and multiply by weight, then matvec
	normalized := make([]float32, len(x))
	for i := range x {
		normalized[i] = x[i] * rms * weight[i]
	}

	MatVec(dst, w, normalized)
}

// fusedRMSNormMatVecAVX2 performs fused RMSNorm + MatVec using AVX2.
func fusedRMSNormMatVecAVX2(dst []float32, w *Mat, x, weight []float32, eps float32) {
	// Compute RMS using SIMD
	var sum archsimd.Float32x8
	i := 0
	for ; i+8 <= len(x); i += 8 {
		vx := archsimd.LoadFloat32x8Slice(x[i:])
		sum = vx.Mul(vx).Add(sum)
	}

	var sumScalar float32
	var tmp [8]float32
	sum.Store(&tmp)
	for j := range 8 {
		sumScalar += tmp[j]
	}

	for ; i < len(x); i++ {
		sumScalar += x[i] * x[i]
	}

	rms := float32(1.0) / float32(math.Sqrt(float64(sumScalar/float32(len(x))+eps)))
	vrms := archsimd.BroadcastFloat32x8(rms)

	// Normalize and matvec in fused manner
	for row := 0; row < w.R; row++ {
		wRow := w.Data[row*w.Stride : row*w.Stride+w.C]
		var acc archsimd.Float32x8
		j := 0
		for ; j+8 <= w.C; j += 8 {
			vx := archsimd.LoadFloat32x8Slice(x[j:])
			vw := archsimd.LoadFloat32x8Slice(weight[j:])
			vwRow := archsimd.LoadFloat32x8Slice(wRow[j:])

			// Normalize: x * rms * weight
			vNorm := vx.Mul(vrms).Mul(vw)
			acc = vwRow.MulAdd(vNorm, acc)
		}

		var sumRow float32
		acc.Store(&tmp)
		for k := range 8 {
			sumRow += tmp[k]
		}

		for ; j < w.C; j++ {
			sumRow += wRow[j] * (x[j] * rms * weight[j])
		}
		dst[row] = sumRow
	}
}

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
