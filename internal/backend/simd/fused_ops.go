//go:build goexperiment.simd

package simd

import (
	"math"

	"github.com/samcharles93/mantle/pkg/mcf"
	"simd/archsimd"
)

// FusedMatVecBias computes dst = W * x + bias in a single pass.
// This avoids writing intermediate results and reading them again for bias addition.
func FusedMatVecBias(dst []float32, w *Mat, x, bias []float32) {
	if w.R == 0 || w.C == 0 || len(bias) == 0 {
		// Fallback to regular matvec if no bias
		MatVec(dst, w, x)
		return
	}

	if w.Raw != nil && w.DType != mcf.DTypeF32 {
		// Quantized weights: use regular path + separate bias add
		MatVec(dst, w, x)
		Add(dst, bias)
		return
	}

	if cpu.HasAVX2 {
		fusedMatVecBiasAVX2(dst, w, x, bias)
		return
	}
	fusedMatVecBiasScalar(dst, w, x, bias)
}

// fusedMatVecBiasScalar computes dst = W * x + bias using scalar operations.
func fusedMatVecBiasScalar(dst []float32, w *Mat, x, bias []float32) {
	for i := 0; i < w.R; i++ {
		row := w.Data[i*w.Stride : i*w.Stride+w.C]
		sum := bias[i] // Start with bias
		for j := 0; j < w.C; j++ {
			sum += row[j] * x[j]
		}
		dst[i] = sum
	}
}

// fusedMatVecBiasAVX2 computes dst = W * x + bias using AVX2 SIMD.
func fusedMatVecBiasAVX2(dst []float32, w *Mat, x, bias []float32) {
	c := w.C
	i := 0

	// Process 4 rows at once for better ILP
	for ; i+3 < w.R; i += 4 {
		row0 := w.Data[i*w.Stride : i*w.Stride+w.C]
		row1 := w.Data[(i+1)*w.Stride : (i+1)*w.Stride+w.C]
		row2 := w.Data[(i+2)*w.Stride : (i+2)*w.Stride+w.C]
		row3 := w.Data[(i+3)*w.Stride : (i+3)*w.Stride+w.C]

		// Initialize accumulators with bias
		acc0 := archsimd.LoadFloat32x8Slice(bias[i:])
		acc1 := archsimd.LoadFloat32x8Slice(bias[i+8:])
		acc2 := archsimd.LoadFloat32x8Slice(bias[i+8:])
		acc3 := archsimd.LoadFloat32x8Slice(bias[i+8:])
		_ = acc1
		_ = acc2
		_ = acc3

		var acc0_2, acc1_2, acc2_2, acc3_2 archsimd.Float32x8
		j := 0

		// Process 16 values per iteration
		for ; j+16 <= c; j += 16 {
			vx0 := archsimd.LoadFloat32x8Slice(x[j:])
			vx1 := archsimd.LoadFloat32x8Slice(x[j+8:])

			vrow0a := archsimd.LoadFloat32x8Slice(row0[j:])
			vrow1a := archsimd.LoadFloat32x8Slice(row1[j:])
			vrow2a := archsimd.LoadFloat32x8Slice(row2[j:])
			vrow3a := archsimd.LoadFloat32x8Slice(row3[j:])

			acc0 = vrow0a.MulAdd(vx0, acc0)
			acc1 = vrow1a.MulAdd(vx0, acc1)
			acc2 = vrow2a.MulAdd(vx0, acc2)
			acc3 = vrow3a.MulAdd(vx0, acc3)

			vrow0b := archsimd.LoadFloat32x8Slice(row0[j+8:])
			vrow1b := archsimd.LoadFloat32x8Slice(row1[j+8:])
			vrow2b := archsimd.LoadFloat32x8Slice(row2[j+8:])
			vrow3b := archsimd.LoadFloat32x8Slice(row3[j+8:])

			acc0_2 = vrow0b.MulAdd(vx1, acc0_2)
			acc1_2 = vrow1b.MulAdd(vx1, acc1_2)
			acc2_2 = vrow2b.MulAdd(vx1, acc2_2)
			acc3_2 = vrow3b.MulAdd(vx1, acc3_2)
		}

		// Handle remaining 8-value chunks
		for ; j+8 <= c; j += 8 {
			vx := archsimd.LoadFloat32x8Slice(x[j:])

			vrow0 := archsimd.LoadFloat32x8Slice(row0[j:])
			acc0 = vrow0.MulAdd(vx, acc0)

			vrow1 := archsimd.LoadFloat32x8Slice(row1[j:])
			acc1 = vrow1.MulAdd(vx, acc1)

			vrow2 := archsimd.LoadFloat32x8Slice(row2[j:])
			acc2 = vrow2.MulAdd(vx, acc2)

			vrow3 := archsimd.LoadFloat32x8Slice(row3[j:])
			acc3 = vrow3.MulAdd(vx, acc3)
		}

		// Combine accumulators
		acc0 = acc0.Add(acc0_2)
		acc1 = acc1.Add(acc1_2)
		acc2 = acc2.Add(acc2_2)
		acc3 = acc3.Add(acc3_2)

		// Horizontal sum and store
		var tmp0 [8]float32
		var tmp1 [8]float32
		var tmp2 [8]float32
		var tmp3 [8]float32
		acc0.Store(&tmp0)
		acc1.Store(&tmp1)
		acc2.Store(&tmp2)
		acc3.Store(&tmp3)

		var sum0, sum1, sum2, sum3 float32
		for k := range 8 {
			sum0 += tmp0[k]
			sum1 += tmp1[k]
			sum2 += tmp2[k]
			sum3 += tmp3[k]
		}

		// Handle remaining elements
		for ; j < c; j++ {
			xv := x[j]
			sum0 += row0[j] * xv
			sum1 += row1[j] * xv
			sum2 += row2[j] * xv
			sum3 += row3[j] * xv
		}

		dst[i] = sum0
		dst[i+1] = sum1
		dst[i+2] = sum2
		dst[i+3] = sum3
	}

	// Fall back to scalar for remaining rows
	for ; i < w.R; i++ {
		row := w.Data[i*w.Stride : i*w.Stride+w.C]
		sum := bias[i]
		j := 0
		for ; j+8 <= c; j += 8 {
			vx := archsimd.LoadFloat32x8Slice(x[j:])
			vrow := archsimd.LoadFloat32x8Slice(row[j:])
			acc := vrow.MulAdd(vx, archsimd.BroadcastFloat32x8(sum))
			var tmp [8]float32
			acc.Store(&tmp)
			for k := range 8 {
				sum += tmp[k]
			}
		}
		for ; j < c; j++ {
			sum += row[j] * x[j]
		}
		dst[i] = sum
	}
}

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
