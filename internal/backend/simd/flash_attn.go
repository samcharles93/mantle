package simd

import (
	"fmt"
	"math"
	"simd/archsimd"
)

// Block sizes for tiling (tuned for L1/L2 cache)
const (
	DefaultBlockQ = 32
	DefaultBlockK = 32
)

// FlashAttention computes scaled dot-product attention using Flash Attention algorithm.
// Q, K, V are laid out as [seqLen, dim] row-major.
func FlashAttention(q, k, v []float32, seqLenQ, seqLenK, dim int, scale float32) []float32 {
	output := make([]float32, seqLenQ*dim)

	if archsimd.X86.AVX512() {
		flashAttentionAVX512(output, q, k, v, seqLenQ, seqLenK, dim, scale)
	} else if archsimd.X86.AVX2() {
		flashAttentionAVX2(output, q, k, v, seqLenQ, seqLenK, dim, scale)
	} else {
		flashAttentionScalar(output, q, k, v, seqLenQ, seqLenK, dim, scale)
	}

	return output
}

// ============================================================
// AVX-512 Implementation (516-bit vectors = 16 float32s)
// ============================================================

func flashAttentionAVX512(output, q, k, v []float32, seqLenQ, seqLenK, dim int, scale float32) {
	// Process query blocks
	for i := 0; i < seqLenQ; i += DefaultBlockQ {
		iEnd := min(i+DefaultBlockQ, seqLenQ)
		blockQ := iEnd - i

		// Block statistics for online softmax
		blockM := make([]float32, blockQ)
		blockL := make([]float32, blockQ)
		for qi := range blockM {
			blockM[qi] = -math.MaxFloat32
		}

		// Flat accumulator for better cache locality
		acc := make([]float32, blockQ*dim)

		// Process key-value blocks
		for j := 0; j < seqLenK; j += DefaultBlockK {
			jEnd := min(j+DefaultBlockK, seqLenK)
			blockK := jEnd - j

			// Compute Q @ K^T * scale
			scores := computeQKTAVX512(q, k, i, iEnd, j, jEnd, dim, scale)

			// Online softmax + accumulate V
			softmaxAccumulateAVX512(acc, scores, v, j, dim, blockM, blockL, blockQ, blockK)
		}

		// Normalize and write output
		for qi := range blockQ {
			invL := float32(1.0) / blockL[qi]
			invLVec := archsimd.BroadcastFloat32x16(invL)

			// Process in chunks of 16 for SIMD
			d := 0
			for ; d <= dim-16; d += 16 {
				rowPtr := (*[16]float32)(acc[(qi*dim + d):])
				vec := archsimd.LoadFloat32x16(rowPtr)
				vec = vec.Mul(invLVec)
				vec.Store((*[16]float32)(output[(i+qi)*dim+d : (i+qi)*dim+d+16]))
			}

			// Remainder
			for ; d < dim; d++ {
				output[(i+qi)*dim+d] = acc[qi*dim+d] * invL
			}
		}
	}
}

func computeQKTAVX512(q, k []float32, iStart, iEnd, jStart, jEnd, dim int, scale float32) [][]float32 {
	blockQ := iEnd - iStart
	blockK := jEnd - jStart
	scores := make([][]float32, blockQ)

	for qi := range blockQ {
		scores[qi] = make([]float32, blockK)
		qRow := q[(iStart+qi)*dim:]

		for ki := range blockK {
			kRow := k[(jStart+ki)*dim:]

			// Dot product using SIMD
			var sum float32
			d := 0
			for ; d <= dim-16; d += 16 {
				qVec := archsimd.LoadFloat32x16((*[16]float32)(qRow[d : d+16]))
				kVec := archsimd.LoadFloat32x16((*[16]float32)(kRow[d : d+16]))
				prod := qVec.Mul(kVec)
				sum += horizontalSumAVX512(prod)
			}

			// Remainder
			for ; d < dim; d++ {
				sum += qRow[d] * kRow[d]
			}

			scores[qi][ki] = sum * scale
		}
	}

	return scores
}

func horizontalSumAVX512(v archsimd.Float32x16) float32 {
	lo := v.GetLo()
	hi := v.GetHi()
	sumVec := lo.Add(hi)

	lo4 := sumVec.GetLo()
	hi4 := sumVec.GetHi()
	sumVec4 := lo4.Add(hi4)

	var arr [4]float32
	sumVec4.Store(&arr)
	return arr[0] + arr[1] + arr[2] + arr[3]
}

func softmaxAccumulateAVX512(acc []float32, scores [][]float32, v []float32,
	jStart, dim int, blockM, blockL []float32, blockQ, blockK int) {

	for qi := range blockQ {
		// Find max in this row
		rowMax := blockM[qi]
		for ki := range blockK {
			if scores[qi][ki] > rowMax {
				rowMax = scores[qi][ki]
			}
		}

		// Compute exp and update sum
		rowSum := float32(0.0)
		expVals := make([]float32, blockK)
		for ki := range blockK {
			expVals[ki] = fastExp(scores[qi][ki] - rowMax)
			rowSum += expVals[ki]
		}

		// Update statistics with online softmax formula
		alpha := fastExp(blockM[qi] - rowMax)
		newL := alpha*blockL[qi] + rowSum

		// Rescale accumulator with vectorized multiply
		scaleFactor := alpha * blockL[qi] / newL
		scaleVec := archsimd.BroadcastFloat32x16(scaleFactor)

		d := 0
		for ; d <= dim-16; d += 16 {
			accPtr := (*[16]float32)(acc[(qi*dim + d):])
			vec := archsimd.LoadFloat32x16(accPtr)
			vec = vec.Mul(scaleVec)
			vec.Store(accPtr)
		}

		// Remainder
		for ; d < dim; d++ {
			acc[qi*dim+d] *= scaleFactor
		}

		// Accumulate weighted values with FMA
		for ki := range blockK {
			weight := expVals[ki]
			weightVec := archsimd.BroadcastFloat32x16(weight)
			vRow := v[(jStart+ki)*dim:]

			d := 0
			for ; d <= dim-16; d += 16 {
				accPtr := (*[16]float32)(acc[(qi*dim + d):])
				vVec := archsimd.LoadFloat32x16((*[16]float32)(vRow[d : d+16]))

				// FMA: acc += weight * v
				prod := vVec.Mul(weightVec)
				accVec := archsimd.LoadFloat32x16(accPtr)
				accVec = accVec.Add(prod)
				accVec.Store(accPtr)
			}

			// Remainder
			for ; d < dim; d++ {
				acc[qi*dim+d] += weight * vRow[d]
			}
		}

		blockM[qi] = rowMax
		blockL[qi] = newL
	}
}

// ============================================================
// AVX2 Implementation (256-bit vectors = 8 float32s)
// ============================================================

func flashAttentionAVX2(output, q, k, v []float32, seqLenQ, seqLenK, dim int, scale float32) {
	// Process query blocks
	for i := 0; i < seqLenQ; i += DefaultBlockQ {
		iEnd := min(i+DefaultBlockQ, seqLenQ)
		blockQ := iEnd - i

		// Block statistics for online softmax
		blockM := make([]float32, blockQ)
		blockL := make([]float32, blockQ)
		for qi := range blockM {
			blockM[qi] = -math.MaxFloat32
		}

		// Flat accumulator for better cache locality
		acc := make([]float32, blockQ*dim)

		// Process key-value blocks
		for j := 0; j < seqLenK; j += DefaultBlockK {
			jEnd := min(j+DefaultBlockK, seqLenK)
			blockK := jEnd - j

			// Compute Q @ K^T * scale
			scores := computeQKTAVX2(q, k, i, iEnd, j, jEnd, dim, scale)

			// Online softmax + accumulate V
			softmaxAccumulateAVX2(acc, scores, v, j, dim, blockM, blockL, blockQ, blockK)
		}

		// Normalize and write output
		for qi := range blockQ {
			invL := float32(1.0) / blockL[qi]
			invLVec := archsimd.BroadcastFloat32x8(invL)

			// Process in chunks of 8 for SIMD
			d := 0
			for ; d <= dim-8; d += 8 {
				accPtr := (*[8]float32)(acc[(qi*dim + d):])
				vec := archsimd.LoadFloat32x8(accPtr)
				vec = vec.Mul(invLVec)
				vec.Store((*[8]float32)(output[(i+qi)*dim+d : (i+qi)*dim+d+8]))
			}

			// Remainder
			for ; d < dim; d++ {
				output[(i+qi)*dim+d] = acc[qi*dim+d] * invL
			}
		}
	}
}

func computeQKTAVX2(q, k []float32, iStart, iEnd, jStart, jEnd, dim int, scale float32) [][]float32 {
	blockQ := iEnd - iStart
	blockK := jEnd - jStart
	scores := make([][]float32, blockQ)

	for qi := range blockQ {
		scores[qi] = make([]float32, blockK)
		qRow := q[(iStart+qi)*dim:]

		for ki := range blockK {
			kRow := k[(jStart+ki)*dim:]

			// Dot product using SIMD
			var sum float32
			d := 0
			for ; d <= dim-8; d += 8 {
				qVec := archsimd.LoadFloat32x8((*[8]float32)(qRow[d : d+8]))
				kVec := archsimd.LoadFloat32x8((*[8]float32)(kRow[d : d+8]))
				prod := qVec.Mul(kVec)
				sum += horizontalSumAVX2(prod)
			}

			// Remainder
			for ; d < dim; d++ {
				sum += qRow[d] * kRow[d]
			}

			scores[qi][ki] = sum * scale
		}
	}

	return scores
}

func horizontalSumAVX2(v archsimd.Float32x8) float32 {
	// Float32x8 -> two Float32x4
	lo := v.GetLo()
	hi := v.GetHi()
	sumVec := lo.Add(hi)

	// Store and sum manually
	var arr [4]float32
	sumVec.Store(&arr)
	return arr[0] + arr[1] + arr[2] + arr[3]
}

func softmaxAccumulateAVX2(acc []float32, scores [][]float32, v []float32,
	jStart, dim int, blockM, blockL []float32, blockQ, blockK int) {

	for qi := range blockQ {
		// Find max in this row
		rowMax := blockM[qi]
		for ki := range blockK {
			if scores[qi][ki] > rowMax {
				rowMax = scores[qi][ki]
			}
		}

		// Compute exp and update sum
		rowSum := float32(0.0)
		expVals := make([]float32, blockK)
		for ki := range blockK {
			expVals[ki] = fastExp(scores[qi][ki] - rowMax)
			rowSum += expVals[ki]
		}

		// Update statistics with online softmax formula
		alpha := fastExp(blockM[qi] - rowMax)
		newL := alpha*blockL[qi] + rowSum

		// Rescale accumulator with vectorized multiply
		scaleFactor := alpha * blockL[qi] / newL
		scaleVec := archsimd.BroadcastFloat32x8(scaleFactor)

		d := 0
		for ; d <= dim-8; d += 8 {
			accPtr := (*[8]float32)(acc[(qi*dim + d):])
			vec := archsimd.LoadFloat32x8(accPtr)
			vec = vec.Mul(scaleVec)
			vec.Store(accPtr)
		}

		// Remainder
		for ; d < dim; d++ {
			acc[qi*dim+d] *= scaleFactor
		}

		// Accumulate weighted values
		for ki := range blockK {
			weight := expVals[ki]
			weightVec := archsimd.BroadcastFloat32x8(weight)
			vRow := v[(jStart+ki)*dim:]

			d := 0
			for ; d <= dim-8; d += 8 {
				accPtr := (*[8]float32)(acc[(qi*dim + d):])
				vVec := archsimd.LoadFloat32x8((*[8]float32)(vRow[d : d+8]))
				// acc += weight * v
				prod := vVec.Mul(weightVec)
				accVec := archsimd.LoadFloat32x8(accPtr)
				accVec = accVec.Add(prod)
				accVec.Store(accPtr)
			}

			// Remainder
			for ; d < dim; d++ {
				acc[qi*dim+d] += weight * vRow[d]
			}
		}

		blockM[qi] = rowMax
		blockL[qi] = newL
	}
}

// ============================================================
// Scalar Fallback Implementation
// ============================================================

func flashAttentionScalar(output, q, k, v []float32, seqLenQ, seqLenK, dim int, scale float32) {
	l := make([]float32, seqLenQ)
	m := make([]float32, seqLenQ)
	for i := range m {
		m[i] = -math.MaxFloat32
	}

	for i := range seqLenQ {
		rowMax := float32(-math.MaxFloat32)

		// Compute attention scores for this query
		scores := make([]float32, seqLenK)
		for j := range seqLenK {
			score := float32(0.0)
			for d := range dim {
				score += q[i*dim+d] * k[j*dim+d]
			}
			score *= scale
			scores[j] = score

			if score > rowMax {
				rowMax = score
			}
		}

		// Compute exp and sum
		rowSum := float32(0.0)
		for j := range seqLenK {
			scores[j] = fastExp(scores[j] - rowMax)
			rowSum += scores[j]
		}

		// Update statistics
		alpha := fastExp(m[i] - rowMax)
		newL := alpha*l[i] + rowSum

		// Rescale output
		for d := range dim {
			output[i*dim+d] *= alpha * l[i] / newL
		}

		// Accumulate weighted values
		for j := range seqLenK {
			weight := scores[j]
			for d := range dim {
				output[i*dim+d] += weight * v[j*dim+d]
			}
		}

		m[i] = rowMax
		l[i] = newL
	}

	// Final normalization
	for i := range seqLenQ {
		invL := float32(1.0) / l[i]
		for d := range dim {
			output[i*dim+d] *= invL
		}
	}
}

// ============================================================
// Multi-Head Attention Wrapper
// ============================================================

// FlashAttentionMultiHead computes multi-head attention.
// Input layout: [batch, heads, seqLen, dim] contiguous in row-major order.
// Output layout: same as input.
func FlashAttentionMultiHead(q, k, v []float32, batch, heads, seqLenQ, seqLenK, headDim int, scale float32) []float32 {
	output := make([]float32, batch*heads*seqLenQ*headDim)

	for b := range batch {
		for h := range heads {
			// Calculate offsets for this batch/head
			qOffset := (b*heads + h) * seqLenQ * headDim
			kvOffset := (b*heads + h) * seqLenK * headDim

			// Slice inputs
			qHead := q[qOffset : qOffset+seqLenQ*headDim]
			kHead := k[kvOffset : kvOffset+seqLenK*headDim]
			vHead := v[kvOffset : kvOffset+seqLenK*headDim]

			// Compute attention for this head
			headOut := FlashAttention(qHead, kHead, vHead, seqLenQ, seqLenK, headDim, scale)

			// Copy to output
			outOffset := qOffset // Output layout matches Q layout
			copy(output[outOffset:outOffset+seqLenQ*headDim], headOut)
		}
	}

	return output
}

// ============================================================
// Utilities
// ============================================================

// GetOptimalBlockSize returns optimal block sizes based on CPU features
func GetOptimalBlockSize() (blockQ, blockK int) {
	if archsimd.X86.AVX512() {
		return 32, 32 // Larger blocks for AVX-512
	} else if archsimd.X86.AVX2() {
		return 32, 32
	}
	return 16, 16 // Smaller blocks for scalar
}

// ValidateFlashAttention checks if inputs are valid
func ValidateFlashAttention(q, k, v []float32, seqLenQ, seqLenK, dim int) error {
	if len(q) < seqLenQ*dim {
		return fmt.Errorf("q buffer too small: expected %d, got %d", seqLenQ*dim, len(q))
	}
	if len(k) < seqLenK*dim {
		return fmt.Errorf("k buffer too small: expected %d, got %d", seqLenK*dim, len(k))
	}
	if len(v) < seqLenK*dim {
		return fmt.Errorf("v buffer too small: expected %d, got %d", seqLenK*dim, len(v))
	}
	if dim <= 0 || seqLenQ <= 0 || seqLenK <= 0 {
		return fmt.Errorf("invalid dimensions: seqLenQ=%d, seqLenK=%d, dim=%d", seqLenQ, seqLenK, dim)
	}
	return nil
}
