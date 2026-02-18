package simd

// Ops defines the matvec operations used by the runtime.
type Ops interface {
	MatVec(dst []float32, w *Mat, x []float32)
	MatVecWithQuant(dst []float32, w *Mat, x []float32, qx *QuantVec)
	RMSNorm(dst, src, weight []float32, eps float32)
	Softmax(x []float32)
	ApplyRoPE(x []float32, nHead, headDim, pos int, invFreq []float64, attentionFactor float32)
	StoreKV(layerIndex, pos, kvStride int, kDst, vDst []float32, kDst16, vDst16 []uint16, kDstQ8, vDstQ8 []int8, kDstQ8S, vDstQ8S []float32, k, v []float32)
}

// DefaultOps provides default CPU-based operations.
type DefaultOps struct{}

func (DefaultOps) MatVec(dst []float32, w *Mat, x []float32) {
	MatVec(dst, w, x)
}

func (DefaultOps) MatVecWithQuant(dst []float32, w *Mat, x []float32, qx *QuantVec) {
	MatVecWithQuant(dst, w, x, qx)
}

func (DefaultOps) FlashAttention(attnOut []float32, layer *Layer, q, k, v []float32, pos, start, nHead, headDim, kvHeads, kvStride int, scale float32) bool {
	// This method is not typically used directly since FlashAttention operates on full sequences
	// Return false to indicate that this specific fast path is not used
	return false
}

func (DefaultOps) FlashAttentionMultiHead(attnOut []float32, layer *Layer, q, k, v []float32, pos, start, nHead, headDim, kvHeads, kvStride int, scale float32) bool {
	// For incremental processing with KV caching, we can't directly use the full FlashAttention
	// implementation since it's designed for full sequence processing.
	// However, we can implement a version that works with the current KV cache structure.

	// Check if we have enough context to meaningfully use FlashAttention
	// For now, return false to indicate that this specific fast path is not implemented
	// for the incremental processing pattern
	return false
}

// Add the method to DefaultOps
func (DefaultOps) IncrementalAttention(attnOut []float32, layer *Layer, q, k, v []float32, pos, start, nHead, headDim, kvHeads, kvStride int, scale float32) bool {
	// For now, return false to indicate that this specific fast path is not implemented
	// This would be where we implement an optimized attention kernel that leverages
	// FlashAttention concepts for the incremental processing pattern
	return false
}

func (DefaultOps) RMSNorm(dst, src, weight []float32, eps float32) {
	RMSNorm(dst, src, weight, eps)
}

func (DefaultOps) Softmax(x []float32) {
	Softmax(x)
}

func (DefaultOps) ApplyRoPE(x []float32, nHead, headDim, pos int, invFreq []float64, attentionFactor float32) {
	ApplyRoPE(x, nHead, headDim, pos, invFreq, attentionFactor)
}

func (DefaultOps) StoreKV(_ int, pos, kvStride int, kDst, vDst []float32, kDst16, vDst16 []uint16, kDstQ8, vDstQ8 []int8, kDstQ8S, vDstQ8S []float32, k, v []float32) {
	if kvStride <= 0 {
		return
	}
	offset := pos * kvStride
	if kDst != nil {
		copy(kDst[offset:], k)
	} else if kDst16 != nil {
		Float32ToFloat16Slice(k, kDst16[offset:])
	} else if kDstQ8 != nil {
		storeQ8(k, kDstQ8[offset:offset+kvStride], kDstQ8S, pos)
	}
	if vDst != nil {
		copy(vDst[offset:], v)
	} else if vDst16 != nil {
		Float32ToFloat16Slice(v, vDst16[offset:])
	} else if vDstQ8 != nil {
		storeQ8(v, vDstQ8[offset:offset+kvStride], vDstQ8S, pos)
	}
}

// storeQ8 quantizes src to int8 with per-position scaling.
func storeQ8(src []float32, dst []int8, scales []float32, pos int) {
	var maxAbs float32
	for _, v := range src {
		if v < 0 {
			if -v > maxAbs {
				maxAbs = -v
			}
		} else if v > maxAbs {
			maxAbs = v
		}
	}
	if maxAbs == 0 {
		scales[pos] = 0
		for i := range dst {
			dst[i] = 0
		}
		return
	}
	scale := maxAbs / 127.0
	scales[pos] = scale
	invScale := 127.0 / maxAbs
	for i, v := range src {
		q := int32(v*invScale + 0.5)
		if v < 0 {
			q = int32(v*invScale - 0.5)
		}
		if q > 127 {
			q = 127
		} else if q < -127 {
			q = -127
		}
		dst[i] = int8(q)
	}
}
