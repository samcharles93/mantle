package simd

// Ops defines the matvec operations used by the runtime.
type Ops interface {
	MatVec(dst []float32, w *Mat, x []float32)
	MatVecWithQuant(dst []float32, w *Mat, x []float32, qx *QuantVec)
	RMSNorm(dst, src, weight []float32, eps float32)
	Softmax(x []float32)
	ApplyRoPE(x []float32, nHead, headDim, pos int, invFreq []float64, attentionFactor float32)
	StoreKV(layerIndex, pos, kvStride int, kDst, vDst []float32, kDst16, vDst16 []uint16, k, v []float32)
}

// DefaultOps provides default CPU-based operations.
type DefaultOps struct{}

func (DefaultOps) MatVec(dst []float32, w *Mat, x []float32) {
	MatVec(dst, w, x)
}

func (DefaultOps) MatVecWithQuant(dst []float32, w *Mat, x []float32, qx *QuantVec) {
	MatVecWithQuant(dst, w, x, qx)
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

func (DefaultOps) StoreKV(_ int, pos, kvStride int, kDst, vDst []float32, kDst16, vDst16 []uint16, k, v []float32) {
	if kvStride <= 0 {
		return
	}
	offset := pos * kvStride
	if kDst != nil {
		copy(kDst[offset:], k)
	} else if kDst16 != nil {
		Float32ToFloat16Slice(k, kDst16[offset:])
	}
	if vDst != nil {
		copy(vDst[offset:], v)
	} else if vDst16 != nil {
		Float32ToFloat16Slice(v, vDst16[offset:])
	}
}

// EnsureOps returns the provided ops or the default implementation.
func EnsureOps(current Ops) Ops {
	if current == nil {
		return DefaultOps{}
	}
	return current
}
