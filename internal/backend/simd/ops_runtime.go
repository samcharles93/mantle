package simd

// Ops defines the matvec operations used by the runtime.
type Ops interface {
	MatVec(dst []float32, w *Mat, x []float32)
	MatVecWithQuant(dst []float32, w *Mat, x []float32, qx *QuantVec)
}

// DefaultOps provides default CPU-based operations.
type DefaultOps struct{}

func (DefaultOps) MatVec(dst []float32, w *Mat, x []float32) {
	MatVec(dst, w, x)
}

func (DefaultOps) MatVecWithQuant(dst []float32, w *Mat, x []float32, qx *QuantVec) {
	MatVecWithQuant(dst, w, x, qx)
}

// EnsureOps returns the provided ops or the default implementation.
func EnsureOps(current Ops) Ops {
	if current == nil {
		return DefaultOps{}
	}
	return current
}
