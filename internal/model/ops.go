package model

import "github.com/samcharles93/mantle/internal/tensor"

type Ops interface {
	MatVec(dst []float32, w *tensor.Mat, x []float32)
	MatVecWithQuant(dst []float32, w *tensor.Mat, x []float32, qx *tensor.QuantVec)
}

type defaultOps struct{}

func (defaultOps) MatVec(dst []float32, w *tensor.Mat, x []float32) {
	tensor.MatVec(dst, w, x)
}

func (defaultOps) MatVecWithQuant(dst []float32, w *tensor.Mat, x []float32, qx *tensor.QuantVec) {
	tensor.MatVecWithQuant(dst, w, x, qx)
}

func ensureOps(current Ops) Ops {
	if current == nil {
		return defaultOps{}
	}
	return current
}
