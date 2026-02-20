package simd

import (
	core "github.com/samcharles93/mantle/internal/backend/core"
	"github.com/samcharles93/mantle/pkg/mcf"
)

func NewMat(r, c int) Mat {
	return core.NewMat(r, c)
}

func NewMatFromData(r, c int, data []float32) Mat {
	return core.NewMatFromData(r, c, data)
}

func NewMatFromRaw(r, c int, dtype mcf.TensorDType, raw []byte) (Mat, error) {
	return core.NewMatFromRaw(r, c, dtype, raw)
}
