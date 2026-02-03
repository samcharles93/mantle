package tensor

import "github.com/samcharles93/mantle/pkg/mcf"

// PrepareQuantVec quantizes x into an int8/block representation suitable for reuse
// across multiple quantized matvec calls with the same input vector.
func PrepareQuantVec(x []float32) *QuantVec {
	if len(x) == 0 {
		return nil
	}
	blocks := (len(x) + 31) / 32
	qx := getQuantVec(blocks)
	quantizeVecBlocksInto(x, blocks, qx.q, qx.q16, qx.scales)
	return qx
}

// ReleaseQuantVec returns a QuantVec to the pool.
func ReleaseQuantVec(qx *QuantVec) {
	putQuantVec(qx)
}

// CanUseQuantVec reports whether a matrix can benefit from a pre-quantized input vector.
func CanUseQuantVec(w *Mat) bool {
	if w == nil || w.Raw == nil {
		return false
	}
	if !mcf.DTypeRequiresAligned64(w.DType) {
		return false
	}
	return cpu.HasAVX2
}
