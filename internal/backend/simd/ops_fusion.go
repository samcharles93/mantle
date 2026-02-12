package simd

// FusedRMSNormMatVec tries to use a fused kernel if available, otherwise falls back to separate ops
func FusedRMSNormMatVec(ops Ops, dst []float32, w *Mat, x, normWeight []float32, eps float32, tmpNorm []float32) {
	// Try fused operation (only works with CUDA ops that implement it)
	type fusedOps interface {
		FusedRMSNormMatVec(out []float32, w *Mat, x, normWeight []float32, eps float32) bool
	}

	if fused, ok := ops.(fusedOps); ok {
		if fused.FusedRMSNormMatVec(dst, w, x, normWeight, eps) {
			return // Fusion succeeded
		}
	}

	// Fallback to separate operations
	ops.RMSNorm(tmpNorm, x, normWeight, eps)
	ops.MatVec(dst, w, tmpNorm)
}
