package simd

// DeviceStateOps is an optional extension of Ops that allows keeping
// the hidden‑state vector (x) on device across layers of a token.
// Implementations that support this can reduce per‑token H2D copies.
type DeviceStateOps interface {
	// BeginToken signals the start of a new token. The host vector x
	// is the current hidden state (after embedding lookup).
	// The implementation may upload x to device and keep it there.
	BeginToken(x []float32)

	// EndToken signals the end of token processing. The host vector x
	// is the final hidden state after all layers; the implementation
	// should download it back to host if needed.
	// If the host x is already up‑to‑date (e.g., because the final
	// logits were computed directly on device), this can be a no‑op.
	EndToken(x []float32)

	// HostStateDirty marks the host vector as updated by a host-side path.
	// Implementations should treat device copies as stale until refreshed.
	HostStateDirty(x []float32)

	// SyncHostState ensures host x reflects the latest state before host ops.
	SyncHostState(x []float32)

	// DeviceAdd performs dst += src on device, where both slices
	// refer to device‑resident buffers (typically the hidden state
	// and an intermediate result). Returns true if the operation
	// was performed on device, false if the caller should fall back
	// to the host Add function.
	DeviceAdd(dst, src []float32) bool

	// DeviceRMSNorm performs RMSNorm on device‑resident buffers.
	// Returns true if the operation was performed on device, false
	// if the caller should fall back to the regular RMSNorm method.
	DeviceRMSNorm(dst, src, weight []float32, eps float32) bool

	// DeviceMatVec performs MatVec on device‑resident buffers.
	// Returns true if the operation was performed on device, false
	// if the caller should fall back to the regular MatVec method.
	DeviceMatVec(dst []float32, w *Mat, x []float32) bool
}
