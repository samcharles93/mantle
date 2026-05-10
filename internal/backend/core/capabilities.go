package core

// OpCapability enumerates optional backend capabilities related to
// graph execution and fused kernels.
type OpCapability int

const (
	// CapUnknown is the zero/default value and represents no capability.
	CapUnknown OpCapability = iota
	CapFusedFFN
	CapFusedAttention
	CapFusedNormResidual
	CapFusedMoE
	CapGraphCompute
	CapGraphCapture
)

func (c OpCapability) String() string {
	switch c {
	case CapFusedFFN:
		return "CapFusedFFN"
	case CapFusedAttention:
		return "CapFusedAttention"
	case CapFusedNormResidual:
		return "CapFusedNormResidual"
	case CapFusedMoE:
		return "CapFusedMoE"
	case CapGraphCompute:
		return "CapGraphCompute"
	case CapGraphCapture:
		return "CapGraphCapture"
	default:
		return "CapUnknown"
	}
}

// Capabilities describes which optional operations a backend supports.
//
// Note: the Ops map is not safe for concurrent mutation. The intended
// usage is to initialise a Capabilities instance at backend creation
// time (single goroutine) and then query it concurrently via
// Supports(). If you require concurrent mutation, wrap access with a
// mutex in the caller; this package purposefully avoids global
// mutable state (T26 will add any global registries).
type Capabilities struct {
	Ops map[OpCapability]bool
}

// Supports reports whether the capability is available. It is safe to
// call concurrently provided the Ops map is not being mutated
// concurrently.
func (c *Capabilities) Supports(cap OpCapability) bool {
	if c == nil || c.Ops == nil {
		return false
	}
	ok, _ := c.Ops[cap]
	return ok
}

// Set marks a capability supported (true) or unsupported (false).
// Callers should avoid concurrent Set() calls without external
// synchronization.
func (c *Capabilities) Set(cap OpCapability, supported bool) {
	if c == nil {
		return
	}
	if c.Ops == nil {
		c.Ops = make(map[OpCapability]bool, 8)
	}
	if supported {
		c.Ops[cap] = true
	} else {
		delete(c.Ops, cap)
	}
}

// DefaultCapabilities returns a conservative CPU baseline: only
// graph compute is available (no fused ops).
func DefaultCapabilities() *Capabilities {
	c := &Capabilities{Ops: make(map[OpCapability]bool, 4)}
	c.Ops[CapGraphCompute] = true
	return c
}

// SIMDCapabilities returns capabilities typically available when the
// SIMD-accelerated CPU path is enabled: fused FFN, fused attention,
// fused norm+residual and basic graph compute.
func SIMDCapabilities() *Capabilities {
	c := &Capabilities{Ops: make(map[OpCapability]bool, 8)}
	c.Ops[CapGraphCompute] = true
	c.Ops[CapFusedFFN] = true
	c.Ops[CapFusedAttention] = true
	c.Ops[CapFusedNormResidual] = true
	return c
}

// CUDACapabilities returns an aggressive capability set used by CUDA
// backends: all fused ops plus graph compute and graph capture.
func CUDACapabilities() *Capabilities {
	c := &Capabilities{Ops: make(map[OpCapability]bool, 12)}
	c.Ops[CapGraphCompute] = true
	c.Ops[CapGraphCapture] = true
	c.Ops[CapFusedFFN] = true
	c.Ops[CapFusedAttention] = true
	c.Ops[CapFusedNormResidual] = true
	c.Ops[CapFusedMoE] = true
	return c
}
