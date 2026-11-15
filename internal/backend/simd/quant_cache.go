package simd

import (
	"sync"
	"sync/atomic"

	instance "github.com/samcharles93/mantle/internal/backend/core"
	"github.com/samcharles93/mantle/pkg/mcf"
)

var quantCacheBuildState = struct {
	mu      sync.Mutex
	enabled atomic.Bool
}{}

func init() {
	quantCacheBuildState.enabled.Store(true)
}

// SetQuantCacheBuildEnabledForLoad toggles quant cache prebuild for subsequent model loads.
// It returns a restore function and serializes changes to avoid cross-load races.
func SetQuantCacheBuildEnabledForLoad(enabled bool) func() {
	quantCacheBuildState.mu.Lock()
	prev := quantCacheBuildState.enabled.Load()
	quantCacheBuildState.enabled.Store(enabled)
	return func() {
		quantCacheBuildState.enabled.Store(prev)
		quantCacheBuildState.mu.Unlock()
	}
}

func quantCacheBuildEnabledForLoad() bool {
	return quantCacheBuildState.enabled.Load()
}

// quantCacheWantedForLoad reports whether the loader should expand a weight
// of dtype dt into the int8 quant cache. Packed 4-bit dtypes never need it:
// the packed matvec and batched kernels read the payload directly and match
// the cached kernels bit for bit, so caching would only double their memory.
func quantCacheWantedForLoad(dt mcf.TensorDType) bool {
	return quantCacheBuildEnabledForLoad() && !isPacked4DType(dt)
}

// BuildQuantCache pre-unpacks a quantized matrix into int8 blocks and per-block scales.
func BuildQuantCache(m *instance.Mat) (*instance.QuantCache, error) {
	return instance.BuildQuantCache(m)
}
