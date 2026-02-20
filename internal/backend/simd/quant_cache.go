package simd

import (
	"sync"
	"sync/atomic"

	instance "github.com/samcharles93/mantle/internal/backend/core"
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

// BuildQuantCache pre-unpacks a quantized matrix into int8 blocks and per-block scales.
func BuildQuantCache(m *instance.Mat) (*instance.QuantCache, error) {
	return instance.BuildQuantCache(m)
}
