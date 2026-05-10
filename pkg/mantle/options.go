package mantle

import "fmt"

// EngineOption configures model loading behaviour in NewEngine.
type EngineOption func(*engineOptions) error

// engineOptions holds the internal configuration for model loading.
type engineOptions struct {
	backend         string
	maxContext      int
	tokenizerJSON   string
	tokenizerConfig string
	chatTemplate    string
	hfConfigFile    string
	cacheTypeK      string
	cacheTypeV      string
	gpuLayers       int
	noSWA           bool
	tileM           int
	tileN           int
	tileK           int
}

// WithBackend selects the execution backend.
// Valid values: "auto" (default), "cpu", "cuda".
func WithBackend(name string) EngineOption {
	return func(o *engineOptions) error {
		switch name {
		case "auto", "cpu", "cuda":
			o.backend = name
			return nil
		default:
			return fmt.Errorf("invalid backend %q: expected auto, cpu, or cuda", name)
		}
	}
}

// WithMaxContext sets the maximum context length for KV cache allocation.
// Use 0 to use the model's native maximum (default).
func WithMaxContext(n int) EngineOption {
	return func(o *engineOptions) error {
		if n < 0 {
			return fmt.Errorf("max context must be >= 0, got %d", n)
		}
		o.maxContext = n
		return nil
	}
}

// WithKVCacheType sets the KV cache data types.
// Valid types: "f32", "f16" (default), "q8_0".
func WithKVCacheType(k, v string) EngineOption {
	return func(o *engineOptions) error {
		o.cacheTypeK = k
		o.cacheTypeV = v
		return nil
	}
}

// WithGPULayers sets the number of layers to offload to the GPU.
// -1 = auto (default), 0 = all layers on CPU.
func WithGPULayers(n int) EngineOption {
	return func(o *engineOptions) error {
		if n < -1 {
			return fmt.Errorf("gpu-layers must be >= -1, got %d", n)
		}
		o.gpuLayers = n
		return nil
	}
}

// WithDisableSWA disables sliding window attention, forcing full-size
// KV cache for all layers.
func WithDisableSWA() EngineOption {
	return func(o *engineOptions) error {
		o.noSWA = true
		return nil
	}
}

// WithTiling sets the GEMM tiling dimensions.
// Defaults: tileM=32, tileN=32, tileK=16.
func WithTiling(tileM, tileN, tileK int) EngineOption {
	return func(o *engineOptions) error {
		if tileM <= 0 || tileN <= 0 || tileK <= 0 {
			return fmt.Errorf("tiling dimensions must be positive, got M=%d N=%d K=%d", tileM, tileN, tileK)
		}
		o.tileM = tileM
		o.tileN = tileN
		o.tileK = tileK
		return nil
	}
}
