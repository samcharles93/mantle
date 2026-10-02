package core

import "github.com/samcharles93/mantle/internal/hostcaps"

// LoadModelOptions controls backend/runtime behavior during model load.
type LoadModelOptions struct {
	CacheTypeK   string
	CacheTypeV   string
	HostCaps     *hostcaps.Snapshot
	TilingConfig TilingConfig
	GpuLayers    int  // -1 auto, 0 all layers on CPU, N first N layers on GPU
	UseGraph     bool // experimental: use graph-based execution

	// HiddenTapLayers is the caller-supplied list of decoder layer indices whose
	// post-layer residual outputs are captured for the DSpark draft model's
	// context taps. -1 selects the embedding output. Empty (the default) disables
	// capture. It is never read from the MCF.
	HiddenTapLayers []int
}
