package core

import "github.com/samcharles93/mantle/internal/hostcaps"

// LoadModelOptions controls backend/runtime behavior during model load.
type LoadModelOptions struct {
	CacheTypeK   string
	CacheTypeV   string
	HostCaps     *hostcaps.Snapshot
	TilingConfig TilingConfig
	GpuLayers    int // -1 auto, 0 all layers on CPU, N first N layers on GPU
}
