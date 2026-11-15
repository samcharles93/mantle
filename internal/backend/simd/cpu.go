package simd

import "github.com/samcharles93/mantle/internal/mcfstore"

type Backend struct{}

func New() *Backend {
	return &Backend{}
}

func (b *Backend) Name() string {
	return "cpu"
}

// loadModelMCF is LoadModelMCF, replaceable in tests.
var loadModelMCF = LoadModelMCF

// LoadModel loads a model for the CPU backend. With opts.NoQuantCache the
// int8 quant cache is not built, so quantized weights stay packed in memory.
// The CUDA backend disables the CPU cache itself through
// bootstrap.LoadSIMDRuntime; taking the toggle here as well would deadlock,
// which is why this lives on the CPU entry point rather than in
// LoadModelMCF.
func (b *Backend) LoadModel(mcfFile *mcfstore.File, cfgBytes []byte, maxContext int, opts LoadModelOptions) (Runtime, error) {
	if opts.NoQuantCache {
		restore := SetQuantCacheBuildEnabledForLoad(false)
		defer restore()
	}
	return loadModelMCF(mcfFile, cfgBytes, maxContext, opts)
}
