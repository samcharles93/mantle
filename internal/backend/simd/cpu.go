package simd

import "github.com/samcharles93/mantle/internal/mcfstore"

type Backend struct{}

func New() *Backend {
	return &Backend{}
}

func (b *Backend) Name() string {
	return "cpu"
}

func (b *Backend) LoadModel(mcfFile *mcfstore.File, cfgBytes []byte, maxContext int, opts LoadModelOptions) (Runtime, error) {
	return LoadModelMCF(mcfFile, cfgBytes, maxContext, opts)
}
