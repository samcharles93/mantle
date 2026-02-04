package cpu

import (
	"github.com/samcharles93/mantle/internal/backend"
	"github.com/samcharles93/mantle/internal/mcfstore"
	"github.com/samcharles93/mantle/internal/model"
)

type Backend struct{}

func New() *Backend {
	return &Backend{}
}

func (b *Backend) Name() string {
	return backend.CPU
}

func (b *Backend) LoadModel(mcfFile *mcfstore.File, cfgBytes []byte, maxContext int) (model.Runtime, error) {
	m, err := model.LoadModelMCF(mcfFile, cfgBytes, maxContext)
	if err != nil {
		return nil, err
	}
	return m, nil
}
