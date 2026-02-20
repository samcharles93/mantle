package bootstrap

import (
	core "github.com/samcharles93/mantle/internal/backend/core"
	"github.com/samcharles93/mantle/internal/backend/simd"
	"github.com/samcharles93/mantle/internal/mcfstore"
)

// LoadSIMDRuntime loads the shared CPU runtime and returns both the runtime
// contract and a core instance view for backend-specific setup.
func LoadSIMDRuntime(mcfFile *mcfstore.File, cfgBytes []byte, maxContext int, opts core.LoadModelOptions, disableQuantCache bool) (core.Runtime, *core.Instance, error) {
	if disableQuantCache {
		restoreQuantCache := simd.SetQuantCacheBuildEnabledForLoad(false)
		defer restoreQuantCache()
	}

	m, err := simd.LoadModelMCF(mcfFile, cfgBytes, maxContext, opts)
	if err != nil {
		return nil, nil, err
	}
	return m, (*core.Instance)(m), nil
}
