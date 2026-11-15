package simd

import (
	"testing"

	"github.com/samcharles93/mantle/internal/mcfstore"
)

func TestCPULoadModelNoQuantCache(t *testing.T) {
	prev := loadModelMCF
	t.Cleanup(func() { loadModelMCF = prev })
	var cacheDuringLoad bool
	loadModelMCF = func(*mcfstore.File, []byte, int, LoadModelOptions) (*Instance, error) {
		cacheDuringLoad = quantCacheBuildEnabledForLoad()
		return nil, nil
	}
	for _, noCache := range []bool{false, true} {
		if _, err := New().LoadModel(nil, nil, 0, LoadModelOptions{NoQuantCache: noCache}); err != nil {
			t.Fatal(err)
		}
		if cacheDuringLoad == noCache {
			t.Errorf("NoQuantCache=%v: cache build enabled during load = %v", noCache, cacheDuringLoad)
		}
		if !quantCacheBuildEnabledForLoad() {
			t.Fatalf("NoQuantCache=%v: cache build left disabled after load", noCache)
		}
	}
}
