package simd

import (
	"testing"

	"github.com/samcharles93/mantle/internal/mcfstore"
	"github.com/samcharles93/mantle/pkg/mcf"
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

func TestQuantCacheWantedForLoad(t *testing.T) {
	gq4, _ := mcf.GQDType(4, 128)
	gq3, _ := mcf.GQDType(3, 64)
	for dt, want := range map[mcf.TensorDType]bool{
		mcf.DTypeQ4: false, mcf.DTypeK4: false, gq4: false,
		mcf.DTypeQ8: true, mcf.DTypeK6: true, mcf.DTypeK3: true, mcf.DTypeK2: true, gq3: true,
	} {
		if got := quantCacheWantedForLoad(dt); got != want {
			t.Errorf("%#x: cache wanted = %v, want %v", dt, got, want)
		}
	}
	restore := SetQuantCacheBuildEnabledForLoad(false)
	defer restore()
	if quantCacheWantedForLoad(mcf.DTypeQ8) {
		t.Error("q8 cache wanted with cache building disabled")
	}
}

func TestLoadModelMCFRejectsInvalidCacheType(t *testing.T) {
	if _, err := LoadModelMCF(nil, nil, 0, LoadModelOptions{CacheTypeK: "q4_0"}); err == nil {
		t.Fatal("LoadModelMCF accepted unsupported CacheTypeK=q4_0")
	}
	if _, err := LoadModelMCF(nil, nil, 0, LoadModelOptions{CacheTypeV: "q4_0"}); err == nil {
		t.Fatal("LoadModelMCF accepted unsupported CacheTypeV=q4_0")
	}
}
