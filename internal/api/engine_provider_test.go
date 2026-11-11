package api

import (
	"context"
	"os"
	"path/filepath"
	"reflect"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/samcharles93/mantle/internal/inference"
)

func TestCachedEngineProviderListModelsFromDir(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	mustWriteFile(t, filepath.Join(dir, "alpha.mcf"), "a")
	mustWriteFile(t, filepath.Join(dir, "beta.mcf"), "b")
	mustWriteFile(t, filepath.Join(dir, "notes.txt"), "x")

	provider := NewCachedEngineProvider(EngineProviderConfig{ModelsPath: dir})
	models, err := provider.ListModels()
	if err != nil {
		t.Fatalf("ListModels() error = %v", err)
	}

	want := []string{"alpha", "beta"}
	if !reflect.DeepEqual(models, want) {
		t.Fatalf("ListModels() = %v, want %v", models, want)
	}
}

func TestCachedEngineProviderListModelsIncludesDefaultModel(t *testing.T) {
	t.Parallel()

	dir := t.TempDir()
	provider := NewCachedEngineProvider(EngineProviderConfig{
		DefaultModelPath: "/models/custom-model.mcf",
		ModelsPath:       dir, // Use empty temp dir to isolate from environment
	})
	models, err := provider.ListModels()
	if err != nil {
		t.Fatalf("ListModels() error = %v", err)
	}

	want := []string{"custom-model"}
	if !reflect.DeepEqual(models, want) {
		t.Fatalf("ListModels() = %v, want %v", models, want)
	}
}

func mustWriteFile(t *testing.T, path, content string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}

// fakeEngine records Close so the cache tests can prove an evicted engine is
// released and a busy one is not.
type fakeEngine struct {
	id     string
	closed atomic.Bool
}

func (f *fakeEngine) Generate(context.Context, *inference.Request, inference.StreamFunc) (*inference.Result, error) {
	return &inference.Result{}, nil
}
func (f *fakeEngine) Capabilities() inference.EngineCapabilities {
	return inference.EngineCapabilities{}
}
func (f *fakeEngine) ResetContext() {}
func (f *fakeEngine) Close() error {
	f.closed.Store(true)
	return nil
}

func newFakeProvider(t *testing.T, maxLoaded int, loads *atomic.Int64) (*CachedEngineProvider, map[string]*fakeEngine) {
	t.Helper()
	engines := map[string]*fakeEngine{}
	var mu sync.Mutex
	return NewCachedEngineProvider(EngineProviderConfig{
		MaxLoaded: maxLoaded,
		LoadFunc: func(_ context.Context, path string, _ int) (*inference.LoadResult, error) {
			loads.Add(1)
			mu.Lock()
			defer mu.Unlock()
			// A fresh engine per load, so an evicted then reloaded model is a
			// new object rather than the closed one.
			e := &fakeEngine{id: path}
			engines[path] = e
			return &inference.LoadResult{Engine: e}, nil
		},
	}), engines
}

func TestCachedEngineProviderEvictsLeastRecentlyUsed(t *testing.T) {
	t.Parallel()
	var loads atomic.Int64
	provider, engines := newFakeProvider(t, 1, &loads)

	use := func(path string) {
		t.Helper()
		if err := provider.WithEngine(context.Background(), path, func(inference.Engine, inference.GenDefaults) error { return nil }); err != nil {
			t.Fatalf("WithEngine(%s): %v", path, err)
		}
	}
	use("/models/a.mcf")
	use("/models/b.mcf")

	if !engines["/models/a.mcf"].closed.Load() {
		t.Fatal("the least recently used engine must be closed when the bound is exceeded")
	}
	if engines["/models/b.mcf"].closed.Load() {
		t.Fatal("the engine in use must stay open")
	}
	if got := loads.Load(); got != 2 {
		t.Fatalf("loads = %d, want 2", got)
	}

	// Re-requesting the evicted model loads it again.
	use("/models/a.mcf")
	if got := loads.Load(); got != 3 {
		t.Fatalf("loads after eviction = %d, want 3", got)
	}
	if engines["/models/a.mcf"].closed.Load() {
		t.Fatal("the reloaded engine must be open")
	}
}

func TestCachedEngineProviderKeepsBusyEngine(t *testing.T) {
	t.Parallel()
	var loads atomic.Int64
	provider, engines := newFakeProvider(t, 1, &loads)

	entered := make(chan struct{})
	release := make(chan struct{})
	firstDone := make(chan error, 1)
	go func() {
		firstDone <- provider.WithEngine(context.Background(), "/models/a.mcf", func(inference.Engine, inference.GenDefaults) error {
			close(entered)
			<-release
			return nil
		})
	}()
	<-entered

	// b is loaded while a is serving; a must not be closed under it. Run the
	// load in the background so a blocking eviction fails the test instead of
	// hanging it.
	bDone := make(chan error, 1)
	go func() {
		bDone <- provider.WithEngine(context.Background(), "/models/b.mcf", func(inference.Engine, inference.GenDefaults) error { return nil })
	}()
	select {
	case err := <-bDone:
		if err != nil {
			t.Fatalf("WithEngine(b): %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("loading another model blocked on an engine that is in use")
	}
	if engines["/models/a.mcf"].closed.Load() {
		t.Fatal("an engine with a request in flight must not be closed")
	}

	close(release)
	if err := <-firstDone; err != nil {
		t.Fatalf("first request: %v", err)
	}

	// Once both are idle, loading a third model evicts to the bound.
	if err := provider.WithEngine(context.Background(), "/models/c.mcf", func(inference.Engine, inference.GenDefaults) error { return nil }); err != nil {
		t.Fatalf("WithEngine(c): %v", err)
	}
	if !engines["/models/a.mcf"].closed.Load() || !engines["/models/b.mcf"].closed.Load() {
		t.Fatal("idle engines beyond the bound must be closed")
	}
	if provider.cacheSizeForTest() != 1 {
		t.Fatalf("cache holds %d engines, want 1", provider.cacheSizeForTest())
	}
}

// cacheSizeForTest reports the number of resident engines.
func (p *CachedEngineProvider) cacheSizeForTest() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return len(p.cache)
}

func TestCachedEngineProviderLoadsOncePerModel(t *testing.T) {
	t.Parallel()
	var loads atomic.Int64
	provider, _ := newFakeProvider(t, 0, &loads)

	var wg sync.WaitGroup
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := provider.WithEngine(context.Background(), "/models/a.mcf", func(inference.Engine, inference.GenDefaults) error { return nil }); err != nil {
				t.Errorf("WithEngine: %v", err)
			}
		}()
	}
	wg.Wait()
	if got := loads.Load(); got != 1 {
		t.Fatalf("loads = %d, want 1 for concurrent requests of one model", got)
	}
}
