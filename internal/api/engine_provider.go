package api

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"

	"github.com/samcharles93/mantle/internal/inference"
)

type EngineProvider interface {
	WithEngine(ctx context.Context, modelID string, fn func(engine inference.Engine, defaults inference.GenDefaults) error) error
}

type EngineProviderConfig struct {
	DefaultModelPath string
	ModelsPath       string
	MaxContext       int
	Loader           inference.Loader

	// MaxLoaded bounds how many model engines stay resident. When a load
	// exceeds the bound, the least recently used engine with no request in
	// flight is closed. A value <= 0 leaves the cache unbounded.
	MaxLoaded int

	// LoadFunc overrides Loader.Load. Tests use it to supply fake engines;
	// production leaves it nil.
	LoadFunc func(ctx context.Context, path string, maxContext int) (*inference.LoadResult, error)
}

type CachedEngineProvider struct {
	cfg     EngineProviderConfig
	mu      sync.Mutex
	cache   map[string]*engineEntry
	loading map[string]*loadGate
	seq     int64
}

type loadGate struct {
	mu    sync.Mutex
	done  bool
	entry *engineEntry
	err   error
}

type engineEntry struct {
	engine   inference.Engine
	defaults inference.GenDefaults
	mu       sync.Mutex

	// users and lastUsed are guarded by CachedEngineProvider.mu. An entry may
	// only be evicted while users is zero, so no request can be handed an
	// engine that has been closed.
	users    int
	lastUsed int64
}

const envMantleModelsDir = "MANTLE_MODELS_DIR"

func NewCachedEngineProvider(cfg EngineProviderConfig) *CachedEngineProvider {
	return &CachedEngineProvider{
		cfg:     cfg,
		cache:   make(map[string]*engineEntry),
		loading: make(map[string]*loadGate),
	}
}

func (p *CachedEngineProvider) WithEngine(ctx context.Context, modelID string, fn func(engine inference.Engine, defaults inference.GenDefaults) error) error {
	path, err := p.resolveModelPath(modelID)
	if err != nil {
		return err
	}
	entry, err := p.acquire(ctx, path)
	if err != nil {
		return err
	}
	defer p.release(entry)

	entry.mu.Lock()
	defer entry.mu.Unlock()
	if err := ctx.Err(); err != nil {
		return err
	}
	return fn(entry.engine, entry.defaults)
}

// acquire returns the cached engine for path, loading it on a miss, and pins
// it so eviction cannot close it under the caller. The caller must release it.
func (p *CachedEngineProvider) acquire(ctx context.Context, path string) (*engineEntry, error) {
	// The cache check and the gate registration share one critical section so a
	// request that misses cannot start a second load for a model another
	// request is caching at the same moment.
	p.mu.Lock()
	if entry, ok := p.cache[path]; ok {
		p.pinLocked(entry)
		p.mu.Unlock()
		return entry, nil
	}
	gate, loading := p.loading[path]
	if !loading {
		gate = &loadGate{}
		p.loading[path] = gate
	}
	p.mu.Unlock()

	gate.mu.Lock()
	if !gate.done {
		load := p.cfg.LoadFunc
		if load == nil {
			load = p.cfg.Loader.Load
		}
		result, err := load(ctx, path, p.cfg.MaxContext)
		if err != nil {
			gate.err = err
		} else {
			gate.entry = &engineEntry{
				engine:   result.Engine,
				defaults: result.GenerationDefaults,
			}
		}
		gate.done = true
	}
	gate.mu.Unlock()

	p.mu.Lock()
	if gate.err != nil {
		delete(p.loading, path)
		p.mu.Unlock()
		return nil, gate.err
	}
	entry := gate.entry
	p.cache[path] = entry
	delete(p.loading, path)
	p.pinLocked(entry)
	victims := p.evictLocked(path)
	p.mu.Unlock()

	closeEntries(victims)
	return entry, nil
}

// pinLocked marks entry as in use. p.mu must be held.
func (p *CachedEngineProvider) pinLocked(entry *engineEntry) {
	entry.users++
	p.seq++
	entry.lastUsed = p.seq
}

func (p *CachedEngineProvider) release(entry *engineEntry) {
	p.mu.Lock()
	entry.users--
	p.mu.Unlock()
}

// evictLocked removes least recently used entries beyond MaxLoaded, skipping
// the entry a request is currently acquiring and any engine still in use.
// p.mu must be held; the returned entries are closed by the caller because
// closing touches files and must not hold the provider lock.
func (p *CachedEngineProvider) evictLocked(keep string) []*engineEntry {
	limit := p.cfg.MaxLoaded
	if limit <= 0 {
		return nil
	}
	var victims []*engineEntry
	for len(p.cache) > limit {
		victimPath := ""
		var oldest *engineEntry
		for path, entry := range p.cache {
			if path == keep || entry.users > 0 {
				continue
			}
			if oldest == nil || entry.lastUsed < oldest.lastUsed {
				oldest, victimPath = entry, path
			}
		}
		if oldest == nil {
			// Every other engine is busy; keep them rather than block the load.
			break
		}
		delete(p.cache, victimPath)
		victims = append(victims, oldest)
	}
	return victims
}

func closeEntries(entries []*engineEntry) {
	for _, entry := range entries {
		if entry.engine == nil {
			continue
		}
		entry.mu.Lock()
		_ = entry.engine.Close() // a failed close only leaks until process exit
		entry.mu.Unlock()
	}
}

func (p *CachedEngineProvider) resolveModelPath(modelID string) (string, error) {
	modelID = strings.TrimSpace(modelID)
	if modelID != "" {
		if looksLikePath(modelID) {
			return filepath.Clean(modelID), nil
		}
		modelsDir := p.modelsDir()
		if modelsDir == "" {
			return "", fmt.Errorf("models-path is required to resolve model %q", modelID)
		}
		if resolved := resolveInDir(modelsDir, modelID); resolved != "" {
			return resolved, nil
		}
		return "", fmt.Errorf("model %q not found in %s", modelID, modelsDir)
	}

	if p.cfg.DefaultModelPath != "" {
		return filepath.Clean(p.cfg.DefaultModelPath), nil
	}
	modelsDir := p.modelsDir()
	if modelsDir == "" {
		return "", fmt.Errorf("model is required")
	}
	models, err := discoverModels(modelsDir)
	if err != nil {
		return "", err
	}
	if len(models) == 1 {
		return models[0], nil
	}
	if len(models) == 0 {
		return "", fmt.Errorf("no .mcf models found in %s", modelsDir)
	}
	return "", fmt.Errorf("multiple models found in %s; specify model", modelsDir)
}

func (p *CachedEngineProvider) modelsDir() string {
	if strings.TrimSpace(p.cfg.ModelsPath) != "" {
		return strings.TrimSpace(p.cfg.ModelsPath)
	}
	return strings.TrimSpace(os.Getenv(envMantleModelsDir))
}

func (p *CachedEngineProvider) ListModels() ([]string, error) {
	seen := map[string]struct{}{}
	models := make([]string, 0)

	add := func(path string) {
		id := modelIDFromPath(path)
		if id == "" {
			return
		}
		if _, ok := seen[id]; ok {
			return
		}
		seen[id] = struct{}{}
		models = append(models, id)
	}

	if strings.TrimSpace(p.cfg.DefaultModelPath) != "" {
		add(p.cfg.DefaultModelPath)
	}

	if dir := p.modelsDir(); dir != "" {
		discovered, err := discoverModels(dir)
		if err != nil {
			return nil, err
		}
		for _, path := range discovered {
			add(path)
		}
	}

	sort.Strings(models)
	return models, nil
}

func looksLikePath(v string) bool {
	if strings.Contains(v, string(filepath.Separator)) {
		return true
	}
	return strings.HasSuffix(strings.ToLower(v), ".mcf")
}

func resolveInDir(dir, name string) string {
	if dir == "" {
		return ""
	}
	cand := filepath.Join(dir, name)
	if fileExists(cand) {
		return cand
	}
	if !strings.HasSuffix(strings.ToLower(name), ".mcf") {
		cand = filepath.Join(dir, name+".mcf")
		if fileExists(cand) {
			return cand
		}
	}
	return ""
}

func discoverModels(dir string) ([]string, error) {
	st, err := os.Stat(dir)
	if err != nil {
		return nil, err
	}
	if !st.IsDir() {
		return nil, fmt.Errorf("models path is not a directory: %s", dir)
	}
	ents, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	models := make([]string, 0, len(ents))
	for _, e := range ents {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		if !strings.HasSuffix(strings.ToLower(name), ".mcf") {
			continue
		}
		models = append(models, filepath.Join(dir, name))
	}
	return models, nil
}

func fileExists(path string) bool {
	if path == "" {
		return false
	}
	_, err := os.Stat(path)
	return err == nil
}

func modelIDFromPath(path string) string {
	base := filepath.Base(strings.TrimSpace(path))
	if base == "." || base == "" {
		return ""
	}
	if strings.HasSuffix(strings.ToLower(base), ".mcf") {
		return base[:len(base)-4]
	}
	return base
}
