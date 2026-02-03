package api

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
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
}

type CachedEngineProvider struct {
	cfg   EngineProviderConfig
	mu    sync.Mutex
	cache map[string]*engineEntry
}

type engineEntry struct {
	engine   inference.Engine
	defaults inference.GenDefaults
	mu       sync.Mutex
}

const envMantleModelsDir = "MANTLE_MODELS_DIR"

func NewCachedEngineProvider(cfg EngineProviderConfig) *CachedEngineProvider {
	return &CachedEngineProvider{
		cfg:   cfg,
		cache: make(map[string]*engineEntry),
	}
}

func (p *CachedEngineProvider) WithEngine(ctx context.Context, modelID string, fn func(engine inference.Engine, defaults inference.GenDefaults) error) error {
	path, err := p.resolveModelPath(modelID)
	if err != nil {
		return err
	}
	entry, err := p.getOrLoad(path)
	if err != nil {
		return err
	}

	entry.mu.Lock()
	defer entry.mu.Unlock()
	if err := ctx.Err(); err != nil {
		return err
	}
	return fn(entry.engine, entry.defaults)
}

func (p *CachedEngineProvider) getOrLoad(path string) (*engineEntry, error) {
	p.mu.Lock()
	entry, ok := p.cache[path]
	p.mu.Unlock()
	if ok {
		return entry, nil
	}

	result, err := p.cfg.Loader.Load(path, p.cfg.MaxContext)
	if err != nil {
		return nil, err
	}
	newEntry := &engineEntry{
		engine:   result.Engine,
		defaults: result.GenerationDefaults,
	}

	p.mu.Lock()
	defer p.mu.Unlock()
	if existing, ok := p.cache[path]; ok {
		return existing, nil
	}
	p.cache[path] = newEntry
	return newEntry, nil
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
