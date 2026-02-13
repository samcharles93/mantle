package api

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
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

	provider := NewCachedEngineProvider(EngineProviderConfig{DefaultModelPath: "/models/custom-model.mcf"})
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
