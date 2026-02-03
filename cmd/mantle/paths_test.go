package main

import (
	"bytes"
	"io"
	"os"
	"path/filepath"
	"testing"
)

func TestResolvePackOut(t *testing.T) {
	t.Run("explicit output wins", func(t *testing.T) {
		inDir := t.TempDir()
		outPath := filepath.Join(t.TempDir(), "nested", "model.mcf")

		got, defaulted, err := resolvePackOut(inDir, outPath)
		if err != nil {
			t.Fatalf("resolvePackOut returned error: %v", err)
		}
		if defaulted {
			t.Fatalf("expected explicit output to not be defaulted")
		}
		if got != filepath.Clean(outPath) {
			t.Fatalf("unexpected output path: got %q want %q", got, filepath.Clean(outPath))
		}
		if _, err := os.Stat(filepath.Dir(got)); err != nil {
			t.Fatalf("expected output directory to exist: %v", err)
		}
	})

	t.Run("env output dir overrides default", func(t *testing.T) {
		envDir := filepath.Join(t.TempDir(), "pack-out")
		t.Setenv(envMantlePackOutDir, envDir)

		inDir := filepath.Join(t.TempDir(), "ModelA")
		got, defaulted, err := resolvePackOut(inDir, "")
		if err != nil {
			t.Fatalf("resolvePackOut returned error: %v", err)
		}
		if !defaulted {
			t.Fatalf("expected output to be defaulted")
		}
		want := filepath.Join(envDir, "ModelA.mcf")
		if got != want {
			t.Fatalf("unexpected output path: got %q want %q", got, want)
		}
	})

	t.Run("default output dir is ./out", func(t *testing.T) {
		wd, err := os.Getwd()
		if err != nil {
			t.Fatalf("getwd: %v", err)
		}
		tmp := t.TempDir()
		if err := os.Chdir(tmp); err != nil {
			t.Fatalf("chdir: %v", err)
		}
		defer func() {
			_ = os.Chdir(wd)
		}()
		t.Setenv(envMantlePackOutDir, "")

		inDir := filepath.Join(tmp, "ModelB")
		got, defaulted, err := resolvePackOut(inDir, "")
		if err != nil {
			t.Fatalf("resolvePackOut returned error: %v", err)
		}
		if !defaulted {
			t.Fatalf("expected output to be defaulted")
		}
		want := filepath.Join(".", "out", "ModelB.mcf")
		if got != want {
			t.Fatalf("unexpected output path: got %q want %q", got, want)
		}
	})
}

func TestDiscoverMCFModelsSorted(t *testing.T) {
	dir := t.TempDir()
	files := []string{"b.mcf", "a.mcf", "ignore.txt"}
	for _, name := range files {
		path := filepath.Join(dir, name)
		if err := os.WriteFile(path, []byte("x"), 0o644); err != nil {
			t.Fatalf("write file %s: %v", name, err)
		}
	}

	got, err := discoverMCFModels(dir)
	if err != nil {
		t.Fatalf("discoverMCFModels returned error: %v", err)
	}
	want := []string{
		filepath.Join(dir, "a.mcf"),
		filepath.Join(dir, "b.mcf"),
	}
	if len(got) != len(want) {
		t.Fatalf("unexpected model count: got %d want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("unexpected ordering at %d: got %q want %q", i, got[i], want[i])
		}
	}
}

func TestResolveRunModelPath(t *testing.T) {
	t.Run("model flag bypasses env", func(t *testing.T) {
		t.Setenv(envMantleModelsDir, "")
		got, err := resolveRunModelPath("/tmp/model.mcf", "", bytes.NewBuffer(nil), io.Discard)
		if err != nil {
			t.Fatalf("resolveRunModelPath returned error: %v", err)
		}
		if got != filepath.Clean("/tmp/model.mcf") {
			t.Fatalf("unexpected model path: got %q", got)
		}
	})

	t.Run("single model selects automatically", func(t *testing.T) {
		dir := t.TempDir()
		only := filepath.Join(dir, "only.mcf")
		if err := os.WriteFile(only, []byte("x"), 0o644); err != nil {
			t.Fatalf("write model: %v", err)
		}
		t.Setenv(envMantleModelsDir, dir)

		prevTTY := stdinIsTTY
		stdinIsTTY = func() bool { return false }
		defer func() { stdinIsTTY = prevTTY }()

		got, err := resolveRunModelPath("", "", bytes.NewBuffer(nil), io.Discard)
		if err != nil {
			t.Fatalf("resolveRunModelPath returned error: %v", err)
		}
		if got != only {
			t.Fatalf("unexpected model path: got %q want %q", got, only)
		}
	})

	t.Run("multiple models requires tty", func(t *testing.T) {
		dir := t.TempDir()
		for _, name := range []string{"a.mcf", "b.mcf"} {
			if err := os.WriteFile(filepath.Join(dir, name), []byte("x"), 0o644); err != nil {
				t.Fatalf("write model %s: %v", name, err)
			}
		}
		t.Setenv(envMantleModelsDir, dir)

		prevTTY := stdinIsTTY
		stdinIsTTY = func() bool { return false }
		defer func() { stdinIsTTY = prevTTY }()

		if _, err := resolveRunModelPath("", "", bytes.NewBuffer(nil), io.Discard); err == nil {
			t.Fatalf("expected error when multiple models and stdin is not a tty")
		}
	})

	t.Run("interactive selection chooses sorted index", func(t *testing.T) {
		dir := t.TempDir()
		a := filepath.Join(dir, "a.mcf")
		b := filepath.Join(dir, "b.mcf")
		if err := os.WriteFile(b, []byte("x"), 0o644); err != nil {
			t.Fatalf("write model b: %v", err)
		}
		if err := os.WriteFile(a, []byte("x"), 0o644); err != nil {
			t.Fatalf("write model a: %v", err)
		}
		t.Setenv(envMantleModelsDir, dir)

		prevTTY := stdinIsTTY
		stdinIsTTY = func() bool { return true }
		defer func() { stdinIsTTY = prevTTY }()

		got, err := resolveRunModelPath("", "", bytes.NewBufferString("2\n"), io.Discard)
		if err != nil {
			t.Fatalf("resolveRunModelPath returned error: %v", err)
		}
		if got != b {
			t.Fatalf("unexpected model selection: got %q want %q", got, b)
		}
	})
}
