package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

func findRepoRoot() (string, error) {
	dir, err := os.Getwd()
	if err != nil {
		return "", err
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir, nil
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return "", fmt.Errorf("go.mod not found")
		}
		dir = parent
	}
}

func copyRepo(src, dst string) error {
	// Clean up old temp dirs first
	removeOldTempDirs()

	return filepath.Walk(src, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		rel, err := filepath.Rel(src, path)
		if err != nil {
			return err
		}
		if rel == "." {
			return nil
		}

		// Skip certain directories
		skipDirs := []string{".git", "work", "tmp", "node_modules", ".vscode", ".idea", "dist", "bin", "out"}
		for _, dir := range skipDirs {
			if rel == dir || strings.HasPrefix(rel, dir+string(filepath.Separator)) {
				if info.IsDir() {
					return filepath.SkipDir
				}
				return nil
			}
		}

		destPath := filepath.Join(dst, rel)

		if info.IsDir() {
			return os.MkdirAll(destPath, info.Mode())
		}

		// Skip large binary files
		if info.Size() > 10*1024*1024 { // 10MB
			return nil
		}

		// Use buffered copy
		return copyFile(path, destPath)
	})
}

func loadBaseline(path string) (*Baseline, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var b Baseline
	if err := json.Unmarshal(data, &b); err != nil {
		return nil, err
	}
	normalizeBaseline(&b)
	return &b, nil
}

func saveBaseline(path string, b *Baseline) error {
	normalizeBaseline(b)
	data, _ := json.MarshalIndent(b, "", "  ")
	return os.WriteFile(path, data, 0644)
}

func collectContext(root, targetRel string) (map[string]string, error) {
	ctx := make(map[string]string)

	readFile := func(path string) (string, error) {
		f, err := os.Open(path)
		if err != nil {
			return "", err
		}
		defer f.Close()

		// Use a buffer to read efficiently
		reader := bufio.NewReader(f)
		var builder strings.Builder
		var totalBytes int64
		const maxBytes = 80000

		for {
			line, err := reader.ReadString('\n')
			if err != nil && err != io.EOF {
				return "", err
			}

			totalBytes += int64(len(line))
			if totalBytes > maxBytes {
				builder.WriteString("\n\n/* ... truncated ... */\n")
				break
			}

			builder.WriteString(line)

			if err == io.EOF {
				break
			}
		}

		return builder.String(), nil
	}

	planPath := filepath.Join(root, planFileName)
	if _, err := os.Stat(planPath); err == nil {
		if c, err := readFile(planPath); err == nil {
			ctx[planFileName] = c
		}
	}

	targetAbs := filepath.Join(root, targetRel)
	if c, err := readFile(targetAbs); err == nil {
		ctx[targetRel] = c
	} else {
		return nil, err
	}

	targetDir := filepath.Dir(targetAbs)
	entries, _ := os.ReadDir(targetDir)

	count := 0
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".go") {
			continue
		}

		relPath := filepath.Join(filepath.Dir(targetRel), e.Name())
		if relPath == targetRel {
			continue
		}

		isTest := strings.HasSuffix(e.Name(), "_test.go")

		if isTest || count < 3 {
			filePath := filepath.Join(targetDir, e.Name())
			if c, err := readFile(filePath); err == nil {
				ctx[relPath] = c
				if !isTest {
					count++
				}
			}
		}
	}

	return ctx, nil
}
