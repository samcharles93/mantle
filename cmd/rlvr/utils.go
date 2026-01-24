package main

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"time"
)

func runCmd(dir, command string, args ...string) error {
	cmd := exec.Command(command, args...)
	cmd.Dir = dir
	if out, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("%s failed: %v\nOutput: %s", command, err, string(out))
	}
	return nil
}

func copyFile(src, dst string) error {
	source, err := os.Open(src)
	if err != nil {
		return err
	}
	defer func() { _ = source.Close() }()

	// Create destination directory if it doesn't exist
	if err := os.MkdirAll(filepath.Dir(dst), 0755); err != nil {
		return err
	}

	dest, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer func() { _ = dest.Close() }()

	// Use buffered copy for efficiency
	buf := bufio.NewReader(source)
	_, err = io.Copy(dest, buf)
	return err
}

func removeOldTempDirs() {
	tmpDir := os.TempDir()
	entries, err := os.ReadDir(tmpDir)
	if err != nil {
		return
	}

	for _, entry := range entries {
		if entry.IsDir() && len(entry.Name()) > 8 && entry.Name()[:8] == "rlvr_cand_" {
			path := filepath.Join(tmpDir, entry.Name())
			info, err := entry.Info()
			if err != nil {
				continue
			}
			// Remove temp dirs older than 1 hour
			if time.Since(info.ModTime()) > time.Hour {
				_ = os.RemoveAll(path)
			}
		}
	}
}
