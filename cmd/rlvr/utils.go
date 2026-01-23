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
	defer source.Close()

	// Create destination directory if it doesn't exist
	if err := os.MkdirAll(filepath.Dir(dst), 0755); err != nil {
		return err
	}

	dest, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer dest.Close()

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
				os.RemoveAll(path)
			}
		}
	}
}

func initGitRepo(dir string) error {
	if _, err := os.Stat(filepath.Join(dir, ".git")); err == nil {
		return nil // Already a git repo
	}

	// Initialize git repo
	if err := runCmd(dir, "git", "init"); err != nil {
		return err
	}
	// Configure user for commits
	_ = runCmd(dir, "git", "config", "user.email", "rlvr@bot")
	_ = runCmd(dir, "git", "config", "user.name", "rlvr")

	// Add all files and commit
	if err := runCmd(dir, "git", "add", "-A"); err != nil {
		return err
	}
	return runCmd(dir, "git", "commit", "-m", "init", "--allow-empty")
}
