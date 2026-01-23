package main

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

func SmartApply(repoRoot, targetRel, llmOutput string) (string, error) {
	clean := cleanOutput(llmOutput)
	fullPath := filepath.Join(repoRoot, targetRel)

	// Handle diff format
	if isDiff(clean) {
		return clean, applyDiff(repoRoot, clean)
	}

	// Handle Go source code
	if isGoSource(clean) {
		oldContent, err := os.ReadFile(fullPath)
		if err != nil && !os.IsNotExist(err) {
			return "", fmt.Errorf("read original: %w", err)
		}

		if err := writeFile(fullPath, clean); err != nil {
			return "", fmt.Errorf("write new: %w", err)
		}

		diff, err := gitDiff(repoRoot, targetRel)
		if err != nil {
			return "", fmt.Errorf("generate diff: %w", err)
		}

		if diff == "" {
			// No changes, restore original if it existed
			if oldContent != nil {
				os.WriteFile(fullPath, oldContent, 0644)
			}
			return "", errors.New("no changes detected")
		}

		return diff, nil
	}

	return "", errors.New("unrecognized output format")
}

func cleanOutput(output string) string {
	clean := strings.TrimSpace(output)
	for _, marker := range []string{"```go", "```diff", "```"} {
		clean = strings.ReplaceAll(clean, marker, "")
	}
	return strings.TrimSpace(clean)
}

func isDiff(output string) bool {
	// Check for standard unified diff markers
	if strings.Contains(output, "--- a/") && strings.Contains(output, "+++ b/") {
		return true
	}

	// Also check for @@ markers which indicate diff hunks
	if strings.Contains(output, "@@") && strings.Contains(output, "-") && strings.Contains(output, "+") {
		return true
	}

	return false
}

func isGoSource(output string) bool {
	return strings.HasPrefix(output, "package ")
}

func applyDiff(repoRoot, patchText string) error {
	// First try strict apply
	cmd := exec.Command("git", "apply", "-")
	cmd.Dir = repoRoot
	cmd.Stdin = strings.NewReader(patchText)

	if _, err := cmd.CombinedOutput(); err != nil {
		cmd = exec.Command("git", "apply", "--ignore-space-change", "--ignore-whitespace", "-")
		cmd.Dir = repoRoot
		cmd.Stdin = strings.NewReader(patchText)

		if output, err := cmd.CombinedOutput(); err != nil {
			return fmt.Errorf("git apply failed: %s", string(output))
		}
	}
	return nil
}

func writeFile(path, content string) error {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}
	return os.WriteFile(path, []byte(content), 0644)
}

func gitDiff(repoRoot, targetRel string) (string, error) {
	cmd := exec.Command("git", "diff", targetRel)
	cmd.Dir = repoRoot
	output, err := cmd.Output()
	if err != nil {
		// Check if git returned non-zero because there's no diff
		if exitErr, ok := err.(*exec.ExitError); ok && exitErr.ExitCode() == 1 {
			return "", nil // No changes
		}
		return "", err
	}
	return string(bytes.TrimSpace(output)), nil
}
