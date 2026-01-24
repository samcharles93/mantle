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

	// Handle apply_patch format
	if isApplyPatch(clean) {
		if err := applyPatchFormat(repoRoot, clean); err != nil {
			return "", err
		}
		return clean, nil
	}

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
				if err := os.WriteFile(fullPath, oldContent, 0644); err != nil {
					return "", fmt.Errorf("restore original: %w", err)
				}
			}
			return "", errors.New("no changes detected")
		}

		return diff, nil
	}

	return "", errors.New("unrecognized output format")
}

func cleanOutput(output string) string {
	clean := strings.TrimSpace(output)
	for _, marker := range []string{"```go", "```diff", "```patch", "```"} {
		clean = strings.ReplaceAll(clean, marker, "")
	}
	return strings.TrimSpace(clean)
}

func isApplyPatch(output string) bool {
	return strings.Contains(output, "*** Begin Patch") && strings.Contains(output, "*** End Patch")
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

func ApplyDiffToRepo(dir, patchText string) error {
	if isApplyPatch(patchText) {
		return applyPatchFormat(dir, patchText)
	}
	return applyDiff(dir, patchText)
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

type patchOpKind int

const (
	patchAdd patchOpKind = iota
	patchDelete
	patchUpdate
)

type patchOp struct {
	kind     patchOpKind
	path     string
	moveTo   string
	addLines []string
	hunks    []patchHunk
}

type patchHunk struct {
	oldLines []string
	newLines []string
	isEOF    bool
}

func applyPatchFormat(repoRoot, patch string) error {
	ops, err := parseApplyPatch(patch)
	if err != nil {
		return err
	}
	for _, op := range ops {
		switch op.kind {
		case patchAdd:
			if err := applyAdd(repoRoot, op.path, op.addLines); err != nil {
				return err
			}
		case patchDelete:
			if err := applyDelete(repoRoot, op.path); err != nil {
				return err
			}
		case patchUpdate:
			if err := applyUpdate(repoRoot, op.path, op.moveTo, op.hunks); err != nil {
				return err
			}
		}
	}
	return nil
}

func parseApplyPatch(patch string) ([]patchOp, error) {
	lines := strings.Split(strings.ReplaceAll(patch, "\r\n", "\n"), "\n")
	if len(lines) < 2 {
		return nil, fmt.Errorf("invalid apply_patch: too few lines")
	}
	if strings.TrimSpace(lines[0]) != "*** Begin Patch" {
		return nil, fmt.Errorf("invalid apply_patch: missing Begin Patch")
	}
	endIdx := -1
	for i := len(lines) - 1; i >= 0; i-- {
		if strings.TrimSpace(lines[i]) == "*** End Patch" {
			endIdx = i
			break
		}
	}
	if endIdx == -1 || endIdx <= 0 {
		return nil, fmt.Errorf("invalid apply_patch: missing End Patch")
	}

	var ops []patchOp
	i := 1
	for i < endIdx {
		line := strings.TrimSpace(lines[i])
		if line == "" {
			i++
			continue
		}
		switch {
		case strings.HasPrefix(line, "*** Add File: "):
			path := strings.TrimSpace(strings.TrimPrefix(line, "*** Add File: "))
			if path == "" {
				return nil, fmt.Errorf("add file missing path")
			}
			i++
			var addLines []string
			for i < endIdx {
				l := lines[i]
				trim := strings.TrimSpace(l)
				if strings.HasPrefix(trim, "*** ") || strings.HasPrefix(trim, "@@") {
					break
				}
				if strings.HasPrefix(l, "+") {
					addLines = append(addLines, l[1:])
				} else if strings.TrimSpace(l) != "" {
					return nil, fmt.Errorf("add file line missing '+' prefix")
				}
				i++
			}
			ops = append(ops, patchOp{kind: patchAdd, path: path, addLines: addLines})
		case strings.HasPrefix(line, "*** Delete File: "):
			path := strings.TrimSpace(strings.TrimPrefix(line, "*** Delete File: "))
			if path == "" {
				return nil, fmt.Errorf("delete file missing path")
			}
			i++
			ops = append(ops, patchOp{kind: patchDelete, path: path})
		case strings.HasPrefix(line, "*** Update File: "):
			path := strings.TrimSpace(strings.TrimPrefix(line, "*** Update File: "))
			if path == "" {
				return nil, fmt.Errorf("update file missing path")
			}
			op := patchOp{kind: patchUpdate, path: path}
			i++
			if i < endIdx && strings.HasPrefix(strings.TrimSpace(lines[i]), "*** Move to: ") {
				op.moveTo = strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(lines[i]), "*** Move to: "))
				i++
			}
			for i < endIdx {
				l := strings.TrimSpace(lines[i])
				if strings.HasPrefix(l, "*** ") {
					break
				}
				if strings.HasPrefix(l, "@@") {
					i++
					h, ni, err := parseHunk(lines, i, endIdx)
					if err != nil {
						return nil, err
					}
					op.hunks = append(op.hunks, h)
					i = ni
					continue
				}
				i++
			}
			if len(op.hunks) == 0 {
				return nil, fmt.Errorf("update file missing hunks for %s", op.path)
			}
			ops = append(ops, op)
		default:
			return nil, fmt.Errorf("invalid apply_patch line: %s", line)
		}
	}
	return ops, nil
}

func parseHunk(lines []string, start, end int) (patchHunk, int, error) {
	var h patchHunk
	i := start
	for i < end {
		line := lines[i]
		trim := strings.TrimSpace(line)
		if strings.HasPrefix(trim, "*** ") || strings.HasPrefix(trim, "@@") {
			break
		}
		if trim == "*** End of File" {
			h.isEOF = true
			i++
			break
		}
		if len(line) == 0 {
			h.oldLines = append(h.oldLines, "")
			h.newLines = append(h.newLines, "")
			i++
			continue
		}
		switch line[0] {
		case ' ':
			h.oldLines = append(h.oldLines, line[1:])
			h.newLines = append(h.newLines, line[1:])
		case '-':
			h.oldLines = append(h.oldLines, line[1:])
		case '+':
			h.newLines = append(h.newLines, line[1:])
		default:
			return patchHunk{}, 0, fmt.Errorf("invalid hunk line: %q", line)
		}
		i++
	}
	return h, i, nil
}

func applyAdd(repoRoot, path string, lines []string) error {
	full := filepath.Join(repoRoot, path)
	if err := os.MkdirAll(filepath.Dir(full), 0755); err != nil {
		return err
	}
	content := strings.Join(lines, "\n")
	if !strings.HasSuffix(content, "\n") {
		content += "\n"
	}
	return os.WriteFile(full, []byte(content), 0644)
}

func applyDelete(repoRoot, path string) error {
	full := filepath.Join(repoRoot, path)
	return os.Remove(full)
}

func applyUpdate(repoRoot, path, moveTo string, hunks []patchHunk) error {
	full := filepath.Join(repoRoot, path)
	data, err := os.ReadFile(full)
	if err != nil {
		return err
	}
	text := string(data)
	hasTrailing := strings.HasSuffix(text, "\n")
	lines := strings.Split(text, "\n")
	if hasTrailing && len(lines) > 0 && lines[len(lines)-1] == "" {
		lines = lines[:len(lines)-1]
	}

	pos := 0
	for _, h := range hunks {
		if len(h.oldLines) == 0 && len(h.newLines) == 0 {
			continue
		}
		var idx int
		if h.isEOF {
			idx = len(lines) - len(h.oldLines)
			if idx < 0 || !sliceEqual(lines[idx:], h.oldLines) {
				return fmt.Errorf("hunk failed to match at EOF")
			}
		} else {
			idx = findSequence(lines, h.oldLines, pos)
			if idx < 0 {
				return fmt.Errorf("hunk failed to match in %s", path)
			}
		}
		lines = applyReplace(lines, idx, len(h.oldLines), h.newLines)
		pos = idx + len(h.newLines)
	}

	if hasTrailing {
		lines = append(lines, "")
	}
	out := strings.Join(lines, "\n")

	dest := full
	if moveTo != "" {
		dest = filepath.Join(repoRoot, moveTo)
		if err := os.MkdirAll(filepath.Dir(dest), 0755); err != nil {
			return err
		}
	}
	if err := os.WriteFile(dest, []byte(out), 0644); err != nil {
		return err
	}
	if moveTo != "" && dest != full {
		return os.Remove(full)
	}
	return nil
}

func findSequence(lines, pattern []string, start int) int {
	if len(pattern) == 0 {
		return start
	}
	for i := start; i+len(pattern) <= len(lines); i++ {
		if sliceEqual(lines[i:i+len(pattern)], pattern) {
			return i
		}
	}
	return -1
}

func sliceEqual(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func applyReplace(lines []string, start, oldLen int, newLines []string) []string {
	out := make([]string, 0, len(lines)-oldLen+len(newLines))
	out = append(out, lines[:start]...)
	out = append(out, newLines...)
	out = append(out, lines[start+oldLen:]...)
	return out
}
