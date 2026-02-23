package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// Tool defines an interface for tools that the agent can call.
type Tool interface {
	Name() string
	Description() string
	Parameters() any
	Execute(ctx context.Context, args json.RawMessage) (string, error)
}

// Registry manages available tools.
type Registry struct {
	tools map[string]Tool
}

func NewRegistry() *Registry {
	return &Registry{tools: make(map[string]Tool)}
}

func (r *Registry) Register(t Tool) {
	r.tools[t.Name()] = t
}

func (r *Registry) Get(name string) Tool {
	return r.tools[name]
}

func (r *Registry) List() []Tool {
	out := make([]Tool, 0, len(r.tools))
	for _, t := range r.tools {
		out = append(out, t)
	}
	return out
}

// Basic Tools Implementation

type WriteFileTool struct {
	Workspace string
}

func (t *WriteFileTool) Name() string        { return "write_file" }
func (t *WriteFileTool) Description() string { return "Write content to a file in the workspace" }
func (t *WriteFileTool) Parameters() any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"filename": map[string]any{"type": "string", "description": "The name of the file to write to"},
			"content":  map[string]any{"type": "string", "description": "The content to write to the file"},
		},
		"required": []string{"filename", "content"},
	}
}

func (t *WriteFileTool) Execute(ctx context.Context, args json.RawMessage) (string, error) {
	var params struct {
		Filename string `json:"filename"`
		Content  string `json:"content"`
	}
	if err := json.Unmarshal(args, &params); err != nil {
		return "", err
	}
	path := filepath.Join(t.Workspace, params.Filename)
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return "", err
	}
	if err := os.WriteFile(path, []byte(params.Content), 0644); err != nil {
		return "", err
	}
	return fmt.Sprintf("Success. Wrote %d bytes to %s.", len(params.Content), params.Filename), nil
}

type ReadFileTool struct {
	Workspace string
}

func (t *ReadFileTool) Name() string        { return "read_file" }
func (t *ReadFileTool) Description() string { return "Read the content of a file in the workspace" }
func (t *ReadFileTool) Parameters() any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"filename": map[string]any{"type": "string", "description": "The name of the file to read"},
		},
		"required": []string{"filename"},
	}
}

func (t *ReadFileTool) Execute(ctx context.Context, args json.RawMessage) (string, error) {
	var params struct {
		Filename string `json:"filename"`
	}
	if err := json.Unmarshal(args, &params); err != nil {
		return "", err
	}
	path := filepath.Join(t.Workspace, params.Filename)
	content, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	return string(content), nil
}

type ExecuteShellTool struct {
	Workspace string
}

func (t *ExecuteShellTool) Name() string        { return "execute_shell" }
func (t *ExecuteShellTool) Description() string { return "Execute a shell command in the workspace" }
func (t *ExecuteShellTool) Parameters() any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"command": map[string]any{"type": "string", "description": "The shell command to execute"},
		},
		"required": []string{"command"},
	}
}

func (t *ExecuteShellTool) Execute(ctx context.Context, args json.RawMessage) (string, error) {
	var params struct {
		Command string `json:"command"`
	}
	if err := json.Unmarshal(args, &params); err != nil {
		return "", err
	}
	cmd := exec.CommandContext(ctx, "bash", "-c", params.Command)
	cmd.Dir = t.Workspace
	output, err := cmd.CombinedOutput()
	if err != nil {
		return string(output), fmt.Errorf("command failed: %w", err)
	}
	return string(output), nil
}

type ListDirectoryTool struct {
	Workspace string
}

func (t *ListDirectoryTool) Name() string        { return "list_directory" }
func (t *ListDirectoryTool) Description() string { return "List files and folders in a directory" }
func (t *ListDirectoryTool) Parameters() any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"path": map[string]any{"type": "string", "description": "The directory path to list (relative to workspace)"},
		},
		"required": []string{"path"},
	}
}

func (t *ListDirectoryTool) Execute(ctx context.Context, args json.RawMessage) (string, error) {
	var params struct {
		Path string `json:"path"`
	}
	if err := json.Unmarshal(args, &params); err != nil {
		return "", err
	}
	fullPath := filepath.Join(t.Workspace, params.Path)
	entries, err := os.ReadDir(fullPath)
	if err != nil {
		return "", err
	}
	var b strings.Builder
	for _, e := range entries {
		if e.IsDir() {
			fmt.Fprintf(&b, "[DIR] %s\n", e.Name())
		} else {
			fmt.Fprintf(&b, "      %s\n", e.Name())
		}
	}
	return b.String(), nil
}
