package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"text/template"
)

// JSONToolDef is the on-disk schema for a user-defined tool.
//
// Example file:
//
//	{
//	  "name": "grep_code",
//	  "description": "Search for patterns in source files",
//	  "parameters": {
//	    "type": "object",
//	    "properties": {
//	      "pattern": {"type": "string", "description": "Regex pattern"},
//	      "path":    {"type": "string", "description": "Directory to search (default: .)"}
//	    },
//	    "required": ["pattern"]
//	  },
//	  "command": "grep -rn '{{.pattern}}' {{if .path}}{{.path}}{{else}}.{{end}} 2>/dev/null | head -50"
//	}
type JSONToolDef struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters"`
	Command     string          `json:"command"`
}

// JSONTool implements Tool by rendering a Go text/template command and running
// it via bash -c inside the agent workspace.
type JSONTool struct {
	def       JSONToolDef
	workspace string
	cmdTmpl   *template.Template
}

func (t *JSONTool) Name() string        { return t.def.Name }
func (t *JSONTool) Description() string { return t.def.Description }

func (t *JSONTool) Parameters() any {
	var p any
	if err := json.Unmarshal(t.def.Parameters, &p); err != nil {
		return map[string]any{"type": "object", "properties": map[string]any{}}
	}
	return p
}

func (t *JSONTool) Execute(ctx context.Context, args json.RawMessage) (string, error) {
	var params map[string]any
	if err := json.Unmarshal(args, &params); err != nil {
		return "", fmt.Errorf("unmarshal args: %w", err)
	}
	var cmdBuf bytes.Buffer
	if err := t.cmdTmpl.Execute(&cmdBuf, params); err != nil {
		return "", fmt.Errorf("render command template: %w", err)
	}
	cmd := exec.CommandContext(ctx, "bash", "-c", cmdBuf.String())
	cmd.Dir = t.workspace
	output, err := cmd.CombinedOutput()
	if err != nil {
		return string(output), fmt.Errorf("command failed: %w", err)
	}
	return string(output), nil
}

// LoadToolsFromDir reads all *.json files from dir and returns a Tool for each.
// If dir does not exist the function returns nil, nil (graceful degradation).
func LoadToolsFromDir(workspace, dir string) ([]Tool, error) {
	if _, err := os.Stat(dir); os.IsNotExist(err) {
		return nil, nil
	}
	matches, err := filepath.Glob(filepath.Join(dir, "*.json"))
	if err != nil {
		return nil, fmt.Errorf("glob tooldefs dir: %w", err)
	}
	var tools []Tool
	for _, path := range matches {
		data, err := os.ReadFile(path)
		if err != nil {
			return nil, fmt.Errorf("read %s: %w", path, err)
		}
		var def JSONToolDef
		if err := json.Unmarshal(data, &def); err != nil {
			return nil, fmt.Errorf("parse %s: %w", path, err)
		}
		tmpl, err := template.New(def.Name).Parse(def.Command)
		if err != nil {
			return nil, fmt.Errorf("parse command template in %s: %w", path, err)
		}
		tools = append(tools, &JSONTool{
			def:       def,
			workspace: workspace,
			cmdTmpl:   tmpl,
		})
	}
	return tools, nil
}
