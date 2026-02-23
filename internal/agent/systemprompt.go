package agent

import (
	"bytes"
	"runtime"
	"text/template"
	"time"
)

// SystemPromptData is the template context injected into the system prompt.
type SystemPromptData struct {
	Workspace string
	OS        string
	Arch      string
	Date      string
	Tools     []ToolInfo
	MaxSteps  int
}

// ToolInfo holds display info for a single tool.
type ToolInfo struct {
	Name        string
	Description string
}

// DefaultSystemPrompt is the built-in system prompt template.
// It is used when no custom template is configured.
const DefaultSystemPrompt = `You are an autonomous agent.

Environment:
- Working directory: {{.Workspace}}
- Platform: {{.OS}}/{{.Arch}}
- Date: {{.Date}}

Available tools:{{range .Tools}}
- {{.Name}}: {{.Description}}{{end}}

Use the available tools to accomplish the goal. Think step by step and call tools as needed. When the goal is fully achieved, summarize what was done.`

// BuildSystemPrompt renders tmplStr with workspace, tool list, and environment info.
func BuildSystemPrompt(tmplStr, workspace string, tools []Tool, maxSteps int) (string, error) {
	data := SystemPromptData{
		Workspace: workspace,
		OS:        runtime.GOOS,
		Arch:      runtime.GOARCH,
		Date:      time.Now().Format("2006-01-02"),
		MaxSteps:  maxSteps,
	}
	for _, t := range tools {
		data.Tools = append(data.Tools, ToolInfo{Name: t.Name(), Description: t.Description()})
	}
	tmpl, err := template.New("system").Parse(tmplStr)
	if err != nil {
		return "", err
	}
	var buf bytes.Buffer
	if err := tmpl.Execute(&buf, data); err != nil {
		return "", err
	}
	return buf.String(), nil
}
