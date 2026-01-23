package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

type Client struct {
	APIKey          string
	BaseURL         string
	Model           string
	Temperature     float64
	ReasoningEffort string
	MaxTokens       int
	HTTP            *http.Client
}

var (
	ErrNoToolCall   = errors.New("model did not call tool propose_change")
	ErrBadToolArgs  = errors.New("tool args missing diff or file")
	ErrBadToolParse = errors.New("failed to parse tool args")
)

func NewClient(key, url, model string, temp float64, reasoning string, tokens int) *Client {
	return &Client{
		APIKey:      key,
		BaseURL:     strings.TrimRight(url, "/"),
		Model:       model,
		Temperature: temp,
		MaxTokens:   tokens,
		HTTP:        &http.Client{Timeout: 3000 * time.Second},
	}
}

type LLMRequest struct {
	Model               string     `json:"model"`
	Messages            []Message  `json:"messages"`
	Temperature         float64    `json:"temperature"`
	MaxCompletionTokens int        `json:"max_completion_tokens,omitempty"`
	ReasoningEffort     string     `json:"reasoning_effort,omitempty"`
	Tools               []Tool     `json:"tools,omitempty"`
	ToolChoice          ToolChoice `json:"tool_choice"`
}

type Message struct {
	Role      string     `json:"role"`
	Content   string     `json:"content,omitempty"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}

type LLMResponse struct {
	Choices []struct {
		Message Message `json:"message"`
	} `json:"choices"`
}

type Tool struct {
	Type     string       `json:"type"`
	Function ToolFunction `json:"function"`
}

type ToolFunction struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters"`
}

type ToolChoice struct {
	Type     string             `json:"type"`
	Function ToolChoiceFunction `json:"function"`
}

type ToolChoiceFunction struct {
	Name string `json:"name"`
}

type ToolCall struct {
	ID       string           `json:"id"`
	Type     string           `json:"type"`
	Function ToolCallFunction `json:"function"`
}

type ToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type ProposedChange struct {
	Summary   string   `json:"summary_of_changes"`
	Reasoning string   `json:"reasoning_for_change"`
	Diff      string   `json:"diff,omitempty"`
	File      string   `json:"file,omitempty"`
	DiffLines []string `json:"diff_lines,omitempty"`
}

func (c *Client) Call(prompt string) (ProposedChange, string, error) {
	tool := Tool{
		Type: "function",
		Function: ToolFunction{
			Name:        "propose_change",
			Description: "Propose a change for the target file. Provide a unified diff (preferred) or full file content.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"summary_of_changes": map[string]any{
						"type":        "string",
						"description": "Provide a short summary of the proposed changes. (1-3 sentences)",
					},
					"reasoning_for_change": map[string]any{
						"type":        "string",
						"description": "What's the reason this change will improve the benchmark? (1 sentence)",
					},
					"diff": map[string]any{
						"type":        "string",
						"description": "Unified diff (git-style) with paths relative to repo root. Use standard format with --- a/file, +++ b/file headers.",
					},
					"file": map[string]any{
						"type":        "string",
						"description": "Complete updated file content.",
					},
				},
				"required": []string{
					"summary_of_changes",
					"reasoning_for_change",
				},
			},
		},
	}

	reqBody := LLMRequest{
		Model: c.Model,
		Messages: []Message{
			{Role: "system", Content: "You are a senior Go performance engineer. Focus on optimizing code for execution speed and memory allocations."},
			{Role: "user", Content: prompt},
		},
		Temperature:         c.Temperature,
		MaxCompletionTokens: c.MaxTokens,
		ReasoningEffort:     c.ReasoningEffort,
		Tools:               []Tool{tool},
		ToolChoice: ToolChoice{
			Type:     "function",
			Function: ToolChoiceFunction{Name: "propose_change"},
		},
	}

	jsonBody, _ := json.Marshal(reqBody)

	req, _ := http.NewRequest("POST", fmt.Sprintf("%s/chat/completions", c.BaseURL), bytes.NewBuffer(jsonBody))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.APIKey)

	resp, err := c.HTTP.Do(req)
	if err != nil {
		return ProposedChange{}, string(jsonBody), err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		return ProposedChange{}, string(jsonBody), fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(body))
	}

	var llmResp LLMResponse
	if err := json.NewDecoder(resp.Body).Decode(&llmResp); err != nil {
		return ProposedChange{}, string(jsonBody), err
	}

	if len(llmResp.Choices) == 0 {
		return ProposedChange{}, string(jsonBody), fmt.Errorf("empty response from model")
	}

	msg := llmResp.Choices[0].Message
	if len(msg.ToolCalls) == 0 {
		return ProposedChange{}, string(jsonBody), ErrNoToolCall
	}
	call := msg.ToolCalls[0]
	if call.Function.Name != "propose_change" {
		return ProposedChange{}, string(jsonBody), fmt.Errorf("unexpected tool call: %s", call.Function.Name)
	}

	var change ProposedChange
	if err := json.Unmarshal([]byte(call.Function.Arguments), &change); err != nil {
		return ProposedChange{}, string(jsonBody), fmt.Errorf("%w: %v", ErrBadToolParse, err)
	}

	// Handle diff_lines if present
	if change.Diff == "" && len(change.DiffLines) > 0 {
		change.Diff = strings.Join(change.DiffLines, "\n")
	}

	// Clean up non-standard diff markers
	change.Diff = cleanDiff(change.Diff)

	if change.Diff == "" && change.File == "" {
		return ProposedChange{}, string(jsonBody), ErrBadToolArgs
	}

	return change, string(jsonBody), nil
}

func cleanDiff(diff string) string {
	if diff == "" {
		return diff
	}

	lines := strings.Split(diff, "\n")
	var result []string
	inValidDiff := false

	for _, line := range lines {
		// Skip non-standard markers
		if strings.HasPrefix(line, "*** Begin Patch") ||
			strings.HasPrefix(line, "*** End Patch") ||
			strings.HasPrefix(line, "*** Update File:") {
			continue
		}

		// Keep everything else
		result = append(result, line)

		// Check if we're in a valid diff section
		if strings.HasPrefix(line, "--- a/") || strings.HasPrefix(line, "+++ b/") {
			inValidDiff = true
		}
	}

	cleaned := strings.Join(result, "\n")

	if !inValidDiff {
		if strings.Contains(cleaned, "@@") {
			lines := strings.Split(cleaned, "\n")
			var hunks []string
			for _, line := range lines {
				if strings.HasPrefix(line, "@@") {
					hunks = append(hunks, line)
				} else if len(hunks) > 0 {
					hunks = append(hunks, line)
				}
			}
			if len(hunks) > 0 {
				cleaned = strings.Join(hunks, "\n")
			}
		}
	}

	return strings.TrimSpace(cleaned)
}

// buildPrompt describes the task; the tool call carries the actual patch.
func buildPrompt(ctx map[string]string, target, benchSpec string) string {
	var sb strings.Builder
	sb.WriteString("You are optimizing a Go LLM inference kernel.\n")
	sb.WriteString("TASK: Propose a change to make the Target File faster while passing all tests.\n")
	sb.WriteString("Use the propose_change tool to return a unified diff (preferred) or full file content.\n")
	sb.WriteString("\nIMPORTANT: If providing a diff, use standard git unified diff format:\n")
	sb.WriteString("  diff --git a/path/to/file b/path/to/file\n")
	sb.WriteString("  --- a/path/to/file\n")
	sb.WriteString("  +++ b/path/to/file\n")
	sb.WriteString("  @@ -start,count +start,count @@\n")
	sb.WriteString("Do NOT use markers like '*** Begin Patch' or '*** Update File:'.\n")
	sb.WriteString("Do NOT include markdown, backticks, or commentary outside the tool call.\n")
	sb.WriteString("\nConstraints:\n")
	sb.WriteString("- Pure Go only (no CGO).\n")
	sb.WriteString("- No stubs, no TODOs.\n")
	sb.WriteString("- Maintain existing public API.\n")
	sb.WriteString(fmt.Sprintf("\nTarget File: %s\n", target))
	sb.WriteString(fmt.Sprintf("Benchmark Gate: %s\n", benchSpec))

	sb.WriteString("\n--- CONTEXT START ---\n")
	for name, content := range ctx {
		sb.WriteString(fmt.Sprintf("\n### FILE: %s\n", name))
		sb.WriteString(content)
	}
	sb.WriteString("\n--- CONTEXT END ---\n")
	sb.WriteString("\nCall the propose_change tool now.")
	return sb.String()
}
