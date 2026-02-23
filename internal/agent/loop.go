package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"regexp"
	"strings"
	"time"

	"github.com/samcharles93/mantle/internal/inference"
	"github.com/samcharles93/mantle/internal/reasoning"
	"github.com/samcharles93/mantle/internal/tokenizer"
)

// ANSI escape codes for terminal formatting.
const (
	ansiReset  = "\033[0m"
	ansiGray   = "\033[90m"
	ansiBold   = "\033[1m"
	ansiGreen  = "\033[32m"
	ansiYellow = "\033[33m"
	ansiRed    = "\033[31m"
	ansiCyan   = "\033[36m"
)

// Loop manages the agentic ReAct cycle for a solo agent.
type Loop struct {
	Engine      inference.Engine
	Registry    *Registry
	MaxSteps    int
	Interactive bool
	Messages    []tokenizer.Message
	// Out is the output writer for display. Nil disables display (swarm workers).
	Out io.Writer

	// Sampling parameters. Zero values use the engine's defaults.
	Temperature float32
	TopP        float32
	TopK        int
}

// Result represents the outcome of an agent run.
type Result struct {
	Success bool
	Output  string
}

// ToolCall is a parsed tool invocation extracted from model output.
type ToolCall struct {
	Name      string
	Arguments json.RawMessage
}

var (
	// toolCallTagRegex matches the <tool_call>{...}</tool_call> format that
	// Qwen3 and similar models emit when a chat template instructs tool use.
	toolCallTagRegex   = regexp.MustCompile(`(?s)<tool_call>\s*(\{.*?\})\s*</tool_call>`)
	toolCallStripRegex = regexp.MustCompile(`(?s)\s*<tool_call>.*?</tool_call>\s*`)

	// reactActionRegex / reactArgsRegex are a ReAct-style fallback for models
	// whose chat template does not define a structured tool-call format.
	reactActionRegex = regexp.MustCompile(`(?i)Action:\s*(\w+)`)
	reactArgsRegex   = regexp.MustCompile(`(?i)↳\s*({.*})`)
)

func (l *Loop) w() io.Writer {
	if l.Out != nil {
		return l.Out
	}
	return io.Discard
}

func (l *Loop) Run(ctx context.Context, goal string) (*Result, error) {
	l.Messages = append(l.Messages, tokenizer.Message{
		Role:    "user",
		Content: goal,
	})

	totalTokens := 0
	totalStart := time.Now()
	w := l.w()

	for i := 1; i <= l.MaxSteps; i++ {
		fmt.Fprintf(w, "\n%s─── Step %d / %d %s%s\n",
			ansiGray, i, l.MaxSteps,
			strings.Repeat("─", max(0, 44-len(fmt.Sprintf("─── Step %d / %d ", i, l.MaxSteps)))),
			ansiReset)

		req := inference.Request{
			Messages:    l.Messages,
			Tools:       l.registryToInferenceTools(),
			Steps:       -1,
			Temperature: float64(l.Temperature),
			TopP:        float64(l.TopP),
			TopK:        l.TopK,
		}

		stepStart := time.Now()
		var (
			splitter reasoning.Splitter
			inThink  bool
		)

		streamFn := func(chunk string) {
			content, think := splitter.Push(chunk)
			if think != "" {
				if !inThink {
					inThink = true
					fmt.Fprintf(w, "\x1b[2;90m[thinking]\n")
				}
				fmt.Fprint(w, think)
			}
			if content != "" {
				if inThink {
					inThink = false
					fmt.Fprintf(w, "\x1b[0m\n\n")
				}
				fmt.Fprint(w, content)
			}
		}

		res, err := l.Engine.Generate(ctx, &req, streamFn)
		if err != nil {
			return nil, err
		}

		if inThink {
			fmt.Fprintf(w, "\x1b[0m")
			inThink = false
		}

		elapsed := time.Since(stepStart)
		totalTokens += res.Stats.TokensGenerated

		fmt.Fprintf(w, "\n%s  %d tok", ansiGray, res.Stats.TokensGenerated)
		if res.Stats.GenerationTPS > 0 {
			fmt.Fprintf(w, " · %.1f TPS", res.Stats.GenerationTPS)
		}
		fmt.Fprintf(w, " · %s%s\n", formatDuration(elapsed), ansiReset)

		// Parse a tool call from the raw output.
		// res.RawText retains <think> and <tool_call> blocks; res.Text has
		// thinking stripped but still contains any <tool_call> text.
		toolCall := l.parseToolCall(res.RawText)

		// Build the assistant message using proper structured fields so the
		// chat template can render subsequent turns correctly.
		if toolCall != nil {
			// Content is whatever the model said before the tool call
			// (after stripping thinking — the template never needs to replay it).
			contentText := strings.TrimSpace(toolCallStripRegex.ReplaceAllString(res.Text, ""))
			var msgContent any
			if contentText != "" {
				msgContent = contentText
			}
			l.Messages = append(l.Messages, tokenizer.Message{
				Role:    "assistant",
				Content: msgContent,
				ToolCalls: []tokenizer.ToolCall{{
					Function: tokenizer.ToolCallFunction{
						Name:      toolCall.Name,
						Arguments: toolCall.Arguments,
					},
				}},
			})
		} else {
			// No tool call: plain text response; the agent is done.
			l.Messages = append(l.Messages, tokenizer.Message{
				Role:    "assistant",
				Content: res.Text,
			})
			finalOutput := strings.TrimSpace(res.Text)
			if finalOutput == "" {
				finalOutput = strings.TrimSpace(res.ReasoningText)
			}
			fmt.Fprintf(w, "\n%s✓ Done%s  %s%d tokens · %s%s\n",
				ansiGreen, ansiReset,
				ansiGray, totalTokens, formatDuration(time.Since(totalStart)), ansiReset)
			return &Result{Success: true, Output: finalOutput}, nil
		}

		// Display the tool call.
		argsPreview := truncate(string(toolCall.Arguments), 120)
		fmt.Fprintf(w, "\n%s▶ %s%s  %s%s%s\n",
			ansiBold, toolCall.Name, ansiReset,
			ansiGray, argsPreview, ansiReset)

		tool := l.Registry.Get(toolCall.Name)
		if tool == nil {
			fmt.Fprintf(w, "%s  ✗ unknown tool %q%s\n", ansiRed, toolCall.Name, ansiReset)
			l.addToolResponse(toolCall.Name, fmt.Sprintf("Error: tool %q not found", toolCall.Name))
			continue
		}

		if l.Interactive {
			if !l.confirmToolExecution(toolCall) {
				fmt.Fprintf(w, "%s  ✗ denied by user%s\n", ansiYellow, ansiReset)
				l.addToolResponse(toolCall.Name, "Error: user denied tool execution")
				continue
			}
		}

		execStart := time.Now()
		observation, execErr := tool.Execute(ctx, toolCall.Arguments)
		execElapsed := time.Since(execStart)

		if execErr != nil {
			fmt.Fprintf(w, "%s  ✗ failed (%s): %v%s\n", ansiRed, formatDuration(execElapsed), execErr, ansiReset)
			l.addToolResponse(toolCall.Name, fmt.Sprintf("Error: %v\nOutput: %s", execErr, observation))
		} else {
			lineCount := strings.Count(observation, "\n") + 1
			preview := truncate(observation, 400)
			fmt.Fprintf(w, "%s◀ %d lines · %s:%s\n", ansiCyan, lineCount, formatDuration(execElapsed), ansiReset)
			for line := range strings.SplitSeq(preview, "\n") {
				fmt.Fprintf(w, "  %s\n", line)
			}
			if len(observation) > 400 {
				fmt.Fprintf(w, "%s  … (%d chars total)%s\n", ansiGray, len(observation), ansiReset)
			}
			l.addToolResponse(toolCall.Name, observation)
		}
	}

	return &Result{Success: false, Output: "max steps reached"}, nil
}

func (l *Loop) registryToInferenceTools() []any {
	tools := l.Registry.List()
	out := make([]any, 0, len(tools))
	for _, t := range tools {
		out = append(out, map[string]any{
			"type": "function",
			"function": map[string]any{
				"name":        t.Name(),
				"description": t.Description(),
				"parameters":  t.Parameters(),
			},
		})
	}
	return out
}

// parseToolCall extracts a tool invocation from model output text.
// It first tries the native <tool_call> format that Qwen3 and other
// template-aware models produce, then falls back to ReAct-style text.
func (l *Loop) parseToolCall(text string) *ToolCall {
	if m := toolCallTagRegex.FindStringSubmatch(text); m != nil {
		var payload struct {
			Name      string          `json:"name"`
			Arguments json.RawMessage `json:"arguments"`
		}
		if err := json.Unmarshal([]byte(m[1]), &payload); err == nil && payload.Name != "" {
			if payload.Arguments == nil {
				payload.Arguments = json.RawMessage("{}")
			}
			return &ToolCall{Name: payload.Name, Arguments: payload.Arguments}
		}
	}

	// ReAct fallback: "Action: name\n↳ {args}"
	actionMatch := reactActionRegex.FindStringSubmatch(text)
	if actionMatch == nil {
		return nil
	}
	var args json.RawMessage
	if m := reactArgsRegex.FindStringSubmatch(text); m != nil {
		args = json.RawMessage(m[1])
	} else {
		args = json.RawMessage("{}")
	}
	return &ToolCall{Name: actionMatch[1], Arguments: args}
}

// addToolResponse appends a tool result message using the structured role
// and Name fields so each architecture's template renders it correctly
// (e.g. Qwen3 wraps in <tool_response>, Gemma3 uses Name for routing).
func (l *Loop) addToolResponse(toolName, content string) {
	l.Messages = append(l.Messages, tokenizer.Message{
		Role:    "tool",
		Name:    toolName,
		Content: content,
	})
}

func (l *Loop) confirmToolExecution(tc *ToolCall) bool {
	fmt.Fprintf(os.Stdout, "  %s✋ execute %s? [Y/n]: %s", ansiYellow, tc.Name, ansiReset)
	var input string
	fmt.Scanln(&input)
	return strings.ToLower(strings.TrimSpace(input)) != "n"
}

func formatDuration(d time.Duration) string {
	if d < time.Second {
		return fmt.Sprintf("%dms", d.Milliseconds())
	}
	return fmt.Sprintf("%.1fs", d.Seconds())
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}
