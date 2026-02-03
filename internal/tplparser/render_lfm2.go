package tplparser

import (
	"fmt"
	"strings"
)

func renderLFM2(opts RenderOptions) (string, bool, error) {
	var b strings.Builder

	if !opts.AddBOS && opts.BOSToken != "" {
		b.WriteString(opts.BOSToken)
	}

	msgs := opts.Messages
	var systemPrompt string
	if len(msgs) > 0 && strings.EqualFold(msgs[0].Role, "system") {
		switch v := msgs[0].Content.(type) {
		case string:
			systemPrompt = v
		default:
			j, err := jsonString(v)
			if err != nil {
				return "", false, fmt.Errorf("lfm2: system content tojson: %w", err)
			}
			systemPrompt = j
		}
		msgs = msgs[1:]
	}

	if len(opts.Tools) > 0 {
		var toolParts []string
		for _, tool := range opts.Tools {
			if s, ok := tool.(string); ok {
				toolParts = append(toolParts, s)
				continue
			}
			j, err := jsonString(tool)
			if err != nil {
				return "", false, fmt.Errorf("lfm2: tool tojson: %w", err)
			}
			toolParts = append(toolParts, j)
		}
		toolList := "List of tools: [" + strings.Join(toolParts, ", ") + "]"
		if systemPrompt == "" {
			systemPrompt = toolList
		} else {
			systemPrompt = systemPrompt + "\n" + toolList
		}
	}

	if systemPrompt != "" {
		b.WriteString("<|im_start|>system\n")
		b.WriteString(systemPrompt)
		b.WriteString("<|im_end|>\n")
	}

	lastAssistant := -1
	for i, m := range msgs {
		if m.Role == "assistant" {
			lastAssistant = i
		}
	}

	for i, m := range msgs {
		b.WriteString("<|im_start|>")
		b.WriteString(m.Role)
		b.WriteString("\n")
		content := m.Content
		if s, ok := content.(string); ok {
			content = s
		} else {
			j, err := jsonString(content)
			if err != nil {
				return "", false, fmt.Errorf("lfm2: content tojson: %w", err)
			}
			content = j
		}

		text := ""
		if s, ok := content.(string); ok {
			text = s
		}
		if m.Role == "assistant" && !opts.KeepPastThinking && i != lastAssistant {
			if cut := strings.LastIndex(text, "</think>"); cut >= 0 {
				text = strings.TrimSpace(text[cut+len("</think>"):])
			}
		}
		b.WriteString(text)
		b.WriteString("<|im_end|>\n")
	}

	if opts.AddGenerationPrompt {
		b.WriteString("<|im_start|>assistant\n")
	}
	return b.String(), true, nil
}
