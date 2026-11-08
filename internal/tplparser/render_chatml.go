package tplparser

import (
	"fmt"
	"strings"
)

func renderChatML(opts RenderOptions) (string, bool, error) {
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
				return "", false, fmt.Errorf("chatml: system content tojson: %w", err)
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
				return "", false, fmt.Errorf("chatml: tool tojson: %w", err)
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
		}

		text := ""
		switch c := content.(type) {
		case string:
			text = c
		case []any:
			// multimodal content blocks: image/video placeholders become the
			// LFM3-VL markers; text blocks inline; anything else is tojson'd
			for _, item := range c {
				block, ok := item.(map[string]any)
				if !ok {
					j, err := jsonString(item)
					if err != nil {
						return "", false, fmt.Errorf("chatml: content block tojson: %w", err)
					}
					text += j
					continue
				}
				switch block["type"] {
				case "image":
					text += "<|vision_start|><|image|><|vision_end|>"
				case "video":
					text += "<|vision_start|><|video|><|vision_end|>"
				case "text":
					if t, ok := block["text"].(string); ok {
						text += t
					}
				default:
					j, err := jsonString(block)
					if err != nil {
						return "", false, fmt.Errorf("chatml: content block tojson: %w", err)
					}
					text += j
				}
			}
		default:
			j, err := jsonString(content)
			if err != nil {
				return "", false, fmt.Errorf("chatml: content tojson: %w", err)
			}
			text = j
		}
		if m.Role == "assistant" && i != lastAssistant {
			if cut := strings.LastIndex(text, "</think>"); cut >= 0 {
				text = strings.TrimSpace(text[cut+len("</think>"):])
			}
		}
		b.WriteString(text)
		b.WriteString("<|im_end|>\n")
	}

	if opts.AddGenerationPrompt {
		b.WriteString("<|im_start|>assistant\n")
		if strings.Contains(opts.Template, "enable_thinking") {
			// templates with an enable_thinking switch emit the think open tag
			// (or the empty block) right after the assistant header
			if opts.DisableThinking {
				b.WriteString("<think></think>")
			} else {
				b.WriteString("<think>")
			}
		} else {
			writeDisabledThinking(&b, opts)
		}
	}
	return b.String(), true, nil
}
