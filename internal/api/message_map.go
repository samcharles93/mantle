package api

import (
	"encoding/json"
	"fmt"

	"github.com/samcharles93/mantle/internal/tokenizer"
)

func responseItemsToMessages(items []ResponseItem) ([]tokenizer.Message, error) {
	if len(items) == 0 {
		return nil, nil
	}
	msgs := make([]tokenizer.Message, 0, len(items))
	for _, item := range items {
		role := item.Role
		if role == "" && item.Type == "message" {
			return nil, fmt.Errorf("message item missing role")
		}
		if role == "" {
			continue
		}
		content, err := responseContentToMessageContent(item)
		if err != nil {
			return nil, err
		}
		msg := tokenizer.Message{
			Role:    role,
			Content: content,
		}
		if len(item.Arguments) > 0 && item.Name != "" {
			tc := tokenizer.ToolCall{
				Type: "function",
				Function: tokenizer.ToolCallFunction{
					Name:      item.Name,
					Arguments: parseToolArguments(item.Arguments),
				},
			}
			msg.ToolCalls = []tokenizer.ToolCall{tc}
		}
		msgs = append(msgs, msg)
	}
	return msgs, nil
}

func responseContentToMessageContent(item ResponseItem) (any, error) {
	if len(item.Content) == 0 {
		if len(item.Output) > 0 {
			return string(item.Output), nil
		}
		return "", nil
	}

	allText := true
	for _, part := range item.Content {
		if part.Type != "input_text" && part.Type != "output_text" && part.Type != "text" {
			allText = false
			break
		}
	}

	if allText && len(item.Content) == 1 {
		return item.Content[0].Text, nil
	}

	blocks := make([]map[string]any, 0, len(item.Content))
	for _, part := range item.Content {
		switch part.Type {
		case "input_text", "output_text", "text":
			blocks = append(blocks, map[string]any{
				"type": "text",
				"text": part.Text,
			})
		case "input_image", "image":
			block := map[string]any{
				"type": "image",
			}
			if part.ImageURL != "" {
				block["image_url"] = part.ImageURL
			}
			if part.Detail != "" {
				block["detail"] = part.Detail
			}
			if part.FileID != "" {
				block["file_id"] = part.FileID
			}
			blocks = append(blocks, block)
		default:
			return nil, fmt.Errorf("unsupported content type %q", part.Type)
		}
	}
	return blocks, nil
}

func parseToolArguments(raw json.RawMessage) any {
	if len(raw) == 0 {
		return nil
	}
	var m map[string]any
	if err := json.Unmarshal(raw, &m); err == nil {
		return m
	}
	return string(raw)
}
