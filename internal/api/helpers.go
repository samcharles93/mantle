package api

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/google/uuid"
	"github.com/labstack/echo/v5"
)

func writeBadRequest(c *echo.Context, msg string) error {
	return writeError(c, http.StatusBadRequest, "invalid_request_error", msg, "", "")
}

func writeNotFound(c *echo.Context, msg string) error {
	return writeError(c, http.StatusNotFound, "not_found_error", msg, "", "")
}

func writeError(c *echo.Context, status int, errType, msg, param, code string) error {
	return c.JSON(status, map[string]any{
		"error": ResponseError{
			Message: msg,
			Type:    errType,
			Code:    code,
			Param:   param,
		},
	})
}

func normalizeInputItems(input any) ([]ResponseItem, error) {
	if input == nil {
		return nil, nil
	}
	switch v := input.(type) {
	case string:
		return []ResponseItem{messageItem("user", v)}, nil
	case []any:
		items := make([]ResponseItem, 0, len(v))
		for _, raw := range v {
			item, err := coerceInputItem(raw)
			if err != nil {
				return nil, err
			}
			items = append(items, item)
		}
		return items, nil
	case []ResponseItem:
		return v, nil
	default:
		return nil, fmt.Errorf("expected string or array")
	}
}

func coerceInputItem(raw any) (ResponseItem, error) {
	switch v := raw.(type) {
	case string:
		return messageItem("user", v), nil
	case map[string]any:
		if role, ok := asString(v["role"]); ok {
			content := v["content"]
			return messageItemFromContent(role, content)
		}
		if typ, ok := asString(v["type"]); ok {
			if typ == "message" {
				role, _ := asString(v["role"])
				return messageItemFromContent(role, v["content"])
			}
		}
		return decodeResponseItem(v)
	default:
		return ResponseItem{}, fmt.Errorf("invalid input item")
	}
}

func decodeResponseItem(m map[string]any) (ResponseItem, error) {
	b, err := json.Marshal(m)
	if err != nil {
		return ResponseItem{}, err
	}
	var item ResponseItem
	if err := json.Unmarshal(b, &item); err != nil {
		return ResponseItem{}, err
	}
	if item.ID == "" {
		item.ID = newInputItemID()
	}
	return item, nil
}

func messageItem(role, text string) ResponseItem {
	return ResponseItem{
		ID:   newInputItemID(),
		Type: "message",
		Role: role,
		Content: []ResponseContent{{
			Type: "input_text",
			Text: text,
		}},
	}
}

func messageItemFromContent(role string, content any) (ResponseItem, error) {
	item := ResponseItem{
		ID:   newInputItemID(),
		Type: "message",
		Role: role,
	}
	switch v := content.(type) {
	case string:
		item.Content = []ResponseContent{{
			Type: "input_text",
			Text: v,
		}}
		return item, nil
	case []any:
		parts := make([]ResponseContent, 0, len(v))
		for _, raw := range v {
			m, ok := raw.(map[string]any)
			if !ok {
				return ResponseItem{}, fmt.Errorf("invalid content part")
			}
			part, err := contentPartFromMap(m)
			if err != nil {
				return ResponseItem{}, err
			}
			parts = append(parts, part)
		}
		item.Content = parts
		return item, nil
	default:
		return ResponseItem{}, fmt.Errorf("invalid message content")
	}
}

func contentPartFromMap(m map[string]any) (ResponseContent, error) {
	typ, _ := asString(m["type"])
	switch typ {
	case "input_text", "text":
		text, _ := asString(m["text"])
		return ResponseContent{Type: "input_text", Text: text}, nil
	case "input_image", "image":
		part := ResponseContent{Type: "input_image"}
		if url, ok := asString(m["image_url"]); ok {
			part.ImageURL = url
		}
		if detail, ok := asString(m["detail"]); ok {
			part.Detail = detail
		}
		if fileID, ok := asString(m["file_id"]); ok {
			part.FileID = fileID
		}
		return part, nil
	case "input_file", "file":
		part := ResponseContent{Type: "input_file"}
		if fileID, ok := asString(m["file_id"]); ok {
			part.FileID = fileID
		}
		if fileURL, ok := asString(m["file_url"]); ok {
			part.FileURL = fileURL
		}
		if fileData, ok := asString(m["file_data"]); ok {
			part.FileData = fileData
		}
		if name, ok := asString(m["filename"]); ok {
			part.Filename = name
		}
		return part, nil
	default:
		return ResponseContent{}, fmt.Errorf("unsupported content type %q", typ)
	}
}

func approximateTokenCount(items []ResponseItem) int {
	count := 0
	for _, item := range items {
		for _, part := range item.Content {
			if part.Type == "input_text" {
				count += len([]rune(part.Text))
			}
		}
	}
	if count == 0 {
		return 0
	}
	return max(count/4, 1)
}

func compactToSingleMessage(items []ResponseItem) []ResponseItem {
	if len(items) == 0 {
		return nil
	}
	var parts []string
	for _, item := range items {
		for _, content := range item.Content {
			if content.Type == "input_text" || content.Type == "output_text" || content.Type == "text" {
				if strings.TrimSpace(content.Text) != "" {
					parts = append(parts, content.Text)
				}
			}
		}
	}
	if len(parts) == 0 {
		return nil
	}
	text := strings.Join(parts, "\n")
	return []ResponseItem{{
		ID:   newInputItemID(),
		Type: "message",
		Role: "assistant",
		Content: []ResponseContent{{
			Type: "output_text",
			Text: text,
		}},
	}}
}

func asString(v any) (string, bool) {
	s, ok := v.(string)
	return s, ok
}

func newInputItemID() string {
	return "item_" + uuid.NewString()
}
