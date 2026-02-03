package tokenizer

import (
	"encoding/json"
	"fmt"
	"os"
)

// LoadMessagesJSON reads a JSON file that is either a messages array
// or an object with a "messages" field.
func LoadMessagesJSON(path string) ([]Message, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var payload any
	if err := json.Unmarshal(raw, &payload); err != nil {
		return nil, fmt.Errorf("parse messages json: %w", err)
	}
	switch v := payload.(type) {
	case []any:
		return decodeMessages(v)
	case map[string]any:
		if msgs, ok := v["messages"]; ok {
			if list, ok := msgs.([]any); ok {
				return decodeMessages(list)
			}
			return nil, fmt.Errorf("messages field must be an array")
		}
		return nil, fmt.Errorf("messages json object missing \"messages\" field")
	default:
		return nil, fmt.Errorf("messages json must be array or object")
	}
}

// LoadToolsJSON reads a JSON file that is either a tools array
// or an object with a "tools" field.
func LoadToolsJSON(path string) ([]any, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var payload any
	if err := json.Unmarshal(raw, &payload); err != nil {
		return nil, fmt.Errorf("parse tools json: %w", err)
	}
	switch v := payload.(type) {
	case []any:
		return v, nil
	case map[string]any:
		if tools, ok := v["tools"]; ok {
			if list, ok := tools.([]any); ok {
				return list, nil
			}
			return nil, fmt.Errorf("tools field must be an array")
		}
		return nil, fmt.Errorf("tools json object missing \"tools\" field")
	default:
		return nil, fmt.Errorf("tools json must be array or object")
	}
}

func decodeMessages(items []any) ([]Message, error) {
	msgs := make([]Message, 0, len(items))
	for i, item := range items {
		b, err := json.Marshal(item)
		if err != nil {
			return nil, fmt.Errorf("encode message %d: %w", i, err)
		}
		var msg Message
		if err := json.Unmarshal(b, &msg); err != nil {
			return nil, fmt.Errorf("decode message %d: %w", i, err)
		}
		msgs = append(msgs, msg)
	}
	return msgs, nil
}
