package tokenizer

// Message represents a chat message for template rendering.
// Content may be a string, map, or slice of blocks depending on the template.
type Message struct {
	Role      string     `json:"role"`
	Content   any        `json:"content,omitempty"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
	Name      string     `json:"name,omitempty"`
}

type ToolCall struct {
	ID       string           `json:"id,omitempty"`
	Type     string           `json:"type,omitempty"`
	Function ToolCallFunction `json:"function"`
}

type ToolCallFunction struct {
	Name      string `json:"name"`
	Arguments any    `json:"arguments,omitempty"`
}

// MessageText returns the content if it is a string.
func MessageText(msg Message) (string, bool) {
	if msg.Content == nil {
		return "", false
	}
	text, ok := msg.Content.(string)
	return text, ok
}
