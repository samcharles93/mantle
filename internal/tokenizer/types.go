package tokenizer

import "github.com/samcharles93/mantle/internal/tplparser"

// Message represents a chat message for template rendering.
// Content may be a string, map, or slice of blocks depending on the template.
type Message = tplparser.Message

type ToolCall = tplparser.ToolCall

type ToolCallFunction = tplparser.ToolCallFunction

// MessageText returns the content if it is a string.
func MessageText(msg Message) (string, bool) {
	if msg.Content == nil {
		return "", false
	}
	text, ok := msg.Content.(string)
	return text, ok
}
