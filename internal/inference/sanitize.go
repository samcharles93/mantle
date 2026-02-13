package inference

import "strings"

// SanitizeAssistantForContext removes reasoning/sentinel artifacts before
// assistant text is fed back into subsequent turns.
func SanitizeAssistantForContext(text string) string {
	s := stripThinkBlocks(text)
	for _, token := range []string{
		"<|im_end|>",
		"<|endoftext|>",
		"<|end_of_text|>",
		"<|eot_id|>",
		"</s>",
	} {
		s = strings.ReplaceAll(s, token, "")
	}
	return strings.TrimSpace(s)
}

func stripThinkBlocks(text string) string {
	source := text
	lower := strings.ToLower(source)
	const (
		openTag  = "<think>"
		closeTag = "</think>"
	)

	var b strings.Builder
	cursor := 0
	for cursor < len(source) {
		start := strings.Index(lower[cursor:], openTag)
		if start < 0 {
			b.WriteString(source[cursor:])
			break
		}
		start += cursor
		b.WriteString(source[cursor:start])

		thinkStart := start + len(openTag)
		end := strings.Index(lower[thinkStart:], closeTag)
		if end < 0 {
			break // drop unclosed think block tail
		}
		cursor = thinkStart + end + len(closeTag)
	}
	return b.String()
}
