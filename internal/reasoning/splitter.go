package reasoning

import "strings"

type SplitResult struct {
	Content   string
	Reasoning string
}

// SplitRaw separates content and reasoning from raw model output using
// DeepSeek-style <think>...</think> tags. If a think block is opened but not
// closed, the remainder is treated as reasoning.
func SplitRaw(raw string) SplitResult {
	source := raw
	lower := strings.ToLower(source)
	const (
		openTag  = "<think>"
		closeTag = "</think>"
	)

	var content strings.Builder
	var reasoning strings.Builder
	cursor := 0

	for cursor < len(source) {
		start := strings.Index(lower[cursor:], openTag)
		if start < 0 {
			content.WriteString(source[cursor:])
			break
		}
		start += cursor
		content.WriteString(source[cursor:start])

		thinkStart := start + len(openTag)
		end := strings.Index(lower[thinkStart:], closeTag)
		if end < 0 {
			reasoning.WriteString(source[thinkStart:])
			break
		}
		end += thinkStart
		reasoning.WriteString(source[thinkStart:end])
		cursor = end + len(closeTag)
	}

	return SplitResult{
		Content:   content.String(),
		Reasoning: reasoning.String(),
	}
}

// Splitter incrementally emits content/reasoning deltas from raw streamed text.
type Splitter struct {
	raw              strings.Builder
	lastContentLen   int
	lastReasoningLen int
}

func (s *Splitter) Push(delta string) (contentDelta, reasoningDelta string) {
	if delta == "" {
		return "", ""
	}
	s.raw.WriteString(delta)
	out := SplitRaw(s.raw.String())

	if s.lastContentLen < len(out.Content) {
		contentDelta = out.Content[s.lastContentLen:]
		s.lastContentLen = len(out.Content)
	}
	if s.lastReasoningLen < len(out.Reasoning) {
		reasoningDelta = out.Reasoning[s.lastReasoningLen:]
		s.lastReasoningLen = len(out.Reasoning)
	}
	return contentDelta, reasoningDelta
}
