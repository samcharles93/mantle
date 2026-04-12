package reasoning

import "strings"

const (
	thinkOpen  = "<think>"
	thinkClose = "</think>"
)

type SplitResult struct {
	Content   string
	Reasoning string
}

type DeltaKind string

const (
	DeltaContent   DeltaKind = "content"
	DeltaReasoning DeltaKind = "reasoning"
)

type Delta struct {
	Kind DeltaKind
	Text string
}

// Splitter incrementally emits ordered content/reasoning deltas from raw
// streamed text. Unlike rescanning the full buffer on every chunk, it tracks
// parser state across Push calls and only inspects newly appended text.
type Splitter struct {
	raw       strings.Builder
	content   strings.Builder
	reasoning strings.Builder
	cursor    int
	inThink   bool

	ReasoningBudget    int
	hasReasoningBudget bool
	reasoningRuneCount int
}

func (s *Splitter) SetReasoningBudget(budget int) {
	s.ReasoningBudget = budget
	s.hasReasoningBudget = true
}

func (s *Splitter) Push(delta string) []Delta {
	if delta == "" {
		return nil
	}
	s.raw.WriteString(delta)
	return s.process(false)
}

// Flush emits any text held back because it could have been the start of a tag
// arriving across chunk boundaries. Must be called when the stream ends to
// avoid dropping a literal partial tag suffix.
func (s *Splitter) Flush() []Delta {
	return s.process(true)
}

func (s *Splitter) Result() SplitResult {
	return SplitResult{
		Content:   s.content.String(),
		Reasoning: s.reasoning.String(),
	}
}

func (s *Splitter) process(flush bool) []Delta {
	raw := s.raw.String()
	if s.cursor >= len(raw) {
		return nil
	}
	lower := strings.ToLower(raw)
	i := s.cursor
	var out []Delta

	for i < len(raw) {
		if s.inThink {
			end := strings.Index(lower[i:], thinkClose)
			if end < 0 {
				emitEnd := len(raw)
				if !flush {
					emitEnd -= partialSuffixLen(lower[i:], thinkClose)
				}
				if emitEnd > i {
					s.appendReasoning(raw[i:emitEnd], &out)
				}
				i = emitEnd
				break
			}
			end += i
			s.appendReasoning(raw[i:end], &out)
			i = end + len(thinkClose)
			s.inThink = false
			continue
		}

		start := strings.Index(lower[i:], thinkOpen)
		if start < 0 {
			emitEnd := len(raw)
			if !flush {
				emitEnd -= partialSuffixLen(lower[i:], thinkOpen)
			}
			if emitEnd > i {
				s.appendContent(raw[i:emitEnd], &out)
			}
			i = emitEnd
			break
		}
		start += i
		if start > i {
			s.appendContent(raw[i:start], &out)
		}
		i = start + len(thinkOpen)
		s.inThink = true
	}

	s.cursor = i
	return out
}

func (s *Splitter) appendContent(text string, out *[]Delta) {
	if text == "" {
		return
	}
	s.content.WriteString(text)
	*out = append(*out, Delta{Kind: DeltaContent, Text: text})
}

func (s *Splitter) appendReasoning(text string, out *[]Delta) {
	trimmed := s.limitReasoning(text)
	if trimmed == "" {
		return
	}
	s.reasoning.WriteString(trimmed)
	*out = append(*out, Delta{Kind: DeltaReasoning, Text: trimmed})
}

func (s *Splitter) limitReasoning(text string) string {
	if !s.hasReasoningBudget || s.ReasoningBudget < 0 {
		return text
	}
	remaining := s.ReasoningBudget - s.reasoningRuneCount
	if remaining <= 0 {
		return ""
	}
	runes := []rune(text)
	if len(runes) <= remaining {
		s.reasoningRuneCount += len(runes)
		return text
	}
	s.reasoningRuneCount = s.ReasoningBudget
	return string(runes[:remaining])
}

// partialSuffixLen returns the length of the longest suffix of text that is a
// proper prefix of tag (i.e., the end of text could be the start of tag arriving
// across chunk boundaries). Returns 0 if no such suffix exists.
func partialSuffixLen(text, tag string) int {
	maxN := min(len(tag)-1, len(text))
	for n := maxN; n >= 1; n-- {
		if strings.HasSuffix(text, tag[:n]) {
			return n
		}
	}
	return 0
}
