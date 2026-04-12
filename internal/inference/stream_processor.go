package inference

import (
	"strings"

	"github.com/samcharles93/mantle/internal/reasoning"
)

type streamProcessor struct {
	reasoningEnabled bool
	visible          strings.Builder
	splitter         reasoning.Splitter
}

func newStreamProcessor(format string, budget int) *streamProcessor {
	p := &streamProcessor{
		reasoningEnabled: !strings.EqualFold(strings.TrimSpace(format), "none"),
	}
	if p.reasoningEnabled && budget >= 0 {
		p.splitter.SetReasoningBudget(budget)
	}
	return p
}

func (p *streamProcessor) Push(rawDelta string) []StreamChunk {
	if rawDelta == "" {
		return nil
	}
	if !p.reasoningEnabled {
		p.visible.WriteString(rawDelta)
		return []StreamChunk{{
			Type:  StreamChunkTextDelta,
			Delta: rawDelta,
		}}
	}

	return p.mapDeltas(p.splitter.Push(rawDelta))
}

func (p *streamProcessor) Flush() []StreamChunk {
	if !p.reasoningEnabled {
		return nil
	}
	return p.mapDeltas(p.splitter.Flush())
}

func (p *streamProcessor) Result() Result {
	if !p.reasoningEnabled {
		return Result{
			Text:          p.visible.String(),
			ReasoningText: "",
		}
	}

	split := p.splitter.Result()
	return Result{
		Text:          split.Content,
		ReasoningText: split.Reasoning,
	}
}

func (p *streamProcessor) mapDeltas(deltas []reasoning.Delta) []StreamChunk {
	if len(deltas) == 0 {
		return nil
	}

	out := make([]StreamChunk, 0, len(deltas))
	for _, delta := range deltas {
		switch delta.Kind {
		case reasoning.DeltaContent:
			out = append(out, StreamChunk{
				Type:  StreamChunkTextDelta,
				Delta: delta.Text,
			})
		case reasoning.DeltaReasoning:
			out = append(out, StreamChunk{
				Type:  StreamChunkReasoningDelta,
				Delta: delta.Text,
			})
		}
	}
	return out
}
