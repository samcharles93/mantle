package inference

import "testing"

func TestStreamProcessorPreservesOrderedDeltas(t *testing.T) {
	t.Parallel()

	p := newStreamProcessor("auto", -1)
	chunks := p.Push("A<think>R</think>B")
	chunks = append(chunks, p.Flush()...)

	if len(chunks) != 3 {
		t.Fatalf("expected 3 chunks, got %#v", chunks)
	}
	if chunks[0].Type != StreamChunkTextDelta || chunks[0].Delta != "A" {
		t.Fatalf("chunk 0 got %#v", chunks[0])
	}
	if chunks[1].Type != StreamChunkReasoningDelta || chunks[1].Delta != "R" {
		t.Fatalf("chunk 1 got %#v", chunks[1])
	}
	if chunks[2].Type != StreamChunkTextDelta || chunks[2].Delta != "B" {
		t.Fatalf("chunk 2 got %#v", chunks[2])
	}

	result := p.Result()
	if result.Text != "AB" || result.ReasoningText != "R" {
		t.Fatalf("unexpected result %#v", result)
	}
}

func TestStreamProcessorAppliesReasoningBudgetBeforeStreaming(t *testing.T) {
	t.Parallel()

	p := newStreamProcessor("auto", 0)
	chunks := p.Push("<think>secret</think>visible")
	chunks = append(chunks, p.Flush()...)

	if len(chunks) != 1 {
		t.Fatalf("expected 1 chunk, got %#v", chunks)
	}
	if chunks[0].Type != StreamChunkTextDelta || chunks[0].Delta != "visible" {
		t.Fatalf("unexpected chunk %#v", chunks[0])
	}

	result := p.Result()
	if result.Text != "visible" {
		t.Fatalf("text got %q want %q", result.Text, "visible")
	}
	if result.ReasoningText != "" {
		t.Fatalf("reasoning got %q want empty", result.ReasoningText)
	}
}

func TestStreamProcessorReasoningFormatNoneTreatsRawAsVisibleText(t *testing.T) {
	t.Parallel()

	p := newStreamProcessor("none", 0)
	chunks := p.Push("<think>secret</think>visible")
	chunks = append(chunks, p.Flush()...)

	if len(chunks) != 1 {
		t.Fatalf("expected 1 chunk, got %#v", chunks)
	}
	if chunks[0].Type != StreamChunkTextDelta || chunks[0].Delta != "<think>secret</think>visible" {
		t.Fatalf("unexpected chunk %#v", chunks[0])
	}

	result := p.Result()
	if result.Text != "<think>secret</think>visible" {
		t.Fatalf("text got %q", result.Text)
	}
	if result.ReasoningText != "" {
		t.Fatalf("reasoning got %q want empty", result.ReasoningText)
	}
}
