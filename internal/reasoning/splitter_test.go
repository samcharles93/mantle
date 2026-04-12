package reasoning

import (
	"fmt"
	"math/rand"
	"strings"
	"testing"
)

func TestSplitterPush(t *testing.T) {
	t.Parallel()

	var s Splitter

	deltas := s.Push(thinkOpen + "abc")
	if len(deltas) != 1 || deltas[0].Kind != DeltaReasoning || deltas[0].Text != "abc" {
		t.Fatalf("first delta got %#v", deltas)
	}

	deltas = s.Push(thinkClose + "Hello")
	if len(deltas) != 1 || deltas[0].Kind != DeltaContent || deltas[0].Text != "Hello" {
		t.Fatalf("second delta got %#v", deltas)
	}
}

func TestSplitterIncrementalExpectedResults(t *testing.T) {
	t.Parallel()

	rng := rand.New(rand.NewSource(42))

	cases := []struct {
		name          string
		input         string
		wantContent   string
		wantReasoning string
	}{
		{name: "plain text", input: "plain text no tags", wantContent: "plain text no tags"},
		{name: "single block", input: thinkOpen + "reasoning" + thinkClose + "content", wantContent: "content", wantReasoning: "reasoning"},
		{name: "open block", input: thinkOpen + "open think", wantReasoning: "open think"},
		{name: "multiple blocks", input: "A" + thinkOpen + "r1" + thinkClose + "B" + thinkOpen + "r2" + thinkClose + "C", wantContent: "ABC", wantReasoning: "r1r2"},
		{name: "mixed open and closed", input: thinkOpen + "R1" + thinkClose + "X" + thinkOpen + "R2", wantContent: "X", wantReasoning: "R1R2"},
		{name: "surrounded block", input: "before" + thinkOpen + "mid" + thinkClose + "after", wantContent: "beforeafter", wantReasoning: "mid"},
	}

	for _, tc := range cases {
		for trial := range 50 {
			chunks := chunkRandom(tc.input, rng)
			var s Splitter
			var allContent, allReasoning string
			for _, ch := range chunks {
				allContent, allReasoning = appendDeltaText(allContent, allReasoning, s.Push(ch))
			}
			allContent, allReasoning = appendDeltaText(allContent, allReasoning, s.Flush())
			if allContent != tc.wantContent {
				t.Fatalf("%s content mismatch for trial %d: got %q want %q", tc.name, trial, allContent, tc.wantContent)
			}
			if allReasoning != tc.wantReasoning {
				t.Fatalf("%s reasoning mismatch for trial %d: got %q want %q", tc.name, trial, allReasoning, tc.wantReasoning)
			}
		}
	}
}

func TestSplitterTagAcrossChunks(t *testing.T) {
	t.Parallel()

	var s Splitter

	deltas := s.Push("hello")
	if len(deltas) != 1 || deltas[0].Kind != DeltaContent || deltas[0].Text != "hello" {
		t.Fatalf("step 1: got %#v", deltas)
	}

	deltas = s.Push(thinkOpen[:3])
	if len(deltas) != 0 {
		t.Fatalf("step 2 (partial open tag): got %#v", deltas)
	}

	deltas = s.Push(thinkOpen[3:] + "reasoning" + thinkClose[:4])
	if len(deltas) != 1 || deltas[0].Kind != DeltaReasoning || deltas[0].Text != "reasoning" {
		t.Fatalf("step 3 (tag completed + reasoning): got %#v", deltas)
	}

	deltas = s.Push(thinkClose[4:] + "more")
	if len(deltas) != 1 || deltas[0].Kind != DeltaContent || deltas[0].Text != "more" {
		t.Fatalf("step 4 (close tag + content): got %#v", deltas)
	}
}

func TestSplitterFlush(t *testing.T) {
	t.Parallel()

	var s Splitter

	deltas := s.Push("hello" + thinkOpen[:3])
	if len(deltas) != 1 || deltas[0].Kind != DeltaContent || deltas[0].Text != "hello" {
		t.Fatalf("step 1: got %#v", deltas)
	}

	deltas = s.Flush()
	if len(deltas) != 1 || deltas[0].Kind != DeltaContent || deltas[0].Text != thinkOpen[:3] {
		t.Fatalf("step 2 flush: got %#v", deltas)
	}
}

func TestSplitterBudgetZero(t *testing.T) {
	t.Parallel()

	var s Splitter
	s.SetReasoningBudget(0)

	deltas := s.Push(thinkOpen + "secret reasoning" + thinkClose + "visible")
	if got := joinByKind(deltas, DeltaReasoning); got != "" {
		t.Fatalf("budget=0 should suppress all reasoning, got %q", got)
	}
	if got := joinByKind(deltas, DeltaContent); got != "visible" {
		t.Fatalf("budget=0 content got %q want %q", got, "visible")
	}
}

func TestSplitterBudgetPartial(t *testing.T) {
	t.Parallel()

	var s Splitter
	s.SetReasoningBudget(5)

	deltas := s.Push(thinkOpen + "ABCDEFGHIJ" + thinkClose + "hello")
	reasoning := joinByKind(deltas, DeltaReasoning)
	if len([]rune(reasoning)) != 5 {
		t.Fatalf("budget=5 should emit 5 reasoning runes, got %d (%q)", len([]rune(reasoning)), reasoning)
	}
	if got := joinByKind(deltas, DeltaContent); got != "hello" {
		t.Fatalf("content got %q want %q", got, "hello")
	}
}

func TestSplitterBudgetNegative(t *testing.T) {
	t.Parallel()

	var s Splitter
	s.SetReasoningBudget(-1)

	deltas := s.Push(thinkOpen + "all reasoning" + thinkClose + "content")
	if got := joinByKind(deltas, DeltaReasoning); got != "all reasoning" {
		t.Fatalf("budget=-1 should emit all reasoning, got %q", got)
	}
	if got := joinByKind(deltas, DeltaContent); got != "content" {
		t.Fatalf("content got %q want %q", got, "content")
	}
}

func TestSplitterPreservesDeltaOrder(t *testing.T) {
	t.Parallel()

	var s Splitter
	deltas := s.Push("A" + thinkOpen + "R" + thinkClose + "B")

	if len(deltas) != 3 {
		t.Fatalf("expected 3 deltas, got %#v", deltas)
	}
	if deltas[0].Kind != DeltaContent || deltas[0].Text != "A" {
		t.Fatalf("delta 0 got %#v", deltas[0])
	}
	if deltas[1].Kind != DeltaReasoning || deltas[1].Text != "R" {
		t.Fatalf("delta 1 got %#v", deltas[1])
	}
	if deltas[2].Kind != DeltaContent || deltas[2].Text != "B" {
		t.Fatalf("delta 2 got %#v", deltas[2])
	}
}

func chunkRandom(s string, rng *rand.Rand) []string {
	if len(s) == 0 {
		return nil
	}
	var chunks []string
	i := 0
	for i < len(s) {
		remain := len(s) - i
		n := rng.Intn(min(remain, 5)) + 1
		chunks = append(chunks, s[i:i+n])
		i += n
	}
	return chunks
}

func BenchmarkSplitterPushIncremental(b *testing.B) {
	input := buildBenchmarkInput(2000)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var s Splitter
		for _, ch := range input.chunks {
			_ = s.Push(ch)
		}
		_ = s.Flush()
	}
}

type benchInput struct {
	chunks []string
}

func buildBenchmarkInput(nTokens int) benchInput {
	rng := rand.New(rand.NewSource(123))

	var chunks []string
	inThink := false
	for i := range nTokens {
		if !inThink && rng.Float64() < 0.05 {
			chunks = append(chunks, thinkOpen)
			inThink = true
		} else if inThink && rng.Float64() < 0.02 {
			chunks = append(chunks, thinkClose)
			inThink = false
		} else {
			chunks = append(chunks, fmt.Sprintf("tok%d ", i))
		}
	}
	return benchInput{chunks: chunks}
}

func appendDeltaText(content, reasoning string, deltas []Delta) (string, string) {
	for _, delta := range deltas {
		switch delta.Kind {
		case DeltaContent:
			content += delta.Text
		case DeltaReasoning:
			reasoning += delta.Text
		}
	}
	return content, reasoning
}

func joinByKind(deltas []Delta, kind DeltaKind) string {
	var b strings.Builder
	for _, delta := range deltas {
		if delta.Kind == kind {
			b.WriteString(delta.Text)
		}
	}
	return b.String()
}
