package simd

import (
	"slices"
	"testing"
)

func TestForwardTokensMatchesSequentialLogits(t *testing.T) {
	const (
		vocabSize = 10
		embdDim   = 8
	)
	tokens := []int{1, 3, 2}
	batched := newTestInstance(vocabSize, embdDim, 1)
	sequential := newTestInstance(vocabSize, embdDim, 1)
	batched.Layers[0].SharedKVSource = -1
	sequential.Layers[0].SharedKVSource = -1

	got, err := (*Instance)(batched).ForwardTokens(tokens)
	if err != nil {
		t.Fatalf("ForwardTokens: %v", err)
	}
	if len(got) != len(tokens) {
		t.Fatalf("logit rows = %d, want %d", len(got), len(tokens))
	}
	for i, tok := range tokens {
		want, err := (*Instance)(sequential).ForwardToken(tok)
		if err != nil {
			t.Fatalf("ForwardToken(%d): %v", tok, err)
		}
		if !floatsEqual(got[i], want, 1e-5) {
			t.Fatalf("logits at position %d differ: got %v, want %v", i, got[i], want)
		}
	}
	if batched.Pos != sequential.Pos {
		t.Fatalf("position = %d, want %d", batched.Pos, sequential.Pos)
	}
	first := slices.Clone(got[0])
	if _, err := (*Instance)(batched).ForwardToken(4); err != nil {
		t.Fatalf("next ForwardToken: %v", err)
	}
	if !slices.Equal(got[0], first) {
		t.Fatal("earlier logits changed after the next token")
	}
}

func TestPrefillTokensMatchesSequentialFinalLogits(t *testing.T) {
	tokens := []int{1, 3, 2}
	prefill := newTestInstance(10, 8, 1)
	sequential := newTestInstance(10, 8, 1)
	prefill.Layers[0].SharedKVSource = -1
	sequential.Layers[0].SharedKVSource = -1

	got, err := (*Instance)(prefill).PrefillTokens(tokens)
	if err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	var want []float32
	for _, tok := range tokens {
		want, err = (*Instance)(sequential).ForwardToken(tok)
		if err != nil {
			t.Fatalf("ForwardToken(%d): %v", tok, err)
		}
	}
	if !floatsEqual(got, want, 1e-5) {
		t.Fatalf("final logits differ: got %v, want %v", got, want)
	}
	if prefill.Pos != sequential.Pos {
		t.Fatalf("position = %d, want %d", prefill.Pos, sequential.Pos)
	}
}
