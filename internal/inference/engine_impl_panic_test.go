package inference

import (
	"context"
	"strings"
	"testing"

	"github.com/samcharles93/mantle/internal/tokenizer"
)

type panicResetModel struct{}

func (panicResetModel) ForwardToken(id int) ([]float32, error) {
	return []float32{1, 0}, nil
}

func (panicResetModel) Reset() {
	panic("reset boom")
}

type okModel struct{}

func (okModel) ForwardToken(id int) ([]float32, error) {
	return []float32{1, 0}, nil
}

func (okModel) Reset() {}

type panicEncodeTokenizer struct{}

func (panicEncodeTokenizer) Encode(string) ([]int, error) { panic("encode boom") }
func (panicEncodeTokenizer) Decode([]int) (string, error) { return "", nil }
func (panicEncodeTokenizer) TokenString(int) string       { return "" }
func (panicEncodeTokenizer) Decoder() []string            { return []string{"", "a"} }

type okTokenizer struct{}

func (okTokenizer) Encode(string) ([]int, error) { return []int{1}, nil }
func (okTokenizer) Decode([]int) (string, error) { return "", nil }
func (okTokenizer) TokenString(int) string       { return "" }
func (okTokenizer) Decoder() []string            { return []string{"", "a"} }

func TestEngineImplGenerateReusesGenerator(t *testing.T) {
	t.Parallel()

	e := &EngineImpl{
		model:           okModel{},
		tokenizer:       okTokenizer{},
		tokenizerConfig: tokenizer.TokenizerConfig{},
	}
	// First call creates the generator
	_, err := e.Generate(context.Background(), &Request{
		Messages: []tokenizer.Message{{Role: "user", Content: "hello"}},
		Steps:    0,
	}, nil)
	if err != nil {
		t.Fatalf("unexpected error on first call: %v", err)
	}
	if e.generator == nil {
		t.Fatal("expected generator to be stored on EngineImpl")
	}

	// Second call reuses the generator (no reset needed since tokens prefix-match)
	gen := e.generator
	_, err = e.Generate(context.Background(), &Request{
		Messages: []tokenizer.Message{{Role: "user", Content: "hello"}},
		Steps:    0,
	}, nil)
	if err != nil {
		t.Fatalf("unexpected error on second call: %v", err)
	}
	if e.generator != gen {
		t.Fatal("expected generator to be reused")
	}
}

func TestEngineImplResetContext(t *testing.T) {
	t.Parallel()

	e := &EngineImpl{
		model:           panicResetModel{},
		tokenizer:       okTokenizer{},
		tokenizerConfig: tokenizer.TokenizerConfig{},
	}
	// First call succeeds — no reset needed
	_, err := e.Generate(context.Background(), &Request{
		Messages: []tokenizer.Message{{Role: "user", Content: "hello"}},
		Steps:    0,
	}, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// ResetContext on a model that panics on Reset — verify it propagates
	defer func() {
		if r := recover(); r == nil {
			t.Fatal("expected panic from ResetContext")
		}
	}()
	e.ResetContext()
}

func TestEngineImplGenerateConvertsEncodePanic(t *testing.T) {
	t.Parallel()

	e := &EngineImpl{
		model:           okModel{},
		tokenizer:       panicEncodeTokenizer{},
		tokenizerConfig: tokenizer.TokenizerConfig{},
	}
	_, err := e.Generate(context.Background(), &Request{
		Messages: []tokenizer.Message{{Role: "user", Content: "hello"}},
		Steps:    0,
	}, nil)
	if err == nil {
		t.Fatalf("expected error")
	}
	if !strings.Contains(err.Error(), "panic in Encode") {
		t.Fatalf("unexpected error: %v", err)
	}
}
