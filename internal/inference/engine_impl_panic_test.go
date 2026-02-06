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

func TestEngineImplGenerateConvertsResetPanic(t *testing.T) {
	t.Parallel()

	e := &EngineImpl{
		model:           panicResetModel{},
		tokenizer:       okTokenizer{},
		tokenizerConfig: tokenizer.TokenizerConfig{},
	}
	_, err := e.Generate(context.Background(), &Request{
		Messages: []tokenizer.Message{{Role: "user", Content: "hello"}},
		Steps:    0,
	}, nil)
	if err == nil {
		t.Fatalf("expected error")
	}
	if !strings.Contains(err.Error(), "panic in Reset") {
		t.Fatalf("unexpected error: %v", err)
	}
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
