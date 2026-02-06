package inference

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/samcharles93/mantle/internal/logits"
)

type panicModel struct{}

func (panicModel) ForwardToken(int) ([]float32, error) {
	panic("boom")
}

func (panicModel) Reset() {}

type errOnSecondModel struct {
	calls int
}

func (m *errOnSecondModel) ForwardToken(int) ([]float32, error) {
	m.calls++
	if m.calls == 2 {
		return nil, errors.New("forced forward failure")
	}
	return []float32{1, 0}, nil
}

func (m *errOnSecondModel) Reset() {
	m.calls = 0
}

func newGreedySampler() *logits.Sampler {
	return logits.NewSampler(logits.SamplerConfig{
		Seed:        1,
		Temperature: 1.0,
		TopK:        1,
		TopP:        1.0,
	})
}

func TestGenerateConvertsForwardPanicToError(t *testing.T) {
	t.Parallel()

	_, err := Generate(panicModel{}, newGreedySampler(), []int{1}, 1, nil)
	if err == nil {
		t.Fatalf("expected error")
	}
	if !strings.Contains(err.Error(), "panic in ForwardToken") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestRunWithContextReturnsForwardError(t *testing.T) {
	t.Parallel()

	g := &Generator{
		Model:   &errOnSecondModel{},
		Sampler: newGreedySampler(),
	}

	_, _, err := g.RunWithContext(context.Background(), []int{1}, 2, nil)
	if err == nil {
		t.Fatalf("expected forward error")
	}
	if !strings.Contains(err.Error(), "forced forward failure") {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestRunWithContextConvertsSamplerPanicToError(t *testing.T) {
	t.Parallel()

	g := &Generator{
		Model:   &errOnSecondModel{},
		Sampler: nil, // nil receiver panic in Sample should be converted to an error
	}

	_, _, err := g.RunWithContext(context.Background(), []int{1}, 1, nil)
	if err == nil {
		t.Fatalf("expected sampler error")
	}
	if !strings.Contains(err.Error(), "panic in Sample") {
		t.Fatalf("unexpected error: %v", err)
	}
}
