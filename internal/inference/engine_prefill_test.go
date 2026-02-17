package inference

import (
	"context"
	"errors"
	"testing"
)

type prefillModel struct {
	forwardCalls int
	prefillCalls int
	prefillErr   error
	batchCalls   int
	batchErr     error
}

func (m *prefillModel) ForwardToken(_ int) ([]float32, error) {
	m.forwardCalls++
	return []float32{1, 0}, nil
}

func (m *prefillModel) PrefillTokens(_ []int) ([]float32, error) {
	m.prefillCalls++
	if m.prefillErr != nil {
		return nil, m.prefillErr
	}
	return []float32{1, 0}, nil
}

func (m *prefillModel) ForwardTokens(_ []int) ([][]float32, error) {
	m.batchCalls++
	if m.batchErr != nil {
		return nil, m.batchErr
	}
	return [][]float32{{1, 0}}, nil
}

func (m *prefillModel) Reset() {}

func TestRunWithContextUsesPrefillForwarder(t *testing.T) {
	t.Parallel()

	model := &prefillModel{}
	g := &Generator{
		Model:   model,
		Sampler: newGreedySampler(),
	}

	toks, stats, err := g.RunWithContext(context.Background(), []int{1, 2}, 1, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if model.prefillCalls != 1 {
		t.Fatalf("expected 1 prefill call, got %d", model.prefillCalls)
	}
	if model.forwardCalls != 1 {
		t.Fatalf("expected 1 forward call for generated token, got %d", model.forwardCalls)
	}
	if stats.PromptTokens != 2 {
		t.Fatalf("expected prompt tokens=2, got %d", stats.PromptTokens)
	}
	if len(toks) != 3 {
		t.Fatalf("expected 3 context tokens (2 prompt + 1 generated), got %d", len(toks))
	}
}

func TestRunWithContextFallsBackWhenPrefillFails(t *testing.T) {
	t.Parallel()

	model := &prefillModel{
		prefillErr: errors.New("prefill unavailable"),
		batchErr:   errors.New("batch unavailable"),
	}
	g := &Generator{
		Model:   model,
		Sampler: newGreedySampler(),
	}

	_, stats, err := g.RunWithContext(context.Background(), []int{1, 2}, 0, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if model.prefillCalls != 1 {
		t.Fatalf("expected 1 prefill attempt, got %d", model.prefillCalls)
	}
	if model.forwardCalls != 2 {
		t.Fatalf("expected fallback token-by-token prefill (2 calls), got %d", model.forwardCalls)
	}
	if stats.PromptTokens != 2 {
		t.Fatalf("expected prompt tokens=2, got %d", stats.PromptTokens)
	}
}

func TestRunWithContextPrefersPrefillOverBatchForward(t *testing.T) {
	t.Parallel()

	model := &prefillModel{batchErr: errors.New("batch unsupported")}
	g := &Generator{
		Model:   model,
		Sampler: newGreedySampler(),
	}

	_, stats, err := g.RunWithContext(context.Background(), []int{1, 2}, 0, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if model.prefillCalls != 1 {
		t.Fatalf("expected prefill call, got %d", model.prefillCalls)
	}
	if model.batchCalls != 0 {
		t.Fatalf("expected batch path not to run when prefill is available, got %d", model.batchCalls)
	}
	if stats.PromptTokens != 2 {
		t.Fatalf("expected prompt tokens=2, got %d", stats.PromptTokens)
	}
}
