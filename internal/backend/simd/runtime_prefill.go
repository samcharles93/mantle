package simd

import (
	"errors"
	"fmt"
)

// PrefillTokens advances model state for a prompt token slice and returns
// logits for the final prompt token.
func (m *Instance) PrefillTokens(tokens []int) ([]float32, error) {
	if len(tokens) == 0 {
		return nil, fmt.Errorf("no tokens to prefill")
	}
	if len(tokens) > m.MaxContext-m.Pos {
		return nil, fmt.Errorf("context length exceeded: %d + %d > %d", m.Pos, len(tokens), m.MaxContext)
	}
	// Plain dense models run the prompt through the batched path, in chunks of at
	// most MaxBatch positions: only the final row's logits are returned, and
	// every row's KV state is written for the decode steps that follow. Anything
	// the gate refuses falls through to the per-token loop below, which stays the
	// correctness oracle.
	var last []float32
	err := m.forEachBatchLogits(tokens, func(logits []float32) { last = logits })
	if err == nil {
		return last, nil
	}
	if !errors.Is(err, errBatchIneligible) {
		return nil, err
	}
	for _, tok := range tokens[:len(tokens)-1] {
		if _, err := m.ForwardTokenGreedy(tok); err != nil {
			return nil, err
		}
	}
	return m.ForwardToken(tokens[len(tokens)-1])
}
