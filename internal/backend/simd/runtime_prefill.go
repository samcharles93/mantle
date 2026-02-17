package simd

import "fmt"

// PrefillTokens advances model state for a prompt token slice and returns
// logits for the final prompt token.
//
// It prefers the SIMD batch path when available and falls back to sequential
// token processing for architectures unsupported by ForwardTokens (for example
// recurrent/Mamba/MoE models).
func (m *Instance) PrefillTokens(tokens []int) ([]float32, error) {
	if len(tokens) == 0 {
		return nil, fmt.Errorf("no tokens to prefill")
	}
	if len(tokens) == 1 {
		return m.ForwardToken(tokens[0])
	}

	if batchLogits, err := m.ForwardTokens(tokens); err == nil && len(batchLogits) > 0 {
		return batchLogits[len(batchLogits)-1], nil
	}

	for i, tok := range tokens {
		if i == len(tokens)-1 {
			return m.ForwardToken(tok)
		}
		if _, err := m.ForwardTokenGreedy(tok); err != nil {
			return nil, err
		}
	}
	return nil, fmt.Errorf("unreachable prefill state")
}
