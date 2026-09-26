package simd

import "fmt"

// PrefillTokens advances model state for a prompt token slice and returns
// logits for the final prompt token.
func (m *Instance) PrefillTokens(tokens []int) ([]float32, error) {
	if len(tokens) == 0 {
		return nil, fmt.Errorf("no tokens to prefill")
	}
	if len(tokens) > m.MaxContext-m.Pos {
		return nil, fmt.Errorf("context length exceeded: %d + %d > %d", m.Pos, len(tokens), m.MaxContext)
	}
	for _, tok := range tokens[:len(tokens)-1] {
		if _, err := m.ForwardTokenGreedy(tok); err != nil {
			return nil, err
		}
	}
	return m.ForwardToken(tokens[len(tokens)-1])
}
