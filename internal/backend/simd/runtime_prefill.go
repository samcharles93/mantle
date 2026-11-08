package simd

import (
	"errors"
	"fmt"
)

// PrefillTokens advances model state for a prompt token slice and returns
// logits for the final prompt token.
func (m *Instance) PrefillTokens(tokens []int) ([]float32, error) {
	return m.prefillTokens(tokens, nil)
}

// PrefillTokensEmbeds is PrefillTokens with optional per-position input
// embedding overrides (embeds parallel to tokens; nil entries or a nil slice
// use the token embedding). Used to splice vision embeddings at image-token
// positions during multimodal prefill.
func (m *Instance) PrefillTokensEmbeds(tokens []int, embeds [][]float32) ([]float32, error) {
	if embeds != nil && len(embeds) != len(tokens) {
		return nil, fmt.Errorf("embeds length %d != tokens %d", len(embeds), len(tokens))
	}
	return m.prefillTokens(tokens, embeds)
}

func (m *Instance) prefillTokens(tokens []int, embeds [][]float32) ([]float32, error) {
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
	err := m.forEachBatchLogitsEmbeds(tokens, embeds, emitLastRow, func(logits []float32) { last = logits })
	if err == nil {
		return last, nil
	}
	if !errors.Is(err, errBatchIneligible) {
		return nil, err
	}
	embedAt := func(i int) []float32 {
		if embeds != nil && i < len(embeds) {
			return embeds[i]
		}
		return nil
	}
	for i, tok := range tokens[:len(tokens)-1] {
		if e := embedAt(i); e != nil {
			if _, err := m.ForwardTokenEmbed(tok, e); err != nil {
				return nil, err
			}
		} else if _, err := m.ForwardTokenGreedy(tok); err != nil {
			return nil, err
		}
	}
	if e := embedAt(len(tokens) - 1); e != nil {
		return m.ForwardTokenEmbed(tokens[len(tokens)-1], e)
	}
	return m.ForwardToken(tokens[len(tokens)-1])
}
