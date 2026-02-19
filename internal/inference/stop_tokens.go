package inference

import (
	"slices"

	"github.com/samcharles93/mantle/internal/tokenizer"
)

// BuildStopTokens returns the set of token IDs that should halt generation.
// The primary source is cfg.EOSTokenID (from the tokenizer config).
// extraIDs comes from generation_config.json's eos_token_id, which may list
// additional turn-end tokens (e.g. Gemma's <end_of_turn>=106, Llama3's special ids).
func BuildStopTokens(cfg tokenizer.TokenizerConfig, extraIDs []int) []int {
	stopTokens := []int{}
	addUnique := func(id int) {
		if id < 0 {
			return
		}
		if slices.Contains(stopTokens, id) {
			return
		}
		stopTokens = append(stopTokens, id)
	}
	addUnique(cfg.EOSTokenID)
	for _, id := range extraIDs {
		addUnique(id)
	}
	return stopTokens
}
