package inference

import (
	"strings"

	"github.com/samcharles93/mantle/internal/tokenizer"
)

func BuildStopTokens(tok tokenizer.Tokenizer, cfg tokenizer.TokenizerConfig) []int {
	stopTokens := []int{cfg.EOSTokenID}
	if cfg.EOSTokenID < 0 {
		stopTokens = stopTokens[:0]
	}

	addUnique := func(id int) {
		if id < 0 {
			return
		}
		for _, v := range stopTokens {
			if v == id {
				return
			}
		}
		stopTokens = append(stopTokens, id)
	}

	isKnownStopTokenString := func(s string) bool {
		switch strings.ToLower(strings.TrimSpace(s)) {
		case "<|im_end|>", "<|eot_id|>", "</s>":
			return true
		default:
			return false
		}
	}

	// Legacy fallback for tokenizers that don't expose a decoder table.
	if isKnownStopTokenString(tok.TokenString(2)) {
		addUnique(2)
	}

	for id, token := range cfg.Tokens {
		if isKnownStopTokenString(token) {
			addUnique(id)
		}
	}

	for id, token := range tok.Decoder() {
		if isKnownStopTokenString(token) {
			addUnique(id)
		}
	}
	return stopTokens
}
