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

	token2 := strings.ToLower(strings.TrimSpace(tok.TokenString(2)))
	if token2 == "<|endoftext|>" || token2 == "<|end_of_text|>" || token2 == "</s>" {
		addUnique(2)
	}

	for id, token := range cfg.Tokens {
		if token == "<|im_end|>" {
			addUnique(id)
			break
		}
	}

	for id, token := range tok.Decoder() {
		if token == "<|im_end|>" {
			addUnique(id)
			break
		}
	}
	return stopTokens
}
