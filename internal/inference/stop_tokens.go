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

	if t, ok := tok.(interface{ TokenString(int) string }); ok {
		token2 := strings.ToLower(strings.TrimSpace(t.TokenString(2)))
		if token2 == "<|endoftext|>" || token2 == "<|end_of_text|>" || token2 == "</s>" {
			if cfg.EOSTokenID != 2 && cfg.EOSTokenID >= 0 {
				stopTokens = append(stopTokens, 2)
			} else if cfg.EOSTokenID < 0 {
				stopTokens = append(stopTokens, 2)
			}
		}
	}
	return stopTokens
}
