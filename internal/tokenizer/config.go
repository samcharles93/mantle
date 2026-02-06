package tokenizer

type TokenizerConfig struct {
	Model        string
	Pre          string
	AddBOS       bool
	AddEOS       bool
	BOSTokenID   int
	EOSTokenID   int
	PADTokenID   int
	UNKTokenID   int
	Tokens       []string
	Merges       []string
	TokenTypes   []int32
	ChatTemplate string
}

// TokenString returns the string for a token id when available.
func (t TokenizerConfig) TokenString(id int) string {
	if id < 0 || id >= len(t.Tokens) {
		return ""
	}
	return t.Tokens[id]
}
