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

func (t TokenizerConfig) BuildGPT2() (*GPT2Tokenizer, error) {
	unk := -1
	if t.UNKTokenID != 0 {
		unk = t.UNKTokenID
	}
	return NewGPT2(t.Tokens, t.Merges, t.Pre, t.AddBOS, t.AddEOS, t.BOSTokenID, t.EOSTokenID, unk)
}

// TokenString returns the string for a token id when available.
func (t TokenizerConfig) TokenString(id int) string {
	if id < 0 || id >= len(t.Tokens) {
		return ""
	}
	return t.Tokens[id]
}
