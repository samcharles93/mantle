package tokenizer

// Tokenizer defines the minimal interface used by the CLI.
type Tokenizer interface {
	Encode(text string) ([]int, error)
	Decode(ids []int) (string, error)
}
