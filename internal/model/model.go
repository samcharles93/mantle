package model

// Model represents a generative language model capable of autoregressive inference.
type Model interface {
	// ForwardToken advances the model by one token and returns the logits for the next token.
	ForwardToken(id int) ([]float32, error)
	// Reset clears the model's internal state (KV cache, etc.)
	Reset()
}
