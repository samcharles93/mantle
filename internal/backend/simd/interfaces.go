package simd

// Model represents a generative language model capable of autoregressive inference.
type Model interface {
	// ForwardToken advances the model by one token and returns the logits for the next token.
	ForwardToken(id int) ([]float32, error)
	// Reset clears the model's internal state (KV cache, etc.).
	Reset()
}

// Runtime extends Model with access to mutable configuration needed by the caller.
// Backends should implement this interface so callers can apply runtime overrides
// (e.g. RoPE scaling, cache dtype) without depending on concrete types.
type Runtime interface {
	Model
	ModelConfig() *ModelConfig
	UpdateRoPE()
}
