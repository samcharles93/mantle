package model

// Runtime extends Model with access to mutable configuration needed by the caller.
// Backends should implement this interface so callers can apply runtime overrides
// (e.g. RoPE scaling, cache dtype) without depending on concrete types.
type Runtime interface {
	Model
	ModelConfig() *ModelConfig
	UpdateRoPE()
}
