package graph

import "github.com/samcharles93/mantle/internal/backend/core"

// Builder constructs a static forward-pass computation graph for a single token
// from a model's configuration and instance state. The returned graph is a
// topology-only description: it does not reference runtime state such as KV
// caches or position, and is built once per model.
type Builder interface {
	// BuildGraph constructs a complete forward-pass graph for a single token,
	// using the model's configuration and instance state.
	BuildGraph(cfg *core.Config, inst *core.Instance) (*Graph, error)
}
