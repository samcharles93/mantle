package graph

import "unsafe"

// ComputeContext carries per-invocation parameters that vary per token
// but are NOT baked into the graph structure.
type ComputeContext struct {
	Pos    int            // current token position in the sequence
	KVLen  int            // current KV cache length (for attention)
	Token  int            // current token ID being processed
	Stream unsafe.Pointer // backend-specific stream handle (opaque, nil on CPU)
}
