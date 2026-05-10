package simd

import "github.com/samcharles93/mantle/internal/graph"

// fusedAttention is the backend entry point for OpFusedAttention graph nodes.
// Since attention is already a single block-level kernel in the SIMD backend,
// this delegates to the existing Attention() function. The "fused" aspect is
// the graph-level relabeling (see graph.FuseAttention) and the backend dispatch
// choosing this fast path.
func fusedAttention(m *Instance, layer *Layer, input []float32, pos int, _ graph.AttentionParams) []float32 {
	return Attention(m, layer, input, pos)
}
