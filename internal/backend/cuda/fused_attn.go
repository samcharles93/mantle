//go:build cuda

package cuda

import (
	"github.com/samcharles93/mantle/internal/graph"
)

// fusedAttentionBlock dispatches a fused QKV→attention→Wo pipeline via the
// existing computeAttention fast path. The graph-level relabeling (T12) sets
// node.Op = OpFusedAttention to signal this capability; the backend already
// uses QKVAttentionProjection for the entire block in a single pass.
func (gr *GraphRuntime) fusedAttentionBlock(
	ctx *graph.ComputeContext,
	params graph.AttentionParams,
	x []float32,
) ([]float32, error) {
	return gr.computeAttention(ctx, params, x)
}
