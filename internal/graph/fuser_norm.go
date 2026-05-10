package graph

// FuseNormResidual searches for a simple linear pattern: OpAdd -> OpAdd
// where the first OpAdd has exactly one use (consumed by the second OpAdd).
// When found, it replaces the two nodes with a single OpFusedNormResidual
// node that preserves the output TensorID of the second node and carries
// over Params from the first node when available.
func FuseNormResidual(g *Graph) bool {
	if g == nil || len(g.Nodes) < 2 {
		return false
	}

	// Scan nodes for the pattern
	for i := 0; i < len(g.Nodes)-1; i++ {
		if FuseLinear(g, i, []OpType{OpAdd, OpAdd}) {
			// Ensure the first node has exactly 1 use (FuseLinear already checks this,
			// but we re-check to be explicit about the placeholder pattern semantics).
			if countNodeUses(g, i) != 1 {
				continue
			}

			first := g.Nodes[i]
			second := g.Nodes[i+1]

			// Build fused node
			fused := Node{
				Op:     OpFusedNormResidual,
				Branch: second.Branch, // choose last node's branch for dispatch
				Name:   first.Name + "+" + second.Name + ":fused_norm_residual",
				Input:  append([]TensorID(nil), first.Input...),
				Output: second.Output, // preserve final output
				Params: first.Params,  // carry params from first if meaningful
			}

			// Rebuild node list: nodes before i, fused node, nodes after i+1
			newNodes := make([]Node, 0, len(g.Nodes)-1)
			newNodes = append(newNodes, g.Nodes[:i]...)
			newNodes = append(newNodes, fused)
			if i+2 < len(g.Nodes) {
				newNodes = append(newNodes, g.Nodes[i+2:]...)
			}
			g.Nodes = newNodes
			return true
		}
	}
	return false
}
