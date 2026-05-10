package graph

// FuseMoE scans the graph and relabels all OpMoEBlock nodes to OpFusedMoE.
// The MoE block is already a single node in the graph; "fusion" is a
// capability flag that tells the backend it can execute the block as a
// fused kernel rather than dispatching individual sub-operations.
//
// All node fields (Branch, Name, Input, Output, Params) are preserved.
// Returns true if any nodes were relabeled.
func FuseMoE(g *Graph) bool {
	if g == nil || len(g.Nodes) == 0 {
		return false
	}

	changed := false
	for i := range g.Nodes {
		if g.Nodes[i].Op == OpMoEBlock {
			g.Nodes[i].Op = OpFusedMoE
			changed = true
		}
	}
	return changed
}
