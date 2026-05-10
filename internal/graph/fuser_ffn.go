package graph

// matchFusedFFN checks whether the 4 node indices form a fusible FFN DAG:
//
//	gate (OpFFNBlock) ─┐
//	                    ├─ add (OpAdd) ── down (OpFFNBlock)
//	up   (OpFFNBlock) ─┘
//
// Gate and up must share the same Input[0] tensor ID.
// The subgraph must be self-contained (intermediate uses all internal).
func matchFusedFFN(g *Graph, nodeIndices []int) bool {
	if len(nodeIndices) != 4 {
		return false
	}
	gateIdx, upIdx, addIdx := nodeIndices[0], nodeIndices[1], nodeIndices[2]
	ops := []OpType{OpFFNBlock, OpFFNBlock, OpAdd, OpFFNBlock}
	outputs := []int{3} // down is the output

	if !FuseSubgraph(g, nodeIndices, ops, outputs) {
		return false
	}

	// Gate and up must share the same first input.
	gate := &g.Nodes[gateIdx]
	up := &g.Nodes[upIdx]
	if len(gate.Input) == 0 || len(up.Input) == 0 {
		return false
	}
	if gate.Input[0] != up.Input[0] {
		return false
	}

	// Verify single-use intermediates via countNodeUses.
	if countNodeUses(g, gateIdx) > 1 {
		return false
	}
	if countNodeUses(g, upIdx) > 1 {
		return false
	}
	if countNodeUses(g, addIdx) > 1 {
		return false
	}

	return true
}

// replaceWithFusedFFN replaces the 4 FFN nodes with a single OpFusedFFN node.
// The fused node preserves:
//   - Input: shared input tensor from gate/up
//   - Output: output tensor from down
//   - Params: FFNParams with Activation inherited from the gate node
func replaceWithFusedFFN(g *Graph, indices []int) {
	if len(indices) != 4 {
		return
	}
	gateIdx, _, _, downIdx := indices[0], indices[1], indices[2], indices[3]

	gate := g.Nodes[gateIdx]
	down := g.Nodes[downIdx]

	ffnParams, _ := gate.Params.(FFNParams)

	fused := Node{
		Op:     OpFusedFFN,
		Branch: BranchFFN,
		Name:   "fused_ffn",
		Input:  append([]TensorID(nil), gate.Input...),
		Output: down.Output,
		Params: ffnParams,
	}

	// Build new node slice by removing the 4 intermediate nodes
	// and inserting the fused node at the position of the first.
	minIdx := indices[0]
	for _, idx := range indices {
		if idx < minIdx {
			minIdx = idx
		}
	}
	// Find max index
	maxIdx := indices[0]
	for _, idx := range indices {
		if idx > maxIdx {
			maxIdx = idx
		}
	}

	newNodes := make([]Node, 0, len(g.Nodes)-3)
	newNodes = append(newNodes, g.Nodes[:minIdx]...)
	newNodes = append(newNodes, fused)
	newNodes = append(newNodes, g.Nodes[maxIdx+1:]...)
	g.Nodes = newNodes
}

// FuseFFN scans the graph for the FFN DAG pattern (gate+up+add+down)
// and replaces the first occurrence with a single OpFusedFFN node.
// Returns true if a fusion was performed.
func FuseFFN(g *Graph) bool {
	for i := 0; i+3 < len(g.Nodes); i++ {
		indices := []int{i, i + 1, i + 2, i + 3}
		if matchFusedFFN(g, indices) {
			replaceWithFusedFFN(g, indices)
			return true
		}
	}
	return false
}
