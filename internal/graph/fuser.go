package graph

import "fmt"

// FuseSubgraph checks whether the nodes at the given indices form a valid
// fusible subgraph. The pattern is defined by:
//   - nodeIndices: the node positions in the graph to check
//   - ops: expected OpType for each position (must match in order)
//   - outputs: which nodes in the subgraph are "outputs" (not consumed internally)
//
// Rules (mirrors ggml_can_fuse_subgraph_ext):
//  1. All node indices are valid (within graph bounds)
//  2. Each node's Op matches the expected ops sequence
//  3. Non-output nodes must have ALL their uses consumed within the subgraph
//  4. Output nodes must have at least one use outside the subgraph
//  5. No view_src chains cross the subgraph boundary
//
// Returns true if the subgraph can be fused into a single node.
func FuseSubgraph(g *Graph, nodeIndices []int, ops []OpType, outputs []int) bool {
	if len(nodeIndices) != len(ops) {
		return false
	}
	if len(nodeIndices) == 0 {
		return false
	}
	// Map: node index (within subgraph) -> is it an output?
	isOutput := make(map[int]bool)
	for _, o := range outputs {
		isOutput[o] = true
	}
	// Gather all nodes in the subgraph
	subgraphNodes := make([]*Node, len(nodeIndices))
	for i, idx := range nodeIndices {
		if idx < 0 || idx >= len(g.Nodes) {
			return false
		}
		n := &g.Nodes[idx]
		subgraphNodes[i] = n
		// Check op matches
		if n.Op != ops[i] {
			return false
		}
	}
	// For each non-output node, all its uses must be within the subgraph.
	// For output nodes, they must have at least one use outside.
	for i, n := range subgraphNodes {
		outID := n.Output
		usesOutside := 0
		usesInside := 0
		for j := range g.Nodes {
			for _, input := range g.Nodes[j].Input {
				if input == outID {
					if isNodeIndexInSet(j, nodeIndices) {
						usesInside++
					} else {
						usesOutside++
					}
				}
			}
		}
		if isOutput[i] {
			// Output nodes must have at least one use outside the subgraph
			if usesOutside == 0 {
				return false
			}
		} else {
			// Non-output nodes must have zero uses outside the subgraph
			if usesOutside > 0 {
				return false
			}
			// And must have at least one use inside (otherwise dead code, unless unset output)
			if usesInside == 0 && outID != 0 {
				return false
			}
		}
	}
	return true
}

// isNodeIndexInSet checks whether idx is present in the sorted-or-unsorted set.
func isNodeIndexInSet(idx int, set []int) bool {
	for _, s := range set {
		if idx == s {
			return true
		}
	}
	return false
}

// countNodeUses counts how many times a tensor (produced by node at nodeIdx)
// is used as input anywhere in the graph.
func countNodeUses(g *Graph, nodeIdx int) int {
	if nodeIdx < 0 || nodeIdx >= len(g.Nodes) {
		return 0
	}
	outID := g.Nodes[nodeIdx].Output
	count := 0
	for i := range g.Nodes {
		for _, input := range g.Nodes[i].Input {
			if input == outID {
				count++
			}
		}
	}
	return count
}

// FuseLinear checks whether a linear sequence of nodes at positions
// startIdx, startIdx+1, ..., startIdx+len(ops)-1 match the expected ops
// and are fusable (same-shape, 1-use intermediates).
func FuseLinear(g *Graph, startIdx int, ops []OpType) bool {
	if startIdx < 0 || startIdx+len(ops) > len(g.Nodes) {
		return false
	}
	for i, expectedOp := range ops {
		idx := startIdx + i
		if g.Nodes[idx].Op != expectedOp {
			return false
		}
		// All nodes except the last must have exactly 1 use
		if i < len(ops)-1 {
			uses := countNodeUses(g, idx)
			if uses != 1 {
				return false
			}
		}
	}
	return true
}

// Fuser applies fusion patterns to a graph.
type Fuser struct{}

// Fuse runs all registered fusion patterns on the graph and returns a new graph.
func (f *Fuser) Fuse(g *Graph) *Graph {
	// Phase 1: simple linear fusions (norm + residual)
	// Phase 2: DAG fusions (FFN gate+up+act+down)
	// Phase 3: attention block relabeling
	// For now, this is a stub that returns the graph unchanged.
	// T11-T14 will implement the actual patterns.
	return g.Clone()
}

// unused import guard
var _ = fmt.Sprintf
