package graph

import "testing"

func TestFuseNormResidual(t *testing.T) {
	// Build graph: add1 -> add2, where add1 output used only by add2
	g := &Graph{
		Nodes: []Node{
			{Op: OpEmbed, Branch: BranchEmbed, Name: "embed", Input: []TensorID{0}, Output: 1, Params: EmbedParams{VocabSize: 10, EmbDim: 8}},
			{Op: OpAdd, Branch: BranchFFN, Name: "add1", Input: []TensorID{1, 0}, Output: 2, Params: FFNParams{}},
			{Op: OpAdd, Branch: BranchFFN, Name: "add2", Input: []TensorID{2, 1}, Output: 3, Params: FFNParams{}},
		},
	}

	ok := FuseNormResidual(g)
	if !ok {
		t.Fatalf("expected fuse to succeed")
	}
	if len(g.Nodes) != 2 {
		t.Fatalf("expected 2 nodes after fuse, got %d", len(g.Nodes))
	}
	if g.Nodes[1].Op != OpFusedNormResidual {
		t.Fatalf("expected fused op at position 1, got %v", g.Nodes[1].Op)
	}
	// Ensure fused node preserved output of second node
	if g.Nodes[1].Output != 3 {
		t.Fatalf("expected fused node output to be 3, got %d", g.Nodes[1].Output)
	}
}
