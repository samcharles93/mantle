package graph

import "testing"

func TestGraphValidate(t *testing.T) {
	g := &Graph{}
	// Empty graph should fail
	if err := g.Validate(); err == nil {
		t.Fatal("expected error for empty graph")
	}

	// Single embed node with valid structure
	g.Nodes = []Node{
		{Op: OpEmbed, Branch: BranchEmbed, Name: "embed", Input: []TensorID{NewTensorID()}, Output: NewTensorID(), Params: EmbedParams{VocabSize: 100, EmbDim: 16}},
	}
	if err := g.Validate(); err != nil {
		t.Fatalf("valid graph rejected: %v", err)
	}

	// Node referencing non-existent input
	g2 := &Graph{
		Nodes: []Node{
			{Op: OpEmbed, Branch: BranchEmbed, Name: "embed", Input: []TensorID{NewTensorID()}, Output: NewTensorID(), Params: EmbedParams{VocabSize: 100, EmbDim: 16}},
			{Op: OpAttentionBlock, Branch: BranchAttention, Name: "attn", Input: []TensorID{999}, Output: NewTensorID(), Params: AttentionParams{LayerIndex: 0, HeadDim: 8, NHeadKV: 2}},
		},
	}
	if err := g2.Validate(); err == nil {
		t.Fatal("expected error for missing input tensor reference")
	}
}

func TestGraphClone(t *testing.T) {
	ResetTensorID()
	g := &Graph{
		Nodes: []Node{
			{Op: OpEmbed, Branch: BranchEmbed, Name: "e", Input: []TensorID{1}, Output: 2, Params: EmbedParams{VocabSize: 10, EmbDim: 4}},
			{Op: OpAttentionBlock, Branch: BranchAttention, Name: "a", Input: []TensorID{2}, Output: 3, Params: AttentionParams{LayerIndex: 0, HeadDim: 4, NHeadKV: 1}},
		},
		UID: 42,
	}
	c := g.Clone()
	if c == nil {
		t.Fatal("Clone returned nil")
	}
	if c.UID != g.UID {
		t.Fatalf("UID mismatch: %d vs %d", c.UID, g.UID)
	}
	if len(c.Nodes) != len(g.Nodes) {
		t.Fatalf("node count mismatch: %d vs %d", len(c.Nodes), len(g.Nodes))
	}
	// Modify clone; original must be unaffected.
	c.Nodes[0].Name = "modified"
	if g.Nodes[0].Name == "modified" {
		t.Fatal("Clone mutation propagated to original")
	}
}

func TestGraphCloneNil(t *testing.T) {
	var g *Graph
	if g.Clone() != nil {
		t.Fatal("nil graph clone must return nil")
	}
}

func TestNodeValidate(t *testing.T) {
	tests := []struct {
		name    string
		node    Node
		wantErr bool
	}{
		{
			name: "valid attention",
			node: Node{Op: OpAttentionBlock, Branch: BranchAttention, Name: "attn", Input: []TensorID{1}, Output: 2, Params: AttentionParams{LayerIndex: 0}},
		},
		{
			name:    "missing params",
			node:    Node{Op: OpAttentionBlock, Branch: BranchAttention, Name: "attn", Input: []TensorID{1}, Output: 2, Params: nil},
			wantErr: true,
		},
		{
			name:    "wrong params type",
			node:    Node{Op: OpFFNBlock, Branch: BranchFFN, Name: "ffn", Input: []TensorID{1}, Output: 2, Params: AttentionParams{}},
			wantErr: true,
		},
		{
			name:    "no input",
			node:    Node{Op: OpEmbed, Branch: BranchEmbed, Name: "e", Input: []TensorID{}, Output: 1, Params: EmbedParams{}},
			wantErr: true,
		},
		{
			name:    "zero output",
			node:    Node{Op: OpEmbed, Branch: BranchEmbed, Name: "e", Input: []TensorID{1}, Output: 0, Params: EmbedParams{}},
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.node.Validate()
			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}
