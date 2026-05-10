package graph

import "testing"

func TestMatchFusedFFN(t *testing.T) {
	// Construct FFN DAG: gate and up share same input, followed by add, then down.
	g := &Graph{
		Nodes: []Node{
			// 0: embed (provides shared input)
			{Op: OpEmbed, Branch: BranchEmbed, Name: "embed", Input: []TensorID{0}, Output: 1, Params: EmbedParams{VocabSize: 100, EmbDim: 16}},
			// 1: gate (OpFFNBlock, input=1 same as up)
			{Op: OpFFNBlock, Branch: BranchFFN, Name: "gate", Input: []TensorID{1}, Output: 2, Params: FFNParams{Activation: "silu"}},
			// 2: up (OpFFNBlock, input=1 same as gate)
			{Op: OpFFNBlock, Branch: BranchFFN, Name: "up", Input: []TensorID{1}, Output: 3, Params: FFNParams{}},
			// 3: add (gate + up → gated activation)
			{Op: OpAdd, Branch: BranchFFN, Name: "gated", Input: []TensorID{2, 3}, Output: 4, Params: FFNParams{}},
			// 4: down (OpFFNBlock)
			{Op: OpFFNBlock, Branch: BranchFFN, Name: "down", Input: []TensorID{4}, Output: 5, Params: FFNParams{}},
			// 5: next node (consumer of down, provides outside use)
			{Op: OpOutput, Branch: BranchOutput, Name: "out", Input: []TensorID{5}, Output: 6, Params: OutputParams{}},
		},
	}

	// Indices 1-4 form the FFN pattern: gate, up, add, down
	if !matchFusedFFN(g, []int{1, 2, 3, 4}) {
		t.Fatal("expected FFN pattern to match")
	}
}

func TestMatchFusedFFNMismatchedOps(t *testing.T) {
	g := &Graph{
		Nodes: []Node{
			{Op: OpEmbed, Branch: BranchEmbed, Name: "embed", Input: []TensorID{0}, Output: 1, Params: EmbedParams{VocabSize: 100, EmbDim: 16}},
			{Op: OpAttentionBlock, Branch: BranchAttention, Name: "attn", Input: []TensorID{1}, Output: 2, Params: AttentionParams{NHeadKV: 4, HeadDim: 64}},
			{Op: OpFFNBlock, Branch: BranchFFN, Name: "up", Input: []TensorID{1}, Output: 3, Params: FFNParams{}},
			{Op: OpAdd, Branch: BranchFFN, Name: "gated", Input: []TensorID{2, 3}, Output: 4, Params: FFNParams{}},
			{Op: OpFFNBlock, Branch: BranchFFN, Name: "down", Input: []TensorID{4}, Output: 5, Params: FFNParams{}},
			{Op: OpOutput, Branch: BranchOutput, Name: "out", Input: []TensorID{5}, Output: 6, Params: OutputParams{}},
		},
	}

	// First node is OpAttentionBlock, not OpFFNBlock
	if matchFusedFFN(g, []int{1, 2, 3, 4}) {
		t.Fatal("expected mismatch: first node is not OpFFNBlock")
	}
}

func TestMatchFusedFFNDifferentInputs(t *testing.T) {
	g := &Graph{
		Nodes: []Node{
			{Op: OpEmbed, Branch: BranchEmbed, Name: "embed", Input: []TensorID{0}, Output: 1, Params: EmbedParams{VocabSize: 100, EmbDim: 16}},
			// gate and up have different inputs → not the FFN pattern
			{Op: OpFFNBlock, Branch: BranchFFN, Name: "gate", Input: []TensorID{1}, Output: 2, Params: FFNParams{Activation: "silu"}},
			{Op: OpFFNBlock, Branch: BranchFFN, Name: "up", Input: []TensorID{99}, Output: 3, Params: FFNParams{}},
			{Op: OpAdd, Branch: BranchFFN, Name: "gated", Input: []TensorID{2, 3}, Output: 4, Params: FFNParams{}},
			{Op: OpFFNBlock, Branch: BranchFFN, Name: "down", Input: []TensorID{4}, Output: 5, Params: FFNParams{}},
			{Op: OpOutput, Branch: BranchOutput, Name: "out", Input: []TensorID{5}, Output: 6, Params: OutputParams{}},
		},
	}

	if matchFusedFFN(g, []int{1, 2, 3, 4}) {
		t.Fatal("expected mismatch: gate and up have different inputs")
	}
}

func TestReplaceWithFusedFFN(t *testing.T) {
	g := &Graph{
		Nodes: []Node{
			{Op: OpEmbed, Branch: BranchEmbed, Name: "embed", Input: []TensorID{0}, Output: 1, Params: EmbedParams{VocabSize: 100, EmbDim: 16}},
			{Op: OpFFNBlock, Branch: BranchFFN, Name: "gate", Input: []TensorID{1}, Output: 2, Params: FFNParams{Activation: "silu"}},
			{Op: OpFFNBlock, Branch: BranchFFN, Name: "up", Input: []TensorID{1}, Output: 3, Params: FFNParams{}},
			{Op: OpAdd, Branch: BranchFFN, Name: "gated", Input: []TensorID{2, 3}, Output: 4, Params: FFNParams{}},
			{Op: OpFFNBlock, Branch: BranchFFN, Name: "down", Input: []TensorID{4}, Output: 5, Params: FFNParams{}},
			{Op: OpOutput, Branch: BranchOutput, Name: "out", Input: []TensorID{5}, Output: 6, Params: OutputParams{}},
		},
	}

	replaceWithFusedFFN(g, []int{1, 2, 3, 4})

	// After replacement: embed, fused_ffn, out → 3 nodes
	if len(g.Nodes) != 3 {
		t.Fatalf("expected 3 nodes after fusion, got %d", len(g.Nodes))
	}

	// Check the fused node at index 1
	fused := g.Nodes[1]
	if fused.Op != OpFusedFFN {
		t.Errorf("expected OpFusedFFN, got %v", fused.Op)
	}
	if fused.Branch != BranchFFN {
		t.Errorf("expected BranchFFN, got %v", fused.Branch)
	}
	if len(fused.Input) != 1 || fused.Input[0] != 1 {
		t.Errorf("expected input [1], got %v", fused.Input)
	}
	if fused.Output != 5 {
		t.Errorf("expected output 5, got %d", fused.Output)
	}
	ffnParams, ok := fused.Params.(FFNParams)
	if !ok {
		t.Fatalf("expected FFNParams, got %T", fused.Params)
	}
	if ffnParams.Activation != "silu" {
		t.Errorf("expected activation 'silu', got %q", ffnParams.Activation)
	}
}

func TestFuseFFN(t *testing.T) {
	// Test full scan and replace via FuseFFN.
	g := &Graph{
		Nodes: []Node{
			{Op: OpEmbed, Branch: BranchEmbed, Name: "embed", Input: []TensorID{0}, Output: 1, Params: EmbedParams{VocabSize: 100, EmbDim: 16}},
			{Op: OpFFNBlock, Branch: BranchFFN, Name: "gate", Input: []TensorID{1}, Output: 2, Params: FFNParams{Activation: "silu"}},
			{Op: OpFFNBlock, Branch: BranchFFN, Name: "up", Input: []TensorID{1}, Output: 3, Params: FFNParams{}},
			{Op: OpAdd, Branch: BranchFFN, Name: "gated", Input: []TensorID{2, 3}, Output: 4, Params: FFNParams{}},
			{Op: OpFFNBlock, Branch: BranchFFN, Name: "down", Input: []TensorID{4}, Output: 5, Params: FFNParams{}},
			{Op: OpOutput, Branch: BranchOutput, Name: "out", Input: []TensorID{5}, Output: 6, Params: OutputParams{}},
		},
	}

	if !FuseFFN(g) {
		t.Fatal("expected FuseFFN to return true")
	}

	if len(g.Nodes) != 3 {
		t.Fatalf("expected 3 nodes after fusion, got %d", len(g.Nodes))
	}

	fused := g.Nodes[1]
	if fused.Op != OpFusedFFN {
		t.Errorf("expected OpFusedFFN, got %v", fused.Op)
	}
}

func TestFuseFFNNoMatch(t *testing.T) {
	g := &Graph{
		Nodes: []Node{
			{Op: OpEmbed, Branch: BranchEmbed, Name: "embed", Input: []TensorID{0}, Output: 1, Params: EmbedParams{VocabSize: 100, EmbDim: 16}},
			{Op: OpAttentionBlock, Branch: BranchAttention, Name: "attn", Input: []TensorID{1}, Output: 2, Params: AttentionParams{NHeadKV: 4, HeadDim: 64}},
			{Op: OpOutput, Branch: BranchOutput, Name: "out", Input: []TensorID{2}, Output: 3, Params: OutputParams{}},
		},
	}

	if FuseFFN(g) {
		t.Fatal("expected FuseFFN to return false (no FFN pattern)")
	}
}
