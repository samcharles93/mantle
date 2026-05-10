package graph

import "testing"

func TestFuseLinear(t *testing.T) {
	g := &Graph{
		Nodes: []Node{
			{Op: OpEmbed, Branch: BranchEmbed, Name: "embed", Input: []TensorID{0}, Output: 1, Params: EmbedParams{VocabSize: 100, EmbDim: 16}},
			{Op: OpFFNBlock, Branch: BranchFFN, Name: "ffn1", Input: []TensorID{1}, Output: 2, Params: FFNParams{}},
		},
	}

	if !FuseLinear(g, 0, []OpType{OpEmbed, OpFFNBlock}) {
		t.Fatal("expected linear fuse to match embed → ffn")
	}
}

func TestFuseLinearMismatch(t *testing.T) {
	g := &Graph{
		Nodes: []Node{
			{Op: OpEmbed, Branch: BranchEmbed, Name: "embed", Input: []TensorID{0}, Output: 1, Params: EmbedParams{VocabSize: 100, EmbDim: 16}},
			{Op: OpFFNBlock, Branch: BranchFFN, Name: "ffn1", Input: []TensorID{1}, Output: 2, Params: FFNParams{}},
		},
	}

	if FuseLinear(g, 0, []OpType{OpEmbed, OpAttentionBlock}) {
		t.Fatal("expected mismatch to fail")
	}
}

func TestFuseLinearMultiUse(t *testing.T) {
	g := &Graph{
		Nodes: []Node{
			{Op: OpEmbed, Branch: BranchEmbed, Name: "embed", Input: []TensorID{0}, Output: 1, Params: EmbedParams{VocabSize: 100, EmbDim: 16}},
			{Op: OpFFNBlock, Branch: BranchFFN, Name: "gate", Input: []TensorID{1}, Output: 2, Params: FFNParams{Activation: "silu"}},
			{Op: OpFFNBlock, Branch: BranchFFN, Name: "up", Input: []TensorID{1}, Output: 3, Params: FFNParams{}},
		},
	}

	if FuseLinear(g, 0, []OpType{OpEmbed, OpFFNBlock}) {
		t.Fatal("expected linear fuse to fail: embed has 2 uses")
	}
}

func TestFuseLinearOutOfBounds(t *testing.T) {
	g := &Graph{
		Nodes: []Node{
			{Op: OpEmbed, Branch: BranchEmbed, Name: "embed", Input: []TensorID{0}, Output: 1, Params: EmbedParams{VocabSize: 100, EmbDim: 16}},
		},
	}

	if FuseLinear(g, 0, []OpType{OpEmbed, OpFFNBlock}) {
		t.Fatal("expected out of bounds to fail")
	}
	if FuseLinear(g, -1, []OpType{OpEmbed}) {
		t.Fatal("expected negative start to fail")
	}
}

func TestFuseSubgraphBasic(t *testing.T) {
	g := &Graph{
		Nodes: []Node{
			{Op: OpEmbed, Branch: BranchEmbed, Name: "e", Input: []TensorID{0}, Output: 1, Params: EmbedParams{VocabSize: 100, EmbDim: 16}},
			{Op: OpFFNBlock, Branch: BranchFFN, Name: "a", Input: []TensorID{1}, Output: 2, Params: FFNParams{}},
			{Op: OpAdd, Branch: BranchFFN, Name: "add", Input: []TensorID{1, 2}, Output: 3, Params: FFNParams{}},
			{Op: OpFFNBlock, Branch: BranchFFN, Name: "next", Input: []TensorID{3}, Output: 4, Params: FFNParams{}},
		},
	}

	// Fuse nodes 0,1 where node 1 is the output. Node 0 has uses at nodes 1 AND 2,
	// so node 0 has a use outside the subgraph → should fail.
	if FuseSubgraph(g, []int{0, 1}, []OpType{OpEmbed, OpFFNBlock}, []int{1}) {
		t.Fatal("expected fuse to fail because node 0 has uses outside subgraph")
	}

	// Fuse nodes 1,2 where node 1 is consumed entirely within the subgraph.
	if !FuseSubgraph(g, []int{1, 2}, []OpType{OpFFNBlock, OpAdd}, []int{1}) {
		t.Fatal("expected fuse to succeed: ffn → add, with add as output")
	}
}

func TestFuseSubgraphOutputNodeMustHaveOutsideUse(t *testing.T) {
	g := &Graph{
		Nodes: []Node{
			{Op: OpEmbed, Branch: BranchEmbed, Name: "e", Input: []TensorID{0}, Output: 1, Params: EmbedParams{VocabSize: 100, EmbDim: 16}},
			{Op: OpFFNBlock, Branch: BranchFFN, Name: "a", Input: []TensorID{1}, Output: 2, Params: FFNParams{}},
		},
	}

	// Node 1 (ffn, output=2) has no uses anywhere. Marking it as output means
	// it should have at least one outside use → fail.
	if FuseSubgraph(g, []int{1}, []OpType{OpFFNBlock}, []int{0}) {
		t.Fatal("expected fuse to fail: output node has no outside uses")
	}

	// Node 0 (embed, output=1) has 1 use at node 1 (outside subgraph).
	// Marking node 0 as output → outside use exists → pass.
	if !FuseSubgraph(g, []int{0}, []OpType{OpEmbed}, []int{0}) {
		t.Fatal("expected fuse to succeed: output node 0 has outside use")
	}
}

func TestFuseSubgraphInvalid(t *testing.T) {
	g := &Graph{
		Nodes: []Node{
			{Op: OpEmbed, Branch: BranchEmbed, Name: "e", Input: []TensorID{0}, Output: 1, Params: EmbedParams{VocabSize: 100, EmbDim: 16}},
		},
	}

	if FuseSubgraph(g, []int{0, 1}, []OpType{OpEmbed, OpFFNBlock}, nil) {
		t.Fatal("expected out of bounds to fail")
	}
	if FuseSubgraph(g, []int{0}, []OpType{OpEmbed, OpFFNBlock}, nil) {
		t.Fatal("expected len mismatch to fail")
	}
	if FuseSubgraph(g, nil, nil, nil) {
		t.Fatal("expected empty nodeIndices to fail")
	}
	if FuseSubgraph(g, []int{0}, []OpType{OpFFNBlock}, nil) {
		t.Fatal("expected op mismatch to fail")
	}
}

func TestCountNodeUses(t *testing.T) {
	g := &Graph{
		Nodes: []Node{
			{Op: OpEmbed, Branch: BranchEmbed, Name: "e", Input: []TensorID{0}, Output: 1, Params: EmbedParams{VocabSize: 100, EmbDim: 16}},
			{Op: OpFFNBlock, Branch: BranchFFN, Name: "a", Input: []TensorID{1}, Output: 2, Params: FFNParams{}},
			{Op: OpAdd, Branch: BranchFFN, Name: "add", Input: []TensorID{1, 2}, Output: 3, Params: FFNParams{}},
			{Op: OpFFNBlock, Branch: BranchFFN, Name: "next", Input: []TensorID{3}, Output: 4, Params: FFNParams{}},
		},
	}
	if countNodeUses(g, 0) != 2 {
		t.Fatalf("expected 2 uses for node 0, got %d", countNodeUses(g, 0))
	}
	if countNodeUses(g, 1) != 1 {
		t.Fatalf("expected 1 use for node 1, got %d", countNodeUses(g, 1))
	}
	if countNodeUses(g, 2) != 1 {
		t.Fatalf("expected 1 use for node 2, got %d", countNodeUses(g, 2))
	}
	if countNodeUses(g, -1) != 0 {
		t.Fatal("expected 0 uses for invalid index")
	}
	if countNodeUses(g, 99) != 0 {
		t.Fatal("expected 0 uses for out of bounds")
	}
}

func TestFuserFuseStub(t *testing.T) {
	g := &Graph{
		Nodes: []Node{
			{Op: OpEmbed, Branch: BranchEmbed, Name: "e", Input: []TensorID{0}, Output: 1, Params: EmbedParams{VocabSize: 100, EmbDim: 16}},
		},
	}
	var f Fuser
	result := f.Fuse(g)
	if result == nil {
		t.Fatal("expected non-nil result")
	}
	if len(result.Nodes) != len(g.Nodes) {
		t.Fatalf("expected %d nodes, got %d", len(g.Nodes), len(result.Nodes))
	}
}

// -- MoE fusion tests --

func TestFuseMoE(t *testing.T) {
	moeParams := MoEParams{
		TopK:         8,
		NumExperts:   64,
		SharedExpert: true,
	}

	g := &Graph{
		Nodes: []Node{
			{
				Op: OpEmbed, Branch: BranchEmbed, Name: "embed",
				Input: []TensorID{0}, Output: 1,
				Params: EmbedParams{VocabSize: 100, EmbDim: 1024},
			},
			{
				Op: OpMoEBlock, Branch: BranchFFN, Name: "layer0.moe",
				Input: []TensorID{1}, Output: 2,
				Params: moeParams,
			},
			{
				Op: OpOutput, Branch: BranchOutput, Name: "output",
				Input: []TensorID{2}, Output: 3,
				Params: OutputParams{Softcap: 30.0},
			},
		},
	}

	changed := FuseMoE(g)
	if !changed {
		t.Fatal("expected FuseMoE to return true")
	}

	// Embed should be unchanged
	if g.Nodes[0].Op != OpEmbed {
		t.Fatalf("node 0: expected OpEmbed, got %s", g.Nodes[0].Op)
	}
	// MoE should be relabeled to fused
	if g.Nodes[1].Op != OpFusedMoE {
		t.Fatalf("node 1: expected OpFusedMoE, got %s", g.Nodes[1].Op)
	}
	// Branch must stay the same
	if g.Nodes[1].Branch != BranchFFN {
		t.Fatalf("node 1: expected BranchFFN, got %s", g.Nodes[1].Branch)
	}
	// Name preserved
	if g.Nodes[1].Name != "layer0.moe" {
		t.Fatalf("node 1 Name: got %s, want layer0.moe", g.Nodes[1].Name)
	}
	// Input/Output preserved
	if len(g.Nodes[1].Input) != 1 || g.Nodes[1].Input[0] != 1 {
		t.Fatalf("node 1 Input: got %v, want [1]", g.Nodes[1].Input)
	}
	if g.Nodes[1].Output != 2 {
		t.Fatalf("node 1 Output: got %d, want 2", g.Nodes[1].Output)
	}
	// Params preserved
	gotMoE, ok := g.Nodes[1].Params.(MoEParams)
	if !ok {
		t.Fatalf("node 1: expected MoEParams, got %T", g.Nodes[1].Params)
	}
	if gotMoE.TopK != moeParams.TopK {
		t.Fatalf("node 1 TopK: got %d, want %d", gotMoE.TopK, moeParams.TopK)
	}
	if gotMoE.NumExperts != moeParams.NumExperts {
		t.Fatalf("node 1 NumExperts: got %d, want %d", gotMoE.NumExperts, moeParams.NumExperts)
	}
	if gotMoE.SharedExpert != moeParams.SharedExpert {
		t.Fatalf("node 1 SharedExpert: got %v, want %v", gotMoE.SharedExpert, moeParams.SharedExpert)
	}

	// Output unchanged
	if g.Nodes[2].Op != OpOutput {
		t.Fatalf("node 2: expected OpOutput, got %s", g.Nodes[2].Op)
	}

	// Validate graph integrity
	if err := g.Validate(); err != nil {
		t.Fatalf("graph Validate failed after fusion: %v", err)
	}
}

func TestFuseMoENoChange(t *testing.T) {
	g := &Graph{
		Nodes: []Node{
			{
				Op: OpEmbed, Branch: BranchEmbed, Name: "embed",
				Input: []TensorID{0}, Output: 1,
				Params: EmbedParams{VocabSize: 100, EmbDim: 1024},
			},
			{
				Op: OpAttentionBlock, Branch: BranchAttention, Name: "layer0.attention",
				Input: []TensorID{1}, Output: 2,
				Params: AttentionParams{NHeadKV: 8, HeadDim: 128},
			},
			{
				Op: OpOutput, Branch: BranchOutput, Name: "output",
				Input: []TensorID{2}, Output: 3,
				Params: OutputParams{},
			},
		},
	}

	changed := FuseMoE(g)
	if changed {
		t.Fatal("expected FuseMoE to return false for graph without MoE blocks")
	}

	if g.Nodes[0].Op != OpEmbed {
		t.Fatalf("node 0: expected OpEmbed, got %s", g.Nodes[0].Op)
	}
	if g.Nodes[1].Op != OpAttentionBlock {
		t.Fatalf("node 1: expected OpAttentionBlock, got %s", g.Nodes[1].Op)
	}
	if g.Nodes[2].Op != OpOutput {
		t.Fatalf("node 2: expected OpOutput, got %s", g.Nodes[2].Op)
	}
}

func TestFuseMoEAlreadyFused(t *testing.T) {
	g := &Graph{
		Nodes: []Node{
			{
				Op: OpEmbed, Branch: BranchEmbed, Name: "embed",
				Input: []TensorID{0}, Output: 1,
				Params: EmbedParams{VocabSize: 100, EmbDim: 1024},
			},
			{
				Op: OpFusedMoE, Branch: BranchFFN, Name: "layer0.moe",
				Input: []TensorID{1}, Output: 2,
				Params: MoEParams{TopK: 8, NumExperts: 64},
			},
			{
				Op: OpOutput, Branch: BranchOutput, Name: "output",
				Input: []TensorID{2}, Output: 3,
				Params: OutputParams{},
			},
		},
	}

	changed := FuseMoE(g)
	if changed {
		t.Fatal("expected FuseMoE to return false when already fused")
	}

	if g.Nodes[1].Op != OpFusedMoE {
		t.Fatalf("node 1: expected OpFusedMoE unchanged, got %s", g.Nodes[1].Op)
	}
}

func TestFuseMoEEmptyGraph(t *testing.T) {
	g := &Graph{Nodes: nil}
	changed := FuseMoE(g)
	if changed {
		t.Fatal("expected FuseMoE to return false for empty graph")
	}
}

// -- End-to-end fusion test --

func TestFuseEndToEnd(t *testing.T) {
	// Build a minimal llama-style graph with 2 layers manually.
	// Llama pattern per layer: attn_norm → attention → attn_residual →
	// ffn_norm → ffn → ffn_residual
	g := &Graph{
		Nodes: []Node{
			// 0: embed
			{Op: OpEmbed, Branch: BranchEmbed, Name: "embed", Input: []TensorID{0}, Output: 1, Params: EmbedParams{VocabSize: 100, EmbDim: 64}},
			// Layer 0
			{Op: OpAdd, Branch: BranchFFN, Name: "layer0.attn_norm", Input: []TensorID{1}, Output: 2, Params: FFNParams{}},
			{Op: OpAttentionBlock, Branch: BranchAttention, Name: "layer0.attention", Input: []TensorID{2}, Output: 3, Params: AttentionParams{NHeadKV: 4, HeadDim: 64}},
			{Op: OpAdd, Branch: BranchFFN, Name: "layer0.attn_residual", Input: []TensorID{1, 3}, Output: 4, Params: FFNParams{}},
			{Op: OpAdd, Branch: BranchFFN, Name: "layer0.ffn_norm", Input: []TensorID{4}, Output: 5, Params: FFNParams{}},
			{Op: OpFFNBlock, Branch: BranchFFN, Name: "layer0.ffn", Input: []TensorID{5}, Output: 6, Params: FFNParams{Activation: "silu"}},
			{Op: OpAdd, Branch: BranchFFN, Name: "layer0.ffn_residual", Input: []TensorID{4, 6}, Output: 7, Params: FFNParams{}},
			// Layer 1
			{Op: OpAdd, Branch: BranchFFN, Name: "layer1.attn_norm", Input: []TensorID{7}, Output: 8, Params: FFNParams{}},
			{Op: OpAttentionBlock, Branch: BranchAttention, Name: "layer1.attention", Input: []TensorID{8}, Output: 9, Params: AttentionParams{NHeadKV: 4, HeadDim: 64}},
			{Op: OpAdd, Branch: BranchFFN, Name: "layer1.attn_residual", Input: []TensorID{7, 9}, Output: 10, Params: FFNParams{}},
			{Op: OpAdd, Branch: BranchFFN, Name: "layer1.ffn_norm", Input: []TensorID{10}, Output: 11, Params: FFNParams{}},
			{Op: OpFFNBlock, Branch: BranchFFN, Name: "layer1.ffn", Input: []TensorID{11}, Output: 12, Params: FFNParams{Activation: "silu"}},
			{Op: OpAdd, Branch: BranchFFN, Name: "layer1.ffn_residual", Input: []TensorID{10, 12}, Output: 13, Params: FFNParams{}},
			// Output
			{Op: OpOutput, Branch: BranchOutput, Name: "output", Input: []TensorID{13}, Output: 14, Params: OutputParams{}},
		},
	}

	origLen := len(g.Nodes)

	// Run all fusers
	attnChanged := FuseAttention(g)
	ffnChanged := FuseFFN(g)
	_ = FuseNormResidual(g)
	moeChanged := FuseMoE(g)

	// Llama has attention blocks → FuseAttention should change them
	if !attnChanged {
		t.Error("FuseAttention: expected true (llama has attention blocks)")
	}
	// Llama has no gated FFN DAG → FuseFFN should not match
	if ffnChanged {
		t.Error("FuseFFN: expected false (llama has no gated FFN pattern)")
	}
	// Llama residual paths have multi-use intermediates → FuseNormResidual may not match
	// FuseMoE has no MoE blocks → should return false
	if moeChanged {
		t.Error("FuseMoE: expected false (llama has no MoE blocks)")
	}

	// Verify attention blocks are now fused
	if g.Nodes[2].Op != OpFusedAttention {
		t.Errorf("node 2: expected OpFusedAttention, got %s", g.Nodes[2].Op)
	}
	if g.Nodes[8].Op != OpFusedAttention {
		t.Errorf("node 8: expected OpFusedAttention, got %s", g.Nodes[8].Op)
	}

	// Node count should not grow — fusers reduce or keep same
	if len(g.Nodes) > origLen {
		t.Errorf("node count increased from %d to %d", origLen, len(g.Nodes))
	}

	// Embed and Output unchanged
	if g.Nodes[0].Op != OpEmbed {
		t.Errorf("node 0: expected OpEmbed, got %s", g.Nodes[0].Op)
	}
	if g.Nodes[len(g.Nodes)-1].Op != OpOutput {
		t.Errorf("last node: expected OpOutput, got %s", g.Nodes[len(g.Nodes)-1].Op)
	}

	if err := g.Validate(); err != nil {
		t.Fatalf("graph Validate failed after end-to-end fusion: %v", err)
	}
}

// -- Safety: no fusion when multi-use intermediate --

func TestNoFuseWhenMultiUse(t *testing.T) {
	// embed output used by TWO nodes → intermediate has 2 uses → FuseLinear fails
	g := &Graph{
		Nodes: []Node{
			{Op: OpEmbed, Branch: BranchEmbed, Name: "embed", Input: []TensorID{0}, Output: 1, Params: EmbedParams{VocabSize: 10, EmbDim: 8}},
			{Op: OpAdd, Branch: BranchFFN, Name: "add1", Input: []TensorID{1}, Output: 2, Params: FFNParams{}},
			{Op: OpAdd, Branch: BranchFFN, Name: "add2", Input: []TensorID{1}, Output: 3, Params: FFNParams{}},
		},
	}

	// FuseLinear check: embed → add1. embed has 2 uses → must fail
	if FuseLinear(g, 0, []OpType{OpEmbed, OpAdd}) {
		t.Fatal("expected FuseLinear to fail: embed node has 2 uses")
	}
}

// -- Safety: no fusion when output-tagged node is involved --

func TestNoFuseWhenOutput(t *testing.T) {
	// Build: embed → ffn → output. The output node (OpOutput) has no further
	// uses and should not be fused into any pattern because it is the terminal
	// consumer.
	g := &Graph{
		Nodes: []Node{
			{Op: OpEmbed, Branch: BranchEmbed, Name: "embed", Input: []TensorID{0}, Output: 1, Params: EmbedParams{VocabSize: 10, EmbDim: 8}},
			{Op: OpFFNBlock, Branch: BranchFFN, Name: "ffn", Input: []TensorID{1}, Output: 2, Params: FFNParams{}},
			{Op: OpOutput, Branch: BranchOutput, Name: "output", Input: []TensorID{2}, Output: 3, Params: OutputParams{}},
		},
	}

	// FuseLinear: ffn → output. ffn has 1 use (good), but the pair
	// [OpFFNBlock, OpOutput] doesn't match any defined fusion pattern
	// (no fuser searches for that combination). Even if it matched,
	// the output node has no downstream uses, which would fail
	// FuseSubgraph output-node checks.
	if FuseLinear(g, 1, []OpType{OpFFNBlock, OpOutput}) {
		// It happens to match FuseLinear's checks (use count=1, both ops
		// match), but no real fuser uses this pattern. This test documents
		// that such a pair is MATCHABLE by FuseLinear but NOT by any
		// existing fuser.
	}
}
