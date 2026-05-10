package graph

import (
	"slices"
	"testing"
)

func TestFuseAttention(t *testing.T) {
	attnParams := AttentionParams{
		NHeadKV:          8,
		HeadDim:          128,
		KVStride:         128,
		SlidingWin:       0,
		LayerIndex:       0,
		InvFreq:          []float64{1.0, 0.5},
		AttnScale:        0.088,
		AttnLogitSoftcap: 50.0,
	}

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
				Params: attnParams,
			},
			{
				Op: OpFFNBlock, Branch: BranchFFN, Name: "layer0.ffn",
				Input: []TensorID{2}, Output: 3,
				Params: FFNParams{Activation: "silu"},
			},
			{
				Op: OpOutput, Branch: BranchOutput, Name: "output",
				Input: []TensorID{3}, Output: 4,
				Params: OutputParams{Softcap: 30.0},
			},
		},
	}

	changed := FuseAttention(g)
	if !changed {
		t.Fatal("expected FuseAttention to return true")
	}

	// Embed should be unchanged
	if g.Nodes[0].Op != OpEmbed {
		t.Fatalf("node 0: expected OpEmbed, got %s", g.Nodes[0].Op)
	}
	// Attention should be relabeled to fused
	if g.Nodes[1].Op != OpFusedAttention {
		t.Fatalf("node 1: expected OpFusedAttention, got %s", g.Nodes[1].Op)
	}
	// Branch must stay the same
	if g.Nodes[1].Branch != BranchAttention {
		t.Fatalf("node 1: expected BranchAttention, got %s", g.Nodes[1].Branch)
	}
	// Params must be preserved
	gotAttn, ok := g.Nodes[1].Params.(AttentionParams)
	if !ok {
		t.Fatalf("node 1: expected AttentionParams, got %T", g.Nodes[1].Params)
	}
	if gotAttn.NHeadKV != attnParams.NHeadKV {
		t.Fatalf("node 1 NHeadKV: got %d, want %d", gotAttn.NHeadKV, attnParams.NHeadKV)
	}
	if gotAttn.HeadDim != attnParams.HeadDim {
		t.Fatalf("node 1 HeadDim: got %d, want %d", gotAttn.HeadDim, attnParams.HeadDim)
	}
	if gotAttn.KVStride != attnParams.KVStride {
		t.Fatalf("node 1 KVStride: got %d, want %d", gotAttn.KVStride, attnParams.KVStride)
	}
	if gotAttn.LayerIndex != attnParams.LayerIndex {
		t.Fatalf("node 1 LayerIndex: got %d, want %d", gotAttn.LayerIndex, attnParams.LayerIndex)
	}
	if gotAttn.AttnScale != attnParams.AttnScale {
		t.Fatalf("node 1 AttnScale: got %f, want %f", gotAttn.AttnScale, attnParams.AttnScale)
	}
	if gotAttn.AttnLogitSoftcap != attnParams.AttnLogitSoftcap {
		t.Fatalf("node 1 AttnLogitSoftcap: got %f, want %f", gotAttn.AttnLogitSoftcap, attnParams.AttnLogitSoftcap)
	}
	if !slices.Equal(gotAttn.InvFreq, attnParams.InvFreq) {
		t.Fatalf("node 1 InvFreq: got %v, want %v", gotAttn.InvFreq, attnParams.InvFreq)
	}
	// Name preserved
	if g.Nodes[1].Name != "layer0.attention" {
		t.Fatalf("node 1 Name: got %s, want layer0.attention", g.Nodes[1].Name)
	}
	// Input/Output preserved
	if len(g.Nodes[1].Input) != 1 || g.Nodes[1].Input[0] != 1 {
		t.Fatalf("node 1 Input: got %v, want [1]", g.Nodes[1].Input)
	}
	if g.Nodes[1].Output != 2 {
		t.Fatalf("node 1 Output: got %d, want 2", g.Nodes[1].Output)
	}

	// FFN and Output unchanged
	if g.Nodes[2].Op != OpFFNBlock {
		t.Fatalf("node 2: expected OpFFNBlock, got %s", g.Nodes[2].Op)
	}
	if g.Nodes[3].Op != OpOutput {
		t.Fatalf("node 3: expected OpOutput, got %s", g.Nodes[3].Op)
	}

	// Validate graph integrity
	if err := g.Validate(); err != nil {
		t.Fatalf("graph Validate failed after fusion: %v", err)
	}
}

func TestFuseAttentionMultipleLayers(t *testing.T) {
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
				Params: AttentionParams{LayerIndex: 0, NHeadKV: 8, HeadDim: 128},
			},
			{
				Op: OpFFNBlock, Branch: BranchFFN, Name: "layer0.ffn",
				Input: []TensorID{2}, Output: 3,
				Params: FFNParams{},
			},
			{
				Op: OpAttentionBlock, Branch: BranchAttention, Name: "layer1.attention",
				Input: []TensorID{3}, Output: 4,
				Params: AttentionParams{LayerIndex: 1, NHeadKV: 8, HeadDim: 128},
			},
			{
				Op: OpFFNBlock, Branch: BranchFFN, Name: "layer1.ffn",
				Input: []TensorID{4}, Output: 5,
				Params: FFNParams{},
			},
			{
				Op: OpOutput, Branch: BranchOutput, Name: "output",
				Input: []TensorID{5}, Output: 6,
				Params: OutputParams{},
			},
		},
	}

	changed := FuseAttention(g)
	if !changed {
		t.Fatal("expected FuseAttention to return true")
	}

	if g.Nodes[1].Op != OpFusedAttention {
		t.Fatalf("node 1: expected OpFusedAttention, got %s", g.Nodes[1].Op)
	}
	if g.Nodes[3].Op != OpFusedAttention {
		t.Fatalf("node 3: expected OpFusedAttention, got %s", g.Nodes[3].Op)
	}

	// Non-attention nodes unchanged
	for i, want := range []OpType{OpEmbed, OpFusedAttention, OpFFNBlock, OpFusedAttention, OpFFNBlock, OpOutput} {
		if g.Nodes[i].Op != want {
			t.Fatalf("node %d: expected %s, got %s", i, want, g.Nodes[i].Op)
		}
	}

	if err := g.Validate(); err != nil {
		t.Fatalf("graph Validate failed: %v", err)
	}
}

func TestFuseAttentionNoChange(t *testing.T) {
	// Graph with no attention blocks at all
	g := &Graph{
		Nodes: []Node{
			{
				Op: OpEmbed, Branch: BranchEmbed, Name: "embed",
				Input: []TensorID{0}, Output: 1,
				Params: EmbedParams{VocabSize: 100, EmbDim: 1024},
			},
			{
				Op: OpOutput, Branch: BranchOutput, Name: "output",
				Input: []TensorID{1}, Output: 2,
				Params: OutputParams{},
			},
		},
	}

	changed := FuseAttention(g)
	if changed {
		t.Fatal("expected FuseAttention to return false for graph without attention blocks")
	}

	// All nodes unchanged
	if g.Nodes[0].Op != OpEmbed {
		t.Fatalf("node 0: expected OpEmbed, got %s", g.Nodes[0].Op)
	}
	if g.Nodes[1].Op != OpOutput {
		t.Fatalf("node 1: expected OpOutput, got %s", g.Nodes[1].Op)
	}
}

func TestFuseAttentionAlreadyFused(t *testing.T) {
	// Graph where attention is already fused — should return false (idempotent)
	g := &Graph{
		Nodes: []Node{
			{
				Op: OpEmbed, Branch: BranchEmbed, Name: "embed",
				Input: []TensorID{0}, Output: 1,
				Params: EmbedParams{VocabSize: 100, EmbDim: 1024},
			},
			{
				Op: OpFusedAttention, Branch: BranchAttention, Name: "layer0.attention",
				Input: []TensorID{1}, Output: 2,
				Params: AttentionParams{LayerIndex: 0, NHeadKV: 8, HeadDim: 128},
			},
			{
				Op: OpOutput, Branch: BranchOutput, Name: "output",
				Input: []TensorID{2}, Output: 3,
				Params: OutputParams{},
			},
		},
	}

	changed := FuseAttention(g)
	if changed {
		t.Fatal("expected FuseAttention to return false when already fused")
	}

	if g.Nodes[1].Op != OpFusedAttention {
		t.Fatalf("node 1: expected OpFusedAttention unchanged, got %s", g.Nodes[1].Op)
	}
}

func TestFuseAttentionEmptyGraph(t *testing.T) {
	g := &Graph{Nodes: nil}
	changed := FuseAttention(g)
	if changed {
		t.Fatal("expected FuseAttention to return false for empty graph")
	}
}
