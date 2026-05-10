package archs

import (
	"testing"

	"github.com/samcharles93/mantle/internal/backend/core"
	"github.com/samcharles93/mantle/internal/graph"
)

func TestBuildQwen3Graph(t *testing.T) {
	inst := &core.Instance{
		Config: &core.ModelConfig{
			Config: core.Config{
				BlockCount:        4,
				EmbeddingLength:   16,
				HeadCount:         4,
				HeadDim:           8,
				VocabSize:         100,
				NumDenseLayers:    2,
				NumExperts:        8,
				NumExpertsPerTok:  2,
				NumSharedExperts:  1,
				FinalLogitSoftcap: 30.0,
			},
		},
		Layers: []core.Layer{
			{HeadKV: 2, HeadDim: 8, AttnWindow: 0, FFNActivation: "silu"},
			{HeadKV: 2, HeadDim: 8, AttnWindow: 0, FFNActivation: "silu"},
			{HeadKV: 2, HeadDim: 8, AttnWindow: 0, FFNActivation: "silu", MoE: &core.MoELayer{}},
			{HeadKV: 2, HeadDim: 8, AttnWindow: 0, FFNActivation: "silu", MoE: &core.MoELayer{}},
		},
		MaxKVStride: 16,
	}

	b := &Qwen3Builder{}
	g, err := b.BuildGraph(&inst.Config.Config, inst)
	if err != nil {
		t.Fatalf("BuildGraph failed: %v", err)
	}

	// embed + (4 layers * 6 nodes each) + output
	expected := 1 + 4*6 + 1
	if len(g.Nodes) != expected {
		t.Fatalf("expected %d nodes, got %d", expected, len(g.Nodes))
	}

	if err := g.Validate(); err != nil {
		t.Fatalf("graph validation failed: %v", err)
	}

	// Verify first two layers use OpFFNBlock, last two use OpMoEBlock.
	// Node layout per layer: attn_norm, attention, attn_residual, ffn_norm, ffn/moe, ffn/moe_residual
	for i := 0; i < inst.Config.Config.BlockCount; i++ {
		ffnIdx := 1 + i*6 + 4 // index of ffn/moe node within g.Nodes
		node := g.Nodes[ffnIdx]
		if i < inst.Config.Config.NumDenseLayers {
			if node.Op != graph.OpFFNBlock {
				t.Errorf("layer %d: expected OpFFNBlock, got %v", i, node.Op)
			}
		} else {
			if node.Op != graph.OpMoEBlock {
				t.Errorf("layer %d: expected OpMoEBlock, got %v", i, node.Op)
			}
		}
	}
}
