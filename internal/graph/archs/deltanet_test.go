package archs

import (
	"testing"

	"github.com/samcharles93/mantle/internal/backend/core"
	"github.com/samcharles93/mantle/internal/graph"
)

func TestBuildDeltaNetGraph(t *testing.T) {
	inst := &core.Instance{
		Config: &core.ModelConfig{
			Config: core.Config{
				BlockCount:        4,
				EmbeddingLength:   16,
				HeadCount:         4,
				HeadDim:           8,
				VocabSize:         100,
				FinalLogitSoftcap: 30.0,
			},
		},
		Layers: []core.Layer{
			{DeltaNet: &core.DeltaNetLayer{}, FFNActivation: "silu"},
			{DeltaNet: &core.DeltaNetLayer{}, FFNActivation: "silu"},
			{DeltaNet: &core.DeltaNetLayer{}, FFNActivation: "silu"},
			{DeltaNet: &core.DeltaNetLayer{}, FFNActivation: "silu"},
		},
		MaxKVStride: 16,
	}

	b := &DeltaNetBuilder{}
	g, err := b.BuildGraph(&inst.Config.Config, inst)
	if err != nil {
		t.Fatalf("BuildGraph failed: %v", err)
	}

	// Expected: embed + 4*(deltanet + attn_residual + ffn_norm + ffn + ffn_residual) + output
	// = 1 + 4*5 + 1 = 22
	expected := 1 + 4*5 + 1
	if len(g.Nodes) != expected {
		t.Fatalf("expected %d nodes, got %d", expected, len(g.Nodes))
	}

	if err := g.Validate(); err != nil {
		t.Fatalf("graph validation failed: %v", err)
	}

	// Verify no attention blocks, all deltanet
	for _, n := range g.Nodes {
		if n.Op == graph.OpAttentionBlock {
			t.Fatalf("deltanet graph should not have OpAttentionBlock nodes, found at %q", n.Name)
		}
	}
}
