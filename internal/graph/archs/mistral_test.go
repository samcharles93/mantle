package archs

import (
	"testing"

	"github.com/samcharles93/mantle/internal/backend/core"
	"github.com/samcharles93/mantle/internal/graph"
)

func TestBuildMistralGraph(t *testing.T) {
	inst := &core.Instance{
		Config: &core.ModelConfig{
			Config: core.Config{
				BlockCount:        4,
				EmbeddingLength:   16,
				HeadCount:         4,
				HeadDim:           8,
				VocabSize:         100,
				SlidingWindow:     128,
				FinalLogitSoftcap: 30.0,
			},
		},
		Layers: []core.Layer{
			{HeadKV: 2, HeadDim: 8, AttnWindow: 128, FFNActivation: "silu"},
			{HeadKV: 2, HeadDim: 8, AttnWindow: 128, FFNActivation: "silu"},
			{HeadKV: 2, HeadDim: 8, AttnWindow: 128, FFNActivation: "silu"},
			{HeadKV: 2, HeadDim: 8, AttnWindow: 128, FFNActivation: "silu"},
		},
		MaxKVStride: 16,
	}

	b := &MistralBuilder{}
	g, err := b.BuildGraph(&inst.Config.Config, inst)
	if err != nil {
		t.Fatalf("BuildGraph failed: %v", err)
	}

	expected := 1 + 4*6 + 1
	if len(g.Nodes) != expected {
		t.Fatalf("expected %d nodes, got %d", expected, len(g.Nodes))
	}

	if err := g.Validate(); err != nil {
		t.Fatalf("graph validation failed: %v", err)
	}

	// Verify sliding window propagated
	for i, n := range g.Nodes {
		if n.Op == graph.OpAttentionBlock {
			params := n.Params.(graph.AttentionParams)
			if params.SlidingWin != 128 {
				t.Fatalf("layer %d: expected SlidingWin=128, got %d", i, params.SlidingWin)
			}
		}
	}
}
