package archs

import (
	"testing"

	"github.com/samcharles93/mantle/internal/backend/core"
	"github.com/samcharles93/mantle/internal/graph"
)

func TestBuildLlamaGraph(t *testing.T) {
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
			{HeadKV: 2, HeadDim: 8, AttnWindow: 0, FFNActivation: "silu"},
			{HeadKV: 2, HeadDim: 8, AttnWindow: 0, FFNActivation: "silu"},
			{HeadKV: 2, HeadDim: 8, AttnWindow: 0, FFNActivation: "silu"},
			{HeadKV: 2, HeadDim: 8, AttnWindow: 0, FFNActivation: "silu"},
		},
		MaxKVStride: 16,
	}

	b := &LlamaBuilder{}
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

	if g.Nodes[0].Op != graph.OpEmbed {
		t.Fatalf("first node should be OpEmbed, got %v", g.Nodes[0].Op)
	}

	last := g.Nodes[len(g.Nodes)-1]
	if last.Op != graph.OpOutput {
		t.Fatalf("last node should be OpOutput, got %v", last.Op)
	}
}
