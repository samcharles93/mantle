package archs

import (
	"testing"

	"github.com/samcharles93/mantle/internal/backend/core"
	"github.com/samcharles93/mantle/internal/graph"
)

func TestBuildMambaGraph(t *testing.T) {
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
			{IsRecurrent: true},
			{IsRecurrent: true},
			{IsRecurrent: true},
			{IsRecurrent: true},
		},
		MaxKVStride: 16,
	}

	b := &MambaBuilder{}
	g, err := b.BuildGraph(&inst.Config.Config, inst)
	if err != nil {
		t.Fatalf("BuildGraph failed: %v", err)
	}

	expected := 1 + 4*2 + 1
	if len(g.Nodes) != expected {
		t.Fatalf("expected %d nodes, got %d", expected, len(g.Nodes))
	}

	if err := g.Validate(); err != nil {
		t.Fatalf("graph validation failed: %v", err)
	}

	for _, n := range g.Nodes {
		if n.Op == graph.OpAttentionBlock {
			t.Fatalf("mamba graph should not have OpAttentionBlock nodes, found at %q", n.Name)
		}
	}

	for i, n := range g.Nodes {
		if i == 0 || i == len(g.Nodes)-1 {
			continue
		}
		if n.Op != graph.OpMambaBlock && n.Op != graph.OpAdd {
			t.Fatalf("unexpected op %v at node %d", n.Op, i)
		}
	}
}
