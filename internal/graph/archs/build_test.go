package archs

import (
	"testing"

	"github.com/samcharles93/mantle/internal/backend/core"
	"github.com/samcharles93/mantle/internal/graph"
)

func makeLayers(n int, headKV, headDim, attnWindow int, act string) []core.Layer {
	layers := make([]core.Layer, n)
	for i := 0; i < n; i++ {
		layers[i] = core.Layer{
			HeadKV:        headKV,
			HeadDim:       headDim,
			AttnWindow:    attnWindow,
			FFNActivation: act,
		}
	}
	return layers
}

func TestAllArchitectures(t *testing.T) {
	cases := []struct {
		name     string
		builder  graph.Builder
		cfg      core.Config
		inst     *core.Instance
		expected int // expected node count
		firstOp  graph.OpType
		lastOp   graph.OpType
	}{
		{
			name:    "llama",
			builder: &LlamaBuilder{},
			cfg: core.Config{
				BlockCount:        4,
				EmbeddingLength:   16,
				HeadCount:         4,
				HeadDim:           8,
				VocabSize:         100,
				FinalLogitSoftcap: 30.0,
			},
			inst: &core.Instance{
				Config:      &core.ModelConfig{Config: core.Config{BlockCount: 4, EmbeddingLength: 16, HeadCount: 4, HeadDim: 8, VocabSize: 100, FinalLogitSoftcap: 30.0}},
				Layers:      makeLayers(4, 2, 8, 0, "silu"),
				MaxKVStride: 16,
			},
			expected: 26, // 1 embed + 4*6 + 1 output
			firstOp:  graph.OpEmbed,
			lastOp:   graph.OpOutput,
		},
		{
			name:    "qwen3",
			builder: &Qwen3Builder{},
			cfg: core.Config{
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
			inst: func() *core.Instance {
				inst := &core.Instance{Config: &core.ModelConfig{Config: core.Config{BlockCount: 4, EmbeddingLength: 16, HeadCount: 4, HeadDim: 8, VocabSize: 100, NumDenseLayers: 2, NumExperts: 8, NumExpertsPerTok: 2, NumSharedExperts: 1, FinalLogitSoftcap: 30.0}}}
				layers := makeLayers(4, 2, 8, 0, "silu")
				// mark two layers as MoE
				layers[2].MoE = &core.MoELayer{}
				layers[3].MoE = &core.MoELayer{}
				inst.Layers = layers
				inst.MaxKVStride = 16
				return inst
			}(),
			expected: 26,
			firstOp:  graph.OpEmbed,
			lastOp:   graph.OpOutput,
		},
		{
			name:    "mistral",
			builder: &MistralBuilder{},
			cfg: core.Config{
				BlockCount:        4,
				EmbeddingLength:   16,
				HeadCount:         4,
				HeadDim:           8,
				VocabSize:         100,
				FinalLogitSoftcap: 30.0,
			},
			inst: &core.Instance{
				Config:      &core.ModelConfig{Config: core.Config{BlockCount: 4, EmbeddingLength: 16, HeadCount: 4, HeadDim: 8, VocabSize: 100, FinalLogitSoftcap: 30.0}},
				Layers:      makeLayers(4, 2, 8, 2, "silu"),
				MaxKVStride: 16,
			},
			expected: 26,
			firstOp:  graph.OpEmbed,
			lastOp:   graph.OpOutput,
		},
		{
			name:    "mamba",
			builder: &MambaBuilder{},
			cfg: core.Config{
				BlockCount:        4,
				EmbeddingLength:   16,
				HeadCount:         4,
				HeadDim:           8,
				VocabSize:         100,
				FinalLogitSoftcap: 30.0,
			},
			inst: &core.Instance{
				Config:      &core.ModelConfig{Config: core.Config{BlockCount: 4, EmbeddingLength: 16, HeadCount: 4, HeadDim: 8, VocabSize: 100, FinalLogitSoftcap: 30.0}},
				Layers:      makeLayers(4, 0, 0, 0, ""),
				MaxKVStride: 16,
			},
			expected: 10, // 1 embed + 4*2 + 1 output = 10
			firstOp:  graph.OpEmbed,
			lastOp:   graph.OpOutput,
		},
		{
			name:    "deltanet",
			builder: &DeltaNetBuilder{},
			cfg: core.Config{
				BlockCount:        4,
				EmbeddingLength:   16,
				HeadCount:         4,
				HeadDim:           8,
				VocabSize:         100,
				FinalLogitSoftcap: 30.0,
			},
			inst: &core.Instance{
				Config:      &core.ModelConfig{Config: core.Config{BlockCount: 4, EmbeddingLength: 16, HeadCount: 4, HeadDim: 8, VocabSize: 100, FinalLogitSoftcap: 30.0}},
				Layers:      makeLayers(4, 0, 0, 0, "silu"),
				MaxKVStride: 16,
			},
			expected: 22, // 1 embed + 4*5 + 1 output = 22
			firstOp:  graph.OpEmbed,
			lastOp:   graph.OpOutput,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			g, err := tc.builder.BuildGraph(&tc.cfg, tc.inst)
			if err != nil {
				t.Fatalf("BuildGraph failed: %v", err)
			}
			if len(g.Nodes) != tc.expected {
				t.Fatalf("expected %d nodes, got %d", tc.expected, len(g.Nodes))
			}
			if err := g.Validate(); err != nil {
				t.Fatalf("validation failed: %v", err)
			}
			if g.Nodes[0].Op != tc.firstOp {
				t.Fatalf("first node: expected %v, got %v", tc.firstOp, g.Nodes[0].Op)
			}
			last := g.Nodes[len(g.Nodes)-1]
			if last.Op != tc.lastOp {
				t.Fatalf("last node: expected %v, got %v", tc.lastOp, last.Op)
			}
		})
	}
}

func TestDeterministicBuilder(t *testing.T) {
	inst := &core.Instance{
		Config:      &core.ModelConfig{Config: core.Config{BlockCount: 4, EmbeddingLength: 16, HeadCount: 4, HeadDim: 8, VocabSize: 100, FinalLogitSoftcap: 30.0}},
		Layers:      makeLayers(4, 2, 8, 0, "silu"),
		MaxKVStride: 16,
	}
	b := &LlamaBuilder{}

	var first *graph.Graph
	for i := 0; i < 100; i++ {
		g, err := b.BuildGraph(&inst.Config.Config, inst)
		if err != nil {
			t.Fatalf("iteration %d: BuildGraph failed: %v", i, err)
		}
		if first == nil {
			first = g
			continue
		}
		if len(g.Nodes) != len(first.Nodes) {
			t.Fatalf("iteration %d: node count changed", i)
		}
		for j := range g.Nodes {
			if g.Nodes[j].Op != first.Nodes[j].Op {
				t.Fatalf("iteration %d: node %d op changed", i, j)
			}
		}
	}
}
