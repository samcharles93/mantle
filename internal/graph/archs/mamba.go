package archs

import (
	"fmt"

	"github.com/samcharles93/mantle/internal/backend/core"
	"github.com/samcharles93/mantle/internal/graph"
)

// MambaBuilder constructs a graph for pure Mamba (SSM) architectures.
// Mamba layers replace both attention and FFN in a single recurrent block.
type MambaBuilder struct{}

// BuildGraph builds a computation graph for a Mamba model.
// Each layer consists of an OpMambaBlock followed by a residual OpAdd.
func (b *MambaBuilder) BuildGraph(cfg *core.Config, inst *core.Instance) (*graph.Graph, error) {
	graph.ResetTensorID()
	g := &graph.Graph{UID: 0}

	embedID := graph.NewTensorID()
	g.AddNode(graph.Node{
		Op:     graph.OpEmbed,
		Branch: graph.BranchEmbed,
		Name:   "embed",
		Input:  []graph.TensorID{0},
		Output: embedID,
		Params: graph.EmbedParams{
			VocabSize: cfg.VocabSize,
			EmbDim:    cfg.EmbeddingLength,
		},
	})

	prev := embedID
	for i := 0; i < cfg.BlockCount; i++ {
		mambaOut := graph.NewTensorID()
		g.AddNode(graph.Node{
			Op:     graph.OpMambaBlock,
			Branch: graph.BranchMamba,
			Name:   fmt.Sprintf("layer%d.mamba", i),
			Input:  []graph.TensorID{prev},
			Output: mambaOut,
			Params: graph.MambaParams{},
		})

		residual := graph.NewTensorID()
		g.AddNode(graph.Node{
			Op:     graph.OpAdd,
			Branch: graph.BranchFFN,
			Name:   fmt.Sprintf("layer%d.residual", i),
			Input:  []graph.TensorID{prev, mambaOut},
			Output: residual,
			Params: graph.FFNParams{},
		})
		prev = residual
	}

	output := graph.NewTensorID()
	g.AddNode(graph.Node{
		Op:     graph.OpOutput,
		Branch: graph.BranchOutput,
		Name:   "output",
		Input:  []graph.TensorID{prev},
		Output: output,
		Params: graph.OutputParams{
			Softcap: inst.Config.Config.FinalLogitSoftcap,
		},
	})

	return g, nil
}
