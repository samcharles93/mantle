package archs

import (
	"fmt"

	"github.com/samcharles93/mantle/internal/backend/core"
	"github.com/samcharles93/mantle/internal/graph"
)

type DeltaNetBuilder struct{}

func (b *DeltaNetBuilder) BuildGraph(cfg *core.Config, inst *core.Instance) (*graph.Graph, error) {
	graph.ResetTensorID()
	g := &graph.Graph{UID: 0}

	embedID := graph.NewTensorID()
	g.AddNode(graph.Node{Op: graph.OpEmbed, Branch: graph.BranchEmbed, Name: "embed", Input: []graph.TensorID{0}, Output: embedID, Params: graph.EmbedParams{VocabSize: cfg.VocabSize, EmbDim: cfg.EmbeddingLength}})

	prev := embedID
	for i := 0; i < cfg.BlockCount; i++ {
		// DeltaNet replaces attention block
		deltaOut := graph.NewTensorID()
		g.AddNode(graph.Node{Op: graph.OpDeltaNetBlock, Branch: graph.BranchDeltaNet, Name: fmt.Sprintf("layer%d.deltanet", i), Input: []graph.TensorID{prev}, Output: deltaOut, Params: graph.DeltaNetParams{}})

		// Residual add
		residual1 := graph.NewTensorID()
		g.AddNode(graph.Node{Op: graph.OpAdd, Branch: graph.BranchFFN, Name: fmt.Sprintf("layer%d.attn_residual", i), Input: []graph.TensorID{prev, deltaOut}, Output: residual1, Params: graph.FFNParams{}})

		// FFN block (if present in architecture)
		normed := graph.NewTensorID()
		g.AddNode(graph.Node{Op: graph.OpAdd, Branch: graph.BranchFFN, Name: fmt.Sprintf("layer%d.ffn_norm", i), Input: []graph.TensorID{residual1}, Output: normed, Params: graph.FFNParams{}})

		ffnOut := graph.NewTensorID()
		g.AddNode(graph.Node{Op: graph.OpFFNBlock, Branch: graph.BranchFFN, Name: fmt.Sprintf("layer%d.ffn", i), Input: []graph.TensorID{normed}, Output: ffnOut, Params: graph.FFNParams{Activation: inst.Layers[i].FFNActivation}})

		residual2 := graph.NewTensorID()
		g.AddNode(graph.Node{Op: graph.OpAdd, Branch: graph.BranchFFN, Name: fmt.Sprintf("layer%d.ffn_residual", i), Input: []graph.TensorID{residual1, ffnOut}, Output: residual2, Params: graph.FFNParams{}})
		prev = residual2
	}

	output := graph.NewTensorID()
	g.AddNode(graph.Node{Op: graph.OpOutput, Branch: graph.BranchOutput, Name: "output", Input: []graph.TensorID{prev}, Output: output, Params: graph.OutputParams{Softcap: inst.Config.Config.FinalLogitSoftcap}})
	return g, nil
}
