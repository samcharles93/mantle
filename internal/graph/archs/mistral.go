package archs

import (
	"fmt"

	"github.com/samcharles93/mantle/internal/backend/core"
	"github.com/samcharles93/mantle/internal/graph"
)

// MistralBuilder constructs a computation graph for Mistral-architecture models.
// Mistral is llama-like but uses sliding window attention on some layers.
// The layer's AttnWindow field is set for sliding window layers.
type MistralBuilder struct{}

// BuildGraph builds the full computation graph for a Mistral model.
func (b *MistralBuilder) BuildGraph(cfg *core.Config, inst *core.Instance) (*graph.Graph, error) {
	graph.ResetTensorID()
	g := &graph.Graph{UID: 0}

	embedID := graph.NewTensorID()
	g.AddNode(graph.Node{Op: graph.OpEmbed, Branch: graph.BranchEmbed, Name: "embed", Input: []graph.TensorID{0}, Output: embedID, Params: graph.EmbedParams{VocabSize: cfg.VocabSize, EmbDim: cfg.EmbeddingLength}})

	prev := embedID
	for i := 0; i < cfg.BlockCount; i++ {
		layer := &inst.Layers[i]
		normed := graph.NewTensorID()
		g.AddNode(graph.Node{Op: graph.OpAdd, Branch: graph.BranchFFN, Name: fmt.Sprintf("layer%d.attn_norm", i), Input: []graph.TensorID{prev}, Output: normed, Params: graph.FFNParams{}})

		attnOut := graph.NewTensorID()
		g.AddNode(graph.Node{Op: graph.OpAttentionBlock, Branch: graph.BranchAttention, Name: fmt.Sprintf("layer%d.attention", i), Input: []graph.TensorID{normed}, Output: attnOut, Params: graph.AttentionParams{
			NHeadKV:    layer.HeadKV,
			HeadDim:    layer.HeadDim,
			KVStride:   inst.MaxKVStride,
			SlidingWin: layer.AttnWindow,
			LayerIndex: i,
			InvFreq:    layer.RopeInvFreq,
			AttnScale:  layer.AttnScale,
		}})

		residual1 := graph.NewTensorID()
		g.AddNode(graph.Node{Op: graph.OpAdd, Branch: graph.BranchFFN, Name: fmt.Sprintf("layer%d.attn_residual", i), Input: []graph.TensorID{prev, attnOut}, Output: residual1, Params: graph.FFNParams{}})

		normed2 := graph.NewTensorID()
		g.AddNode(graph.Node{Op: graph.OpAdd, Branch: graph.BranchFFN, Name: fmt.Sprintf("layer%d.ffn_norm", i), Input: []graph.TensorID{residual1}, Output: normed2, Params: graph.FFNParams{}})

		ffnOut := graph.NewTensorID()
		g.AddNode(graph.Node{Op: graph.OpFFNBlock, Branch: graph.BranchFFN, Name: fmt.Sprintf("layer%d.ffn", i), Input: []graph.TensorID{normed2}, Output: ffnOut, Params: graph.FFNParams{Activation: layer.FFNActivation}})

		residual2 := graph.NewTensorID()
		g.AddNode(graph.Node{Op: graph.OpAdd, Branch: graph.BranchFFN, Name: fmt.Sprintf("layer%d.ffn_residual", i), Input: []graph.TensorID{residual1, ffnOut}, Output: residual2, Params: graph.FFNParams{}})
		prev = residual2
	}

	output := graph.NewTensorID()
	g.AddNode(graph.Node{Op: graph.OpOutput, Branch: graph.BranchOutput, Name: "output", Input: []graph.TensorID{prev}, Output: output, Params: graph.OutputParams{Softcap: cfg.FinalLogitSoftcap}})
	return g, nil
}
