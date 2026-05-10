package archs

import (
	"fmt"

	"github.com/samcharles93/mantle/internal/backend/core"
	"github.com/samcharles93/mantle/internal/graph"
)

// LlamaBuilder builds a static forward-pass computation graph for a standard
// Llama-family transformer. The graph is topology-only and built once per model.
type LlamaBuilder struct{}

// BuildGraph constructs a complete forward-pass graph for a single token.
func (b *LlamaBuilder) BuildGraph(cfg *core.Config, inst *core.Instance) (*graph.Graph, error) {
	graph.ResetTensorID()
	g := &graph.Graph{UID: 0}

	// Embed node: produces hidden state from token ID.
	embedID := graph.NewTensorID()
	g.AddNode(graph.Node{
		Op:     graph.OpEmbed,
		Branch: graph.BranchEmbed,
		Name:   "embed",
		Input:  []graph.TensorID{0}, // 0 = sentinel for token input
		Output: embedID,
		Params: graph.EmbedParams{
			VocabSize: cfg.VocabSize,
			EmbDim:    cfg.EmbeddingLength,
		},
	})

	prev := embedID
	for i := 0; i < cfg.BlockCount; i++ {
		layer := &inst.Layers[i]
		// Attention pre-norm
		normed := graph.NewTensorID()
		g.AddNode(graph.Node{
			Op:     graph.OpAdd, // RMSNorm + residual handled by runtime; Add marks the residual path
			Branch: graph.BranchFFN,
			Name:   fmt.Sprintf("layer%d.attn_norm", i),
			Input:  []graph.TensorID{prev},
			Output: normed,
			Params: graph.FFNParams{}, // placeholder — norm params from layer config
		})

		// Attention block
		attnOut := graph.NewTensorID()
		g.AddNode(graph.Node{
			Op:     graph.OpAttentionBlock,
			Branch: graph.BranchAttention,
			Name:   fmt.Sprintf("layer%d.attention", i),
			Input:  []graph.TensorID{normed},
			Output: attnOut,
			Params: graph.AttentionParams{
				NHeadKV:    layer.HeadKV,
				HeadDim:    layer.HeadDim,
				KVStride:   inst.MaxKVStride,
				SlidingWin: layer.AttnWindow,
				LayerIndex: i,
				InvFreq:    layer.RopeInvFreq,
				AttnScale:  layer.AttnScale,
			},
		})

		// Residual add (attention output + previous state)
		residual1 := graph.NewTensorID()
		g.AddNode(graph.Node{
			Op:     graph.OpAdd,
			Branch: graph.BranchFFN,
			Name:   fmt.Sprintf("layer%d.attn_residual", i),
			Input:  []graph.TensorID{prev, attnOut},
			Output: residual1,
			Params: graph.FFNParams{},
		})

		// FFN pre-norm
		normed2 := graph.NewTensorID()
		g.AddNode(graph.Node{
			Op:     graph.OpAdd,
			Branch: graph.BranchFFN,
			Name:   fmt.Sprintf("layer%d.ffn_norm", i),
			Input:  []graph.TensorID{residual1},
			Output: normed2,
			Params: graph.FFNParams{},
		})

		// FFN block
		ffnOut := graph.NewTensorID()
		g.AddNode(graph.Node{
			Op:     graph.OpFFNBlock,
			Branch: graph.BranchFFN,
			Name:   fmt.Sprintf("layer%d.ffn", i),
			Input:  []graph.TensorID{normed2},
			Output: ffnOut,
			Params: graph.FFNParams{
				Activation: layer.FFNActivation,
				HasBias:    cfg.AttentionBias,
			},
		})

		// Residual add (FFN output + previous state)
		residual2 := graph.NewTensorID()
		g.AddNode(graph.Node{
			Op:     graph.OpAdd,
			Branch: graph.BranchFFN,
			Name:   fmt.Sprintf("layer%d.ffn_residual", i),
			Input:  []graph.TensorID{residual1, ffnOut},
			Output: residual2,
			Params: graph.FFNParams{},
		})

		prev = residual2
	}

	// Output head
	output := graph.NewTensorID()
	g.AddNode(graph.Node{
		Op:     graph.OpOutput,
		Branch: graph.BranchOutput,
		Name:   "output",
		Input:  []graph.TensorID{prev},
		Output: output,
		Params: graph.OutputParams{
			Softcap: cfg.FinalLogitSoftcap,
		},
	})

	return g, nil
}
