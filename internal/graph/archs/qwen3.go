package archs

import (
	"fmt"

	"github.com/samcharles93/mantle/internal/backend/core"
	"github.com/samcharles93/mantle/internal/graph"
)

type Qwen3Builder struct{}

func (b *Qwen3Builder) BuildGraph(cfg *core.Config, inst *core.Instance) (*graph.Graph, error) {
	graph.ResetTensorID()
	g := &graph.Graph{UID: 0}

	embedID := graph.NewTensorID()
	g.AddNode(graph.Node{
		Op:     graph.OpEmbed,
		Branch: graph.BranchEmbed,
		Name:   "embed",
		Input:  []graph.TensorID{0},
		Output: embedID,
		Params: graph.EmbedParams{VocabSize: cfg.VocabSize, EmbDim: cfg.EmbeddingLength},
	})

	prev := embedID
	for i := 0; i < cfg.BlockCount; i++ {
		layer := &inst.Layers[i]

		normed := graph.NewTensorID()
		g.AddNode(graph.Node{
			Op:     graph.OpAdd,
			Branch: graph.BranchFFN,
			Name:   fmt.Sprintf("layer%d.attn_norm", i),
			Input:  []graph.TensorID{prev},
			Output: normed,
			Params: graph.FFNParams{},
		})

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

		residual1 := graph.NewTensorID()
		g.AddNode(graph.Node{
			Op:     graph.OpAdd,
			Branch: graph.BranchFFN,
			Name:   fmt.Sprintf("layer%d.attn_residual", i),
			Input:  []graph.TensorID{prev, attnOut},
			Output: residual1,
			Params: graph.FFNParams{},
		})

		normed2 := graph.NewTensorID()
		g.AddNode(graph.Node{
			Op:     graph.OpAdd,
			Branch: graph.BranchFFN,
			Name:   fmt.Sprintf("layer%d.ffn_norm", i),
			Input:  []graph.TensorID{residual1},
			Output: normed2,
			Params: graph.FFNParams{},
		})

		// Qwen3: some layers are MoE, others are standard FFN.
		isMoE := layer.MoE != nil || (cfg.NumDenseLayers > 0 && i >= cfg.NumDenseLayers)
		if isMoE {
			moeOut := graph.NewTensorID()
			g.AddNode(graph.Node{
				Op:     graph.OpMoEBlock,
				Branch: graph.BranchFFN,
				Name:   fmt.Sprintf("layer%d.moe", i),
				Input:  []graph.TensorID{normed2},
				Output: moeOut,
				Params: graph.MoEParams{
					TopK:         cfg.NumExpertsPerTok,
					NumExperts:   cfg.NumExperts,
					SharedExpert: cfg.NumSharedExperts > 0,
				},
			})
			residual2 := graph.NewTensorID()
			g.AddNode(graph.Node{
				Op:     graph.OpAdd,
				Branch: graph.BranchFFN,
				Name:   fmt.Sprintf("layer%d.moe_residual", i),
				Input:  []graph.TensorID{residual1, moeOut},
				Output: residual2,
				Params: graph.FFNParams{},
			})
			prev = residual2
		} else {
			ffnOut := graph.NewTensorID()
			g.AddNode(graph.Node{
				Op:     graph.OpFFNBlock,
				Branch: graph.BranchFFN,
				Name:   fmt.Sprintf("layer%d.ffn", i),
				Input:  []graph.TensorID{normed2},
				Output: ffnOut,
				Params: graph.FFNParams{Activation: layer.FFNActivation},
			})
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
	}

	output := graph.NewTensorID()
	g.AddNode(graph.Node{
		Op:     graph.OpOutput,
		Branch: graph.BranchOutput,
		Name:   "output",
		Input:  []graph.TensorID{prev},
		Output: output,
		Params: graph.OutputParams{Softcap: cfg.FinalLogitSoftcap},
	})
	return g, nil
}
