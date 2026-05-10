package simd

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/samcharles93/mantle/internal/graph"
)

// Compute executes a computation graph node-by-node for a single token
// using the SIMD backend. Each graph node dispatches to an existing
// kernel function based on its Branch field.
//
// The graph must contain at least one BranchOutput node that produces
// the final logits. Intermediate activations are tracked via TensorID
// slots stored in the model's scratch buffers.
func (m *Instance) Compute(ctx *graph.ComputeContext, g *graph.Graph) ([]float32, error) {
	if len(g.Nodes) == 0 {
		return nil, fmt.Errorf("Compute: empty graph")
	}

	tensors := make(map[graph.TensorID][]float32)
	layerIdx := 0

	for _, node := range g.Nodes {
		switch node.Branch {
		case graph.BranchEmbed:
			x, err := initializeTokenInput(m, ctx.Token)
			if err != nil {
				return nil, fmt.Errorf("%s: %w", node.Name, err)
			}
			tensors[node.Output] = x

		case graph.BranchAttention:
			input, err := getTensorInput(tensors, node, 0)
			if err != nil {
				return nil, err
			}
			attnParams, ok := node.Params.(graph.AttentionParams)
			if !ok {
				return nil, fmt.Errorf("%s: expected AttentionParams, got %T", node.Name, node.Params)
			}
			li := attnParams.LayerIndex
			if li < 0 || li >= len(m.Layers) {
				return nil, fmt.Errorf("%s: layer index %d out of range [0, %d)", node.Name, li, len(m.Layers))
			}
			var out []float32
			if node.Op == graph.OpFusedAttention {
				out = fusedAttention(m, &m.Layers[li], input, ctx.Pos, attnParams)
			} else {
				out = Attention(m, &m.Layers[li], input, ctx.Pos)
			}
			tensors[node.Output] = out
			layerIdx = li + 1

		case graph.BranchFFN:
			// Determine whether this is a norm+add node (OpAdd with two inputs),
			// a fused norm+residual node, or a true FFN/MoE block.
			input, err := getTensorInput(tensors, node, 0)
			if err != nil {
				return nil, err
			}

			// OpAdd with two inputs: element-wise residual add.
			if node.Op == graph.OpAdd && len(node.Input) >= 2 {
				residual, err := getTensorInput(tensors, node, 1)
				if err != nil {
					return nil, err
				}
				// Copy input to output and add residual.
				n := len(input)
				out := make([]float32, n)
				copy(out, input)
				Add(out, residual)
				tensors[node.Output] = out
				continue
			}

			// OpFusedNormResidual: apply norm to input, then add residual.
			if node.Op == graph.OpFusedNormResidual && len(node.Input) >= 2 {
				residual, err := getTensorInput(tensors, node, 1)
				if err != nil {
					return nil, err
				}
				li := extractLayerIndex(node.Name, layerIdx)
				if li < 0 || li >= len(m.Layers) {
					return nil, fmt.Errorf("%s: layer index %d out of range", node.Name, li)
				}
				layer := &m.Layers[li]
				normWeight := layer.AttnNorm
				if len(normWeight) == 0 {
					normWeight = layer.FfnNorm
				}
				RMSNorm(m.Scratch.Tmp, input, normWeight, m.RMSEpsilon)
				out := make([]float32, len(m.Scratch.Tmp))
				copy(out, m.Scratch.Tmp)
				Add(out, residual)
				tensors[node.Output] = out
				layerIdx = li + 1
				continue
			}

			// OpAdd with a _norm name and single input: standalone RMSNorm.
			// This matches the llama/llama-like builder convention where attn_norm
			// and ffn_norm are emitted as OpAdd nodes with a single input.
			if node.Op == graph.OpAdd && len(node.Input) == 1 {
				name := node.Name
				if strings.Contains(name, "_norm") || strings.Contains(name, ".norm") {
					li := extractLayerIndex(name, layerIdx)
					if li < 0 || li >= len(m.Layers) {
						return nil, fmt.Errorf("%s: layer index %d out of range", name, li)
					}
					layer := &m.Layers[li]
					var weight []float32
					switch {
					case strings.Contains(name, "attn_norm"):
						weight = layer.AttnNorm
					case strings.Contains(name, "post_attn_norm"):
						weight = layer.PostAttnNorm
					case strings.Contains(name, "ffn_norm"):
						weight = layer.FfnNorm
					case strings.Contains(name, "post_ffn_norm"):
						weight = layer.PostFfnNorm
					default:
						return nil, fmt.Errorf("%s: cannot determine norm weight from name", name)
					}
					if len(weight) == 0 {
						return nil, fmt.Errorf("%s: norm weight is empty", name)
					}
					RMSNorm(m.Scratch.Tmp, input, weight, m.RMSEpsilon)
					out := make([]float32, len(m.Scratch.Tmp))
					copy(out, m.Scratch.Tmp)
					tensors[node.Output] = out
					layerIdx = li + 1
					continue
				}
			}

			// Standard FFN or MoE block.
			li := extractLayerIndex(node.Name, layerIdx)
			if li < 0 || li >= len(m.Layers) {
				return nil, fmt.Errorf("%s: layer index %d out of range", node.Name, li)
			}
			layer := &m.Layers[li]

			// OpFusedFFN: fused gate+up+activation+down in a single call.
			if node.Op == graph.OpFusedFFN {
				ffnParams, ok := node.Params.(graph.FFNParams)
				if !ok {
					return nil, fmt.Errorf("%s: expected FFNParams for OpFusedFFN, got %T", node.Name, node.Params)
				}
				out := fusedFFN(m, layer, input, ffnParams.Activation)
				tensors[node.Output] = out
				layerIdx = li + 1
				continue
			}

			switch node.Params.(type) {
			case graph.MoEParams:
				out := MoE(m, layer, input)
				tensors[node.Output] = out
			default:
				out := FFN(m, layer, input)
				tensors[node.Output] = out
			}
			layerIdx = li + 1

		case graph.BranchMamba:
			input, err := getTensorInput(tensors, node, 0)
			if err != nil {
				return nil, err
			}
			li := extractLayerIndex(node.Name, layerIdx)
			if li < 0 || li >= len(m.Layers) {
				return nil, fmt.Errorf("%s: layer index %d out of range", node.Name, li)
			}
			out := Mamba(m, &m.Layers[li], input)
			tensors[node.Output] = out
			layerIdx = li + 1

		case graph.BranchDeltaNet:
			input, err := getTensorInput(tensors, node, 0)
			if err != nil {
				return nil, err
			}
			li := extractLayerIndex(node.Name, layerIdx)
			if li < 0 || li >= len(m.Layers) {
				return nil, fmt.Errorf("%s: layer index %d out of range", node.Name, li)
			}
			out := DeltaNet(m, &m.Layers[li], input)
			tensors[node.Output] = out
			layerIdx = li + 1

		case graph.BranchOutput:
			input, err := getTensorInput(tensors, node, 0)
			if err != nil {
				return nil, err
			}
			outParams, ok := node.Params.(graph.OutputParams)
			if !ok {
				return nil, fmt.Errorf("%s: expected OutputParams, got %T", node.Name, node.Params)
			}

			// Output norm + projection.
			RMSNorm(m.Scratch.Tmp, input, m.OutputNorm, m.RMSEpsilon)
			m.Ops().MatVec(m.Scratch.Logits, m.Output, m.Scratch.Tmp)

			if scale := m.Config.Config.LMHeadMultiplier; scale != 0 && scale != 1 {
				s := float32(scale)
				for i := range m.Scratch.Logits {
					m.Scratch.Logits[i] *= s
				}
			}
			if softcap := outParams.Softcap; softcap > 0 {
				for i := range m.Scratch.Logits {
					m.Scratch.Logits[i] = fastTanh(m.Scratch.Logits[i]/softcap) * softcap
				}
			}

			logits := make([]float32, len(m.Scratch.Logits))
			copy(logits, m.Scratch.Logits)
			return logits, nil

		default:
			return nil, fmt.Errorf("%s: unsupported branch %v", node.Name, node.Branch)
		}
	}

	return nil, fmt.Errorf("Compute: graph has no BranchOutput node")
}

// getTensorInput retrieves the tensor value for the given input index in a node.
func getTensorInput(tensors map[graph.TensorID][]float32, node graph.Node, idx int) ([]float32, error) {
	if idx >= len(node.Input) {
		return nil, fmt.Errorf("%s: input index %d out of range (have %d inputs)", node.Name, idx, len(node.Input))
	}
	tid := node.Input[idx]
	if tid == 0 {
		return nil, fmt.Errorf("%s: input %d is sentinel (0) — missing connection", node.Name, idx)
	}
	val := tensors[tid]
	if val == nil {
		return nil, fmt.Errorf("%s: tensor %d not produced by any prior node", node.Name, tid)
	}
	return val, nil
}

// extractLayerIndex attempts to parse "layerN." from a node name.
// Falls back to the provided default when parsing fails.
func extractLayerIndex(name string, fallback int) int {
	if !strings.HasPrefix(name, "layer") {
		return fallback
	}
	rest := name[5:] // skip "layer"
	dotIdx := strings.IndexByte(rest, '.')
	if dotIdx <= 0 {
		return fallback
	}
	n, err := strconv.Atoi(rest[:dotIdx])
	if err != nil {
		return fallback
	}
	return n
}
