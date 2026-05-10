//go:build cuda

package cuda

import (
	"fmt"
	"strings"

	instance "github.com/samcharles93/mantle/internal/backend/core"
	"github.com/samcharles93/mantle/internal/graph"
)

// GraphRuntime wraps a cudaRuntime with the core.Instance it was loaded from,
// enabling graph-based computation that needs direct access to model weights.
type GraphRuntime struct {
	*cudaRuntime
	inst       *instance.Instance
	GraphCache *CUDAGraphCache
}

// NewGraphRuntime creates a graph-compute wrapper from an existing cudaRuntime
// and the core.Instance it governs. The instance must be the same one that was
// used during LoadModel (obtained via bootstrap.LoadSIMDRuntime).
func NewGraphRuntime(r *cudaRuntime, inst *instance.Instance) *GraphRuntime {
	return &GraphRuntime{
		cudaRuntime: r,
		inst:        inst,
		GraphCache:  &CUDAGraphCache{graphs: make(map[uint64]*CUDAGraph)},
	}
}

// Compute executes a computation graph using the CUDA backend, returning
// the final output tensor (logits). Fused operation nodes fall back to
// their non-fused equivalents; fused CUDA kernels will be added in T20/T21.
func (gr *GraphRuntime) Compute(ctx *graph.ComputeContext, g *graph.Graph) ([]float32, error) {
	if gr.cudaRuntime == nil || gr.model == nil {
		return nil, fmt.Errorf("cuda runtime is closed")
	}
	if gr.inst == nil {
		return nil, fmt.Errorf("cuda compute: no core instance bound")
	}

	if err := g.Validate(); err != nil {
		return nil, fmt.Errorf("cuda compute: invalid graph: %w", err)
	}

	tensors := make(map[graph.TensorID][]float32)
	currentLayer := 0

	for i, node := range g.Nodes {
		out, err := gr.computeNode(ctx, node, tensors, &currentLayer)
		if err != nil {
			return nil, fmt.Errorf("cuda compute node %d %q: %w", i, node.Name, err)
		}
		tensors[node.Output] = out
	}

	last := g.Nodes[len(g.Nodes)-1]
	logits, ok := tensors[last.Output]
	if !ok {
		return nil, fmt.Errorf("cuda compute: output tensor %d not produced", last.Output)
	}
	return logits, nil
}

func (gr *GraphRuntime) computeNode(
	ctx *graph.ComputeContext,
	node graph.Node,
	tensors map[graph.TensorID][]float32,
	currentLayer *int,
) ([]float32, error) {

	getInput := func(idx int) ([]float32, error) {
		if idx >= len(node.Input) {
			return nil, fmt.Errorf("input index %d out of range (have %d inputs)", idx, len(node.Input))
		}
		tid := node.Input[idx]
		if tid == 0 {
			return nil, fmt.Errorf("input tensor 0 is a sentinel — no data available")
		}
		v, ok := tensors[tid]
		if !ok {
			return nil, fmt.Errorf("input tensor %d not available", tid)
		}
		return v, nil
	}

	switch node.Branch {
	case graph.BranchEmbed:
		return computeEmbed(gr.inst, ctx.Token, node.Params.(graph.EmbedParams))

	case graph.BranchAttention:
		ap := node.Params.(graph.AttentionParams)
		*currentLayer = ap.LayerIndex
		x, err := getInput(0)
		if err != nil {
			return nil, err
		}
		if node.Op == graph.OpFusedAttention {
			return gr.fusedAttentionBlock(ctx, ap, x)
		}
		return gr.computeAttention(ctx, ap, x)

	case graph.BranchFFN:
		return gr.computeFFNNode(ctx, node, tensors, currentLayer)

	case graph.BranchMamba:
		x, err := getInput(0)
		if err != nil {
			return nil, err
		}
		return gr.computeMamba(ctx, node, x, currentLayer)

	case graph.BranchDeltaNet:
		x, err := getInput(0)
		if err != nil {
			return nil, err
		}
		return gr.computeDeltaNet(ctx, node, x, currentLayer)

	case graph.BranchOutput:
		x, err := getInput(0)
		if err != nil {
			return nil, err
		}
		return gr.computeOutput(node.Params.(graph.OutputParams), x)

	default:
		return nil, fmt.Errorf("unsupported branch %s", node.Branch)
	}
}

// --- Embed ---

func computeEmbed(inst *instance.Instance, token int, params graph.EmbedParams) ([]float32, error) {
	if inst.Embeddings == nil {
		return nil, fmt.Errorf("embed: no embedding matrix")
	}
	if token < 0 || token >= inst.Embeddings.R {
		return nil, fmt.Errorf("embed: token %d out of range [0, %d)", token, inst.Embeddings.R)
	}
	embDim := inst.Embeddings.C
	out := make([]float32, embDim)
	inst.Embeddings.RowTo(out, token)
	return out, nil
}

// --- Attention ---

func (gr *GraphRuntime) computeAttention(
	ctx *graph.ComputeContext,
	params graph.AttentionParams,
	x []float32,
) ([]float32, error) {
	if params.LayerIndex < 0 || params.LayerIndex >= len(gr.inst.Layers) {
		return nil, fmt.Errorf("attention: layer index %d out of range", params.LayerIndex)
	}
	layer := &gr.inst.Layers[params.LayerIndex]

	nHead := gr.inst.HeadCount
	headDim := params.HeadDim
	if headDim <= 0 {
		headDim = layer.HeadDim
	}
	kvHeads := params.NHeadKV
	if kvHeads <= 0 {
		kvHeads = layer.HeadKV
	}
	kvStride := params.KVStride
	if kvStride <= 0 {
		kvStride = gr.inst.MaxKVStride
	}

	invFreq := params.InvFreq
	if len(invFreq) == 0 {
		invFreq = layer.RopeInvFreq
	}
	ropeAttnScale := params.AttnScale
	if ropeAttnScale == 0 {
		ropeAttnScale = layer.RopeAttnScale
	}
	applyRope := !layer.NoRoPE && len(invFreq) > 0

	embDim := gr.inst.Embeddings.C
	if len(x) < embDim {
		return nil, fmt.Errorf("attention: input len %d < embd %d", len(x), embDim)
	}

	projDim := layer.Wo.R
	out := make([]float32, projDim)

	if ok := gr.ops.QKVAttentionProjection(
		out, layer, layer, x[:embDim],
		ctx.Pos, 0, nHead, headDim, kvHeads, kvStride,
		params.AttnScale, gr.inst.RMSEpsilon, params.AttnLogitSoftcap,
		invFreq, ropeAttnScale, applyRope,
		true,
	); ok {
		if err := gr.ops.FlushBlockResult(); err != nil {
			return nil, fmt.Errorf("attention: flush after QKV pipeline: %w", err)
		}
		return out, nil
	}

	if err := gr.computeAttentionFallback(ctx, layer, params, x, out); err != nil {
		return nil, fmt.Errorf("attention fallback: %w", err)
	}
	return out, nil
}

func (gr *GraphRuntime) computeAttentionFallback(
	ctx *graph.ComputeContext,
	layer *instance.Layer,
	params graph.AttentionParams,
	x []float32,
	out []float32,
) error {
	logits, err := gr.model.ForwardToken(ctx.Token)
	if err != nil {
		return err
	}
	_ = logits
	return fmt.Errorf("attention fallback not fully implemented: use QKVAttentionProjection fast path")
}

// --- FFN / Add / Norm (all share BranchFFN) ---

func (gr *GraphRuntime) computeFFNNode(
	ctx *graph.ComputeContext,
	node graph.Node,
	tensors map[graph.TensorID][]float32,
	currentLayer *int,
) ([]float32, error) {
	getInput := func(idx int) ([]float32, error) {
		if idx >= len(node.Input) {
			return nil, fmt.Errorf("input index %d out of range", idx)
		}
		v, ok := tensors[node.Input[idx]]
		if !ok {
			return nil, fmt.Errorf("input tensor %d not available", node.Input[idx])
		}
		return v, nil
	}

	switch node.Op {
	case graph.OpFFNBlock:
		x, err := getInput(0)
		if err != nil {
			return nil, err
		}
		return gr.computeFFNBlock(node, x, currentLayer)

	case graph.OpFusedFFN:
		x, err := getInput(0)
		if err != nil {
			return nil, err
		}
		return gr.fusedFFNBlock(node, x, currentLayer)

	case graph.OpMoEBlock, graph.OpFusedMoE:
		x, err := getInput(0)
		if err != nil {
			return nil, err
		}
		return gr.computeMoEBlock(node, x, currentLayer)

	case graph.OpAdd:
		return gr.computeAddNode(node, tensors, currentLayer)

	default:
		return nil, fmt.Errorf("ffn branch: unsupported op %s", node.Op)
	}
}

func (gr *GraphRuntime) computeFFNBlock(
	node graph.Node,
	x []float32,
	currentLayer *int,
) ([]float32, error) {
	layerIdx := parseLayerIndex(node.Name, *currentLayer)
	*currentLayer = layerIdx

	if layerIdx < 0 || layerIdx >= len(gr.inst.Layers) {
		return nil, fmt.Errorf("ffn: layer index %d out of range", layerIdx)
	}
	layer := &gr.inst.Layers[layerIdx]

	out := make([]float32, layer.FfnDown.R)
	if gr.ops.FFNBlock(layer, x, out) {
		return out, nil
	}
	return nil, fmt.Errorf("ffn: CUDA fast path unavailable for layer %d", layerIdx)
}

func (gr *GraphRuntime) computeMoEBlock(
	node graph.Node,
	x []float32,
	currentLayer *int,
) ([]float32, error) {
	layerIdx := parseLayerIndex(node.Name, *currentLayer)
	*currentLayer = layerIdx

	if layerIdx < 0 || layerIdx >= len(gr.inst.Layers) {
		return nil, fmt.Errorf("moe: layer index %d out of range", layerIdx)
	}
	layer := &gr.inst.Layers[layerIdx]
	if layer.MoE == nil {
		return nil, fmt.Errorf("moe: layer %d has no MoE config", layerIdx)
	}

	out := make([]float32, gr.inst.Embeddings.C)
	if gr.ops.MoEBlock(layer.MoE, x, out, instance.MoEConfig{}) {
		return out, nil
	}
	return nil, fmt.Errorf("moe: CUDA fast path unavailable for layer %d", layerIdx)
}

func (gr *GraphRuntime) computeAddNode(
	node graph.Node,
	tensors map[graph.TensorID][]float32,
	currentLayer *int,
) ([]float32, error) {
	name := node.Name

	layerIdx := parseLayerIndex(name, *currentLayer)
	*currentLayer = layerIdx

	if strings.Contains(name, "_norm") || strings.Contains(name, ".norm") {
		return gr.computeNorm(node, tensors, layerIdx, name)
	}

	if strings.Contains(name, "_residual") || strings.Contains(name, ".residual") {
		if len(node.Input) < 2 {
			return nil, fmt.Errorf("residual add %q: expected 2 inputs, got %d", name, len(node.Input))
		}
		x, ok := tensors[node.Input[0]]
		if !ok {
			return nil, fmt.Errorf("residual: input tensor %d not available", node.Input[0])
		}
		y, ok := tensors[node.Input[1]]
		if !ok {
			return nil, fmt.Errorf("residual: input tensor %d not available", node.Input[1])
		}
		if len(x) != len(y) {
			return nil, fmt.Errorf("residual add %q: shape mismatch (%d vs %d)", name, len(x), len(y))
		}
		out := make([]float32, len(x))
		for i := range out {
			out[i] = x[i] + y[i]
		}
		return out, nil
	}

	return nil, fmt.Errorf("add node %q: unrecognised pattern (expected _norm or _residual)", name)
}

func (gr *GraphRuntime) computeNorm(
	node graph.Node,
	tensors map[graph.TensorID][]float32,
	layerIdx int,
	name string,
) ([]float32, error) {
	if layerIdx < 0 || layerIdx >= len(gr.inst.Layers) {
		return nil, fmt.Errorf("norm %q: layer %d out of range", name, layerIdx)
	}
	layer := &gr.inst.Layers[layerIdx]

	x, ok := tensors[node.Input[0]]
	if !ok {
		return nil, fmt.Errorf("norm %q: input tensor %d not available", name, node.Input[0])
	}

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
		return nil, fmt.Errorf("norm %q: cannot determine weight from name", name)
	}

	if len(weight) != len(x) {
		return nil, fmt.Errorf("norm %q: weight len %d != input len %d", name, len(weight), len(x))
	}

	dst := make([]float32, len(x))
	gr.ops.RMSNorm(dst, x, weight, gr.inst.RMSEpsilon)
	return dst, nil
}

// --- Mamba ---

func (gr *GraphRuntime) computeMamba(
	ctx *graph.ComputeContext,
	node graph.Node,
	x []float32,
	currentLayer *int,
) ([]float32, error) {
	layerIdx := parseLayerIndex(node.Name, *currentLayer)
	*currentLayer = layerIdx

	if layerIdx < 0 || layerIdx >= len(gr.inst.Layers) {
		return nil, fmt.Errorf("mamba: layer index %d out of range", layerIdx)
	}
	layer := &gr.inst.Layers[layerIdx]
	if layer.Mamba == nil {
		return nil, fmt.Errorf("mamba: layer %d has no Mamba config", layerIdx)
	}

	out := make([]float32, layer.Mamba.OutProj.R)
	cfg := instance.MambaConfig{
		SSMInMultiplier:     float32(gr.inst.Config.Config.SSMInMultiplier),
		SSMOutMultiplier:    float32(gr.inst.Config.Config.SSMOutMultiplier),
		TimeStepMin:         float32(gr.inst.Config.Config.TimeStepMin),
		TimeStepMax:         float32(gr.inst.Config.Config.TimeStepMax),
		TimeStepFloor:       float32(gr.inst.Config.Config.TimeStepFloor),
		MambaRMSNorm:        gr.inst.Config.Config.MambaRMSNorm,
		MambaNormBeforeGate: gr.inst.Config.Config.MambaNormBeforeGate,
		RMSEpsilon:          gr.inst.RMSEpsilon,
	}
	if gr.ops.MambaBlock(layer.Mamba, x, out, cfg) {
		return out, nil
	}
	return nil, fmt.Errorf("mamba: CUDA fast path unavailable for layer %d", layerIdx)
}

// --- DeltaNet ---

func (gr *GraphRuntime) computeDeltaNet(
	ctx *graph.ComputeContext,
	node graph.Node,
	x []float32,
	currentLayer *int,
) ([]float32, error) {
	layerIdx := parseLayerIndex(node.Name, *currentLayer)
	*currentLayer = layerIdx

	if layerIdx < 0 || layerIdx >= len(gr.inst.Layers) {
		return nil, fmt.Errorf("deltanet: layer index %d out of range", layerIdx)
	}
	layer := &gr.inst.Layers[layerIdx]
	if layer.DeltaNet == nil {
		return nil, fmt.Errorf("deltanet: layer %d has no DeltaNet config", layerIdx)
	}

	out := make([]float32, layer.DeltaNet.OutProj.R)
	cfg := instance.DeltaNetConfig{
		RMSEpsilon: gr.inst.RMSEpsilon,
	}
	if gr.ops.DeltaNetBlock(layer.DeltaNet, x, out, cfg) {
		return out, nil
	}
	return nil, fmt.Errorf("deltanet: CUDA fast path unavailable for layer %d", layerIdx)
}

// --- Output ---

func (gr *GraphRuntime) computeOutput(
	params graph.OutputParams,
	x []float32,
) ([]float32, error) {
	if gr.inst.Output == nil {
		return nil, fmt.Errorf("output: no output projection matrix")
	}

	embDim := gr.inst.Embeddings.C
	if len(x) < embDim {
		return nil, fmt.Errorf("output: input len %d < embd %d", len(x), embDim)
	}

	normed := make([]float32, embDim)
	gr.ops.RMSNorm(normed, x[:embDim], gr.inst.OutputNorm, gr.inst.RMSEpsilon)

	vocabSize := gr.inst.Output.R
	logits := make([]float32, vocabSize)
	gr.ops.MatVec(logits, gr.inst.Output, normed)

	if params.Softcap > 0 {
		for i := range logits {
			logits[i] = softcap(logits[i], params.Softcap)
		}
	}

	return logits, nil
}

// --- Helpers ---

// parseLayerIndex extracts a layer index from a node name like "layer3.ffn".
// Falls back to defaultIdx when parsing fails.
func parseLayerIndex(name string, defaultIdx int) int {
	idx := strings.Index(name, "layer")
	if idx < 0 {
		return defaultIdx
	}
	rest := name[idx+5:]
	end := 0
	for end < len(rest) && rest[end] >= '0' && rest[end] <= '9' {
		end++
	}
	if end == 0 {
		return defaultIdx
	}
	var n int
	for _, c := range rest[:end] {
		n = n*10 + int(c-'0')
	}
	return n
}

// tanhApprox is a rational approximation: tanh(x) ≈ x / (1 + 0.3333·x²).
// Used for the optional logit softcap in output projection.
func tanhApprox(x float64) float32 {
	if x > 10 {
		return 1.0
	}
	if x < -10 {
		return -1.0
	}
	x2 := x * x
	return float32(x / (1.0 + 0.3333*x2))
}

func softcap(x, cap float32) float32 {
	if cap <= 0 {
		return x
	}
	return cap * tanhApprox(float64(x)/float64(cap))
}
