package simd

import (
	"math"
	"testing"

	"github.com/samcharles93/mantle/internal/backend/core"
	"github.com/samcharles93/mantle/internal/graph"
	"github.com/samcharles93/mantle/internal/graph/archs"
	"github.com/samcharles93/mantle/pkg/mcf"
)

func newTestInstance(vocabSize, embdDim, nLayers int) *core.Instance {
	headDim := 4
	nHead := embdDim / headDim
	if nHead < 1 {
		nHead = 1
		headDim = embdDim
	}
	kvHeads := nHead
	ffnDim := embdDim * 3 / 2
	if ffnDim < 1 {
		ffnDim = embdDim
	}
	kvStride := kvHeads * headDim
	maxCtx := 64

	m := &core.Instance{
		Config: &core.ModelConfig{
			Config: core.Config{
				VocabSize:           vocabSize,
				EmbeddingLength:     embdDim,
				HeadCount:           nHead,
				HeadDim:             headDim,
				BlockCount:          nLayers,
				FFNLength:           ffnDim,
				HiddenAct:           "silu",
				FinalLogitSoftcap:   0,
				LMHeadMultiplier:    0,
				EmbeddingMultiplier: 0,
			},
		},
		Embeddings: &core.Mat{
			R:      vocabSize,
			C:      embdDim,
			Stride: embdDim,
			DType:  mcf.DTypeF32,
			Data:   make([]float32, vocabSize*embdDim),
		},
		OutputNorm: make([]float32, embdDim),
		Output: &core.Mat{
			R:      vocabSize,
			C:      embdDim,
			Stride: embdDim,
			DType:  mcf.DTypeF32,
			Data:   make([]float32, vocabSize*embdDim),
		},
		Layers:      make([]core.Layer, nLayers),
		HeadDim:     headDim,
		HeadCount:   nHead,
		MaxKVStride: kvStride,
		MaxContext:  maxCtx,
		RMSEpsilon:  1e-5,
		Scratch: core.ScratchBuffers{
			X:        make([]float32, embdDim),
			Tmp:      make([]float32, embdDim),
			Tmp2:     make([]float32, max(embdDim, ffnDim)),
			Q:        make([]float32, nHead*headDim),
			K:        make([]float32, kvStride),
			V:        make([]float32, kvStride),
			AttnOut:  make([]float32, nHead*headDim),
			AttnProj: make([]float32, embdDim),
			Scores:   make([]float32, nHead*maxCtx),
			FfnUp:    make([]float32, ffnDim),
			FfnGate:  make([]float32, ffnDim),
			FfnAct:   make([]float32, ffnDim),
			Logits:   make([]float32, vocabSize),
		},
	}

	for i := range m.Embeddings.Data {
		m.Embeddings.Data[i] = float32(i+1) * 0.01
	}
	for i := range m.OutputNorm {
		m.OutputNorm[i] = 1.0
	}
	for i := range m.Output.Data {
		m.Output.Data[i] = float32(i+1) * 0.01
	}

	for li := 0; li < nLayers; li++ {
		layer := &m.Layers[li]
		layer.HeadKV = kvHeads
		layer.HeadDim = headDim
		layer.NoRoPE = true
		layer.FFNActivation = "silu"
		layer.AttnNorm = make([]float32, embdDim)
		layer.FfnNorm = make([]float32, embdDim)
		for i := range layer.AttnNorm {
			layer.AttnNorm[i] = 1.0
		}
		for i := range layer.FfnNorm {
			layer.FfnNorm[i] = 1.0
		}

		wqData := make([]float32, (nHead*headDim)*embdDim)
		for i := range wqData {
			wqData[i] = float32(i+1) * 0.005
		}
		layer.Wq = &core.Mat{R: nHead * headDim, C: embdDim, Stride: embdDim, DType: mcf.DTypeF32, Data: wqData}

		wkData := make([]float32, kvStride*embdDim)
		for i := range wkData {
			wkData[i] = float32(i+1) * 0.004
		}
		layer.Wk = &core.Mat{R: kvStride, C: embdDim, Stride: embdDim, DType: mcf.DTypeF32, Data: wkData}

		wvData := make([]float32, kvStride*embdDim)
		for i := range wvData {
			wvData[i] = float32(i+1) * 0.003
		}
		layer.Wv = &core.Mat{R: kvStride, C: embdDim, Stride: embdDim, DType: mcf.DTypeF32, Data: wvData}

		woData := make([]float32, embdDim*(nHead*headDim))
		for i := range woData {
			woData[i] = float32(i+1) * 0.002
		}
		layer.Wo = &core.Mat{R: embdDim, C: nHead * headDim, Stride: nHead * headDim, DType: mcf.DTypeF32, Data: woData}

		fUpData := make([]float32, ffnDim*embdDim)
		for i := range fUpData {
			fUpData[i] = float32(i+1) * 0.001
		}
		layer.FfnUp = &core.Mat{R: ffnDim, C: embdDim, Stride: embdDim, DType: mcf.DTypeF32, Data: fUpData}

		fGateData := make([]float32, ffnDim*embdDim)
		for i := range fGateData {
			fGateData[i] = float32(i+1) * 0.0015
		}
		layer.FfnGate = &core.Mat{R: ffnDim, C: embdDim, Stride: embdDim, DType: mcf.DTypeF32, Data: fGateData}

		fDownData := make([]float32, embdDim*ffnDim)
		for i := range fDownData {
			fDownData[i] = float32(i+1) * 0.0008
		}
		layer.FfnDown = &core.Mat{R: embdDim, C: ffnDim, Stride: ffnDim, DType: mcf.DTypeF32, Data: fDownData}

		kcacheSize := kvStride * maxCtx
		layer.AttnCache = core.AttnCache{
			K:        make([]float32, kcacheSize),
			V:        make([]float32, kcacheSize),
			KvStride: kvStride,
			CacheLen: maxCtx,
			Cap:      maxCtx,
		}
	}

	return m
}

func floatsEqual(a, b []float32, tol float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		diff := float32(math.Abs(float64(a[i] - b[i])))
		if diff > tol {
			return false
		}
	}
	return true
}

func TestGraphComputeEquivalentToForwardToken_EmbedOutput(t *testing.T) {
	const vocabSize = 10
	const embdDim = 8

	m := newTestInstance(vocabSize, embdDim, 0)
	m.BindDefaultOps()
	inst := (*Instance)(m)

	graph.ResetTensorID()
	embedID := graph.NewTensorID()
	outputID := graph.NewTensorID()

	g := &graph.Graph{
		Nodes: []graph.Node{
			{
				Op:     graph.OpEmbed,
				Branch: graph.BranchEmbed,
				Name:   "embed",
				Input:  []graph.TensorID{0},
				Output: embedID,
				Params: graph.EmbedParams{VocabSize: vocabSize, EmbDim: embdDim},
			},
			{
				Op:     graph.OpOutput,
				Branch: graph.BranchOutput,
				Name:   "output",
				Input:  []graph.TensorID{embedID},
				Output: outputID,
				Params: graph.OutputParams{Softcap: 0},
			},
		},
	}

	ctx := &graph.ComputeContext{Token: 5, Pos: 0}
	graphLogits, err := inst.Compute(ctx, g)
	if err != nil {
		t.Fatalf("Compute failed: %v", err)
	}

	inst.Reset()
	forwardLogits, err := inst.ForwardToken(5)
	if err != nil {
		t.Fatalf("ForwardToken failed: %v", err)
	}

	if len(graphLogits) != vocabSize {
		t.Fatalf("graph logits: expected %d, got %d", vocabSize, len(graphLogits))
	}
	if len(forwardLogits) != vocabSize {
		t.Fatalf("forward logits: expected %d, got %d", vocabSize, len(forwardLogits))
	}

	if !floatsEqual(graphLogits, forwardLogits, 1e-7) {
		t.Errorf("logits mismatch (embed→output):\n  graph: %v\n  forward: %v", graphLogits, forwardLogits)
	}
}

func TestGraphComputeEquivalentToForwardToken_AttnFFN(t *testing.T) {
	const vocabSize = 10
	const embdDim = 8

	m := newTestInstance(vocabSize, embdDim, 1)
	m.BindDefaultOps()

	b := &archs.LlamaBuilder{}
	g, err := b.BuildGraph(&m.Config.Config, m)
	if err != nil {
		t.Fatalf("BuildGraph failed: %v", err)
	}
	if err := g.Validate(); err != nil {
		t.Fatalf("validation failed: %v", err)
	}

	inst := (*Instance)(m)
	ctx := &graph.ComputeContext{Token: 5, Pos: 0}
	graphLogits, err := inst.Compute(ctx, g)
	if err != nil {
		t.Fatalf("Compute failed: %v", err)
	}

	inst.Reset()
	forwardLogits, err := inst.ForwardToken(5)
	if err != nil {
		t.Fatalf("ForwardToken failed: %v", err)
	}

	if len(graphLogits) != vocabSize {
		t.Fatalf("graph logits: expected %d, got %d", vocabSize, len(graphLogits))
	}
	if len(forwardLogits) != vocabSize {
		t.Fatalf("forward logits: expected %d, got %d", vocabSize, len(forwardLogits))
	}

    if !floatsEqual(graphLogits, forwardLogits, 2e-3) {
        firstDiff := -1
        for i := range graphLogits {
            diff := float32(math.Abs(float64(graphLogits[i] - forwardLogits[i])))
            if diff > 2e-3 {
                firstDiff = i
                break
            }
        }
        if firstDiff >= 0 {
			t.Errorf("logits mismatch at index %d (attention+ffn): graph[%d]=%v forward[%d]=%v diff=%v",
				firstDiff, firstDiff, graphLogits[firstDiff], firstDiff, forwardLogits[firstDiff],
				graphLogits[firstDiff]-forwardLogits[firstDiff])
		} else {
			t.Errorf("logits mismatch (attention+ffn)")
		}
	}
}

func TestGraphComputeStepByStep(t *testing.T) {
	const vocabSize = 10
	const embdDim = 8

	m := newTestInstance(vocabSize, embdDim, 1)
	m.BindDefaultOps()
	inst := (*Instance)(m)
	tok := 5

	xInit, err := initializeTokenInput(inst, tok)
	if err != nil {
		t.Fatalf("init: %v", err)
	}
	_ = xInit

	graph.ResetTensorID()
	embedID := graph.NewTensorID()
	normID := graph.NewTensorID()
	attnID := graph.NewTensorID()
	residual1ID := graph.NewTensorID()
	norm2ID := graph.NewTensorID()
	ffnID := graph.NewTensorID()
	residual2ID := graph.NewTensorID()
	outputID := graph.NewTensorID()

	g := &graph.Graph{
		Nodes: []graph.Node{
			{Op: graph.OpEmbed, Branch: graph.BranchEmbed, Name: "embed", Input: []graph.TensorID{0}, Output: embedID, Params: graph.EmbedParams{VocabSize: vocabSize, EmbDim: embdDim}},
			{Op: graph.OpAdd, Branch: graph.BranchFFN, Name: "layer0.attn_norm", Input: []graph.TensorID{embedID}, Output: normID, Params: graph.FFNParams{}},
			{Op: graph.OpAttentionBlock, Branch: graph.BranchAttention, Name: "layer0.attention", Input: []graph.TensorID{normID}, Output: attnID, Params: graph.AttentionParams{LayerIndex: 0}},
			{Op: graph.OpAdd, Branch: graph.BranchFFN, Name: "layer0.attn_residual", Input: []graph.TensorID{embedID, attnID}, Output: residual1ID, Params: graph.FFNParams{}},
			{Op: graph.OpAdd, Branch: graph.BranchFFN, Name: "layer0.ffn_norm", Input: []graph.TensorID{residual1ID}, Output: norm2ID, Params: graph.FFNParams{}},
			{Op: graph.OpFFNBlock, Branch: graph.BranchFFN, Name: "layer0.ffn", Input: []graph.TensorID{norm2ID}, Output: ffnID, Params: graph.FFNParams{Activation: "silu"}},
			{Op: graph.OpAdd, Branch: graph.BranchFFN, Name: "layer0.ffn_residual", Input: []graph.TensorID{residual1ID, ffnID}, Output: residual2ID, Params: graph.FFNParams{}},
			{Op: graph.OpOutput, Branch: graph.BranchOutput, Name: "output", Input: []graph.TensorID{residual2ID}, Output: outputID, Params: graph.OutputParams{Softcap: 0}},
		},
	}

	ctx := &graph.ComputeContext{Token: tok, Pos: 0}
	graphLogits, err := inst.Compute(ctx, g)
	if err != nil {
		t.Fatalf("Compute failed: %v", err)
	}

	inst.Reset()
	forwardLogits, err := inst.ForwardToken(tok)
	if err != nil {
		t.Fatalf("ForwardToken failed: %v", err)
	}

    if !floatsEqual(graphLogits, forwardLogits, 2e-3) {
        firstDiff := -1
        for i := range graphLogits {
            diff := float32(math.Abs(float64(graphLogits[i] - forwardLogits[i])))
            if diff > 2e-3 {
                firstDiff = i
                break
            }
        }
        t.Errorf("logits mismatch at index %d:", firstDiff)
        for i := range graphLogits {
            t.Logf("  [%d] graph=%.10f forward=%.10f diff=%.10e",
                i, graphLogits[i], forwardLogits[i], graphLogits[i]-forwardLogits[i])
        }
    }
}
