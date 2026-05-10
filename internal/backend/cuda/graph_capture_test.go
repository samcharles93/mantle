//go:build cuda

package cuda

import (
	"testing"

	"github.com/samcharles93/mantle/internal/backend/core"
	instance "github.com/samcharles93/mantle/internal/backend/core"
	"github.com/samcharles93/mantle/internal/backend/cuda/native"
	"github.com/samcharles93/mantle/internal/graph"
)

// testModelForCapture wraps *core.Instance to satisfy core.Runtime.
type testModelForCapture struct {
	*core.Instance
}

func (m *testModelForCapture) ForwardToken(id int) ([]float32, error) { return nil, nil }
func (m *testModelForCapture) Reset()                                 {}
func (m *testModelForCapture) UpdateRoPE()                            {}

func TestGraphCaptureStub(t *testing.T) {
	count, err := native.DeviceCount()
	if err != nil {
		t.Skipf("no CUDA device: %v", err)
	}
	if count < 1 {
		t.Skip("no CUDA device available")
	}

	stream, err := native.NewStream()
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	defer func() { _ = stream.Destroy() }()

	blas, err := native.NewBlasHandle(stream)
	if err != nil {
		t.Fatalf("NewBlasHandle: %v", err)
	}
	defer func() { _ = blas.Destroy() }()

	ops := NewOps(stream, blas)
	defer func() { _ = ops.Close() }()

	const (
		embDim    = 8
		ffnDim    = 16
		vocabSize = 16
	)

	ffnUp := core.NewMat(ffnDim, embDim)
	ffnGate := core.NewMat(ffnDim, embDim)
	ffnDown := core.NewMat(embDim, ffnDim)
	for i := 0; i < ffnDim*embDim; i++ {
		ffnUp.Data[i] = float32(i+1) / 100.0
		ffnGate.Data[i] = float32(i+1) / 110.0
	}
	for i := 0; i < embDim*ffnDim; i++ {
		ffnDown.Data[i] = float32(i+1) / 120.0
	}

	embeddings := core.NewMat(vocabSize, embDim)
	for i := 0; i < vocabSize*embDim; i++ {
		embeddings.Data[i] = float32(i+1) / 10.0
	}

	output := core.NewMat(vocabSize, embDim)
	for i := 0; i < vocabSize*embDim; i++ {
		output.Data[i] = float32(i+1) / 100.0
	}

	outputNorm := make([]float32, embDim)
	for i := range outputNorm {
		outputNorm[i] = 1.0
	}

	layers := []instance.Layer{
		{
			FfnUp:       &ffnUp,
			FfnGate:     &ffnGate,
			FfnDown:     &ffnDown,
			FfnNorm:     make([]float32, embDim),
			PostFfnNorm: make([]float32, embDim),
			AttnNorm:    make([]float32, embDim),
			Wq:          &core.Mat{},
			Wk:          &core.Mat{},
			Wv:          &core.Mat{},
			Wo:          &core.Mat{},
		},
	}

	inst := &core.Instance{
		Config: &core.ModelConfig{
			Arch: "test",
			Config: core.Config{
				VocabSize:       vocabSize,
				EmbeddingLength: embDim,
				BlockCount:      1,
			},
		},
		Embeddings:  &embeddings,
		Output:      &output,
		OutputNorm:  outputNorm,
		Layers:      layers,
		RMSEpsilon:  1e-6,
		HeadCount:   1,
		MaxKVStride: embDim,
	}
	inst.SetOps(ops)

	if err := ops.PreloadModelWeights(inst); err != nil {
		t.Fatalf("PreloadModelWeights failed: %v", err)
	}

	rt := &cudaRuntime{
		model:  &testModelForCapture{Instance: inst},
		ops:    ops,
		stream: stream,
		blas:   blas,
	}
	gr := NewGraphRuntime(rt, inst)

	graph.ResetTensorID()
	g := &graph.Graph{UID: 42}

	embedID := graph.NewTensorID()
	g.AddNode(graph.Node{
		Op:     graph.OpEmbed,
		Branch: graph.BranchEmbed,
		Name:   "embed",
		Input:  []graph.TensorID{0},
		Output: embedID,
		Params: graph.EmbedParams{
			VocabSize: vocabSize,
			EmbDim:    embDim,
		},
	})

	ffn1ID := graph.NewTensorID()
	g.AddNode(graph.Node{
		Op:     graph.OpFusedFFN,
		Branch: graph.BranchFFN,
		Name:   "layer0.ffn",
		Input:  []graph.TensorID{embedID},
		Output: ffn1ID,
		Params: graph.FFNParams{
			Activation: "silu",
			HasBias:    false,
		},
	})

	ffn2ID := graph.NewTensorID()
	g.AddNode(graph.Node{
		Op:     graph.OpFusedFFN,
		Branch: graph.BranchFFN,
		Name:   "layer1.ffn",
		Input:  []graph.TensorID{ffn1ID},
		Output: ffn2ID,
		Params: graph.FFNParams{
			Activation: "silu",
			HasBias:    false,
		},
	})

	if err := g.Validate(); err != nil {
		t.Fatalf("graph validation failed: %v", err)
	}

	cg, err := gr.CaptureFFNGraph(g, 1, 2)
	if err != nil {
		t.Fatalf("CaptureFFNGraph failed: %v", err)
	}
	if cg == nil {
		t.Fatal("CaptureFFNGraph returned nil CUDAGraph")
	}
	if !cg.captured {
		t.Error("expected CUDAGraph.captured to be true after capture")
	}

	if gr.GraphCache == nil {
		t.Fatal("GraphRuntime.GraphCache is nil")
	}
	cached, ok := gr.GraphCache.Get(g.UID)
	if !ok {
		t.Fatal("captured graph not found in cache")
	}
	if cached != cg {
		t.Error("cached CUDAGraph differs from returned CUDAGraph")
	}

	if err := cg.Execute(); err != nil {
		t.Logf("Execute returned error (expected for stub mode): %v", err)
	}

	t.Log("graph capture stub test passed")
}

func TestCaptureFFNGraphRejectsNonStatic(t *testing.T) {
	count, err := native.DeviceCount()
	if err != nil {
		t.Skipf("no CUDA device: %v", err)
	}
	if count < 1 {
		t.Skip("no CUDA device available")
	}

	stream, err := native.NewStream()
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	defer func() { _ = stream.Destroy() }()

	blas, err := native.NewBlasHandle(stream)
	if err != nil {
		t.Fatalf("NewBlasHandle: %v", err)
	}
	defer func() { _ = blas.Destroy() }()

	ops := NewOps(stream, blas)
	defer func() { _ = ops.Close() }()

	inst := &core.Instance{
		Config: &core.ModelConfig{
			Arch: "test",
			Config: core.Config{
				VocabSize:       16,
				EmbeddingLength: 8,
				BlockCount:      1,
			},
		},
		Embeddings:  &core.Mat{},
		Output:      &core.Mat{},
		OutputNorm:  make([]float32, 8),
		Layers:      []instance.Layer{{}},
		HeadCount:   1,
		MaxKVStride: 8,
	}
	inst.SetOps(ops)

	rt := &cudaRuntime{
		model:  &testModelForCapture{Instance: inst},
		ops:    ops,
		stream: stream,
		blas:   blas,
	}
	gr := NewGraphRuntime(rt, inst)

	graph.ResetTensorID()
	g := &graph.Graph{UID: 99}

	embedID := graph.NewTensorID()
	g.AddNode(graph.Node{
		Op:     graph.OpEmbed,
		Branch: graph.BranchEmbed,
		Name:   "embed",
		Input:  []graph.TensorID{0},
		Output: embedID,
		Params: graph.EmbedParams{VocabSize: 16, EmbDim: 8},
	})

	attnID := graph.NewTensorID()
	g.AddNode(graph.Node{
		Op:     graph.OpFusedAttention,
		Branch: graph.BranchAttention,
		Name:   "layer0.attention",
		Input:  []graph.TensorID{embedID},
		Output: attnID,
		Params: graph.AttentionParams{LayerIndex: 0},
	})

	ffnID := graph.NewTensorID()
	g.AddNode(graph.Node{
		Op:     graph.OpFusedFFN,
		Branch: graph.BranchFFN,
		Name:   "layer0.ffn",
		Input:  []graph.TensorID{attnID},
		Output: ffnID,
		Params: graph.FFNParams{Activation: "silu"},
	})

	if err := g.Validate(); err != nil {
		t.Fatalf("graph validation failed: %v", err)
	}

	_, err = gr.CaptureFFNGraph(g, 1, 2)
	if err == nil {
		t.Error("expected error capturing non-static subgraph with attention node")
	}
	t.Logf("non-static rejection: %v", err)
}

func TestCaptureFFNGraphRangeValidation(t *testing.T) {
	gr := &GraphRuntime{}

	g := &graph.Graph{UID: 100}
	g.AddNode(graph.Node{
		Op:     graph.OpFusedFFN,
		Branch: graph.BranchFFN,
		Name:   "layer0.ffn",
		Input:  []graph.TensorID{0},
		Output: 1,
		Params: graph.FFNParams{Activation: "silu"},
	})

	_, err := gr.CaptureFFNGraph(g, 2, 1)
	if err == nil {
		t.Error("expected error when startIdx > endIdx")
	}

	_, err = gr.CaptureFFNGraph(g, 0, 10)
	if err == nil {
		t.Error("expected error when endIdx >= len(g.Nodes)")
	}

	_, err = gr.CaptureFFNGraph(nil, 0, 0)
	if err == nil {
		t.Error("expected error for nil graph")
	}

	_, err = gr.CaptureFFNGraph(g, 0, 0)
	if err == nil {
		t.Error("expected error for empty subgraph range")
	}

	t.Log("range validation passed")
}

func TestCUDAGraphCacheLookup(t *testing.T) {
	cache := &CUDAGraphCache{
		graphs: make(map[uint64]*CUDAGraph),
	}

	cg := &CUDAGraph{captured: true}
	cache.Put(42, cg)

	got, ok := cache.Get(42)
	if !ok {
		t.Fatal("cache.Get returned miss for key 42")
	}
	if got != cg {
		t.Error("cache.Get returned wrong CUDAGraph")
	}
	if !got.captured {
		t.Error("expected captured=true for cached graph")
	}

	_, ok = cache.Get(99)
	if ok {
		t.Error("cache.Get should return false for unknown key")
	}

	cache.Delete(42)
	_, ok = cache.Get(42)
	if ok {
		t.Error("cache.Get should return false after Delete")
	}
}
