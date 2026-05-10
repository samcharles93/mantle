//go:build cuda

package cuda

import (
	"testing"

	"github.com/samcharles93/mantle/internal/backend/core"
	instance "github.com/samcharles93/mantle/internal/backend/core"
	"github.com/samcharles93/mantle/internal/backend/cuda/native"
	"github.com/samcharles93/mantle/internal/graph"
)

// testModel wraps *core.Instance to satisfy core.Runtime.
type testModelForFFN struct {
	*core.Instance
}

func (m *testModelForFFN) ForwardToken(id int) ([]float32, error) { return nil, nil }
func (m *testModelForFFN) Reset()                                 {}
func (m *testModelForFFN) UpdateRoPE()                            {}

func TestFusedFFNCUDA(t *testing.T) {
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
		tokenID   = 5
	)

	// Create synthetic FFN weight matrices.
	// FfnUp:   ffnDim x embDim  (project up)
	// FfnGate: ffnDim x embDim  (gate projection)
	// FfnDown: embDim x ffnDim  (project down)
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

	// Embedding and output matrices.
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
			FfnUp:   &ffnUp,
			FfnGate: &ffnGate,
			FfnDown: &ffnDown,
			// Minimal attention weights (unused in this test but required for PreloadModelWeights)
			Wq: &core.Mat{},
			Wk: &core.Mat{},
			Wv: &core.Mat{},
			Wo: &core.Mat{},
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

	// Preload weights to device.
	if err := ops.PreloadModelWeights(inst); err != nil {
		t.Fatalf("PreloadModelWeights failed: %v", err)
	}

	rt := &cudaRuntime{
		model:  &testModelForFFN{Instance: inst},
		ops:    ops,
		stream: stream,
		blas:   blas,
	}
	gr := NewGraphRuntime(rt, inst)

	graph.ResetTensorID()
	g := &graph.Graph{}

	// Embedding node.
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

	// Fused FFN node.
	ffnID := graph.NewTensorID()
	g.AddNode(graph.Node{
		Op:     graph.OpFusedFFN,
		Branch: graph.BranchFFN,
		Name:   "layer0.ffn",
		Input:  []graph.TensorID{embedID},
		Output: ffnID,
		Params: graph.FFNParams{
			Activation: "silu",
			HasBias:    false,
		},
	})

	// Output node.
	outputID := graph.NewTensorID()
	g.AddNode(graph.Node{
		Op:     graph.OpOutput,
		Branch: graph.BranchOutput,
		Name:   "output",
		Input:  []graph.TensorID{ffnID},
		Output: outputID,
		Params: graph.OutputParams{
			Softcap: 0,
		},
	})

	ctx := &graph.ComputeContext{
		Pos:   0,
		KVLen: 1,
		Token: tokenID,
	}

	logits, err := gr.Compute(ctx, g)
	if err != nil {
		t.Fatalf("Compute failed: %v", err)
	}

	if len(logits) != vocabSize {
		t.Fatalf("expected %d logits, got %d", vocabSize, len(logits))
	}

	// Verify output is non-zero (the fused FFN should produce reasonable values with synthetic weights).
	nonZero := false
	for _, v := range logits {
		if v != 0 {
			nonZero = true
			break
		}
	}
	if !nonZero {
		t.Errorf("all logits are zero; expected non-zero from fused FFN with synthetic weights")
	}

	t.Logf("fused FFN logits: %v", logits)
}
