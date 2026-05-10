package simd

import (
	"testing"

	"github.com/samcharles93/mantle/internal/backend/core"
	"github.com/samcharles93/mantle/internal/graph"
)

func TestGraphComputeBasic(t *testing.T) {
	const vocabSize = 10
	const embdDim = 8

	m := &core.Instance{
		Config: &core.ModelConfig{
			Config: core.Config{
				VocabSize:       vocabSize,
				EmbeddingLength: embdDim,
			},
		},
		Embeddings: &core.Mat{
			R:      vocabSize,
			C:      embdDim,
			Stride: embdDim,
			Data:   make([]float32, vocabSize*embdDim),
		},
		OutputNorm: make([]float32, embdDim),
		Output: &core.Mat{
			R:      vocabSize,
			C:      embdDim,
			Stride: embdDim,
			Data:   make([]float32, vocabSize*embdDim),
		},
		RMSEpsilon: 1e-5,
		MaxContext: 4096,
		Scratch: core.ScratchBuffers{
			X:      make([]float32, embdDim),
			Tmp:    make([]float32, embdDim),
			Logits: make([]float32, vocabSize),
		},
	}

	m.BindDefaultOps()

	for i := range m.Embeddings.Data {
		m.Embeddings.Data[i] = float32(i+1) * 0.01
	}
	for i := range m.OutputNorm {
		m.OutputNorm[i] = 1.0
	}
	for i := range m.Output.Data {
		m.Output.Data[i] = float32(i+1) * 0.01
	}

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
				Params: graph.EmbedParams{
					VocabSize: vocabSize,
					EmbDim:    embdDim,
				},
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
	inst := (*Instance)(m) // core.Instance → simd.Instance (type alias)
	logits, err := inst.Compute(ctx, g)
	if err != nil {
		t.Fatalf("Compute failed: %v", err)
	}
	if logits == nil {
		t.Fatal("expected non-nil logits")
	}
	if len(logits) != vocabSize {
		t.Fatalf("expected %d logits, got %d", vocabSize, len(logits))
	}
}
