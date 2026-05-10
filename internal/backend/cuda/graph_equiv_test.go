//go:build cuda

package cuda

import (
	"math"
	"testing"

	instance "github.com/samcharles93/mantle/internal/backend/core"
	"github.com/samcharles93/mantle/internal/backend/cuda/native"
	"github.com/samcharles93/mantle/internal/backend/simd"
	"github.com/samcharles93/mantle/internal/graph"
)

func makeEquivInstance(vocabSize, embdDim int) *instance.Instance {
	maxCtx := 64
	m := &instance.Instance{
		Config: &instance.ModelConfig{
			Config: instance.Config{
				VocabSize:           vocabSize,
				EmbeddingLength:     embdDim,
				HeadCount:           1,
				HeadDim:             embdDim,
				BlockCount:          0,
				FFNLength:           embdDim,
				HiddenAct:           "silu",
				FinalLogitSoftcap:   0,
				LMHeadMultiplier:    0,
				EmbeddingMultiplier: 0,
			},
		},
		Embeddings: &instance.Mat{
			R:      vocabSize,
			C:      embdDim,
			Stride: embdDim,
			DType:  0,
			Data:   make([]float32, vocabSize*embdDim),
		},
		OutputNorm: make([]float32, embdDim),
		Output: &instance.Mat{
			R:      vocabSize,
			C:      embdDim,
			Stride: embdDim,
			DType:  0,
			Data:   make([]float32, vocabSize*embdDim),
		},
		Layers:     nil,
		HeadDim:    embdDim,
		HeadCount:  1,
		MaxContext: maxCtx,
		RMSEpsilon: 1e-5,
		Scratch: instance.ScratchBuffers{
			X:      make([]float32, embdDim),
			Tmp:    make([]float32, embdDim),
			Tmp2:   make([]float32, embdDim),
			Logits: make([]float32, vocabSize),
		},
	}

	for i := range vocabSize * embdDim {
		m.Embeddings.Data[i] = float32(i+1) * 0.01
	}
	for i := range embdDim {
		m.OutputNorm[i] = 1.0
	}
	for i := range vocabSize * embdDim {
		m.Output.Data[i] = float32(i+1) * 0.01
	}

	return m
}

func floatsEqualCUDA(a, b []float32, tol float32) bool {
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

func TestGraphComputeEquivalentCUDA(t *testing.T) {
	count, err := native.DeviceCount()
	if err != nil {
		t.Fatalf("DeviceCount: %v", err)
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
		vocabSize = 10
		embdDim   = 8
		tokenID   = 5
	)

	inst := makeEquivInstance(vocabSize, embdDim)
	inst.SetOps(ops)

	sinst := (*simd.Instance)(inst)
	sinst.SetOps(ops)

	rt := &cudaRuntime{
		model:  sinst,
		ops:    ops,
		stream: stream,
		blas:   blas,
	}
	gr := NewGraphRuntime(rt, inst)

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
				Params: graph.OutputParams{
					Softcap: 0,
				},
			},
		},
	}

	ctx := &graph.ComputeContext{
		Token: tokenID,
		Pos:   0,
	}

	graphLogits, err := gr.Compute(ctx, g)
	if err != nil {
		t.Fatalf("Compute failed: %v", err)
	}

	sinst.Reset()
	forwardLogits, err := rt.ForwardToken(tokenID)
	if err != nil {
		t.Fatalf("ForwardToken failed: %v", err)
	}

	if len(graphLogits) != vocabSize {
		t.Fatalf("graph logits: expected %d, got %d", vocabSize, len(graphLogits))
	}
	if len(forwardLogits) != vocabSize {
		t.Fatalf("forward logits: expected %d, got %d", vocabSize, len(forwardLogits))
	}

	if !floatsEqualCUDA(graphLogits, forwardLogits, 2e-3) {
		firstDiff := -1
		for i := range graphLogits {
			diff := float32(math.Abs(float64(graphLogits[i] - forwardLogits[i])))
			if diff > 2e-3 {
				firstDiff = i
				break
			}
		}
		if firstDiff >= 0 {
			t.Errorf("logits mismatch at index %d (embed→output): graph[%d]=%v forward[%d]=%v diff=%v",
				firstDiff, firstDiff, graphLogits[firstDiff],
				firstDiff, forwardLogits[firstDiff],
				graphLogits[firstDiff]-forwardLogits[firstDiff])
		} else {
			t.Errorf("logits mismatch (embed→output):\n  graph: %v\n  forward: %v",
				graphLogits, forwardLogits)
		}
	}
}
