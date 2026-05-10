//go:build goexperiment.simd

package simd

import (
	"testing"

	"github.com/samcharles93/mantle/internal/backend/core"
	"github.com/samcharles93/mantle/internal/graph"
)

func TestFusedFFN(t *testing.T) {
	const (
		embdDim = 4
		ffnDim  = 8
	)

	// Build gate, up, down weight matrices.
	gateData := make([]float32, ffnDim*embdDim)
	upData := make([]float32, ffnDim*embdDim)
	downData := make([]float32, embdDim*ffnDim)

	for i := range gateData {
		gateData[i] = float32(i+1) * 0.01
		upData[i] = float32(i%ffnDim+1) * 0.02
	}
	for i := range downData {
		downData[i] = float32(i+3) * 0.005
	}

	layer := &core.Layer{
		FfnGate: &core.Mat{R: ffnDim, C: embdDim, Stride: embdDim, Data: gateData},
		FfnUp:   &core.Mat{R: ffnDim, C: embdDim, Stride: embdDim, Data: upData},
		FfnDown: &core.Mat{R: embdDim, C: ffnDim, Stride: ffnDim, Data: downData},
	}

	input := []float32{1.0, 0.5, 0.1, 0.0}

	m := &core.Instance{
		Scratch: core.ScratchBuffers{
			FfnGate: make([]float32, ffnDim),
			FfnUp:   make([]float32, ffnDim),
			FfnAct:  make([]float32, ffnDim),
			Tmp2:    make([]float32, embdDim),
		},
	}
	m.BindDefaultOps()

	tests := []struct {
		name       string
		activation string
	}{
		{"silu", "silu"},
		{"gelu", "gelu"},
		{"gelu_erf", "gelu_erf"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			inst := (*Instance)(m)

			// Reset scratch to zeros.
			for i := range m.Scratch.FfnGate {
				m.Scratch.FfnGate[i] = 0
				m.Scratch.FfnUp[i] = 0
				m.Scratch.FfnAct[i] = 0
			}
			for i := range m.Scratch.Tmp2 {
				m.Scratch.Tmp2[i] = 0
			}

			// Compute fused FFN using the new function.
			out := fusedFFN(inst, layer, input, tt.activation)
			if out == nil {
				t.Fatal("fusedFFN returned nil")
			}
			if len(out) != embdDim {
				t.Fatalf("expected %d outputs, got %d", embdDim, len(out))
			}

			// Manually compute expected output.
			// 1. gate projection: gateOut = gate * input
			gateOut := make([]float32, ffnDim)
			for r := range ffnDim {
				var sum float32
				row := gateData[r*embdDim : (r+1)*embdDim]
				for c, v := range row {
					sum += v * input[c]
				}
				gateOut[r] = sum
			}

			// 2. up projection: upOut = up * input
			upOut := make([]float32, ffnDim)
			for r := range ffnDim {
				var sum float32
				row := upData[r*embdDim : (r+1)*embdDim]
				for c, v := range row {
					sum += v * input[c]
				}
				upOut[r] = sum
			}

			// 3. activation(gateOut) * upOut
			actOut := make([]float32, ffnDim)
			switch tt.activation {
			case "silu":
				for i := range actOut {
					actOut[i] = Silu(gateOut[i]) * upOut[i]
				}
			case "gelu", "gelu_erf":
				for i := range actOut {
					actOut[i] = Gelu(gateOut[i]) * upOut[i]
				}
			default:
				t.Fatalf("unknown activation: %s", tt.activation)
			}

			// 4. down projection: downOut = down * actOut
			wantOut := make([]float32, embdDim)
			for r := range embdDim {
				var sum float32
				row := downData[r*ffnDim : (r+1)*ffnDim]
				for c, v := range row {
					sum += v * actOut[c]
				}
				wantOut[r] = sum
			}

			const tol = 1e-5
			for i := range wantOut {
				if diff := out[i] - wantOut[i]; diff > tol || diff < -tol {
					t.Errorf("output[%d]: got %v, want %v (diff=%v)", i, out[i], wantOut[i], diff)
				}
			}
		})
	}
}

// TestFusedFFNGraphCompute verifies fusedFFN is called from Compute()
// when the graph node has OpFusedFFN.
func TestFusedFFNGraphCompute(t *testing.T) {
	const (
		vocabSize = 10
		embdDim   = 4
		ffnDim    = 8
	)

	gateData := make([]float32, ffnDim*embdDim)
	upData := make([]float32, ffnDim*embdDim)
	downData := make([]float32, embdDim*ffnDim)
	for i := range gateData {
		gateData[i] = float32(i+1) * 0.01
		upData[i] = float32(i%ffnDim+1) * 0.02
	}
	for i := range downData {
		downData[i] = float32(i+3) * 0.005
	}

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
		Layers: []core.Layer{
			{
				AttnNorm: make([]float32, embdDim),
				FfnNorm:  make([]float32, embdDim),
				FfnGate:  &core.Mat{R: ffnDim, C: embdDim, Stride: embdDim, Data: gateData},
				FfnUp:    &core.Mat{R: ffnDim, C: embdDim, Stride: embdDim, Data: upData},
				FfnDown:  &core.Mat{R: embdDim, C: ffnDim, Stride: ffnDim, Data: downData},
			},
		},
		Scratch: core.ScratchBuffers{
			X:       make([]float32, embdDim),
			Tmp:     make([]float32, embdDim),
			FfnGate: make([]float32, ffnDim),
			FfnUp:   make([]float32, ffnDim),
			FfnAct:  make([]float32, ffnDim),
			Tmp2:    make([]float32, embdDim),
			Logits:  make([]float32, vocabSize),
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
	for i := range m.Layers[0].AttnNorm {
		m.Layers[0].AttnNorm[i] = 1.0
	}
	for i := range m.Layers[0].FfnNorm {
		m.Layers[0].FfnNorm[i] = 1.0
	}

	graph.ResetTensorID()
	embedID := graph.NewTensorID()
	ffnID := graph.NewTensorID()
	ffnInID := graph.NewTensorID()
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
				Op:     graph.OpFusedNormResidual,
				Branch: graph.BranchFFN,
				Name:   "layer0.norm_residual",
				Input:  []graph.TensorID{embedID, embedID},
				Output: ffnInID,
			},
			{
				Op:     graph.OpFusedFFN,
				Branch: graph.BranchFFN,
				Name:   "layer0.ffn",
				Input:  []graph.TensorID{ffnInID},
				Output: ffnID,
				Params: graph.FFNParams{Activation: "silu"},
			},
			{
				Op:     graph.OpAdd,
				Branch: graph.BranchFFN,
				Name:   "layer0.add",
				Input:  []graph.TensorID{ffnID, embedID},
				Output: outputID,
			},
			{
				Op:     graph.OpOutput,
				Branch: graph.BranchOutput,
				Name:   "output",
				Input:  []graph.TensorID{outputID},
				Output: graph.NewTensorID(),
				Params: graph.OutputParams{Softcap: 0},
			},
		},
	}

	ctx := &graph.ComputeContext{Token: 3, Pos: 0}
	inst := (*Instance)(m)
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
