package simd

import (
	"bytes"
	"math"
	"os"
	"strings"
	"testing"

	"github.com/samcharles93/mantle/internal/mcfstore"
	"github.com/samcharles93/mantle/pkg/mcf"
)

func TestGemma4TraceBranchesMatchLoadedWeights(t *testing.T) {
	if os.Getenv("MANTLE_GEMMA4_DEBUG") == "" {
		t.Skip("set MANTLE_GEMMA4_DEBUG=1 to run local Gemma-4 parity debug test")
	}

	modelPath := "/work/models/mcf/gemma-4-E4B-it.mcf"
	if env := os.Getenv("MANTLE_GEMMA4_MCF"); env != "" {
		modelPath = env
	}
	if _, err := os.Stat(modelPath); err != nil {
		t.Skipf("model not available: %v", err)
	}

	file, err := mcfstore.Open(modelPath)
	if err != nil {
		t.Fatalf("open mcf: %v", err)
	}
	defer func() { _ = file.Close() }()

	cfgBytes := file.SectionData(mcf.SectionHFConfigJSON)
	model, err := LoadModelMCF(file, cfgBytes, 16, LoadModelOptions{})
	if err != nil {
		t.Fatalf("LoadModelMCF: %v", err)
	}

	trace, err := model.TraceTokens([]int{37889, 29104, 236761})
	if err != nil {
		t.Fatalf("TraceTokens: %v", err)
	}

	layer := &model.Layers[0]
	if layer.Gemma4PLE == nil {
		t.Fatal("layer 0 missing Gemma4PLE")
	}

	useGelu := strings.Contains(model.Config.Config.HiddenAct, "gelu")
	for tokIdx := range trace.TokenIDs {
		attnHidden := addVec(trace.HiddenStates[0][tokIdx], trace.AttentionOutputs[0][tokIdx])
		roundBF16SliceInPlace(attnHidden)
		ffnNormed := exactRMSNorm(attnHidden, layer.FfnNorm, model.RMSEpsilon)
		roundBF16SliceInPlace(ffnNormed)
		gate := exactMatVec(layer.FfnGate, ffnNormed)
		roundBF16SliceInPlace(gate)
		up := exactMatVec(layer.FfnUp, ffnNormed)
		roundBF16SliceInPlace(up)
		act := make([]float32, len(gate))
		for i := range act {
			if useGelu {
				actVal := roundBF16Value(gemma4GeluTanhExact(gate[i]))
				act[i] = roundBF16Value(actVal * up[i])
			} else {
				actVal := roundBF16Value(gemma4SiluExact(gate[i]))
				act[i] = roundBF16Value(actVal * up[i])
			}
		}
		ffnRaw := exactMatVec(layer.FfnDown, act)
		roundBF16SliceInPlace(ffnRaw)
		ffnOut := exactRMSNorm(ffnRaw, layer.PostFfnNorm, model.RMSEpsilon)
		roundBF16SliceInPlace(ffnOut)

		assertCloseVec(t, "ffn", tokIdx, trace.FfnOutputs[0][tokIdx], ffnOut, 5e-3)

		postFfnHidden := addVec(attnHidden, ffnOut)
		roundBF16SliceInPlace(postFfnHidden)
		perLayerInput := trace.PerLayerInputsProjected[tokIdx][0]
		pliGate := exactMatVec(layer.Gemma4PLE.InputGate, postFfnHidden)
		roundBF16SliceInPlace(pliGate)
		pliAct := make([]float32, len(pliGate))
		for i := range pliAct {
			if useGelu {
				actVal := roundBF16Value(gemma4GeluTanhExact(pliGate[i]))
				pliAct[i] = roundBF16Value(actVal * perLayerInput[i])
			} else {
				actVal := roundBF16Value(gemma4SiluExact(pliGate[i]))
				pliAct[i] = roundBF16Value(actVal * perLayerInput[i])
			}
		}
		pliRaw := exactMatVec(layer.Gemma4PLE.Projection, pliAct)
		roundBF16SliceInPlace(pliRaw)
		pliOut := exactRMSNorm(pliRaw, layer.Gemma4PLE.PostNorm, model.RMSEpsilon)
		roundBF16SliceInPlace(pliOut)

		assertCloseVec(t, "per_layer", tokIdx, trace.PerLayerResidualOutputs[0][tokIdx], pliOut, 5e-3)
	}
}

func TestGemma4MCFTensorsMatchRawSafetensors(t *testing.T) {
	if os.Getenv("MANTLE_GEMMA4_DEBUG") == "" {
		t.Skip("set MANTLE_GEMMA4_DEBUG=1 to run local Gemma-4 parity debug test")
	}

	modelPath := "/work/models/mcf/gemma-4-E4B-it.mcf"
	if env := os.Getenv("MANTLE_GEMMA4_MCF"); env != "" {
		modelPath = env
	}
	rawDir := "/work/models/raw/gemma-4-E4B-it"
	if env := os.Getenv("MANTLE_GEMMA4_RAW"); env != "" {
		rawDir = env
	}
	if _, err := os.Stat(modelPath); err != nil {
		t.Skipf("mcf model not available: %v", err)
	}
	if _, err := os.Stat(rawDir); err != nil {
		t.Skipf("raw model not available: %v", err)
	}

	file, err := mcfstore.Open(modelPath)
	if err != nil {
		t.Fatalf("open mcf: %v", err)
	}
	defer func() { _ = file.Close() }()

	rawModel, err := mcf.OpenSafetensorsModel(rawDir)
	if err != nil {
		t.Fatalf("OpenSafetensorsModel: %v", err)
	}
	defer func() { _ = rawModel.Close() }()

	names := []string{
		"model.language_model.layers.0.input_layernorm.weight",
		"model.language_model.layers.0.post_attention_layernorm.weight",
		"model.language_model.layers.0.pre_feedforward_layernorm.weight",
		"model.language_model.layers.0.post_feedforward_layernorm.weight",
		"model.language_model.layers.0.post_per_layer_input_norm.weight",
		"model.language_model.layers.0.mlp.gate_proj.weight",
		"model.language_model.layers.0.mlp.up_proj.weight",
		"model.language_model.layers.0.self_attn.o_proj.weight",
		"model.language_model.layers.0.mlp.down_proj.weight",
		"model.language_model.layers.0.per_layer_input_gate.weight",
		"model.language_model.layers.0.per_layer_projection.weight",
	}
	for _, name := range names {
		gotRaw, _, err := file.ReadTensorRaw(name)
		if err != nil {
			t.Fatalf("ReadTensorRaw(%q): %v", name, err)
		}
		r, ref, err := rawModel.TensorReader(name)
		if err != nil {
			t.Fatalf("TensorReader(%q): %v", name, err)
		}
		wantRaw := make([]byte, ref.Info.Size())
		if _, err := r.Read(wantRaw); err != nil {
			t.Fatalf("read raw tensor %q: %v", name, err)
		}
		if !bytes.Equal(gotRaw, wantRaw) {
			t.Fatalf("tensor %q payload mismatch", name)
		}
	}
}

func exactMatVec(mat *Mat, x []float32) []float32 {
	out := make([]float32, mat.R)
	row := make([]float32, mat.C)
	for i := 0; i < mat.R; i++ {
		mat.RowTo(row, i)
		var sum float32
		for j, v := range row {
			sum += v * x[j]
		}
		out[i] = sum
	}
	return out
}

func exactRMSNorm(src, weight []float32, eps float32) []float32 {
	out := make([]float32, len(src))
	var sum float32
	for _, v := range src {
		sum += v * v
	}
	scale := float32(1.0) / float32(math.Sqrt(float64(sum/float32(len(src))+eps)))
	for i := range src {
		out[i] = src[i] * scale * weight[i]
	}
	return out
}

func addVec(a, b []float32) []float32 {
	out := make([]float32, len(a))
	for i := range a {
		out[i] = a[i] + b[i]
	}
	return out
}

func assertCloseVec(t *testing.T, name string, tokIdx int, got, want []float32, maxDiff float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s token %d len=%d want %d", name, tokIdx, len(got), len(want))
	}
	var worst float32
	worstIdx := -1
	for i := range got {
		diff := abs32(got[i] - want[i])
		if diff > worst {
			worst = diff
			worstIdx = i
		}
	}
	if worst > maxDiff {
		t.Fatalf("%s token %d worst[%d]=%g got=%g want=%g", name, tokIdx, worstIdx, worst, got[worstIdx], want[worstIdx])
	}
}
