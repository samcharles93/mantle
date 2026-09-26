package simd

import (
	"testing"

	instance "github.com/samcharles93/mantle/internal/backend/core"
	"github.com/samcharles93/mantle/internal/model"
)

func TestNeedsOneCenteredRMSNorm(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		cfg  *model.HFConfig
		want bool
	}{
		{
			name: "gemma3",
			cfg: &model.HFConfig{
				ModelType:     "gemma3",
				Architectures: []string{"Gemma3ForConditionalGeneration"},
			},
			want: true,
		},
		{
			name: "gemma4",
			cfg: &model.HFConfig{
				ModelType:     "gemma4",
				Architectures: []string{"Gemma4ForConditionalGeneration"},
			},
			want: false,
		},
		{
			name: "gemma3n",
			cfg: &model.HFConfig{
				ModelType:     "gemma3n",
				Architectures: []string{"Gemma3nForConditionalGeneration"},
			},
			want: false,
		},
		{
			name: "qwen3.5",
			cfg: &model.HFConfig{
				ModelType:     "qwen3_5",
				Architectures: []string{"Qwen3_5ForCausalLM"},
			},
			want: true,
		},
	}

	for _, tc := range tests {
		if got := needsOneCenteredRMSNorm(tc.cfg); got != tc.want {
			t.Fatalf("%s: needsOneCenteredRMSNorm=%v want %v", tc.name, got, tc.want)
		}
	}
}

func TestAdjustGemmaNormsLeavesGemma4Unchanged(t *testing.T) {
	t.Parallel()

	m := &Instance{
		OutputNorm: []float32{0.25, 0.5},
		Layers: []instance.Layer{{
			AttnNorm:     []float32{0.1, 0.2},
			PostAttnNorm: []float32{0.3, 0.4},
			FfnNorm:      []float32{0.5, 0.6},
			PostFfnNorm:  []float32{0.7, 0.8},
			AttnQNorm:    []float32{0.9, 1.0},
			AttnKNorm:    []float32{1.1, 1.2},
		}},
	}

	cfg := &model.HFConfig{
		ModelType:     "gemma4",
		Architectures: []string{"Gemma4ForConditionalGeneration"},
	}
	adjustGemmaNorms(m, cfg)

	if m.OutputNorm[0] != 0.25 || m.OutputNorm[1] != 0.5 {
		t.Fatalf("gemma4 output norm modified: %v", m.OutputNorm)
	}
	if m.Layers[0].AttnNorm[0] != 0.1 || m.Layers[0].AttnKNorm[1] != 1.2 {
		t.Fatalf("gemma4 layer norms modified: %#v", m.Layers[0])
	}
}

// TestAdjustGemmaNormsAddsOneForQwen35 pins the Qwen3.5 zero-centered norm
// behaviour: Qwen3_5RMSNorm applies output*(1+weight), so every non-gated norm
// must be shifted by +1. The gated DeltaNet norm uses a plain init-to-ones
// weight (Qwen3_5RMSNormGated) and must stay untouched.
func TestAdjustGemmaNormsAddsOneForQwen35(t *testing.T) {
	t.Parallel()

	m := &Instance{
		OutputNorm: []float32{0.25, 0.5},
		Layers: []instance.Layer{{
			AttnNorm:     []float32{0.1, 0.2},
			PostAttnNorm: []float32{0.3, 0.4},
			FfnNorm:      []float32{0.5, 0.6},
			PostFfnNorm:  []float32{0.7, 0.8},
			AttnQNorm:    []float32{0.9, 1.0},
			AttnKNorm:    []float32{1.1, 1.2},
			DeltaNet:     &instance.DeltaNetLayer{Norm: []float32{0.2, 0.4}},
		}},
	}

	cfg := &model.HFConfig{
		ModelType:     "qwen3_5",
		Architectures: []string{"Qwen3_5ForConditionalGeneration"},
	}
	adjustGemmaNorms(m, cfg)

	if m.OutputNorm[0] != 1.25 || m.OutputNorm[1] != 1.5 {
		t.Fatalf("qwen3.5 output norm not one-centered: %v", m.OutputNorm)
	}
	l := m.Layers[0]
	if l.AttnNorm[0] != 1.1 || l.FfnNorm[1] != 1.6 || l.AttnQNorm[0] != 1.9 || l.AttnKNorm[1] != 2.2 {
		t.Fatalf("qwen3.5 non-gated norms not one-centered: %#v", l)
	}
	if l.DeltaNet.Norm[0] != 0.2 || l.DeltaNet.Norm[1] != 0.4 {
		t.Fatalf("qwen3.5 gated deltanet norm must stay plain: %v", l.DeltaNet.Norm)
	}
}
