package model

import (
	"math"
	"testing"
)

func TestLoadHFConfigBytesMergesTextConfigAttnOutputGate(t *testing.T) {
	raw := []byte(`{
		"model_type": "qwen3_5",
		"architectures": ["Qwen3_5ForConditionalGeneration"],
		"text_config": {
			"hidden_size": 2048,
			"num_attention_heads": 8,
			"attn_output_gate": true
		}
	}`)

	cfg, err := LoadHFConfigBytes(raw)
	if err != nil {
		t.Fatalf("LoadHFConfigBytes: %v", err)
	}
	if !cfg.AttnOutputGate {
		t.Fatalf("expected attn_output_gate to be merged from text_config")
	}
	if cfg.HiddenSize != 2048 {
		t.Fatalf("hidden_size=%d want 2048", cfg.HiddenSize)
	}
	if cfg.NumAttentionHeads != 8 {
		t.Fatalf("num_attention_heads=%d want 8", cfg.NumAttentionHeads)
	}
}

func TestLoadHFConfigBytesMergesGemma4TextConfigAndRopeProfiles(t *testing.T) {
	raw := []byte(`{
		"model_type": "gemma4",
		"architectures": ["Gemma4ForConditionalGeneration"],
		"text_config": {
			"hidden_size": 2560,
			"global_head_dim": 512,
			"hidden_size_per_layer_input": 256,
			"vocab_size_per_layer_input": 262144,
			"num_attention_heads": 8,
			"num_key_value_heads": 2,
			"num_global_key_value_heads": 2,
			"num_hidden_layers": 4,
			"num_kv_shared_layers": 2,
			"attention_k_eq_v": true,
			"enable_moe_block": true,
			"top_k_experts": 8,
			"rope_parameters": {
				"full_attention": {
					"rope_type": "proportional",
					"rope_theta": 1000000.0,
					"partial_rotary_factor": 0.25
				},
				"sliding_attention": {
					"rope_type": "default",
					"rope_theta": 10000.0
				}
			}
		}
	}`)

	cfg, err := LoadHFConfigBytes(raw)
	if err != nil {
		t.Fatalf("LoadHFConfigBytes: %v", err)
	}
	if cfg.GlobalHeadDim != 512 {
		t.Fatalf("global_head_dim=%d want 512", cfg.GlobalHeadDim)
	}
	if cfg.HiddenSizePerLayerInput != 256 {
		t.Fatalf("hidden_size_per_layer_input=%d want 256", cfg.HiddenSizePerLayerInput)
	}
	if cfg.VocabSizePerLayerInput != 262144 {
		t.Fatalf("vocab_size_per_layer_input=%d want 262144", cfg.VocabSizePerLayerInput)
	}
	if cfg.NumKVSharedLayers != 2 || cfg.SharedKVLayers != 2 {
		t.Fatalf("shared kv layers=(%d,%d) want (2,2)", cfg.NumKVSharedLayers, cfg.SharedKVLayers)
	}
	if !cfg.AttentionKEqualV {
		t.Fatalf("expected attention_k_eq_v to merge from text_config")
	}
	if !cfg.EnableMoEBlock || cfg.TopKExperts != 8 {
		t.Fatalf("expected gemma4 moe fields to merge, got enable=%v topk=%d", cfg.EnableMoEBlock, cfg.TopKExperts)
	}
	if len(cfg.RopeParametersByType) != 2 {
		t.Fatalf("rope profiles=%d want 2", len(cfg.RopeParametersByType))
	}
	full := cfg.RopeParametersByType["full_attention"]
	if full == nil || full.RopeType != "proportional" || full.RopeTheta != 1000000.0 || full.PartialRotaryFactor != 0.25 {
		t.Fatalf("unexpected full_attention rope profile: %#v", full)
	}
	slid := cfg.RopeParametersByType["sliding_attention"]
	if slid == nil || slid.RopeType != "default" || slid.RopeTheta != 10000.0 {
		t.Fatalf("unexpected sliding_attention rope profile: %#v", slid)
	}
}

func TestDetectArchAllowsGemma4MoE(t *testing.T) {
	cfg := &HFConfig{
		ModelType:       "gemma4",
		Architectures:   []string{"Gemma4ForConditionalGeneration"},
		EnableMoEBlock:  true,
		NumExperts:      128,
		TopKExperts:     8,
		HiddenSize:      2816,
		NumHiddenLayers: 30,
	}

	spec, err := DetectArch(cfg)
	if err != nil {
		t.Fatalf("DetectArch: %v", err)
	}
	if spec.Name != "gemma4" {
		t.Fatalf("spec=%q want gemma4", spec.Name)
	}
	if got, want := float32(cfg.EmbeddingMultiplier), float32(53); got != want {
		t.Fatalf("embedding_multiplier=%v want %v", got, want)
	}
}

func TestLoadHFConfigBytesSynthesizesGemma3nRopeProfilesAndFields(t *testing.T) {
	raw := []byte(`{
		"model_type": "gemma3n",
		"architectures": ["Gemma3nForConditionalGeneration"],
		"text_config": {
			"model_type": "gemma3n_text",
			"hidden_size": 2048,
			"hidden_size_per_layer_input": 256,
			"vocab_size_per_layer_input": 262144,
			"num_attention_heads": 8,
			"num_key_value_heads": 2,
			"num_hidden_layers": 4,
			"num_kv_shared_layers": 2,
			"head_dim": 256,
			"sliding_window": 512,
			"layer_types": ["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
			"rope_theta": 1000000.0,
			"rope_local_base_freq": 10000.0,
			"activation_sparsity_pattern": [0.95, 0.95, 0.0, 0.0],
			"altup_active_idx": 0,
			"altup_coef_clip": 120.0,
			"altup_correct_scale": true,
			"altup_num_inputs": 4,
			"laurel_rank": 64
		}
	}`)

	cfg, err := LoadHFConfigBytes(raw)
	if err != nil {
		t.Fatalf("LoadHFConfigBytes: %v", err)
	}
	if cfg.HiddenSizePerLayerInput != 256 {
		t.Fatalf("hidden_size_per_layer_input=%d want 256", cfg.HiddenSizePerLayerInput)
	}
	if cfg.VocabSizePerLayerInput != 262144 {
		t.Fatalf("vocab_size_per_layer_input=%d want 262144", cfg.VocabSizePerLayerInput)
	}
	if got := len(cfg.ActivationSparsity); got != 4 {
		t.Fatalf("activation_sparsity len=%d want 4", got)
	}
	if cfg.AltUpCoefClip != 120 {
		t.Fatalf("altup_coef_clip=%v want 120", cfg.AltUpCoefClip)
	}
	if !cfg.AltUpCorrectScale {
		t.Fatalf("expected altup_correct_scale to merge from text_config")
	}
	if cfg.AltUpNumInputs != 4 {
		t.Fatalf("altup_num_inputs=%d want 4", cfg.AltUpNumInputs)
	}
	if cfg.LaurelRank != 64 {
		t.Fatalf("laurel_rank=%d want 64", cfg.LaurelRank)
	}
	if cfg.NumKVSharedLayers != 2 {
		t.Fatalf("num_kv_shared_layers=%d want 2", cfg.NumKVSharedLayers)
	}
	full := cfg.RopeParametersByType["full_attention"]
	if full == nil || full.RopeType != "default" || full.RopeTheta != 1000000 {
		t.Fatalf("unexpected full_attention rope profile: %#v", full)
	}
	slid := cfg.RopeParametersByType["sliding_attention"]
	if slid == nil || slid.RopeType != "default" || slid.RopeTheta != 10000 {
		t.Fatalf("unexpected sliding_attention rope profile: %#v", slid)
	}
}

func TestDetectArchRoundsGemma3nEmbeddingMultiplierToBF16(t *testing.T) {
	cfg := &HFConfig{
		ModelType:     "gemma3n",
		Architectures: []string{"Gemma3nForConditionalGeneration"},
		HiddenSize:    2048,
	}

	spec, err := DetectArch(cfg)
	if err != nil {
		t.Fatalf("DetectArch: %v", err)
	}
	if spec.Name != "gemma3n_text" {
		t.Fatalf("spec=%q want gemma3n_text", spec.Name)
	}
	want := RoundFloat32ToBF16(float32(math.Sqrt(2048)))
	if got := float32(cfg.EmbeddingMultiplier); got != want {
		t.Fatalf("embedding_multiplier=%v want %v", got, want)
	}
}

func TestRoundFloat32ToBF16(t *testing.T) {
	t.Parallel()

	tests := []struct {
		in   float32
		want float32
	}{
		{in: float32(50.596442), want: 50.5},
		{in: float32(1.0 / 50.596442), want: 0.01977539},
		{in: float32(0.70710677), want: 0.70703125},
	}

	for _, tc := range tests {
		if got := RoundFloat32ToBF16(tc.in); got != tc.want {
			t.Fatalf("RoundFloat32ToBF16(%v)=%v want %v", tc.in, got, tc.want)
		}
	}
}
