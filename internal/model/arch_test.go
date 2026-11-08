package model

import (
	"math"
	"strings"
	"testing"
)

func TestDetectArchMiniCPM5BaseAndDSpark(t *testing.T) {
	base := &HFConfig{ModelType: "llama", Architectures: []string{"LlamaForCausalLM"}}
	spec, err := DetectArch(base)
	if err != nil {
		t.Fatalf("detect base model: %v", err)
	}
	if spec.Name != "llama" || spec.Names.Wq(0) != "model.layers.0.self_attn.q_proj.weight" {
		t.Fatalf("unexpected base mapping: %s, %s", spec.Name, spec.Names.Wq(0))
	}

	draft := &HFConfig{ModelType: "qwen3", Architectures: []string{"Qwen3DSparkModel"}}
	if _, err := DetectArch(draft); err == nil || !strings.Contains(err.Error(), "DSpark") {
		t.Fatalf("expected DSpark runtime error, got %v", err)
	}
}

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

func TestDetectArchLFM3(t *testing.T) {
	cfg := &HFConfig{ModelType: "lfm3", Architectures: []string{"Lfm3ForCausalLM"}}
	spec, err := DetectArch(cfg)
	if err != nil {
		t.Fatalf("DetectArch: %v", err)
	}
	if spec.Name != "lfm3" || !spec.ZeroTailPartialRoPE {
		t.Fatalf("got spec %q zeroTail=%v, want lfm3 with zero-tail rope", spec.Name, spec.ZeroTailPartialRoPE)
	}
	n := spec.Names
	got := []string{n.QNormCandidates(2)[0], n.KNormCandidates(2)[0], n.Wo(2), n.AttnGate(2),
		n.PostAttnNormCandidates(2)[0], n.PostFfnNormCandidates(2)[0], n.OutputNorm}
	want := []string{
		"model.layers.2.self_attn.q_norm.weight", "model.layers.2.self_attn.k_norm.weight",
		"model.layers.2.self_attn.o_proj.weight", "model.layers.2.self_attn.gate_proj.weight",
		"model.layers.2.post_operator_norm.weight", "model.layers.2.post_ffn_norm.weight",
		"model.embedding_norm.weight",
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("name %d: got %q, want %q", i, got[i], want[i])
		}
	}
	if lfm2, _ := DetectArch(&HFConfig{ModelType: "lfm2"}); lfm2 == nil || lfm2.Name != "lfm2" {
		t.Fatal("lfm2 must still detect as lfm2")
	}
}

func TestLayerZeroTailRoPEForConfig(t *testing.T) {
	cfg, err := LoadHFConfigBytes([]byte(`{
		"model_type": "lfm3", "hidden_size": 2048, "num_attention_heads": 16, "head_dim": 128,
		"rope_parameters": {
			"full_attention": {"partial_rotary_factor": 0.5, "rope_theta": 1000000.0, "rope_type": "default"},
			"sliding_attention": {"rope_theta": 10000, "rope_type": "default"}
		}
	}`))
	if err != nil {
		t.Fatalf("LoadHFConfigBytes: %v", err)
	}
	full, scale, err := LayerZeroTailRoPEForConfig(cfg, "full_attention", 128)
	if err != nil || scale != 1 {
		t.Fatalf("full: err=%v scale=%v", err, scale)
	}
	sliding, _, err := LayerZeroTailRoPEForConfig(cfg, "sliding_attention", 128)
	if err != nil {
		t.Fatalf("sliding: %v", err)
	}
	if len(full) != 64 || len(sliding) != 64 {
		t.Fatalf("table lengths %d and %d, want 64", len(full), len(sliding))
	}
	for i := range 64 {
		wantFull := 0.0
		if i < 32 {
			wantFull = 1 / math.Pow(1e6, float64(2*i)/128)
		}
		wantSliding := 1 / math.Pow(1e4, float64(2*i)/128)
		if math.Abs(full[i]-wantFull) > 1e-12 || math.Abs(sliding[i]-wantSliding) > 1e-12 {
			t.Fatalf("entry %d: full %g (want %g), sliding %g (want %g)", i, full[i], wantFull, sliding[i], wantSliding)
		}
	}
}

// TestDetectArchLFM3Omni pins the lfm3_omni detection: it must resolve BEFORE
// the plain lfm3 case ("lfm3_omni" contains "lfm3"), reuse the lfm3 text-decoder
// spec under model.language_model., and parse the audio tower config and token
// ids. The omni config.json omits vision_start/end_token_id, so detection
// supplies the tokenizer-pinned defaults (125009/125010).
func TestDetectArchLFM3Omni(t *testing.T) {
	raw := []byte(`{
		"model_type": "lfm3_omni",
		"architectures": ["Lfm3OmniForConditionalGeneration"],
		"image_token_id": 125030, "video_token_id": 125029,
		"audio_token_id": 125031, "audio_start_token_id": 125012, "audio_out_token_id": 125032,
		"text_config": {"hidden_size": 2048, "num_hidden_layers": 24, "vocab_size": 128000,
			"bos_token_id": 124894, "eos_token_id": 124900, "pad_token_id": 124893},
		"audio_encoder_config": {
			"model_type": "lfm3_omni_encoder", "conv_channels": 256, "hidden_size": 768,
			"num_mel_bins": 128, "sliding_window": 32,
			"text_config": {"model_type": "lfm2", "hidden_size": 768, "intermediate_size": 2560,
				"block_auto_adjust_ff_dim": true, "block_ffn_dim_multiplier": 1.0, "block_multiple_of": 256,
				"num_hidden_layers": 17, "num_attention_heads": 8, "num_key_value_heads": 4,
				"conv_L_cache": 3, "conv_bias": false, "norm_eps": 1e-5,
				"layer_types": ["conv", "conv", "full_attention"],
				"rope_parameters": {"rope_type": "default", "rope_theta": 1000000.0}}
		}
	}`)
	cfg, err := LoadHFConfigBytes(raw)
	if err != nil {
		t.Fatalf("LoadHFConfigBytes: %v", err)
	}
	spec, err := DetectArch(cfg)
	if err != nil {
		t.Fatalf("DetectArch: %v", err)
	}
	if spec.Name != "lfm3_omni" {
		t.Fatalf("spec=%q want lfm3_omni (must not fall through to plain lfm3)", spec.Name)
	}
	n := spec.Names
	if got := n.Embedding; got != "model.language_model.embed_tokens.weight" {
		t.Fatalf("embedding name %q, want model.language_model. prefixed", got)
	}
	if got := n.Wo(2); got != "model.language_model.layers.2.self_attn.o_proj.weight" {
		t.Fatalf("Wo(2)=%q, want model.language_model. prefixed", got)
	}
	if !spec.ZeroTailPartialRoPE {
		t.Fatalf("lfm3_omni must keep the lfm3 zero-tail RoPE semantics")
	}
	if cfg.AudioTokenID != 125031 || cfg.AudioOutTokenID != 125032 || cfg.AudioStartTokenID != 125012 {
		t.Fatalf("audio token ids: %d/%d/%d", cfg.AudioTokenID, cfg.AudioOutTokenID, cfg.AudioStartTokenID)
	}
	if cfg.VisionStartTokenID != 125009 || cfg.VisionEndTokenID != 125010 {
		t.Fatalf("vision marker defaults: %d/%d want 125009/125010", cfg.VisionStartTokenID, cfg.VisionEndTokenID)
	}
	aec := cfg.AudioEncoderConfig
	if aec == nil || aec.HiddenSize != 768 || aec.ConvChannels != 256 || aec.NumMelBins != 128 || aec.SlidingWindow != 32 {
		t.Fatalf("audio_encoder_config not parsed: %#v", aec)
	}
	if aec.TextConfig == nil || aec.TextConfig.NumHiddenLayers != 17 || aec.TextConfig.NumAttentionHeads != 8 ||
		aec.TextConfig.NumKeyValueHeads != 4 || aec.TextConfig.ConvLCache != 3 || aec.TextConfig.ConvBias {
		t.Fatalf("audio lfm text_config not parsed: %#v", aec.TextConfig)
	}
	// text_config merge: the decoder fields come from the nested text_config.
	if cfg.HiddenSize != 2048 || cfg.NumHiddenLayers != 24 || cfg.VocabSize != 128000 {
		t.Fatalf("text_config merge: hidden=%d layers=%d vocab=%d", cfg.HiddenSize, cfg.NumHiddenLayers, cfg.VocabSize)
	}

	// Regression guards: plain lfm3 and lfm3_vl must be unaffected.
	if s, _ := DetectArch(&HFConfig{ModelType: "lfm3", Architectures: []string{"Lfm3ForCausalLM"}}); s == nil || s.Name != "lfm3" {
		t.Fatal("plain lfm3 must still detect as lfm3")
	}
	if s, _ := DetectArch(&HFConfig{ModelType: "lfm3_vl", Architectures: []string{"Lfm3VlForConditionalGeneration"}}); s == nil || s.Name != "lfm3_vl" {
		t.Fatal("lfm3_vl must still detect as lfm3_vl")
	}
}
