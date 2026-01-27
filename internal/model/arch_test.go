package model

import "testing"

func TestDetectArch(t *testing.T) {
	tests := []struct {
		name      string
		cfg       hfConfig
		wantArch  string
		wantError bool
	}{
		{
			name:     "lfm2",
			cfg:      hfConfig{ModelType: "lfm2"},
			wantArch: "lfm2",
		},
		{
			name:     "llama",
			cfg:      hfConfig{ModelType: "llama"},
			wantArch: "llama",
		},
		{
			name:     "qwen",
			cfg:      hfConfig{ModelType: "qwen3"},
			wantArch: "qwen3",
		},
		{
			name:     "gemma",
			cfg:      hfConfig{ModelType: "gemma3"},
			wantArch: "gemma",
		},
		{
			name:     "granite",
			cfg:      hfConfig{ModelType: "granite"},
			wantArch: "granite",
		},
		{
			name:     "mistral3",
			cfg:      hfConfig{ModelType: "mistral3"},
			wantArch: "mistral3",
		},
		{
			name: "afmoe-moe-supported",
			cfg: hfConfig{
				ModelType:        "afmoe",
				NumExperts:       8,
				NumExpertsPerTok: 2,
			},
			wantArch: "afmoe",
		},
		{
			name:      "unknown",
			cfg:       hfConfig{ModelType: "mamba"},
			wantError: true,
		},
		{
			name:      "moe-unsupported",
			cfg:       hfConfig{ModelType: "mistral", NumLocalExperts: 4},
			wantError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			spec, err := detectArch(&tt.cfg)
			if tt.wantError {
				if err == nil {
					t.Fatalf("expected error, got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if spec == nil {
				t.Fatalf("expected spec, got nil")
			}
			if spec.Name != tt.wantArch {
				t.Fatalf("arch mismatch: want %q, got %q", tt.wantArch, spec.Name)
			}
		})
	}
}

func TestBuildHeadKV(t *testing.T) {
	cfg := &hfConfig{
		ModelType:         "llama",
		NumAttentionHeads: 8,
		NumKeyValueHeads:  2,
		NumHiddenLayers:   4,
	}
	spec := llamaSpec()
	out := buildHeadKV(cfg, spec, 4)
	if len(out) != 4 {
		t.Fatalf("unexpected length: %d", len(out))
	}
	for i, v := range out {
		if v != 2 {
			t.Fatalf("head kv[%d]=%d, want 2", i, v)
		}
	}
}

func TestLoadHFConfigBytesTextConfig(t *testing.T) {
	raw := []byte(`{
		"model_type": "mistral3",
		"text_config": {
			"model_type": "mistral",
			"hidden_size": 3072,
			"intermediate_size": 8192,
			"num_hidden_layers": 30,
			"num_attention_heads": 24,
			"num_key_value_heads": 8,
			"rms_norm_eps": 1e-5,
			"rope_parameters": {
				"type": "yarn",
				"rope_type": "yarn",
				"factor": 16,
				"original_max_position_embeddings": 8192,
				"rope_theta": 1000000,
				"low_freq_factor": 2,
				"high_freq_factor": 4
			},
			"max_position_embeddings": 32768,
			"vocab_size": 131072
		}
	}`)

	cfg, err := loadHFConfigBytes(raw)
	if err != nil {
		t.Fatalf("loadHFConfigBytes error: %v", err)
	}
	if cfg.ModelType != "mistral3" {
		t.Fatalf("model_type mismatch: got %q", cfg.ModelType)
	}
	if cfg.HiddenSize != 3072 || cfg.NumHiddenLayers != 30 {
		t.Fatalf("text_config merge missing core fields: hidden=%d layers=%d", cfg.HiddenSize, cfg.NumHiddenLayers)
	}
	if cfg.NumAttentionHeads != 24 || cfg.NumKeyValueHeads != 8 {
		t.Fatalf("text_config merge missing head fields: heads=%d kv=%d", cfg.NumAttentionHeads, cfg.NumKeyValueHeads)
	}
	if cfg.RMSNormEps == 0 || cfg.MaxPosition == 0 || cfg.VocabSize == 0 {
		t.Fatalf(
			"text_config merge missing eps/max_position/vocab: eps=%f maxpos=%d vocab=%d",
			cfg.RMSNormEps, cfg.MaxPosition, cfg.VocabSize,
		)
	}
	if cfg.RopeParameters == nil || cfg.RopeTheta != 1_000_000 {
		t.Fatalf("rope_parameters merge missing rope_theta: rope_theta=%f", cfg.RopeTheta)
	}
	rs := ropeScalingForConfig(cfg)
	if rs != nil {
		t.Fatalf("expected unsupported yarn rope scaling to be skipped, got: %+v", rs)
	}

	spec, err := detectArch(cfg)
	if err != nil {
		t.Fatalf("detectArch error: %v", err)
	}
	if spec.Name != "mistral3" {
		t.Fatalf("arch mismatch: want mistral3, got %s", spec.Name)
	}
}
