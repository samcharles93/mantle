package model

import "testing"

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
