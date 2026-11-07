package model

import "testing"

const dsparkTestConfig = `{
  "architectures": ["Qwen3DSparkModel"],
  "hidden_size": 2048, "intermediate_size": 6144, "num_hidden_layers": 5,
  "num_attention_heads": 16, "num_key_value_heads": 2, "head_dim": 128,
  "vocab_size": 130560, "draft_vocab_size": 130560, "rms_norm_eps": 1e-6,
  "block_size": 7, "mask_token_id": 75982, "target_layer_ids": [1, 10, 20, 30, 39],
  "num_target_layers": 42, "markov_rank": 256, "markov_head_type": "vanilla",
  "projector_type": "dspark", "enable_confidence_head": true, "hidden_act": "silu",
  "rope_parameters": {"rope_theta": 5000000, "rope_type": "default"}
}`

func TestParseDSparkConfig(t *testing.T) {
	c, err := ParseDSparkConfig([]byte(dsparkTestConfig))
	if err != nil {
		t.Fatalf("ParseDSparkConfig: %v", err)
	}
	if c.BlockSize != 7 || c.MaskTokenID != 75982 || c.MarkovRank != 256 || !c.HasConfidence ||
		c.RopeParameters.RopeTheta != 5e6 || len(c.TargetLayerIDs) != 5 {
		t.Fatalf("unexpected config: %+v", c)
	}

	bad := map[string]string{
		"not dspark":     `{"architectures":["LlamaForCausalLM"]}`,
		"rnn markov":     replaceJSON(dsparkTestConfig, `"vanilla"`, `"rnn"`),
		"reduced vocab":  replaceJSON(dsparkTestConfig, `"draft_vocab_size": 130560`, `"draft_vocab_size": 32000`),
		"unsorted taps":  replaceJSON(dsparkTestConfig, `[1, 10, 20, 30, 39]`, `[1, 20, 10, 30, 39]`),
		"oversize block": replaceJSON(dsparkTestConfig, `"block_size": 7`, `"block_size": 64`),
		"bonus anchor":   replaceJSON(dsparkTestConfig, `"hidden_act": "silu"`, `"hidden_act": "silu", "sample_from_anchor": false`),
	}
	for name, raw := range bad {
		if _, err := ParseDSparkConfig([]byte(raw)); err == nil {
			t.Errorf("%s: expected an error", name)
		}
	}
}

func replaceJSON(s, old, new string) string {
	for i := 0; i+len(old) <= len(s); i++ {
		if s[i:i+len(old)] == old {
			return s[:i] + new + s[i+len(old):]
		}
	}
	panic("replaceJSON: " + old + " not found")
}
