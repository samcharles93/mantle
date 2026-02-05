package model

import "fmt"

// Mistral Models
func mistralSpec() *ArchSpec {
	return &ArchSpec{
		Name:          "mistral",
		HasQKNorm:     false,
		UseLayerTypes: false,
		Names: ArchNames{
			Embedding:  "model.embed_tokens.weight",
			OutputNorm: "model.norm.weight",
			OutputCandidates: func() []string {
				return []string{
					"lm_head.weight",
					"model.lm_head.weight",
					"output.weight",
					"model.output.weight",
					"model.embed_tokens.weight",
				}
			},
			AttnNorm: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.input_layernorm.weight", layer)
			},
			FfnNorm: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.post_attention_layernorm.weight", layer)
			},
			AttnNormCandidates: func(layer int) []string {
				return []string{
					fmt.Sprintf("model.layers.%d.input_layernorm.weight", layer),
				}
			},
			FfnNormCandidates: func(layer int) []string {
				return []string{
					fmt.Sprintf("model.layers.%d.post_attention_layernorm.weight", layer),
				}
			},
			QNormCandidates: func(layer int) []string { return nil },
			KNormCandidates: func(layer int) []string { return nil },
			Wq: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.self_attn.q_proj.weight", layer)
			},
			Wk: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.self_attn.k_proj.weight", layer)
			},
			Wv: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.self_attn.v_proj.weight", layer)
			},
			Wo: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.self_attn.o_proj.weight", layer)
			},
			FfnUp: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.mlp.up_proj.weight", layer)
			},
			FfnGate: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.mlp.gate_proj.weight", layer)
			},
			FfnDown: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.mlp.down_proj.weight", layer)
			},
		},
	}
}

// Mistral3
// mistral3Spec describes the text model tensors inside Mistral3 multimodal
// checkpoints. The runtime is text-only, so we intentionally map only the
// language_model.* tensors.
func mistral3Spec() *ArchSpec {
	return &ArchSpec{
		Name:          "mistral3",
		HasQKNorm:     false,
		UseLayerTypes: false,
		Names: ArchNames{
			Embedding:  "language_model.model.embed_tokens.weight",
			OutputNorm: "language_model.model.norm.weight",
			OutputCandidates: func() []string {
				return []string{
					"language_model.lm_head.weight",
					"language_model.model.lm_head.weight",
					"language_model.output.weight",
					"language_model.model.output.weight",
					"language_model.model.embed_tokens.weight",
				}
			},
			AttnNorm: func(layer int) string {
				return fmt.Sprintf("language_model.model.layers.%d.input_layernorm.weight", layer)
			},
			FfnNorm: func(layer int) string {
				return fmt.Sprintf("language_model.model.layers.%d.post_attention_layernorm.weight", layer)
			},
			AttnNormCandidates: func(layer int) []string {
				return []string{
					fmt.Sprintf("language_model.model.layers.%d.input_layernorm.weight", layer),
				}
			},
			FfnNormCandidates: func(layer int) []string {
				return []string{
					fmt.Sprintf("language_model.model.layers.%d.post_attention_layernorm.weight", layer),
				}
			},
			QNormCandidates: func(layer int) []string { return nil },
			KNormCandidates: func(layer int) []string { return nil },
			Wq: func(layer int) string {
				return fmt.Sprintf("language_model.model.layers.%d.self_attn.q_proj.weight", layer)
			},
			Wk: func(layer int) string {
				return fmt.Sprintf("language_model.model.layers.%d.self_attn.k_proj.weight", layer)
			},
			Wv: func(layer int) string {
				return fmt.Sprintf("language_model.model.layers.%d.self_attn.v_proj.weight", layer)
			},
			Wo: func(layer int) string {
				return fmt.Sprintf("language_model.model.layers.%d.self_attn.o_proj.weight", layer)
			},
			FfnUp: func(layer int) string {
				return fmt.Sprintf("language_model.model.layers.%d.mlp.up_proj.weight", layer)
			},
			FfnGate: func(layer int) string {
				return fmt.Sprintf("language_model.model.layers.%d.mlp.gate_proj.weight", layer)
			},
			FfnDown: func(layer int) string {
				return fmt.Sprintf("language_model.model.layers.%d.mlp.down_proj.weight", layer)
			},
		},
	}
}

// Qwen3
func qwen3Spec() *ArchSpec {
	return &ArchSpec{
		Name:          "qwen3",
		HasQKNorm:     true,
		UseLayerTypes: false,
		Names: ArchNames{
			Embedding:  "model.embed_tokens.weight",
			OutputNorm: "model.norm.weight",
			OutputCandidates: func() []string {
				return []string{
					"lm_head.weight",
					"model.lm_head.weight",
					"output.weight",
					"model.output.weight",
					"model.embed_tokens.weight",
				}
			},
			AttnNorm: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.input_layernorm.weight", layer)
			},
			FfnNorm: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.post_attention_layernorm.weight", layer)
			},
			QNormCandidates: func(layer int) []string {
				return []string{
					fmt.Sprintf("model.layers.%d.self_attn.q_norm.weight", layer),
					fmt.Sprintf("model.layers.%d.self_attn.q_layernorm.weight", layer),
				}
			},
			KNormCandidates: func(layer int) []string {
				return []string{
					fmt.Sprintf("model.layers.%d.self_attn.k_norm.weight", layer),
					fmt.Sprintf("model.layers.%d.self_attn.k_layernorm.weight", layer),
				}
			},
			Wq: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.self_attn.q_proj.weight", layer)
			},
			Wk: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.self_attn.k_proj.weight", layer)
			},
			Wv: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.self_attn.v_proj.weight", layer)
			},
			Wo: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.self_attn.o_proj.weight", layer)
			},
			WqBias: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.self_attn.q_proj.bias", layer)
			},
			WkBias: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.self_attn.k_proj.bias", layer)
			},
			WvBias: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.self_attn.v_proj.bias", layer)
			},
			FfnUp: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.mlp.up_proj.weight", layer)
			},
			FfnGate: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.mlp.gate_proj.weight", layer)
			},
			FfnDown: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.mlp.down_proj.weight", layer)
			},
		},
	}
}

// Falcon H1
func falconH1Spec() *ArchSpec {
	return &ArchSpec{
		Name:          "falcon_h1",
		HasQKNorm:     false,
		UseLayerTypes: false,
		RopeLocalOnly: false,
		Names: ArchNames{
			Embedding:  "model.embed_tokens.weight",
			OutputNorm: "model.final_layernorm.weight",
			OutputCandidates: func() []string {
				return []string{
					"model.embed_tokens.weight",
				}
			},
			AttnNormCandidates: func(layer int) []string {
				return []string{
					fmt.Sprintf("model.layers.%d.input_layernorm.weight", layer),
				}
			},
			FfnNormCandidates: func(layer int) []string {
				return []string{
					fmt.Sprintf("model.layers.%d.pre_ff_layernorm.weight", layer),
				}
			},
			Wq:      func(layer int) string { return fmt.Sprintf("model.layers.%d.self_attn.q_proj.weight", layer) },
			Wk:      func(layer int) string { return fmt.Sprintf("model.layers.%d.self_attn.k_proj.weight", layer) },
			Wv:      func(layer int) string { return fmt.Sprintf("model.layers.%d.self_attn.v_proj.weight", layer) },
			Wo:      func(layer int) string { return fmt.Sprintf("model.layers.%d.self_attn.o_proj.weight", layer) },
			FfnUp:   func(layer int) string { return fmt.Sprintf("model.layers.%d.feed_forward.up_proj.weight", layer) },
			FfnGate: func(layer int) string { return fmt.Sprintf("model.layers.%d.feed_forward.gate_proj.weight", layer) },
			FfnDown: func(layer int) string { return fmt.Sprintf("model.layers.%d.feed_forward.down_proj.weight", layer) },
			MambaInProj: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.mamba.in_proj.weight", layer)
			},
			MambaOutProj: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.mamba.out_proj.weight", layer)
			},
			MambaConv: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.mamba.conv1d.weight", layer)
			},
			MambaConvBias: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.mamba.conv1d.bias", layer)
			},
			MambaALog: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.mamba.A_log", layer)
			},
			MambaD: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.mamba.D", layer)
			},
			MambaDTBias: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.mamba.dt_bias", layer)
			},
		},
	}
}

// Llama
func llamaSpec() *ArchSpec {
	return &ArchSpec{
		Name:          "llama",
		HasQKNorm:     false,
		UseLayerTypes: false,
		Names: ArchNames{
			Embedding:  "model.embed_tokens.weight",
			OutputNorm: "model.norm.weight",
			OutputCandidates: func() []string {
				return []string{
					"lm_head.weight",
					"model.lm_head.weight",
					"output.weight",
					"model.output.weight",
					"model.embed_tokens.weight",
				}
			},
			AttnNorm: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.input_layernorm.weight", layer)
			},
			FfnNorm: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.post_attention_layernorm.weight", layer)
			},
			QNormCandidates: func(layer int) []string { return nil },
			KNormCandidates: func(layer int) []string { return nil },
			Wq: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.self_attn.q_proj.weight", layer)
			},
			Wk: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.self_attn.k_proj.weight", layer)
			},
			Wv: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.self_attn.v_proj.weight", layer)
			},
			Wo: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.self_attn.o_proj.weight", layer)
			},
			FfnUp: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.mlp.up_proj.weight", layer)
			},
			FfnGate: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.mlp.gate_proj.weight", layer)
			},
			FfnDown: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.mlp.down_proj.weight", layer)
			},
		},
	}
}

// LFM2
func lfm2Spec() *ArchSpec {
	return &ArchSpec{
		Name:          "lfm2",
		HasQKNorm:     true,
		UseLayerTypes: true,
		Names: ArchNames{
			Embedding:  "model.embed_tokens.weight",
			OutputNorm: "model.embedding_norm.weight",
			OutputCandidates: func() []string {
				return []string{
					"model.output.weight",
					"lm_head.weight",
					"model.lm_head.weight",
					"output.weight",
					"model.embed_tokens.weight",
				}
			},
			AttnNorm: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.operator_norm.weight", layer)
			},
			FfnNorm: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.ffn_norm.weight", layer)
			},
			QNormCandidates: func(layer int) []string {
				return []string{fmt.Sprintf("model.layers.%d.self_attn.q_layernorm.weight", layer)}
			},
			KNormCandidates: func(layer int) []string {
				return []string{fmt.Sprintf("model.layers.%d.self_attn.k_layernorm.weight", layer)}
			},
			Wq: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.self_attn.q_proj.weight", layer)
			},
			Wk: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.self_attn.k_proj.weight", layer)
			},
			Wv: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.self_attn.v_proj.weight", layer)
			},
			Wo: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.self_attn.out_proj.weight", layer)
			},
			FfnUp: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.feed_forward.w3.weight", layer)
			},
			FfnGate: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.feed_forward.w1.weight", layer)
			},
			FfnDown: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.feed_forward.w2.weight", layer)
			},
			ShortConvKernel: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.conv.conv.weight", layer)
			},
			ShortConvInProj: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.conv.in_proj.weight", layer)
			},
			ShortConvOutProj: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.conv.out_proj.weight", layer)
			},
		},
	}
}

// IBM Granite
func graniteSpec() *ArchSpec {
	return &ArchSpec{
		Name:          "granite",
		HasQKNorm:     false,
		UseLayerTypes: false,
		Names: ArchNames{
			Embedding:  "model.embed_tokens.weight",
			OutputNorm: "model.norm.weight",
			OutputCandidates: func() []string {
				return []string{
					"lm_head.weight",
					"model.lm_head.weight",
					"output.weight",
					"model.output.weight",
					"model.embed_tokens.weight",
				}
			},
			AttnNorm: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.input_layernorm.weight", layer)
			},
			FfnNorm: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.post_attention_layernorm.weight", layer)
			},
			AttnNormCandidates: func(layer int) []string {
				return []string{
					fmt.Sprintf("model.layers.%d.input_layernorm.weight", layer),
					fmt.Sprintf("model.layers.%d.attn_norm.weight", layer),
				}
			},
			FfnNormCandidates: func(layer int) []string {
				return []string{
					fmt.Sprintf("model.layers.%d.post_attention_layernorm.weight", layer),
					fmt.Sprintf("model.layers.%d.ffn_norm.weight", layer),
				}
			},
			QNormCandidates: func(layer int) []string { return nil },
			KNormCandidates: func(layer int) []string { return nil },
			Wq: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.self_attn.q_proj.weight", layer)
			},
			Wk: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.self_attn.k_proj.weight", layer)
			},
			Wv: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.self_attn.v_proj.weight", layer)
			},
			Wo: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.self_attn.o_proj.weight", layer)
			},
			FfnUp: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.mlp.up_proj.weight", layer)
			},
			FfnGate: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.mlp.gate_proj.weight", layer)
			},
			FfnDown: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.mlp.down_proj.weight", layer)
			},
		},
	}
}

// Google Gemma
func gemmaSpec() *ArchSpec {
	return &ArchSpec{
		Name:          "gemma",
		HasQKNorm:     true,
		UseLayerTypes: false,
		Names: ArchNames{
			Embedding:  "model.embed_tokens.weight",
			OutputNorm: "model.norm.weight",
			OutputCandidates: func() []string {
				return []string{
					"lm_head.weight",
					"model.lm_head.weight",
					"output.weight",
					"model.output.weight",
					"model.embed_tokens.weight",
				}
			},
			AttnNorm: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.input_layernorm.weight", layer)
			},
			FfnNorm: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.post_attention_layernorm.weight", layer)
			},
			AttnNormCandidates: func(layer int) []string {
				return []string{
					fmt.Sprintf("model.layers.%d.input_layernorm.weight", layer),
				}
			},
			FfnNormCandidates: func(layer int) []string {
				return []string{
					fmt.Sprintf("model.layers.%d.post_attention_layernorm.weight", layer),
				}
			},
			QNormCandidates: func(layer int) []string {
				return []string{
					fmt.Sprintf("model.layers.%d.self_attn.q_norm.weight", layer),
					fmt.Sprintf("model.layers.%d.self_attn.q_layernorm.weight", layer),
				}
			},
			KNormCandidates: func(layer int) []string {
				return []string{
					fmt.Sprintf("model.layers.%d.self_attn.k_norm.weight", layer),
					fmt.Sprintf("model.layers.%d.self_attn.k_layernorm.weight", layer),
				}
			},
			Wq: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.self_attn.q_proj.weight", layer)
			},
			Wk: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.self_attn.k_proj.weight", layer)
			},
			Wv: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.self_attn.v_proj.weight", layer)
			},
			Wo: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.self_attn.o_proj.weight", layer)
			},
			FfnUp: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.mlp.up_proj.weight", layer)
			},
			FfnGate: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.mlp.gate_proj.weight", layer)
			},
			FfnDown: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.mlp.down_proj.weight", layer)
			},
		},
	}
}
