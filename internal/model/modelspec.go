package model

import "fmt"

// Mistral Models
func mistralSpec() *archSpec {
	return &archSpec{
		Name:          "mistral",
		HasQKNorm:     false,
		UseLayerTypes: false,
		Names: archNames{
			embedding:  "model.embed_tokens.weight",
			outputNorm: "model.norm.weight",
			outputCandidates: func() []string {
				return []string{
					"lm_head.weight",
					"model.lm_head.weight",
					"output.weight",
					"model.output.weight",
					"model.embed_tokens.weight",
				}
			},
			attnNorm: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.input_layernorm.weight", layer)
			},
			ffnNorm: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.post_attention_layernorm.weight", layer)
			},
			attnNormCandidates: func(layer int) []string {
				return []string{
					fmt.Sprintf("model.layers.%d.input_layernorm.weight", layer),
				}
			},
			ffnNormCandidates: func(layer int) []string {
				return []string{
					fmt.Sprintf("model.layers.%d.post_attention_layernorm.weight", layer),
				}
			},
			qNormCandidates: func(layer int) []string { return nil },
			kNormCandidates: func(layer int) []string { return nil },
			wq: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.self_attn.q_proj.weight", layer)
			},
			wk: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.self_attn.k_proj.weight", layer)
			},
			wv: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.self_attn.v_proj.weight", layer)
			},
			wo: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.self_attn.o_proj.weight", layer)
			},
			ffnUp: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.mlp.up_proj.weight", layer)
			},
			ffnGate: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.mlp.gate_proj.weight", layer)
			},
			ffnDown: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.mlp.down_proj.weight", layer)
			},
		},
	}
}

// Mistral3
// mistral3Spec describes the text model tensors inside Mistral3 multimodal
// checkpoints. The runtime is text-only, so we intentionally map only the
// language_model.* tensors.
func mistral3Spec() *archSpec {
	return &archSpec{
		Name:          "mistral3",
		HasQKNorm:     false,
		UseLayerTypes: false,
		Names: archNames{
			embedding:  "language_model.model.embed_tokens.weight",
			outputNorm: "language_model.model.norm.weight",
			outputCandidates: func() []string {
				return []string{
					"language_model.lm_head.weight",
					"language_model.model.lm_head.weight",
					"language_model.output.weight",
					"language_model.model.output.weight",
					"language_model.model.embed_tokens.weight",
				}
			},
			attnNorm: func(layer int) string {
				return fmt.Sprintf("language_model.model.layers.%d.input_layernorm.weight", layer)
			},
			ffnNorm: func(layer int) string {
				return fmt.Sprintf("language_model.model.layers.%d.post_attention_layernorm.weight", layer)
			},
			attnNormCandidates: func(layer int) []string {
				return []string{
					fmt.Sprintf("language_model.model.layers.%d.input_layernorm.weight", layer),
				}
			},
			ffnNormCandidates: func(layer int) []string {
				return []string{
					fmt.Sprintf("language_model.model.layers.%d.post_attention_layernorm.weight", layer),
				}
			},
			qNormCandidates: func(layer int) []string { return nil },
			kNormCandidates: func(layer int) []string { return nil },
			wq: func(layer int) string {
				return fmt.Sprintf("language_model.model.layers.%d.self_attn.q_proj.weight", layer)
			},
			wk: func(layer int) string {
				return fmt.Sprintf("language_model.model.layers.%d.self_attn.k_proj.weight", layer)
			},
			wv: func(layer int) string {
				return fmt.Sprintf("language_model.model.layers.%d.self_attn.v_proj.weight", layer)
			},
			wo: func(layer int) string {
				return fmt.Sprintf("language_model.model.layers.%d.self_attn.o_proj.weight", layer)
			},
			ffnUp: func(layer int) string {
				return fmt.Sprintf("language_model.model.layers.%d.mlp.up_proj.weight", layer)
			},
			ffnGate: func(layer int) string {
				return fmt.Sprintf("language_model.model.layers.%d.mlp.gate_proj.weight", layer)
			},
			ffnDown: func(layer int) string {
				return fmt.Sprintf("language_model.model.layers.%d.mlp.down_proj.weight", layer)
			},
		},
	}
}

// Qwen3
func qwen3Spec() *archSpec {
	return &archSpec{
		Name:          "qwen3",
		HasQKNorm:     true,
		UseLayerTypes: false,
		Names: archNames{
			embedding:  "model.embed_tokens.weight",
			outputNorm: "model.norm.weight",
			outputCandidates: func() []string {
				return []string{
					"lm_head.weight",
					"model.lm_head.weight",
					"output.weight",
					"model.output.weight",
					"model.embed_tokens.weight",
				}
			},
			attnNorm: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.input_layernorm.weight", layer)
			},
			ffnNorm: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.post_attention_layernorm.weight", layer)
			},
			qNormCandidates: func(layer int) []string {
				return []string{
					fmt.Sprintf("model.layers.%d.self_attn.q_norm.weight", layer),
					fmt.Sprintf("model.layers.%d.self_attn.q_layernorm.weight", layer),
				}
			},
			kNormCandidates: func(layer int) []string {
				return []string{
					fmt.Sprintf("model.layers.%d.self_attn.k_norm.weight", layer),
					fmt.Sprintf("model.layers.%d.self_attn.k_layernorm.weight", layer),
				}
			},
			wq: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.self_attn.q_proj.weight", layer)
			},
			wk: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.self_attn.k_proj.weight", layer)
			},
			wv: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.self_attn.v_proj.weight", layer)
			},
			wo: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.self_attn.o_proj.weight", layer)
			},
			wqBias: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.self_attn.q_proj.bias", layer)
			},
			wkBias: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.self_attn.k_proj.bias", layer)
			},
			wvBias: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.self_attn.v_proj.bias", layer)
			},
			ffnUp: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.mlp.up_proj.weight", layer)
			},
			ffnGate: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.mlp.gate_proj.weight", layer)
			},
			ffnDown: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.mlp.down_proj.weight", layer)
			},
		},
	}
}

// Llama
func llamaSpec() *archSpec {
	return &archSpec{
		Name:          "llama",
		HasQKNorm:     false,
		UseLayerTypes: false,
		Names: archNames{
			embedding:  "model.embed_tokens.weight",
			outputNorm: "model.norm.weight",
			outputCandidates: func() []string {
				return []string{
					"lm_head.weight",
					"model.lm_head.weight",
					"output.weight",
					"model.output.weight",
					"model.embed_tokens.weight",
				}
			},
			attnNorm: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.input_layernorm.weight", layer)
			},
			ffnNorm: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.post_attention_layernorm.weight", layer)
			},
			qNormCandidates: func(layer int) []string { return nil },
			kNormCandidates: func(layer int) []string { return nil },
			wq: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.self_attn.q_proj.weight", layer)
			},
			wk: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.self_attn.k_proj.weight", layer)
			},
			wv: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.self_attn.v_proj.weight", layer)
			},
			wo: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.self_attn.o_proj.weight", layer)
			},
			ffnUp: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.mlp.up_proj.weight", layer)
			},
			ffnGate: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.mlp.gate_proj.weight", layer)
			},
			ffnDown: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.mlp.down_proj.weight", layer)
			},
		},
	}
}

// LFM2
func lfm2Spec() *archSpec {
	return &archSpec{
		Name:          "lfm2",
		HasQKNorm:     true,
		UseLayerTypes: true,
		Names: archNames{
			embedding:  "model.embed_tokens.weight",
			outputNorm: "model.embedding_norm.weight",
			outputCandidates: func() []string {
				return []string{
					"model.output.weight",
					"lm_head.weight",
					"model.lm_head.weight",
					"output.weight",
					"model.embed_tokens.weight",
				}
			},
			attnNorm: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.operator_norm.weight", layer)
			},
			ffnNorm: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.ffn_norm.weight", layer)
			},
			qNormCandidates: func(layer int) []string {
				return []string{fmt.Sprintf("model.layers.%d.self_attn.q_layernorm.weight", layer)}
			},
			kNormCandidates: func(layer int) []string {
				return []string{fmt.Sprintf("model.layers.%d.self_attn.k_layernorm.weight", layer)}
			},
			wq: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.self_attn.q_proj.weight", layer)
			},
			wk: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.self_attn.k_proj.weight", layer)
			},
			wv: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.self_attn.v_proj.weight", layer)
			},
			wo: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.self_attn.out_proj.weight", layer)
			},
			ffnUp: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.feed_forward.w3.weight", layer)
			},
			ffnGate: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.feed_forward.w1.weight", layer)
			},
			ffnDown: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.feed_forward.w2.weight", layer)
			},
			shortConvKernel: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.conv.conv.weight", layer)
			},
			shortConvInProj: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.conv.in_proj.weight", layer)
			},
			shortConvOutProj: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.conv.out_proj.weight", layer)
			},
		},
	}
}

// IBM Granite
func graniteSpec() *archSpec {
	return &archSpec{
		Name:          "granite",
		HasQKNorm:     false,
		UseLayerTypes: false,
		Names: archNames{
			embedding:  "model.embed_tokens.weight",
			outputNorm: "model.norm.weight",
			outputCandidates: func() []string {
				return []string{
					"lm_head.weight",
					"model.lm_head.weight",
					"output.weight",
					"model.output.weight",
					"model.embed_tokens.weight",
				}
			},
			attnNorm: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.input_layernorm.weight", layer)
			},
			ffnNorm: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.post_attention_layernorm.weight", layer)
			},
			attnNormCandidates: func(layer int) []string {
				return []string{
					fmt.Sprintf("model.layers.%d.input_layernorm.weight", layer),
					fmt.Sprintf("model.layers.%d.attn_norm.weight", layer),
				}
			},
			ffnNormCandidates: func(layer int) []string {
				return []string{
					fmt.Sprintf("model.layers.%d.post_attention_layernorm.weight", layer),
					fmt.Sprintf("model.layers.%d.ffn_norm.weight", layer),
				}
			},
			qNormCandidates: func(layer int) []string { return nil },
			kNormCandidates: func(layer int) []string { return nil },
			wq: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.self_attn.q_proj.weight", layer)
			},
			wk: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.self_attn.k_proj.weight", layer)
			},
			wv: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.self_attn.v_proj.weight", layer)
			},
			wo: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.self_attn.o_proj.weight", layer)
			},
			ffnUp: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.mlp.up_proj.weight", layer)
			},
			ffnGate: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.mlp.gate_proj.weight", layer)
			},
			ffnDown: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.mlp.down_proj.weight", layer)
			},
		},
	}
}

// Google Gemma
func gemmaSpec() *archSpec {
	return &archSpec{
		Name:          "gemma",
		HasQKNorm:     true,
		UseLayerTypes: false,
		Names: archNames{
			embedding:  "model.embed_tokens.weight",
			outputNorm: "model.norm.weight",
			outputCandidates: func() []string {
				return []string{
					"lm_head.weight",
					"model.lm_head.weight",
					"output.weight",
					"model.output.weight",
					"model.embed_tokens.weight",
				}
			},
			attnNorm: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.input_layernorm.weight", layer)
			},
			ffnNorm: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.post_attention_layernorm.weight", layer)
			},
			attnNormCandidates: func(layer int) []string {
				return []string{
					fmt.Sprintf("model.layers.%d.input_layernorm.weight", layer),
				}
			},
			ffnNormCandidates: func(layer int) []string {
				return []string{
					fmt.Sprintf("model.layers.%d.post_attention_layernorm.weight", layer),
				}
			},
			qNormCandidates: func(layer int) []string {
				return []string{
					fmt.Sprintf("model.layers.%d.self_attn.q_norm.weight", layer),
					fmt.Sprintf("model.layers.%d.self_attn.q_layernorm.weight", layer),
				}
			},
			kNormCandidates: func(layer int) []string {
				return []string{
					fmt.Sprintf("model.layers.%d.self_attn.k_norm.weight", layer),
					fmt.Sprintf("model.layers.%d.self_attn.k_layernorm.weight", layer),
				}
			},
			wq: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.self_attn.q_proj.weight", layer)
			},
			wk: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.self_attn.k_proj.weight", layer)
			},
			wv: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.self_attn.v_proj.weight", layer)
			},
			wo: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.self_attn.o_proj.weight", layer)
			},
			ffnUp: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.mlp.up_proj.weight", layer)
			},
			ffnGate: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.mlp.gate_proj.weight", layer)
			},
			ffnDown: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.mlp.down_proj.weight", layer)
			},
		},
	}
}
