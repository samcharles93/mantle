package model

import (
	"fmt"
	"strings"
)

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

func smolLM3Spec() *ArchSpec {
	return &ArchSpec{
		Name:          "smollm3",
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
					"model.embed_tokens.weight", // tied embeddings fallback
				}
			},
			AttnNorm: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.input_layernorm.weight", layer)
			},
			FfnNorm: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.post_attention_layernorm.weight", layer)
			},
			AttnNormCandidates: func(layer int) []string {
				return []string{fmt.Sprintf("model.layers.%d.input_layernorm.weight", layer)}
			},
			FfnNormCandidates: func(layer int) []string {
				return []string{fmt.Sprintf("model.layers.%d.post_attention_layernorm.weight", layer)}
			},
			QNormCandidates: func(layer int) []string { return nil },
			KNormCandidates: func(layer int) []string { return nil },
			Wq:              func(layer int) string { return fmt.Sprintf("model.layers.%d.self_attn.q_proj.weight", layer) },
			Wk:              func(layer int) string { return fmt.Sprintf("model.layers.%d.self_attn.k_proj.weight", layer) },
			Wv:              func(layer int) string { return fmt.Sprintf("model.layers.%d.self_attn.v_proj.weight", layer) },
			Wo:              func(layer int) string { return fmt.Sprintf("model.layers.%d.self_attn.o_proj.weight", layer) },
			FfnUp:           func(layer int) string { return fmt.Sprintf("model.layers.%d.mlp.up_proj.weight", layer) },
			FfnGate:         func(layer int) string { return fmt.Sprintf("model.layers.%d.mlp.gate_proj.weight", layer) },
			FfnDown:         func(layer int) string { return fmt.Sprintf("model.layers.%d.mlp.down_proj.weight", layer) },
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

// Qwen3.5
func qwen35Spec() *ArchSpec {
	return &ArchSpec{
		Name:          "qwen3_5",
		HasQKNorm:     true,
		UseLayerTypes: true,
		Names: ArchNames{
			Embedding:  "model.language_model.embed_tokens.weight",
			OutputNorm: "model.language_model.norm.weight",
			OutputCandidates: func() []string {
				return []string{
					"model.language_model.lm_head.weight",
					"model.language_model.embed_tokens.weight",
				}
			},
			AttnNorm: func(layer int) string {
				return fmt.Sprintf("model.language_model.layers.%d.input_layernorm.weight", layer)
			},
			FfnNorm: func(layer int) string {
				return fmt.Sprintf("model.language_model.layers.%d.post_attention_layernorm.weight", layer)
			},
			QNormCandidates: func(layer int) []string {
				return []string{
					fmt.Sprintf("model.language_model.layers.%d.self_attn.q_norm.weight", layer),
				}
			},
			KNormCandidates: func(layer int) []string {
				return []string{
					fmt.Sprintf("model.language_model.layers.%d.self_attn.k_norm.weight", layer),
				}
			},
			Wq: func(layer int) string {
				return fmt.Sprintf("model.language_model.layers.%d.self_attn.q_proj.weight", layer)
			},
			Wk: func(layer int) string {
				return fmt.Sprintf("model.language_model.layers.%d.self_attn.k_proj.weight", layer)
			},
			Wv: func(layer int) string {
				return fmt.Sprintf("model.language_model.layers.%d.self_attn.v_proj.weight", layer)
			},
			Wo: func(layer int) string {
				return fmt.Sprintf("model.language_model.layers.%d.self_attn.o_proj.weight", layer)
			},
			FfnUp: func(layer int) string {
				return fmt.Sprintf("model.language_model.layers.%d.mlp.up_proj.weight", layer)
			},
			FfnGate: func(layer int) string {
				return fmt.Sprintf("model.language_model.layers.%d.mlp.gate_proj.weight", layer)
			},
			FfnDown: func(layer int) string {
				return fmt.Sprintf("model.language_model.layers.%d.mlp.down_proj.weight", layer)
			},
			DeltaQKVProj: func(layer int) string {
				return fmt.Sprintf("model.language_model.layers.%d.linear_attn.in_proj_qkv.weight", layer)
			},
			DeltaAProj: func(layer int) string {
				return fmt.Sprintf("model.language_model.layers.%d.linear_attn.in_proj_a.weight", layer)
			},
			DeltaBProj: func(layer int) string {
				return fmt.Sprintf("model.language_model.layers.%d.linear_attn.in_proj_b.weight", layer)
			},
			DeltaZProj: func(layer int) string {
				return fmt.Sprintf("model.language_model.layers.%d.linear_attn.in_proj_z.weight", layer)
			},
			DeltaOutProj: func(layer int) string {
				return fmt.Sprintf("model.language_model.layers.%d.linear_attn.out_proj.weight", layer)
			},
			DeltaConv: func(layer int) string {
				return fmt.Sprintf("model.language_model.layers.%d.linear_attn.conv1d.weight", layer)
			},
			DeltaNorm: func(layer int) string {
				return fmt.Sprintf("model.language_model.layers.%d.linear_attn.norm.weight", layer)
			},
			DeltaALog: func(layer int) string {
				return fmt.Sprintf("model.language_model.layers.%d.linear_attn.A_log", layer)
			},
			DeltaDTBias: func(layer int) string {
				return fmt.Sprintf("model.language_model.layers.%d.linear_attn.dt_bias", layer)
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

// LiquidAI LFM3: LFM2's short-conv and attention hybrid with sandwich norms,
// q/k norms, a per-head sigmoid output gate and zero-tail partial RoPE.
func lfm3Spec() *ArchSpec {
	spec := lfm2Spec()
	spec.Name = "lfm3"
	spec.ZeroTailPartialRoPE = true
	n := &spec.Names
	n.QNormCandidates = func(layer int) []string {
		return []string{fmt.Sprintf("model.layers.%d.self_attn.q_norm.weight", layer)}
	}
	n.KNormCandidates = func(layer int) []string {
		return []string{fmt.Sprintf("model.layers.%d.self_attn.k_norm.weight", layer)}
	}
	n.Wo = func(layer int) string {
		return fmt.Sprintf("model.layers.%d.self_attn.o_proj.weight", layer)
	}
	n.AttnGate = func(layer int) string {
		return fmt.Sprintf("model.layers.%d.self_attn.gate_proj.weight", layer)
	}
	n.PostAttnNormCandidates = func(layer int) []string {
		return []string{fmt.Sprintf("model.layers.%d.post_operator_norm.weight", layer)}
	}
	n.PostFfnNormCandidates = func(layer int) []string {
		return []string{fmt.Sprintf("model.layers.%d.post_ffn_norm.weight", layer)}
	}
	return spec
}

// LiquidAI LFM3-VL: the LFM3 text decoder nested under model.language_model,
// plus a vision tower (model.vision_tower.*) and multimodal projector
// (model.multi_modal_projector.*) that the loader attaches separately.
func lfm3VlSpec() *ArchSpec {
	spec := lfm3Spec()
	spec.Name = "lfm3_vl"
	prefixArchNames(&spec.Names, "model.language_model.")
	return spec
}

// prefixArchNames rewrites every tensor name of an arch to live under prefix
// (e.g. "model.language_model.") instead of the top-level "model.".
func prefixArchNames(n *ArchNames, prefix string) {
	rewrite := func(s string) string { return strings.Replace(s, "model.", prefix, 1) }
	str1 := func(f func(int) string) func(int) string {
		if f == nil {
			return nil
		}
		return func(i int) string { return rewrite(f(i)) }
	}
	strs1 := func(f func(int) []string) func(int) []string {
		if f == nil {
			return nil
		}
		return func(i int) []string {
			out := f(i)
			for j := range out {
				out[j] = rewrite(out[j])
			}
			return out
		}
	}
	n.Embedding = rewrite(n.Embedding)
	n.OutputNorm = rewrite(n.OutputNorm)
	if n.OutputCandidates != nil {
		oc := n.OutputCandidates
		n.OutputCandidates = func() []string {
			out := oc()
			for j := range out {
				out[j] = rewrite(out[j])
			}
			return out
		}
	}
	n.AttnNorm = str1(n.AttnNorm)
	n.FfnNorm = str1(n.FfnNorm)
	n.AttnNormCandidates = strs1(n.AttnNormCandidates)
	n.FfnNormCandidates = strs1(n.FfnNormCandidates)
	n.PostAttnNormCandidates = strs1(n.PostAttnNormCandidates)
	n.PostFfnNormCandidates = strs1(n.PostFfnNormCandidates)
	n.QNormCandidates = strs1(n.QNormCandidates)
	n.KNormCandidates = strs1(n.KNormCandidates)
	n.Wq, n.Wk, n.Wv, n.Wo, n.AttnGate = str1(n.Wq), str1(n.Wk), str1(n.Wv), str1(n.Wo), str1(n.AttnGate)
	n.WqBias, n.WkBias, n.WvBias = str1(n.WqBias), str1(n.WkBias), str1(n.WvBias)
	n.FfnUp, n.FfnGate, n.FfnDown, n.FfnGateUp = str1(n.FfnUp), str1(n.FfnGate), str1(n.FfnDown), str1(n.FfnGateUp)
	n.ShortConvKernel, n.ShortConvInProj, n.ShortConvOutProj = str1(n.ShortConvKernel), str1(n.ShortConvInProj), str1(n.ShortConvOutProj)
	n.PerLayerInputGate = str1(n.PerLayerInputGate)
	n.PerLayerInputProj = str1(n.PerLayerInputProj)
	n.PostPerLayerInputNorm = str1(n.PostPerLayerInputNorm)
	n.LayerScalar = str1(n.LayerScalar)
	n.AltUpProjection = str1(n.AltUpProjection)
	n.AltUpUnembedProjection = str1(n.AltUpUnembedProjection)
	n.AltUpCorrectOutputScale = str1(n.AltUpCorrectOutputScale)
	n.AltUpCorrectionCoefs = str1(n.AltUpCorrectionCoefs)
	n.AltUpPredictionCoefs = str1(n.AltUpPredictionCoefs)
	n.AltUpModalityRouter = str1(n.AltUpModalityRouter)
	n.AltUpRouterNorm = str1(n.AltUpRouterNorm)
	n.LaurelLeft, n.LaurelRight, n.LaurelPostNorm = str1(n.LaurelLeft), str1(n.LaurelRight), str1(n.LaurelPostNorm)
	n.MoERouter, n.MoEExpertBias = str1(n.MoERouter), str1(n.MoEExpertBias)
	n.MoESharedUp, n.MoESharedGate, n.MoESharedDown = str1(n.MoESharedUp), str1(n.MoESharedGate), str1(n.MoESharedDown)
	n.DeltaQKVProj, n.DeltaAProj, n.DeltaBProj, n.DeltaZProj, n.DeltaOutProj = str1(n.DeltaQKVProj), str1(n.DeltaAProj), str1(n.DeltaBProj), str1(n.DeltaZProj), str1(n.DeltaOutProj)
	n.DeltaConv, n.DeltaNorm, n.DeltaALog, n.DeltaDTBias = str1(n.DeltaConv), str1(n.DeltaNorm), str1(n.DeltaALog), str1(n.DeltaDTBias)
	n.MambaInProj, n.MambaOutProj, n.MambaConv, n.MambaNorm = str1(n.MambaInProj), str1(n.MambaOutProj), str1(n.MambaConv), str1(n.MambaNorm)
	n.MambaConvBias, n.MambaALog, n.MambaD, n.MambaDTBias = str1(n.MambaConvBias), str1(n.MambaALog), str1(n.MambaD), str1(n.MambaDTBias)
	n.PerLayerEmbedding = rewrite(n.PerLayerEmbedding)
	n.PerLayerModelProjection = rewrite(n.PerLayerModelProjection)
	n.PerLayerProjectionNorm = rewrite(n.PerLayerProjectionNorm)
}

// IBM GraniteMoeHybrid (e.g. granite-4.0-350M): attention layers with a
// shared, tied MLP (shared_mlp, stored once on layer 0) whose input_linear is
// a fused gate+up projection, plus embedding/attention/residual/logit scalars.
func graniteMoeHybridSpec() *ArchSpec {
	// Note: shared_mlp is per-layer (shared between the MoE and dense branch
	// within a layer), not tied across layers.
	return &ArchSpec{
		Name: "granitemoehybrid",
		Names: ArchNames{
			Embedding:  "model.embed_tokens.weight",
			OutputNorm: "model.norm.weight",
			OutputCandidates: func() []string {
				return []string{
					"lm_head.weight",
					"model.lm_head.weight",
					"output.weight",
					"model.output.weight",
					"model.embed_tokens.weight", // tie_word_embeddings
				}
			},
			AttnNorm: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.input_layernorm.weight", layer)
			},
			FfnNorm: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.post_attention_layernorm.weight", layer)
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
			FfnGateUp: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.shared_mlp.input_linear.weight", layer)
			},
			FfnDown: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.shared_mlp.output_linear.weight", layer)
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

// Google Gemma3
func gemma3Spec() *ArchSpec {
	return &ArchSpec{
		Name:          "gemma3_text",
		HasQKNorm:     true,
		UseLayerTypes: false,
		RopeLocalOnly: false,
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
				return fmt.Sprintf("model.layers.%d.pre_feedforward_layernorm.weight", layer)
			},
			AttnNormCandidates: func(layer int) []string {
				return []string{
					fmt.Sprintf("model.layers.%d.input_layernorm.weight", layer),
				}
			},
			FfnNormCandidates: func(layer int) []string {
				return []string{
					fmt.Sprintf("model.layers.%d.pre_feedforward_layernorm.weight", layer),
				}
			},
			PostAttnNormCandidates: func(layer int) []string {
				return []string{
					fmt.Sprintf("model.layers.%d.post_attention_layernorm.weight", layer),
				}
			},
			PostFfnNormCandidates: func(layer int) []string {
				return []string{
					fmt.Sprintf("model.layers.%d.post_feedforward_layernorm.weight", layer),
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

// Google Gemma3 multimodal (Gemma3ForConditionalGeneration).
// Tensors use the language_model.model.* prefix rather than model.*.
func gemma3ConditionalSpec() *ArchSpec {
	const p = "language_model.model."
	return &ArchSpec{
		Name:          "gemma3_text",
		HasQKNorm:     true,
		UseLayerTypes: false,
		RopeLocalOnly: false,
		Names: ArchNames{
			Embedding:  p + "embed_tokens.weight",
			OutputNorm: p + "norm.weight",
			OutputCandidates: func() []string {
				return []string{
					"language_model.lm_head.weight",
					p + "embed_tokens.weight",
				}
			},
			AttnNorm: func(layer int) string {
				return fmt.Sprintf(p+"layers.%d.input_layernorm.weight", layer)
			},
			FfnNorm: func(layer int) string {
				return fmt.Sprintf(p+"layers.%d.pre_feedforward_layernorm.weight", layer)
			},
			AttnNormCandidates: func(layer int) []string {
				return []string{fmt.Sprintf(p+"layers.%d.input_layernorm.weight", layer)}
			},
			FfnNormCandidates: func(layer int) []string {
				return []string{fmt.Sprintf(p+"layers.%d.pre_feedforward_layernorm.weight", layer)}
			},
			PostAttnNormCandidates: func(layer int) []string {
				return []string{fmt.Sprintf(p+"layers.%d.post_attention_layernorm.weight", layer)}
			},
			PostFfnNormCandidates: func(layer int) []string {
				return []string{fmt.Sprintf(p+"layers.%d.post_feedforward_layernorm.weight", layer)}
			},
			QNormCandidates: func(layer int) []string {
				return []string{
					fmt.Sprintf(p+"layers.%d.self_attn.q_norm.weight", layer),
					fmt.Sprintf(p+"layers.%d.self_attn.q_layernorm.weight", layer),
				}
			},
			KNormCandidates: func(layer int) []string {
				return []string{
					fmt.Sprintf(p+"layers.%d.self_attn.k_norm.weight", layer),
					fmt.Sprintf(p+"layers.%d.self_attn.k_layernorm.weight", layer),
				}
			},
			Wq: func(layer int) string {
				return fmt.Sprintf(p+"layers.%d.self_attn.q_proj.weight", layer)
			},
			Wk: func(layer int) string {
				return fmt.Sprintf(p+"layers.%d.self_attn.k_proj.weight", layer)
			},
			Wv: func(layer int) string {
				return fmt.Sprintf(p+"layers.%d.self_attn.v_proj.weight", layer)
			},
			Wo: func(layer int) string {
				return fmt.Sprintf(p+"layers.%d.self_attn.o_proj.weight", layer)
			},
			FfnUp: func(layer int) string {
				return fmt.Sprintf(p+"layers.%d.mlp.up_proj.weight", layer)
			},
			FfnGate: func(layer int) string {
				return fmt.Sprintf(p+"layers.%d.mlp.gate_proj.weight", layer)
			},
			FfnDown: func(layer int) string {
				return fmt.Sprintf(p+"layers.%d.mlp.down_proj.weight", layer)
			},
		},
	}
}

// Google Gemma4 conditional checkpoints use model.language_model.* tensors.
func gemma4Spec() *ArchSpec {
	const p = "model.language_model."
	return &ArchSpec{
		Name:          "gemma4",
		HasQKNorm:     true,
		UseLayerTypes: true,
		RopeLocalOnly: false,
		Names: ArchNames{
			Embedding:  p + "embed_tokens.weight",
			OutputNorm: p + "norm.weight",
			OutputCandidates: func() []string {
				return []string{
					p + "lm_head.weight",
					p + "embed_tokens.weight",
				}
			},
			AttnNorm: func(layer int) string {
				return fmt.Sprintf(p+"layers.%d.input_layernorm.weight", layer)
			},
			FfnNorm: func(layer int) string {
				return fmt.Sprintf(p+"layers.%d.pre_feedforward_layernorm.weight", layer)
			},
			AttnNormCandidates: func(layer int) []string {
				return []string{fmt.Sprintf(p+"layers.%d.input_layernorm.weight", layer)}
			},
			FfnNormCandidates: func(layer int) []string {
				return []string{fmt.Sprintf(p+"layers.%d.pre_feedforward_layernorm.weight", layer)}
			},
			PostAttnNormCandidates: func(layer int) []string {
				return []string{fmt.Sprintf(p+"layers.%d.post_attention_layernorm.weight", layer)}
			},
			PostFfnNormCandidates: func(layer int) []string {
				return []string{fmt.Sprintf(p+"layers.%d.post_feedforward_layernorm.weight", layer)}
			},
			QNormCandidates: func(layer int) []string {
				return []string{
					fmt.Sprintf(p+"layers.%d.self_attn.q_norm.weight", layer),
					fmt.Sprintf(p+"layers.%d.self_attn.q_layernorm.weight", layer),
				}
			},
			KNormCandidates: func(layer int) []string {
				return []string{
					fmt.Sprintf(p+"layers.%d.self_attn.k_norm.weight", layer),
					fmt.Sprintf(p+"layers.%d.self_attn.k_layernorm.weight", layer),
				}
			},
			Wq: func(layer int) string {
				return fmt.Sprintf(p+"layers.%d.self_attn.q_proj.weight", layer)
			},
			Wk: func(layer int) string {
				return fmt.Sprintf(p+"layers.%d.self_attn.k_proj.weight", layer)
			},
			Wv: func(layer int) string {
				return fmt.Sprintf(p+"layers.%d.self_attn.v_proj.weight", layer)
			},
			Wo: func(layer int) string {
				return fmt.Sprintf(p+"layers.%d.self_attn.o_proj.weight", layer)
			},
			FfnUp: func(layer int) string {
				return fmt.Sprintf(p+"layers.%d.mlp.up_proj.weight", layer)
			},
			FfnGate: func(layer int) string {
				return fmt.Sprintf(p+"layers.%d.mlp.gate_proj.weight", layer)
			},
			FfnDown: func(layer int) string {
				return fmt.Sprintf(p+"layers.%d.mlp.down_proj.weight", layer)
			},
			PerLayerEmbedding:       p + "embed_tokens_per_layer.weight",
			PerLayerModelProjection: p + "per_layer_model_projection.weight",
			PerLayerProjectionNorm:  p + "per_layer_projection_norm.weight",
			PerLayerInputGate: func(layer int) string {
				return fmt.Sprintf(p+"layers.%d.per_layer_input_gate.weight", layer)
			},
			PerLayerInputProj: func(layer int) string {
				return fmt.Sprintf(p+"layers.%d.per_layer_projection.weight", layer)
			},
			PostPerLayerInputNorm: func(layer int) string {
				return fmt.Sprintf(p+"layers.%d.post_per_layer_input_norm.weight", layer)
			},
			LayerScalar: func(layer int) string {
				return fmt.Sprintf(p+"layers.%d.layer_scalar", layer)
			},
			Gemma4RouterProj: func(layer int) string {
				return fmt.Sprintf(p+"layers.%d.router.proj.weight", layer)
			},
			Gemma4RouterScale: func(layer int) string {
				return fmt.Sprintf(p+"layers.%d.router.scale", layer)
			},
			Gemma4RouterPerExpertScale: func(layer int) string {
				return fmt.Sprintf(p+"layers.%d.router.per_expert_scale", layer)
			},
			Gemma4ExpertsGateUp: func(layer int) string {
				return fmt.Sprintf(p+"layers.%d.experts.gate_up_proj", layer)
			},
			Gemma4ExpertsDown: func(layer int) string {
				return fmt.Sprintf(p+"layers.%d.experts.down_proj", layer)
			},
			Gemma4PreFfnNorm2: func(layer int) string {
				return fmt.Sprintf(p+"layers.%d.pre_feedforward_layernorm_2.weight", layer)
			},
			Gemma4PostFfnNorm1: func(layer int) string {
				return fmt.Sprintf(p+"layers.%d.post_feedforward_layernorm_1.weight", layer)
			},
			Gemma4PostFfnNorm2: func(layer int) string {
				return fmt.Sprintf(p+"layers.%d.post_feedforward_layernorm_2.weight", layer)
			},
		},
	}
}

// Gemma 3n Models (multimodal - uses model.language_model.* prefix)
func gemma3nSpec() *ArchSpec {
	return &ArchSpec{
		Name:          "gemma3n_text",
		HasQKNorm:     true,
		UseLayerTypes: true,
		RopeLocalOnly: false,
		Names: ArchNames{
			Embedding:  "model.language_model.embed_tokens.weight",
			OutputNorm: "model.language_model.norm.weight",
			OutputCandidates: func() []string {
				return []string{
					"model.language_model.lm_head.weight",
					"lm_head.weight",
					"model.language_model.embed_tokens.weight",
				}
			},
			AttnNorm: func(layer int) string {
				return fmt.Sprintf("model.language_model.layers.%d.input_layernorm.weight", layer)
			},
			FfnNorm: func(layer int) string {
				return fmt.Sprintf("model.language_model.layers.%d.pre_feedforward_layernorm.weight", layer)
			},
			AttnNormCandidates: func(layer int) []string {
				return []string{
					fmt.Sprintf("model.language_model.layers.%d.input_layernorm.weight", layer),
				}
			},
			FfnNormCandidates: func(layer int) []string {
				return []string{
					fmt.Sprintf("model.language_model.layers.%d.pre_feedforward_layernorm.weight", layer),
				}
			},
			PostAttnNormCandidates: func(layer int) []string {
				return []string{
					fmt.Sprintf("model.language_model.layers.%d.post_attention_layernorm.weight", layer),
				}
			},
			PostFfnNormCandidates: func(layer int) []string {
				return []string{
					fmt.Sprintf("model.language_model.layers.%d.post_feedforward_layernorm.weight", layer),
				}
			},
			QNormCandidates: func(layer int) []string {
				return []string{
					fmt.Sprintf("model.language_model.layers.%d.self_attn.q_norm.weight", layer),
				}
			},
			KNormCandidates: func(layer int) []string {
				return []string{
					fmt.Sprintf("model.language_model.layers.%d.self_attn.k_norm.weight", layer),
				}
			},
			Wq: func(layer int) string {
				return fmt.Sprintf("model.language_model.layers.%d.self_attn.q_proj.weight", layer)
			},
			Wk: func(layer int) string {
				return fmt.Sprintf("model.language_model.layers.%d.self_attn.k_proj.weight", layer)
			},
			Wv: func(layer int) string {
				return fmt.Sprintf("model.language_model.layers.%d.self_attn.v_proj.weight", layer)
			},
			Wo: func(layer int) string {
				return fmt.Sprintf("model.language_model.layers.%d.self_attn.o_proj.weight", layer)
			},
			FfnUp: func(layer int) string {
				return fmt.Sprintf("model.language_model.layers.%d.mlp.up_proj.weight", layer)
			},
			FfnGate: func(layer int) string {
				return fmt.Sprintf("model.language_model.layers.%d.mlp.gate_proj.weight", layer)
			},
			FfnDown: func(layer int) string {
				return fmt.Sprintf("model.language_model.layers.%d.mlp.down_proj.weight", layer)
			},
			PerLayerEmbedding:       "model.language_model.embed_tokens_per_layer.weight",
			PerLayerModelProjection: "model.language_model.per_layer_model_projection.weight",
			PerLayerProjectionNorm:  "model.language_model.per_layer_projection_norm.weight",
			PerLayerInputGate: func(layer int) string {
				return fmt.Sprintf("model.language_model.layers.%d.per_layer_input_gate.weight", layer)
			},
			PerLayerInputProj: func(layer int) string {
				return fmt.Sprintf("model.language_model.layers.%d.per_layer_projection.weight", layer)
			},
			PostPerLayerInputNorm: func(layer int) string {
				return fmt.Sprintf("model.language_model.layers.%d.post_per_layer_input_norm.weight", layer)
			},
			AltUpProjection: func(index int) string {
				return fmt.Sprintf("model.language_model.altup_projections.%d.weight", index)
			},
			AltUpUnembedProjection: func(index int) string {
				return fmt.Sprintf("model.language_model.altup_unembed_projections.%d.weight", index)
			},
			AltUpCorrectOutputScale: func(layer int) string {
				return fmt.Sprintf("model.language_model.layers.%d.altup.correct_output_scale", layer)
			},
			AltUpCorrectionCoefs: func(layer int) string {
				return fmt.Sprintf("model.language_model.layers.%d.altup.correction_coefs.weight", layer)
			},
			AltUpPredictionCoefs: func(layer int) string {
				return fmt.Sprintf("model.language_model.layers.%d.altup.prediction_coefs.weight", layer)
			},
			AltUpModalityRouter: func(layer int) string {
				return fmt.Sprintf("model.language_model.layers.%d.altup.modality_router.weight", layer)
			},
			AltUpRouterNorm: func(layer int) string {
				return fmt.Sprintf("model.language_model.layers.%d.altup.router_norm.weight", layer)
			},
			LaurelLeft: func(layer int) string {
				return fmt.Sprintf("model.language_model.layers.%d.laurel.linear_left.weight", layer)
			},
			LaurelRight: func(layer int) string {
				return fmt.Sprintf("model.language_model.layers.%d.laurel.linear_right.weight", layer)
			},
			LaurelPostNorm: func(layer int) string {
				return fmt.Sprintf("model.language_model.layers.%d.laurel.post_laurel_norm.weight", layer)
			},
		},
	}
}

// afmoeSpec
func afmoeSpec() *ArchSpec {
	return &ArchSpec{
		Name:          "afmoe",
		HasQKNorm:     true,
		UseLayerTypes: false,
		RopeLocalOnly: true,
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
			AttnNormCandidates: func(layer int) []string {
				return []string{
					fmt.Sprintf("model.layers.%d.input_layernorm.weight", layer),
				}
			},
			PostAttnNormCandidates: func(layer int) []string {
				return []string{
					fmt.Sprintf("model.layers.%d.post_attention_layernorm.weight", layer),
				}
			},
			FfnNormCandidates: func(layer int) []string {
				return []string{
					fmt.Sprintf("model.layers.%d.pre_mlp_layernorm.weight", layer),
				}
			},
			PostFfnNormCandidates: func(layer int) []string {
				return []string{
					fmt.Sprintf("model.layers.%d.post_mlp_layernorm.weight", layer),
				}
			},
			QNormCandidates: func(layer int) []string {
				return []string{
					fmt.Sprintf("model.layers.%d.self_attn.q_norm.weight", layer),
				}
			},
			KNormCandidates: func(layer int) []string {
				return []string{
					fmt.Sprintf("model.layers.%d.self_attn.k_norm.weight", layer),
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
			AttnGate: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.self_attn.gate_proj.weight", layer)
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
			MoERouter: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.mlp.router.gate.weight", layer)
			},
			MoEExpertBias: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.mlp.expert_bias", layer)
			},
			MoESharedUp: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.mlp.shared_experts.up_proj.weight", layer)
			},
			MoESharedGate: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.mlp.shared_experts.gate_proj.weight", layer)
			},
			MoESharedDown: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.mlp.shared_experts.down_proj.weight", layer)
			},
			MoEExpertUp: func(layer int, expert int) string {
				return fmt.Sprintf("model.layers.%d.mlp.experts.%d.up_proj.weight", layer, expert)
			},
			MoEExpertGate: func(layer int, expert int) string {
				return fmt.Sprintf("model.layers.%d.mlp.experts.%d.gate_proj.weight", layer, expert)
			},
			MoEExpertDown: func(layer int, expert int) string {
				return fmt.Sprintf("model.layers.%d.mlp.experts.%d.down_proj.weight", layer, expert)
			},
		},
	}
}
