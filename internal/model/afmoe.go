package model

import "fmt"

// afmoeSpec maps AFMoE (for example Trinity-Nano) tensor names. Only
// architecture-specific naming lives here; loading and runtime behavior remain
// explicit in shared code.
func afmoeSpec() *archSpec {
	return &archSpec{
		Name:          "afmoe",
		HasQKNorm:     true,
		UseLayerTypes: false,
		RopeLocalOnly: true,
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
			attnNormCandidates: func(layer int) []string {
				return []string{
					fmt.Sprintf("model.layers.%d.input_layernorm.weight", layer),
				}
			},
			postAttnNormCandidates: func(layer int) []string {
				return []string{
					fmt.Sprintf("model.layers.%d.post_attention_layernorm.weight", layer),
				}
			},
			ffnNormCandidates: func(layer int) []string {
				return []string{
					fmt.Sprintf("model.layers.%d.pre_mlp_layernorm.weight", layer),
				}
			},
			postFfnNormCandidates: func(layer int) []string {
				return []string{
					fmt.Sprintf("model.layers.%d.post_mlp_layernorm.weight", layer),
				}
			},
			qNormCandidates: func(layer int) []string {
				return []string{
					fmt.Sprintf("model.layers.%d.self_attn.q_norm.weight", layer),
				}
			},
			kNormCandidates: func(layer int) []string {
				return []string{
					fmt.Sprintf("model.layers.%d.self_attn.k_norm.weight", layer),
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
			attnGate: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.self_attn.gate_proj.weight", layer)
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
			moeRouter: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.mlp.router.gate.weight", layer)
			},
			moeExpertBias: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.mlp.expert_bias", layer)
			},
			moeSharedUp: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.mlp.shared_experts.up_proj.weight", layer)
			},
			moeSharedGate: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.mlp.shared_experts.gate_proj.weight", layer)
			},
			moeSharedDown: func(layer int) string {
				return fmt.Sprintf("model.layers.%d.mlp.shared_experts.down_proj.weight", layer)
			},
			moeExpertUp: func(layer int, expert int) string {
				return fmt.Sprintf("model.layers.%d.mlp.experts.%d.up_proj.weight", layer, expert)
			},
			moeExpertGate: func(layer int, expert int) string {
				return fmt.Sprintf("model.layers.%d.mlp.experts.%d.gate_proj.weight", layer, expert)
			},
			moeExpertDown: func(layer int, expert int) string {
				return fmt.Sprintf("model.layers.%d.mlp.experts.%d.down_proj.weight", layer, expert)
			},
		},
	}
}
