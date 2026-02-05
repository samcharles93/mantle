package model

import "fmt"

// afmoeSpec maps AFMoE (for example Trinity-Nano) tensor names. Only
// architecture-specific naming lives here; loading and runtime behavior remain
// explicit in shared code.
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
