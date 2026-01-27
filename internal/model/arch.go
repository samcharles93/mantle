package model

import (
	"encoding/json"
	"fmt"
	"strings"
)

type hfConfig struct {
	ModelType     string   `json:"model_type"`
	Architectures []string `json:"architectures"`

	BlockDim          int          `json:"block_dim"`
	ConvLCache        int          `json:"conv_L_cache"`
	LayerTypes        []string     `json:"layer_types"`
	NormEps           float64      `json:"norm_eps"`
	MaxPosition       int          `json:"max_position_embeddings"`
	NumAttentionHeads int          `json:"num_attention_heads"`
	NumKeyValueHeads  int          `json:"num_key_value_heads"`
	RopeScaling       *ropeScaling `json:"rope_scaling"`
	RopeParameters    *ropeParams  `json:"rope_parameters"`

	// MoE / AFMoE-specific fields.
	MoEIntermediateSize int     `json:"moe_intermediate_size"`
	NumDenseLayers      int     `json:"num_dense_layers"`
	NumSharedExperts    int     `json:"num_shared_experts"`
	RouteScale          float64 `json:"route_scale"`
	SlidingWindow       int     `json:"sliding_window"`
	GlobalAttnEveryN    int     `json:"global_attn_every_n_layers"`
	AttentionBias       bool    `json:"attention_bias"`
	MuPEnabled          bool    `json:"mup_enabled"`

	HiddenSize       int     `json:"hidden_size"`
	IntermediateSize int     `json:"intermediate_size"`
	NumHiddenLayers  int     `json:"num_hidden_layers"`
	HeadDim          int     `json:"head_dim"`
	RMSNormEps       float64 `json:"rms_norm_eps"`
	LayerNormEps     float64 `json:"layer_norm_eps"`
	VocabSize        int     `json:"vocab_size"`
	RopeTheta        float64 `json:"rope_theta"`

	NumLocalExperts   int `json:"num_local_experts"`
	NumExperts        int `json:"num_experts"`
	NExpert           int `json:"n_expert"`
	NumExpertsPerTok  int `json:"num_experts_per_tok"`
	MoENumExperts     int `json:"moe_num_experts"`
	MoENumExpertsUsed int `json:"moe_num_experts_used"`
}

type ropeScaling struct {
	Type                          string  `json:"type"`
	RopeType                      string  `json:"rope_type"`
	Factor                        float64 `json:"factor"`
	OriginalMaxPositionEmbeddings int     `json:"original_max_position_embeddings"`
	LowFreqFactor                 float64 `json:"low_freq_factor"`
	HighFreqFactor                float64 `json:"high_freq_factor"`
}

// ropeParams captures the rope_parameters schema used by newer Mistral/Ministral
// configs. We only consume the fields needed by this runtime.
type ropeParams struct {
	Type                          string  `json:"type"`
	RopeType                      string  `json:"rope_type"`
	Factor                        float64 `json:"factor"`
	OriginalMaxPositionEmbeddings int     `json:"original_max_position_embeddings"`
	RopeTheta                     float64 `json:"rope_theta"`

	LowFreqFactor  float64 `json:"low_freq_factor"`
	HighFreqFactor float64 `json:"high_freq_factor"`

	// Additional rope/yarn fields we currently parse-but-do-not-use directly.
	BetaFast          float64 `json:"beta_fast"`
	BetaSlow          float64 `json:"beta_slow"`
	MScale            float64 `json:"mscale"`
	MScaleAllDim      float64 `json:"mscale_all_dim"`
	Llama4ScalingBeta float64 `json:"llama_4_scaling_beta"`
}

type archNames struct {
	embedding        string
	outputNorm       string
	outputCandidates func() []string

	attnNorm               func(layer int) string
	ffnNorm                func(layer int) string
	attnNormCandidates     func(layer int) []string
	ffnNormCandidates      func(layer int) []string
	postAttnNormCandidates func(layer int) []string
	postFfnNormCandidates  func(layer int) []string

	qNormCandidates func(layer int) []string
	kNormCandidates func(layer int) []string

	wq       func(layer int) string
	wk       func(layer int) string
	wv       func(layer int) string
	wo       func(layer int) string
	attnGate func(layer int) string

	ffnUp   func(layer int) string
	ffnGate func(layer int) string
	ffnDown func(layer int) string

	// MoE tensors (optional).
	moeRouter     func(layer int) string
	moeExpertBias func(layer int) string
	moeSharedUp   func(layer int) string
	moeSharedGate func(layer int) string
	moeSharedDown func(layer int) string
	moeExpertUp   func(layer int, expert int) string
	moeExpertGate func(layer int, expert int) string
	moeExpertDown func(layer int, expert int) string

	shortConvKernel  func(layer int) string
	shortConvInProj  func(layer int) string
	shortConvOutProj func(layer int) string
}

type archSpec struct {
	Name          string
	HasQKNorm     bool
	UseLayerTypes bool
	RopeLocalOnly bool

	Names archNames
}

func loadHFConfigBytes(raw []byte) (*hfConfig, error) {
	var cfg hfConfig
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return nil, err
	}
	if err := mergeTextConfigMissing(&cfg, raw); err != nil {
		return nil, err
	}
	if cfg.HiddenSize == 0 && cfg.BlockDim > 0 {
		cfg.HiddenSize = cfg.BlockDim
	}
	return &cfg, nil
}

// mergeTextConfigMissing fills missing hfConfig fields from a nested text_config
// object when present. This is needed for multimodal configs (for example
// mistral3) that place text model parameters under text_config.
func mergeTextConfigMissing(dst *hfConfig, raw []byte) error {
	if dst == nil || len(raw) == 0 {
		return nil
	}

	var top map[string]json.RawMessage
	if err := json.Unmarshal(raw, &top); err != nil {
		return err
	}
	textRaw, ok := top["text_config"]
	if !ok || len(textRaw) == 0 {
		return nil
	}

	var textCfg hfConfig
	if err := json.Unmarshal(textRaw, &textCfg); err != nil {
		return err
	}

	// Do not override model identity fields; only fill missing numeric/struct data.
	if dst.BlockDim == 0 && textCfg.BlockDim > 0 {
		dst.BlockDim = textCfg.BlockDim
	}
	if dst.ConvLCache == 0 && textCfg.ConvLCache > 0 {
		dst.ConvLCache = textCfg.ConvLCache
	}
	if len(dst.LayerTypes) == 0 && len(textCfg.LayerTypes) > 0 {
		dst.LayerTypes = textCfg.LayerTypes
	}
	if dst.NormEps == 0 && textCfg.NormEps > 0 {
		dst.NormEps = textCfg.NormEps
	}
	if dst.MaxPosition == 0 && textCfg.MaxPosition > 0 {
		dst.MaxPosition = textCfg.MaxPosition
	}
	if dst.NumAttentionHeads == 0 && textCfg.NumAttentionHeads > 0 {
		dst.NumAttentionHeads = textCfg.NumAttentionHeads
	}
	if dst.NumKeyValueHeads == 0 && textCfg.NumKeyValueHeads > 0 {
		dst.NumKeyValueHeads = textCfg.NumKeyValueHeads
	}
	if dst.RopeScaling == nil && textCfg.RopeScaling != nil {
		dst.RopeScaling = textCfg.RopeScaling
	}
	if dst.RopeParameters == nil && textCfg.RopeParameters != nil {
		dst.RopeParameters = textCfg.RopeParameters
	}
	if dst.HiddenSize == 0 && textCfg.HiddenSize > 0 {
		dst.HiddenSize = textCfg.HiddenSize
	}
	if dst.IntermediateSize == 0 && textCfg.IntermediateSize > 0 {
		dst.IntermediateSize = textCfg.IntermediateSize
	}
	if dst.NumHiddenLayers == 0 && textCfg.NumHiddenLayers > 0 {
		dst.NumHiddenLayers = textCfg.NumHiddenLayers
	}
	if dst.HeadDim == 0 && textCfg.HeadDim > 0 {
		dst.HeadDim = textCfg.HeadDim
	}
	if dst.RMSNormEps == 0 && textCfg.RMSNormEps > 0 {
		dst.RMSNormEps = textCfg.RMSNormEps
	}
	if dst.LayerNormEps == 0 && textCfg.LayerNormEps > 0 {
		dst.LayerNormEps = textCfg.LayerNormEps
	}
	if dst.VocabSize == 0 && textCfg.VocabSize > 0 {
		dst.VocabSize = textCfg.VocabSize
	}
	if dst.RopeTheta == 0 && textCfg.RopeTheta > 0 {
		dst.RopeTheta = textCfg.RopeTheta
	}
	if dst.RopeTheta == 0 && textCfg.RopeParameters != nil && textCfg.RopeParameters.RopeTheta > 0 {
		dst.RopeTheta = textCfg.RopeParameters.RopeTheta
	}

	if dst.MoEIntermediateSize == 0 && textCfg.MoEIntermediateSize > 0 {
		dst.MoEIntermediateSize = textCfg.MoEIntermediateSize
	}
	if dst.NumDenseLayers == 0 && textCfg.NumDenseLayers > 0 {
		dst.NumDenseLayers = textCfg.NumDenseLayers
	}
	if dst.NumSharedExperts == 0 && textCfg.NumSharedExperts > 0 {
		dst.NumSharedExperts = textCfg.NumSharedExperts
	}
	if dst.RouteScale == 0 && textCfg.RouteScale > 0 {
		dst.RouteScale = textCfg.RouteScale
	}
	if dst.SlidingWindow == 0 && textCfg.SlidingWindow > 0 {
		dst.SlidingWindow = textCfg.SlidingWindow
	}
	if dst.GlobalAttnEveryN == 0 && textCfg.GlobalAttnEveryN > 0 {
		dst.GlobalAttnEveryN = textCfg.GlobalAttnEveryN
	}
	if !dst.AttentionBias && textCfg.AttentionBias {
		dst.AttentionBias = true
	}
	if !dst.MuPEnabled && textCfg.MuPEnabled {
		dst.MuPEnabled = true
	}

	if dst.NumLocalExperts == 0 && textCfg.NumLocalExperts > 0 {
		dst.NumLocalExperts = textCfg.NumLocalExperts
	}
	if dst.NumExperts == 0 && textCfg.NumExperts > 0 {
		dst.NumExperts = textCfg.NumExperts
	}
	if dst.NExpert == 0 && textCfg.NExpert > 0 {
		dst.NExpert = textCfg.NExpert
	}
	if dst.NumExpertsPerTok == 0 && textCfg.NumExpertsPerTok > 0 {
		dst.NumExpertsPerTok = textCfg.NumExpertsPerTok
	}
	if dst.MoENumExperts == 0 && textCfg.MoENumExperts > 0 {
		dst.MoENumExperts = textCfg.MoENumExperts
	}
	if dst.MoENumExpertsUsed == 0 && textCfg.MoENumExpertsUsed > 0 {
		dst.MoENumExpertsUsed = textCfg.MoENumExpertsUsed
	}

	return nil
}

func detectArch(cfg *hfConfig) (*archSpec, error) {
	if cfg == nil {
		return nil, fmt.Errorf("nil config")
	}

	modelType := strings.ToLower(strings.TrimSpace(cfg.ModelType))
	archs := make([]string, 0, len(cfg.Architectures))
	for _, arch := range cfg.Architectures {
		archs = append(archs, strings.ToLower(arch))
	}

	hasArch := func(substr string) bool {
		if strings.Contains(modelType, substr) {
			return true
		}
		for _, arch := range archs {
			if strings.Contains(arch, substr) {
				return true
			}
		}
		return false
	}

	// Most MoE models are not supported yet. AFMoE is handled explicitly.
	if hasMoE(cfg) && !hasArch("afmoe") {
		return nil, fmt.Errorf("moe models are not supported by this runtime")
	}

	switch {
	case hasArch("lfm"):
		return lfm2Spec(), nil
	case hasArch("qwen3"):
		return qwen3Spec(), nil
	case hasArch("afmoe"):
		return afmoeSpec(), nil
	case hasArch("gemma3"):
		return gemmaSpec(), nil
	case hasArch("granite"):
		return graniteSpec(), nil
	case hasArch("mistral3"):
		return mistral3Spec(), nil
	case hasArch("mistral"):
		return mistralSpec(), nil
	case hasArch("llama"):
		return llamaSpec(), nil
	default:
		return nil, fmt.Errorf("unsupported model_type %q (architectures=%v)", cfg.ModelType, cfg.Architectures)
	}
}

func hasMoE(cfg *hfConfig) bool {
	if cfg == nil {
		return false
	}
	if cfg.NumLocalExperts > 0 || cfg.NumExperts > 0 || cfg.NExpert > 0 {
		return true
	}
	if cfg.NumExpertsPerTok > 0 || cfg.MoENumExperts > 0 || cfg.MoENumExpertsUsed > 0 {
		return true
	}
	for _, lt := range cfg.LayerTypes {
		if strings.Contains(strings.ToLower(lt), "moe") {
			return true
		}
	}
	return false
}
