package model

import (
	"encoding/json"
	"fmt"
	"slices"
	"strings"
)

// DSparkMaxBlockSize bounds a draft block. Draft blocks and their verification
// batches run through the small-batch kernels, which take at most 16 rows.
const DSparkMaxBlockSize = 16

// DSparkConfig is the shape of a DSpark draft checkpoint, read from its
// config.json. These are training-time properties of the weights; whether to
// speculate and how far is decided by the caller, not by this config.
type DSparkConfig struct {
	Architectures    []string `json:"architectures"`
	HiddenSize       int      `json:"hidden_size"`
	IntermediateSize int      `json:"intermediate_size"`
	NumLayers        int      `json:"num_hidden_layers"`
	NumHeads         int      `json:"num_attention_heads"`
	NumKVHeads       int      `json:"num_key_value_heads"`
	HeadDim          int      `json:"head_dim"`
	VocabSize        int      `json:"vocab_size"`
	DraftVocabSize   int      `json:"draft_vocab_size"`
	RMSNormEps       float64  `json:"rms_norm_eps"`
	BlockSize        int      `json:"block_size"`
	MaskTokenID      int      `json:"mask_token_id"`
	TargetLayerIDs   []int    `json:"target_layer_ids"`
	NumTargetLayers  int      `json:"num_target_layers"`
	MarkovRank       int      `json:"markov_rank"`
	MarkovHeadType   string   `json:"markov_head_type"`
	ProjectorType    string   `json:"projector_type"`
	HasConfidence    bool     `json:"enable_confidence_head"`
	HiddenAct        string   `json:"hidden_act"`
	RopeParameters   struct {
		RopeTheta float64 `json:"rope_theta"`
		RopeType  string  `json:"rope_type"`
	} `json:"rope_parameters"`
	SampleFromAnchor *bool `json:"sample_from_anchor"`
}

// ParseDSparkConfig decodes and validates a DSpark draft config.json. Only the
// variant Mantle implements is accepted: a vanilla Markov head, the dspark
// projector, full draft vocabulary, SiLU and default RoPE, drafting from the
// anchor slot.
func ParseDSparkConfig(raw []byte) (*DSparkConfig, error) {
	var c DSparkConfig
	if err := json.Unmarshal(raw, &c); err != nil {
		return nil, fmt.Errorf("dspark config: %w", err)
	}
	if !slices.Contains(c.Architectures, "Qwen3DSparkModel") {
		return nil, fmt.Errorf("dspark config: architectures %v is not Qwen3DSparkModel", c.Architectures)
	}
	switch {
	case c.HiddenSize <= 0, c.IntermediateSize <= 0, c.NumLayers <= 0, c.NumHeads <= 0,
		c.NumKVHeads <= 0, c.HeadDim <= 0, c.VocabSize <= 0, c.MarkovRank <= 0:
		return nil, fmt.Errorf("dspark config: non-positive model dimension")
	case c.NumHeads%c.NumKVHeads != 0:
		return nil, fmt.Errorf("dspark config: %d heads not divisible by %d kv heads", c.NumHeads, c.NumKVHeads)
	case c.HeadDim%2 != 0:
		return nil, fmt.Errorf("dspark config: odd head_dim %d", c.HeadDim)
	case c.BlockSize <= 0 || c.BlockSize > DSparkMaxBlockSize:
		return nil, fmt.Errorf("dspark config: block_size %d outside [1,%d]", c.BlockSize, DSparkMaxBlockSize)
	case c.MaskTokenID < 0 || c.MaskTokenID >= c.VocabSize:
		return nil, fmt.Errorf("dspark config: mask_token_id %d outside vocab", c.MaskTokenID)
	case len(c.TargetLayerIDs) == 0:
		return nil, fmt.Errorf("dspark config: no target_layer_ids")
	case c.DraftVocabSize != 0 && c.DraftVocabSize != c.VocabSize:
		return nil, fmt.Errorf("dspark config: reduced draft vocabulary is not supported")
	case c.MarkovHeadType != "" && c.MarkovHeadType != "vanilla":
		return nil, fmt.Errorf("dspark config: markov_head_type %q is not supported", c.MarkovHeadType)
	case c.ProjectorType != "" && c.ProjectorType != "dspark":
		return nil, fmt.Errorf("dspark config: projector_type %q is not supported", c.ProjectorType)
	case c.HiddenAct != "" && c.HiddenAct != "silu":
		return nil, fmt.Errorf("dspark config: hidden_act %q is not supported", c.HiddenAct)
	case c.RopeParameters.RopeType != "" && c.RopeParameters.RopeType != "default":
		return nil, fmt.Errorf("dspark config: rope_type %q is not supported", c.RopeParameters.RopeType)
	case c.SampleFromAnchor != nil && !*c.SampleFromAnchor:
		return nil, fmt.Errorf("dspark config: bonus-anchor drafting is not supported")
	}
	for i := 1; i < len(c.TargetLayerIDs); i++ {
		if c.TargetLayerIDs[i] <= c.TargetLayerIDs[i-1] {
			return nil, fmt.Errorf("dspark config: target_layer_ids %v must be strictly increasing", c.TargetLayerIDs)
		}
	}
	if c.RMSNormEps <= 0 {
		c.RMSNormEps = 1e-6
	}
	if c.RopeParameters.RopeTheta <= 0 {
		c.RopeParameters.RopeTheta = 10000
	}
	return &c, nil
}

// DSparkDraftSpec is the tensor naming of a Qwen3 DSpark draft. Its decoder
// layers are standard Qwen3 layers stored without the "model." prefix. It has
// no embedding or LM head: the caller supplies the paired target's.
func DSparkDraftSpec() *ArchSpec {
	spec := qwen3Spec()
	spec.Name = "qwen3_dspark"
	n := &spec.Names
	strip := func(f func(int) string) func(int) string {
		return func(l int) string { return strings.TrimPrefix(f(l), "model.") }
	}
	stripAll := func(f func(int) []string) func(int) []string {
		return func(l int) []string {
			names := f(l)
			for i, s := range names {
				names[i] = strings.TrimPrefix(s, "model.")
			}
			return names
		}
	}
	n.Embedding = ""
	n.OutputCandidates = nil
	n.OutputNorm = "norm.weight"
	n.AttnNorm, n.FfnNorm = strip(n.AttnNorm), strip(n.FfnNorm)
	n.QNormCandidates, n.KNormCandidates = stripAll(n.QNormCandidates), stripAll(n.KNormCandidates)
	n.Wq, n.Wk, n.Wv, n.Wo = strip(n.Wq), strip(n.Wk), strip(n.Wv), strip(n.Wo)
	n.WqBias, n.WkBias, n.WvBias = strip(n.WqBias), strip(n.WkBias), strip(n.WvBias)
	n.FfnUp, n.FfnGate, n.FfnDown = strip(n.FfnUp), strip(n.FfnGate), strip(n.FfnDown)
	return spec
}
