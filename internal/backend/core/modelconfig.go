package core

import (
	"fmt"
	"strings"

	"github.com/samcharles93/mantle/internal/model"
	"github.com/samcharles93/mantle/internal/tokenizer"
)

type Config struct {
	BlockCount         int
	EmbeddingLength    int
	FFNLength          int
	HeadCount          int
	HeadDim            int
	RotaryDim          int
	HeadCountKV        []int
	RMSEpsilon         float64
	RopeFreqBase       float64
	RopeFreqBaseLocal  float64
	RopeScaling        *RopeScaling
	ContextLength      int
	VocabSize          int
	ShortConvLCache    int
	DeltaKeyDim        int
	DeltaValueDim      int
	DeltaNumKeyHeads   int
	DeltaNumValueHeads int
	DeltaHeadKeyDim    int
	DeltaHeadValueDim  int
	DeltaConvKernel    int

	EmbeddingMultiplier    float64
	ResidualMultiplier     float32
	LMHeadMultiplier       float64
	AttentionInMultiplier  float64
	AttentionOutMultiplier float64
	SSMInMultiplier        float64
	SSMOutMultiplier       float64
	SSMMultipliers         []float64
	AttnLogitSoftcap       float32
	FinalLogitSoftcap      float32

	MambaInner          int
	MambaHeadCount      int
	MambaHeadDim        int
	MambaDState         int
	MambaGroups         int
	MambaConvChannels   int
	MambaConvKernel     int
	MambaRMSNorm        bool
	MambaNormBeforeGate bool
	MambaChunkSize      int
	TimeStepMin         float64
	TimeStepMax         float64
	TimeStepFloor       float64

	// Optional architecture features.
	NumDenseLayers      int
	MoEIntermediateSize int
	NumExperts          int
	NumExpertsPerTok    int
	NumSharedExperts    int
	RouteScale          float64
	SlidingWindow       int
	LayerTypes          []string
	MuPEnabled          bool
	AttentionBias       bool
	CacheTypeK          string
	CacheTypeV          string
	HiddenAct           string
}

// ResidualScale returns the residual multiplier, normalizing the zero value
// (unset) to 1 so hand-built instances behave like loaded models.
func (c *Config) ResidualScale() float32 {
	if c == nil || c.ResidualMultiplier == 0 {
		return 1
	}
	return c.ResidualMultiplier
}

const (
	CacheTypeF32  = "f32"
	CacheTypeF16  = "f16"
	CacheTypeBF16 = "bf16" // Reserved/future
	CacheTypeQ8_0 = "q8_0"
	CacheTypeQ4_0 = "q4_0" // Reserved/future
)

// ResolveCacheType normalizes and validates a KV cache data type string for the
// CPU/general runtime. An empty value defaults to defaultType.
func ResolveCacheType(flagName, raw, defaultType string) (string, error) {
	ct := strings.ToLower(strings.TrimSpace(raw))
	if ct == "" {
		ct = defaultType
	}
	switch ct {
	case CacheTypeF32, CacheTypeF16, CacheTypeQ8_0:
		return ct, nil
	default:
		return "", fmt.Errorf("unsupported %s %q (supported: f32, f16, q8_0)", flagName, raw)
	}
}

// ResolveCUDACacheType normalizes and validates a KV cache data type string for
// the CUDA backend, where device KV caches support f16 and q8_0. An empty value
// defaults to f16.
func ResolveCUDACacheType(flagName, raw string) (string, error) {
	ct := strings.ToLower(strings.TrimSpace(raw))
	if ct == "" {
		ct = CacheTypeF16
	}
	switch ct {
	case CacheTypeF16, CacheTypeQ8_0:
		return ct, nil
	default:
		return "", fmt.Errorf("unsupported %s %q for cuda backend (supported: f16, q8_0)", flagName, raw)
	}
}

// RopeScaling is the config-level rotary scaling description, shared with
// the model package rather than mirrored.
type RopeScaling = model.RopeScaling

type ModelConfig struct {
	Arch      string
	Config    Config
	Tokenizer tokenizer.TokenizerConfig
	// Vision is non-nil for multimodal (vision-language) models.
	Vision *VisionConfig
	// Audio is non-nil for audio-language models (e.g. LFM3-Omni).
	Audio *AudioConfig
}

// MambaConfig captures the runtime parameters required to execute a Mamba
// block on any backend. It is the shared contract used by the SIMD fast-path
// hook and the CUDA backend implementation.
type MambaConfig struct {
	SSMInMultiplier     float32
	SSMOutMultiplier    float32
	TimeStepMin         float32
	TimeStepMax         float32
	TimeStepFloor       float32
	MambaRMSNorm        bool
	MambaNormBeforeGate bool
	RMSEpsilon          float32
}

// DeltaNetConfig captures the runtime parameters required to execute a Gated
// DeltaNet block on any backend. It is the shared contract used by the SIMD
// fast-path hook and the CUDA backend implementation.
type DeltaNetConfig struct {
	RMSEpsilon float32
}

// MoEConfig captures the runtime parameters required to execute a MoE block on
// any backend. It is the shared contract used by the SIMD fast-path hook and
// the CUDA backend implementation.
type MoEConfig struct{}
