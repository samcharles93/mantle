package core

import "github.com/samcharles93/mantle/internal/tokenizer"

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
	FlashAttention      bool
	HiddenAct           string
}

const (
	CacheTypeF32  = "f32"
	CacheTypeF16  = "f16"
	CacheTypeBF16 = "bf16" // Reserved/future
	CacheTypeQ8_0 = "q8_0"
	CacheTypeQ4_0 = "q4_0" // Reserved/future
)

type RopeScaling struct {
	Type            string
	Factor          float64
	OrigMaxCtx      int
	LowFactor       float64
	HighFactor      float64
	AttentionFactor float64
	BetaFast        float64
	BetaSlow        float64
	MScale          float64
	MScaleAllDim    float64
	Truncate        bool
	HasTruncate     bool
}

type ModelConfig struct {
	Arch      string
	Config    Config
	Tokenizer tokenizer.TokenizerConfig
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
