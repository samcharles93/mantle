package simd

import "github.com/samcharles93/mantle/internal/tokenizer"

type Config struct {
	BlockCount      int
	EmbeddingLength int
	FFNLength       int
	HeadCount       int
	HeadDim         int
	HeadCountKV     []int
	RMSEpsilon      float64
	RopeFreqBase    float64
	RopeScaling     *RopeScaling
	ContextLength   int
	VocabSize       int
	ShortConvLCache int

	EmbeddingMultiplier    float64
	LMHeadMultiplier       float64
	AttentionInMultiplier  float64
	AttentionOutMultiplier float64
	SSMInMultiplier        float64
	SSMOutMultiplier       float64
	SSMMultipliers         []float64

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
}

const (
	CacheTypeF32  = "f32"
	CacheTypeF16  = "f16"
	CacheTypeBF16 = "bf16" // Reserved/future
	CacheTypeQ8_0 = "q8_0" // Reserved/future
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
