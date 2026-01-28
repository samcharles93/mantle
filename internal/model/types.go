package model

import (
	"sync"

	"github.com/samcharles93/mantle/internal/tensor"
	"github.com/samcharles93/mantle/internal/tokenizer"
)

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
}

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

type Instance struct {
	Config        *ModelConfig
	Embeddings    *tensor.Mat
	OutputNorm    []float32
	Output        *tensor.Mat
	Layers        []Layer
	MaxContext    int
	Pos           int
	RMSEpsilon    float32
	HeadDim       int
	HeadCount     int
	MaxHeadKV     int
	ropeInvFreq   []float64
	ropeAttnScale float32
	muPScale      float32
	ropeLocalOnly bool

	attnPoolOnce sync.Once
	attnPool     *attnPool

	scratch scratchBuffers
}

type Layer struct {
	IsRecurrent bool
	HeadKV      int
	AttnType    string
	AttnWindow  int

	AttnNorm     []float32
	PostAttnNorm []float32
	FfnNorm      []float32
	PostFfnNorm  []float32
	AttnQNorm    []float32
	AttnKNorm    []float32

	Wq *tensor.Mat
	Wk *tensor.Mat
	Wv *tensor.Mat
	Wo *tensor.Mat
	// Optional attention gating (AFMoE).
	AttnGate *tensor.Mat

	ShortConvKernel  *tensor.Mat
	ShortConvInProj  *tensor.Mat
	ShortConvOutProj *tensor.Mat
	ShortConvState   shortConvState

	FfnUp   *tensor.Mat
	FfnGate *tensor.Mat
	FfnDown *tensor.Mat
	MoE     *moeLayer

	AttnCache attnCache
}

type attnCache struct {
	k        []float32
	v        []float32
	kvStride int
}

type shortConvState struct {
	buf       []float32
	kernelLen int
}

type moeExpert struct {
	Up   *tensor.Mat
	Gate *tensor.Mat
	Down *tensor.Mat
}

type moeShared struct {
	Up           *tensor.Mat
	Gate         *tensor.Mat
	Down         *tensor.Mat
	Intermediate int
}

type moeLayer struct {
	Router     *tensor.Mat
	ExpertBias []float32
	Shared     moeShared
	Experts    []moeExpert
	TopK       int
	RouteScale float32
}

type scratchBuffers struct {
	x         []float32
	tmp       []float32
	tmp2      []float32
	q         []float32
	k         []float32
	v         []float32
	attnOut   []float32
	attnProj  []float32
	attnGate  []float32
	scores    []float32
	ffnUp     []float32
	ffnGate   []float32
	ffnAct    []float32
	moeAccum  []float32
	routerRaw []float32
	routerSel []float32
	routerIdx []int
	routerW   []float32
	scProj    []float32
	scBx      []float32
	scConv    []float32
	logits    []float32
}
