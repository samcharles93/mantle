package model

import (
	"github.com/samcharles93/mantle/internal/tensor"
	"github.com/samcharles93/mantle/internal/tokenizer"
)

type Config struct {
	BlockCount      int
	EmbeddingLength int
	FFNLength       int
	HeadCount       int
	HeadCountKV     []int
	RMSEpsilon      float64
	RopeFreqBase    float64
	ContextLength   int
	VocabSize       int
	ShortConvLCache int
}

type ModelConfig struct {
	Arch      string
	Config    Config
	Tokenizer tokenizer.TokenizerConfig
}

type Model struct {
	Config      *ModelConfig
	Embeddings  *tensor.Mat
	OutputNorm  []float32
	Output      *tensor.Mat
	Layers      []Layer
	MaxContext  int
	Pos         int
	RMSEpsilon  float32
	HeadDim     int
	HeadCount   int
	MaxHeadKV   int
	ropeInvFreq []float64

	scratch scratchBuffers
}

type Layer struct {
	IsRecurrent bool
	HeadKV      int

	AttnNorm  []float32
	FfnNorm   []float32
	AttnQNorm []float32
	AttnKNorm []float32

	Wq *tensor.Mat
	Wk *tensor.Mat
	Wv *tensor.Mat
	Wo *tensor.Mat

	ShortConvKernel  *tensor.Mat
	ShortConvInProj  *tensor.Mat
	ShortConvOutProj *tensor.Mat
	ShortConvState   shortConvState

	FfnUp   *tensor.Mat
	FfnGate *tensor.Mat
	FfnDown *tensor.Mat

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

type scratchBuffers struct {
	x        []float32
	tmp      []float32
	tmp2     []float32
	q        []float32
	k        []float32
	v        []float32
	attnOut  []float32
	attnProj []float32
	scores   []float32
	ffnUp    []float32
	ffnGate  []float32
	ffnAct   []float32
	scProj   []float32
	scBx     []float32
	scConv   []float32
	logits   []float32
}
