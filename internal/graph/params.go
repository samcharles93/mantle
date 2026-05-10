package graph

type NodeParams interface {
	nodeParams()
}

type AttentionParams struct {
	NHeadKV          int
	HeadDim          int
	KVStride         int
	SlidingWin       int
	LayerIndex       int
	InvFreq          []float64
	AttnScale        float32
	AttnLogitSoftcap float32
}

func (AttentionParams) nodeParams() {}

type FFNParams struct {
	Activation string
	HasBias    bool
}

func (FFNParams) nodeParams() {}

type MoEParams struct {
	TopK         int
	NumExperts   int
	SharedExpert bool
}

func (MoEParams) nodeParams() {}

type MambaParams struct{}

func (MambaParams) nodeParams() {}

type DeltaNetParams struct{}

func (DeltaNetParams) nodeParams() {}

type EmbedParams struct {
	VocabSize int
	EmbDim    int
}

func (EmbedParams) nodeParams() {}

type OutputParams struct {
	Softcap float32
}

func (OutputParams) nodeParams() {}
