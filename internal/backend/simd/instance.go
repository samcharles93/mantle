package simd

import "sync"

// Instance holds the CPU backend runtime state for a loaded model.
// It implements model.Runtime interface.
type Instance struct {
	Config        *ModelConfig
	Embeddings    *Mat
	OutputNorm    []float32
	Output        *Mat
	Layers        []Layer
	MaxContext    int
	Pos           int
	RMSEpsilon    float32
	HeadDim       int
	HeadCount     int
	MaxHeadKV     int
	RopeInvFreq   []float64
	RopeAttnScale float32
	MuPScale      float32
	RopeLocalOnly bool

	attnPoolOnce sync.Once
	attnPool     *AttnPool

	Scratch ScratchBuffers
	ops     Ops
}

// Layer represents a single transformer layer with all its parameters.
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

	Wq *Mat
	Wk *Mat
	Wv *Mat
	Wo *Mat

	// Optional QKV biases
	WqBias []float32
	WkBias []float32
	WvBias []float32

	// Optional attention gating (AFMoE)
	AttnGate *Mat

	ShortConvKernel  *Mat
	ShortConvInProj  *Mat
	ShortConvOutProj *Mat
	ShortConvState   ShortConvState

	FfnUp   *Mat
	FfnGate *Mat
	FfnDown *Mat
	MoE     *MoELayer

	AttnCache AttnCache

	Mamba *MambaLayer
}

// AttnCache holds KV cache for attention mechanism.
type AttnCache struct {
	K        []float32
	V        []float32
	K16      []uint16
	V16      []uint16
	KvStride int
}

// ShortConvState holds state for recurrent attention convolution.
type ShortConvState struct {
	Buf       []float32
	KernelLen int
}

// MoEExpert represents a single expert in mixture of experts.
type MoEExpert struct {
	Up   *Mat
	Gate *Mat
	Down *Mat
}

// MoEShared represents shared expert in mixture of experts.
type MoEShared struct {
	Up           *Mat
	Gate         *Mat
	Down         *Mat
	Intermediate int
}

// MoELayer represents a mixture of experts layer.
type MoELayer struct {
	Router     *Mat
	ExpertBias []float32
	Shared     MoEShared
	Experts    []MoEExpert
	TopK       int
	RouteScale float32
}

// MambaLayer holds the parameters and state for a Mamba-2 SSM block.
type MambaLayer struct {
	InProj   *Mat
	OutProj  *Mat
	Conv     *Mat
	ConvBias []float32
	ALog     []float32
	D        []float32
	DTBias   []float32
	Norm     []float32

	Inner        int
	HeadCount    int
	HeadDim      int
	DState       int
	Groups       int
	GroupSize    int
	ConvKernel   int
	ConvChannels int

	ConvState []float32
	SSMState  []float32
}

// ScratchBuffers holds temporary buffers for computation.
type ScratchBuffers struct {
	X         []float32
	Tmp       []float32
	Tmp2      []float32
	Q         []float32
	K         []float32
	V         []float32
	AttnOut   []float32
	AttnProj  []float32
	AttnGate  []float32
	Scores    []float32
	FfnUp     []float32
	FfnGate   []float32
	FfnAct    []float32
	MoeAccum  []float32
	RouterRaw []float32
	RouterSel []float32
	RouterIdx []int
	RouterW   []float32
	ScProj    []float32
	ScBx      []float32
	ScConv    []float32
	Logits    []float32

	MambaIn   []float32
	MambaProj []float32
	MambaConv []float32
	MambaZ    []float32
	MambaX    []float32
	MambaB    []float32
	MambaC    []float32
	MambaDT   []float32
	MambaY    []float32
	MambaOut  []float32
}

// Ops returns the ops interface for this instance.
func (m *Instance) Ops() Ops {
	if m.ops == nil {
		return DefaultOps{}
	}
	return m.ops
}

// SetOps sets the ops implementation for this instance.
func (m *Instance) SetOps(ops Ops) {
	if m == nil {
		return
	}
	m.ops = ops
}

// ModelConfig returns the model configuration (implements model.Runtime).
func (m *Instance) ModelConfig() *ModelConfig {
	if m == nil {
		return nil
	}
	return m.Config
}

// GetAttnPool returns the attention pool, initializing it if needed.
func (m *Instance) GetAttnPool() *AttnPool {
	return m.getAttnPool()
}

func (m *Instance) getAttnPool() *AttnPool {
	if m.attnPool == nil {
		m.initAttnPool()
	}
	return m.attnPool
}

func (m *Instance) initAttnPool() {
	m.attnPoolOnce.Do(func() {
		m.attnPool = NewAttnPool(AttnWorkersFor(m.HeadCount), m.MaxContext)
	})
}
