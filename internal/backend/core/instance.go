package core

import (
	"sync"

	"github.com/samcharles93/mantle/internal/hostcaps"
)

// Instance holds the CPU backend runtime state for a loaded model.
type Instance struct {
	Config             *ModelConfig
	Embeddings         *Mat
	OutputNorm         []float32
	Output             *Mat
	Layers             []Layer
	MaxContext         int
	Pos                int
	RMSEpsilon         float32
	HeadDim            int
	HeadCount          int
	MaxHeadKV          int
	RopeInvFreq        []float64
	RopeAttnScale      float32
	RopeInvFreqLocal   []float64
	RopeAttnScaleLocal float32
	MuPScale           float32
	RopeLocalOnly      bool
	RopeCosTable       []float32 // Precomputed cosine values for RoPE
	RopeSinTable       []float32 // Precomputed sine values for RoPE
	TilingConfig       TilingConfig

	attnPoolOnce sync.Once
	attnPool     *AttnPool

	Scratch ScratchBuffers
	ops     Ops

	hostCaps *hostcaps.Snapshot

	// effectiveContextLen constrains KV cache allocation to this length (0 = use model max)
	effectiveContextLen int
}

// Layer represents a single transformer layer with all its parameters.
type Layer struct {
	IsRecurrent bool
	HeadKV      int
	AttnType    string
	AttnWindow  int
	NoRoPE      bool // skip RoPE positional encoding for this layer
	FusedQGate  bool // Q and Attention Gate are fused into a single weight matrix (2*qDim rows)

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

	DeltaNet *DeltaNetLayer

	FfnUp   *Mat
	FfnGate *Mat
	FfnDown *Mat

	AttnCache AttnCache

	// MoE support
	MoE *MoELayer

	// Mamba support
	Mamba *MambaLayer
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
	InProj       *Mat
	OutProj      *Mat
	Conv         *Mat
	ConvBias     []float32
	ALog         []float32
	D            []float32
	DTBias       []float32
	Norm         []float32
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

// DeltaNetLayer holds the weights and recurrent state for a Gated DeltaNet block.
type DeltaNetLayer struct {
	QKVProj *Mat
	AProj   *Mat
	BProj   *Mat
	ZProj   *Mat
	OutProj *Mat
	Conv    *Mat
	Norm    []float32
	ALog    []float32
	DTBias  []float32

	NumKeyHeads   int
	NumValueHeads int
	HeadKeyDim    int
	HeadValueDim  int
	KeyDim        int
	ValueDim      int

	ConvState      []float32
	RecurrentState []float32
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
	DeltaQKV  []float32
	DeltaA    []float32
	DeltaB    []float32
	DeltaZ    []float32
	DeltaQ    []float32
	DeltaK    []float32
	DeltaV    []float32
	DeltaOut  []float32
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
	if m == nil {
		return DefaultOps{}
	}
	m.bindDefaultOps()
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

func (m *Instance) setHostCapabilities(caps *hostcaps.Snapshot) {
	if m == nil || caps == nil {
		return
	}
	m.hostCaps = caps
}

// SetHostCapabilities binds detected host capabilities to this instance.
func (m *Instance) SetHostCapabilities(caps *hostcaps.Snapshot) {
	m.setHostCapabilities(caps)
}

// ModelConfig returns the model configuration.
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

// SetEffectiveContextLength constrains KV cache allocation to this length.
// This is used at inference time to prevent allocating cache for the model's
// full maximum context window when fewer tokens will be generated.
// Set to 0 to use the model's MaxContext (default).
func (m *Instance) SetEffectiveContextLength(ctxLen int) {
	if m == nil {
		return
	}
	m.effectiveContextLen = ctxLen

	// Propagate to ops backend if available
	if opsWithContext, ok := m.ops.(interface{ SetEffectiveContextLength(int) }); ok {
		opsWithContext.SetEffectiveContextLength(ctxLen)
	}
}

// GetEffectiveContextLength returns the constrained context length for KV cache allocation.
// Returns 0 if no constraint is set (use model max).
func (m *Instance) GetEffectiveContextLength() int {
	if m == nil {
		return 0
	}
	return m.effectiveContextLen
}
