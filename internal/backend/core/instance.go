package core

import (
	"fmt"
	"math"
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
	Gemma4PerLayer     *Gemma4PerLayerInputModel
	MaxContext         int
	Pos                int
	RMSEpsilon         float32
	HeadDim            int
	HeadCount          int
	MaxHeadKV          int
	MaxQDim            int
	MaxKVStride        int
	RopeInvFreq        []float64
	RopeAttnScale      float32
	RopeInvFreqLocal   []float64
	RopeAttnScaleLocal float32
	MuPScale           float32
	RopeLocalOnly      bool
	RopeCosTable       []float32 // Precomputed cosine values for RoPE
	RopeSinTable       []float32 // Precomputed sine values for RoPE
	TilingConfig       TilingConfig

	// HiddenTaps captures decoder-layer residual outputs for the DSpark draft
	// model's context taps. Disabled until SetHiddenTapLayers is called; see
	// HiddenTaps for the buffer contract.
	HiddenTaps HiddenTaps

	attnPoolOnce sync.Once
	attnPool     *AttnPool

	// MaxBatch is the largest number of positions one batched prefill call can
	// process. It sizes the Batch* scratch buffers; 0 disables the batched path.
	MaxBatch int
	Scratch  ScratchBuffers
	ops      Ops

	hostCaps *hostcaps.Snapshot

	// effectiveContextLen constrains KV cache allocation to this length (0 = use model max)
	effectiveContextLen int
}

// Layer represents a single transformer layer with all its parameters.
type Layer struct {
	IsRecurrent    bool
	HeadKV         int
	HeadDim        int
	KVHeadDim      int
	AttnType       string
	AttnWindow     int
	AttnScale      float32
	NoRoPE         bool // skip RoPE positional encoding for this layer
	FusedQGate     bool // Q and Attention Gate are fused into a single weight matrix (2*qDim rows)
	ValueFromKey   bool
	ApplyVNorm     bool
	SharedKVSource int
	StoreFullKV    bool
	RopeInvFreq    []float64
	RopeAttnScale  float32
	LayerScale     float32
	FFNActivation  string

	RoundActivationsBF16 bool

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
	MoE       *MoELayer
	Gemma4MoE *Gemma4MoELayer
	Gemma4PLE *Gemma4PLELayer

	// Mamba support
	Mamba *MambaLayer
}

type Gemma4PerLayerInputModel struct {
	Embeddings      *Mat
	Projection      *Mat
	ProjectionNorm  []float32
	HiddenSize      int
	LayerCount      int
	EmbeddingScale  float32
	ProjectionScale float32
	InputScale      float32
}

type Gemma4PLELayer struct {
	InputGate  *Mat
	Projection *Mat
	PostNorm   []float32
}

type Gemma4MoEExpert struct {
	GateUp *Mat
	Down   *Mat
}

type Gemma4MoELayer struct {
	RouterProj           *Mat
	RouterScale          []float32
	RouterPerExpertScale []float32
	PreNorm2             []float32
	PostNorm1            []float32
	PostNorm2            []float32
	Experts              []Gemma4MoEExpert
	TopK                 int
	Intermediate         int
	ScalarRootSize       float32
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
	X             []float32
	Tmp           []float32
	Tmp2          []float32
	Q             []float32
	K             []float32
	V             []float32
	AttnOut       []float32
	AttnProj      []float32
	AttnGate      []float32
	Scores        []float32
	FfnUp         []float32
	FfnGate       []float32
	FfnAct        []float32
	MoeAccum      []float32
	RouterRaw     []float32
	RouterSel     []float32
	RouterIdx     []int
	RouterW       []float32
	RouterTop     []float32
	ScProj        []float32
	ScBx          []float32
	ScConv        []float32
	DeltaQKV      []float32
	DeltaConv     []float32
	DeltaA        []float32
	DeltaB        []float32
	DeltaZ        []float32
	DeltaQ        []float32
	DeltaK        []float32
	DeltaV        []float32
	DeltaOut      []float32
	Logits        []float32
	PerLayerTok   []float32
	PerLayerProj  []float32
	PerLayerInput []float32

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

	// Batched prefill scratch. Each buffer holds [MaxBatch, width] rows for the
	// same quantity as its single-token counterpart above, so a whole prompt can
	// be projected with one dense GEMM per weight matrix.
	BatchX       []float32 // [MaxBatch, embd] hidden state
	BatchNorm    []float32 // [MaxBatch, embd] pre-norm output
	BatchProj    []float32 // [MaxBatch, embd] attention, then FFN, block output
	BatchQ       []float32 // [MaxBatch, qDim]
	BatchK       []float32 // [MaxBatch, kvStride]
	BatchV       []float32 // [MaxBatch, kvStride]
	BatchAttnOut []float32 // [MaxBatch, qDim]
	BatchFfnUp   []float32 // [MaxBatch, ffn]
	BatchFfnGate []float32 // [MaxBatch, ffn]
	BatchFfnAct  []float32 // [MaxBatch, ffn]
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

// GraphCompute is a no-op default implementation for backends that don't
// implement graph-based execution yet.
func (m *Instance) GraphCompute(_ any, _ any) ([]float32, error) {
	return nil, fmt.Errorf("GraphCompute not implemented for this backend")
}

// HiddenTaps captures the post-layer residual output (the "hidden state") of
// selected decoder layers for the current forward. It supplies the context taps
// the DSpark draft model consumes.
//
// Capture is opt-in and off by default: with no Layers configured no buffer is
// allocated and each per-layer capture hook is one length check. The buffer
// holds only the rows of the most recent forward — it is deliberately not a
// whole-context cache. At 4096 tokens a five-tap MiniCPM5 context would need
// about 160 MiB, whereas one eight-row verification block costs 320 KiB.
//
// The layer list is caller-supplied on purpose: AGENTS.md forbids inferring
// runtime behaviour from container contents, so it is never read from the MCF.
type HiddenTaps struct {
	// Layers lists the tapped decoder layer indices in concatenation order: slot
	// i of the buffer is Layers[i]'s output. -1 selects the embedding output (the
	// input to decoder layer 0). The list is strictly increasing, matching the
	// reference's target_layer_ids order.
	Layers []int
	// Slots maps a decoder layer index to its tap slot, or -1 when that layer is
	// not tapped. It is the per-layer hot-path lookup.
	Slots []int
	// EmbeddingSlot is the tap slot for layer index -1, or -1 when the embedding
	// output is not tapped.
	EmbeddingSlot int
	// Host holds len(Layers)*Capacity*Hidden floats laid out slot-major: row r of
	// slot s starts at (s*Capacity+r)*Hidden. A forward overwrites it; nothing
	// accumulates across forwards.
	Host []float32
	// Hidden is the residual width (the model's embedding length).
	Hidden int
	// Capacity is the number of rows Host holds per tap.
	Capacity int
	// Rows is the number of rows captured by the most recent forward. It is
	// authoritative: a consumer that expected more rows than this is looking at
	// an incomplete capture — for example a prompt that fell back to the
	// single-token path, which captures one row per step.
	Rows int
}

// Enabled reports whether hidden-tap capture is configured.
func (t *HiddenTaps) Enabled() bool {
	return t != nil && len(t.Layers) > 0
}

// SlotFor returns the tap slot for decoder layer layerIdx, or -1 when that
// layer is not tapped. layerIdx == -1 selects the embedding output.
func (t *HiddenTaps) SlotFor(layerIdx int) int {
	if !t.Enabled() {
		return -1
	}
	if layerIdx == -1 {
		return t.EmbeddingSlot
	}
	if layerIdx < 0 || layerIdx >= len(t.Slots) {
		return -1
	}
	return t.Slots[layerIdx]
}

// StoreRow records the Hidden-wide residual x as row row of tap slot slot. It
// ignores an untapped slot or an out-of-range row, so a forward larger than
// Capacity leaves Rows short of the requested row count instead of silently
// wrapping onto earlier rows.
func (t *HiddenTaps) StoreRow(slot, row int, x []float32) {
	if slot < 0 || row < 0 || row >= t.Capacity || len(x) < t.Hidden {
		return
	}
	base := (slot*t.Capacity + row) * t.Hidden
	copy(t.Host[base:base+t.Hidden], x[:t.Hidden])
	if row+1 > t.Rows {
		t.Rows = row + 1
	}
}

// SetHiddenTapLayers enables capture of decoder-layer residual outputs. layers
// are decoder layer indices in concatenation order; -1 selects the embedding
// output. The list must be strictly increasing and every entry must lie in
// [-1, len(m.Layers)-1]; capacity is the number of rows one forward may capture.
// An empty list disables capture and releases the buffer.
func (m *Instance) SetHiddenTapLayers(layers []int, capacity int) error {
	if m == nil {
		return fmt.Errorf("hidden taps: nil instance")
	}
	if len(layers) == 0 {
		m.HiddenTaps = HiddenTaps{}
		return nil
	}
	if m.Config == nil || m.Config.Config.EmbeddingLength <= 0 {
		return fmt.Errorf("hidden taps: model has no embedding length")
	}
	if capacity <= 0 {
		return fmt.Errorf("hidden taps: capacity must be positive, got %d", capacity)
	}
	prev := -2
	for _, l := range layers {
		if l < -1 || l >= len(m.Layers) {
			return fmt.Errorf("hidden tap layer %d out of range [-1, %d]", l, len(m.Layers)-1)
		}
		if l <= prev {
			return fmt.Errorf("hidden tap layers must be strictly increasing: %d follows %d", l, prev)
		}
		prev = l
	}
	hidden := m.Config.Config.EmbeddingLength
	if capacity > math.MaxInt/hidden {
		return fmt.Errorf("hidden taps: row size overflows int")
	}
	rowFloats := capacity * hidden
	if len(layers) > math.MaxInt/rowFloats {
		return fmt.Errorf("hidden taps: buffer size overflows int")
	}
	taps := HiddenTaps{
		Layers:        append([]int(nil), layers...),
		Slots:         make([]int, len(m.Layers)),
		EmbeddingSlot: -1,
		Hidden:        hidden,
		Capacity:      capacity,
		Host:          make([]float32, rowFloats*len(layers)),
	}
	for i := range taps.Slots {
		taps.Slots[i] = -1
	}
	for slot, l := range taps.Layers {
		if l == -1 {
			taps.EmbeddingSlot = slot
		} else {
			taps.Slots[l] = slot
		}
	}
	m.HiddenTaps = taps
	return nil
}
