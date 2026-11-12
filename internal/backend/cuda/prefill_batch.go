//go:build cuda

package cuda

import (
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
	"sync"
	"unsafe"

	instance "github.com/samcharles93/mantle/internal/backend/core"
	"github.com/samcharles93/mantle/internal/backend/cuda/native"
	"github.com/samcharles93/mantle/pkg/mcf"
)

// This file implements a device-resident BATCHED prefill path for the CUDA
// backend. It is deliberately conservative: a strict pre-flight gate decides
// whether a prompt can be run as a single [N,hidden] batch, and anything it
// cannot prove safe falls back to the existing per-token sequential path
// (cudaRuntime.PrefillTokens) unchanged.
//
// The batched path owns its own [N,...] device buffers and never touches the
// single-slot persistent hidden-state mechanism (BeginToken/EndToken/xPersistDev)
// used by the decode path. The only shared device state it mutates is the
// per-layer KV cache (via the same row-store kernels the sequential path uses)
// and o.lastDevKVPos, which is advanced to the last stored position.

const batchedPrefillEnv = "MANTLE_CUDA_BATCHED_PREFILL"

// batchedPrefillEnabled reports whether the batched prefill path is allowed to
// run. It defaults to enabled and can be disabled with the
// MANTLE_CUDA_BATCHED_PREFILL kill-switch ("0"/"false"/"off"/"no").
func batchedPrefillEnabled() bool {
	v, ok := os.LookupEnv(batchedPrefillEnv)
	if !ok {
		return true
	}
	trimmed := strings.TrimSpace(v)
	if b, err := strconv.ParseBool(trimmed); err == nil {
		return b
	}
	switch strings.ToLower(trimmed) {
	case "0", "false", "off", "no", "n":
		return false
	default:
		return true
	}
}

// PrefillTokens shadows the embedded cudaRuntime method. It runs the whole
// prompt as one device-resident batch when the conservative gate permits, and
// otherwise delegates to the untouched sequential implementation so that path
// remains the correctness oracle.
func (gr *GraphRuntime) PrefillTokens(tokens []int) ([]float32, error) {
	if gr == nil || gr.cudaRuntime == nil {
		return nil, fmt.Errorf("cuda runtime is closed")
	}
	if !batchedPrefillEnabled() || gr.inst == nil || gr.ops == nil ||
		gr.cudaRuntime.model == nil || !gr.batchedPrefillEligible(tokens) {
		return gr.cudaRuntime.PrefillTokens(tokens)
	}

	plan, err := gr.newBatchedPrefillPlan(tokens)
	if err != nil || plan == nil {
		// Pre-flight failed before any KV write; the sequential path is safe.
		return gr.cudaRuntime.PrefillTokens(tokens)
	}
	defer plan.free()

	return plan.run()
}

// PrefillTokensEmbeds is PrefillTokens with optional per-position input
// embedding overrides (multimodal image tokens). The batched device path
// applies the overrides during its host embedding gather, so they ride the
// same single H2D upload as the token embeddings.
func (gr *GraphRuntime) PrefillTokensEmbeds(tokens []int, embeds [][]float32) ([]float32, error) {
	if gr == nil || gr.cudaRuntime == nil {
		return nil, fmt.Errorf("cuda runtime is closed")
	}
	if embeds == nil {
		return gr.PrefillTokens(tokens)
	}
	if !batchedPrefillEnabled() || gr.inst == nil || gr.ops == nil ||
		gr.cudaRuntime.model == nil || !gr.batchedPrefillEligible(tokens) {
		return gr.cudaRuntime.PrefillTokensEmbeds(tokens, embeds)
	}
	plan, err := gr.newBatchedPrefillPlanEmbeds(tokens, embeds)
	if err != nil || plan == nil {
		return gr.cudaRuntime.PrefillTokensEmbeds(tokens, embeds)
	}
	defer plan.free()
	return plan.run()
}

// batchedPrefillEligible is the conservative pre-flight gate. It returns false
// unless every layer is a plain dense transformer block whose device kernels
// the batched path can reproduce. It never mutates any state.
func (gr *GraphRuntime) batchedPrefillEligible(tokens []int) bool {
	if gr == nil || gr.inst == nil || gr.ops == nil || gr.cudaRuntime == nil || gr.cudaRuntime.model == nil {
		return false
	}
	if len(tokens) < 2 {
		return false
	}
	m := gr.inst
	o := gr.ops
	if m.Config == nil || m.Config.Config.EmbeddingLength <= 0 || m.Config.Config.VocabSize <= 0 {
		return false
	}
	if m.Gemma4PerLayer != nil {
		return false
	}
	if m.HeadCount <= 0 || m.HeadDim <= 0 {
		return false
	}
	if m.Pos < 0 || m.Pos+len(tokens) > m.MaxContext {
		return false
	}
	if o.blas == (native.BlasHandle{}) || o.stream == (native.Stream{}) {
		return false
	}
	// Note: models that set FinalLogitSoftcap or LMHeadMultiplier are eligible.
	// The sequential oracle applies both (simd/runtime.go ForwardToken forces
	// the pending device result to host before postprocessing), and the batched
	// output head applies the same transform via batchedLogitPostprocess, so the
	// two agree. These used to be excluded because the sequential path's
	// flushLastResult clobbered its own postprocessing.
	if !useAttentionInnerFastPath() || !useFFNFastPath() {
		return false
	}
	hidden := m.Config.Config.EmbeddingLength
	if m.Embeddings == nil || m.Embeddings.C != hidden {
		return false
	}
	if m.Output == nil || m.Output.C != hidden || m.Output.R != m.Config.Config.VocabSize {
		return false
	}
	if !batchedWeightDense(o, m.Output) {
		return false
	}
	if len(m.Layers) == 0 {
		return false
	}
	for _, tok := range tokens {
		if tok < 0 || tok >= m.Config.Config.VocabSize {
			return false
		}
	}
	// All layers must share one KV stride and FFN width: the scratch buffers
	// are allocated as [N, max] row-major and indexed by the per-layer stride,
	// which is only valid when every layer's stride equals the maximum.
	headDim := m.HeadDim
	// The KV stride comes from the first attention layer; recurrent (conv)
	// layers carry no KV cache and may sit at index 0.
	wantKvStride := 0
	wantInterm := 0
	for i := range m.Layers {
		l := &m.Layers[i]
		if !l.IsRecurrent && wantKvStride == 0 {
			wantKvStride = l.HeadKV * headDim
		}
		if wantInterm == 0 && l.FfnUp != nil {
			wantInterm = l.FfnUp.R
		}
	}
	if wantKvStride == 0 {
		wantKvStride = m.MaxKVStride
	}
	for i := range m.Layers {
		l := &m.Layers[i]
		// Recurrent (short-conv) layers carry no KV cache; only the FFN width
		// must match the shared buffers.
		if !l.IsRecurrent && l.HeadKV*headDim != wantKvStride {
			return false
		}
		if l.FfnUp == nil || l.FfnUp.R != wantInterm {
			return false
		}
		if !gr.batchedLayerEligible(l) {
			return false
		}
	}
	return true
}

func (gr *GraphRuntime) batchedLayerEligible(layer *instance.Layer) bool {
	m := gr.inst
	o := gr.ops
	hidden := m.Config.Config.EmbeddingLength
	headDim := m.HeadDim
	qDim := m.HeadCount * headDim
	kvStride := layer.HeadKV * headDim

	if layer.Mamba != nil || layer.DeltaNet != nil {
		return false
	}
	if layer.MoE != nil || layer.Gemma4MoE != nil || layer.Gemma4PLE != nil {
		return false
	}
	if layer.LayerScale != 1 {
		// runDecoderLayers routes any non-unit LayerScale (including the zero
		// value) through the Gemma4 BF16-rounding FFN path, which this batched
		// implementation does not reproduce.
		return false
	}
	if layer.SharedKVSource >= 0 {
		return false
	}
	if layer.FusedQGate {
		return false
	}
	if layer.ValueFromKey || layer.ApplyVNorm {
		return false
	}
	if layer.RoundActivationsBF16 {
		return false
	}
	if layer.AttnType != "" && layer.AttnType != "full_attention" && layer.AttnType != "sliding_attention" && !layer.IsRecurrent {
		return false
	}
	if layer.IsRecurrent {
		// Short-conv block: in_proj [3*hidden, hidden], out_proj
		// [hidden, hidden], f32 kernel [hidden, klen], state (klen-1)*hidden.
		k := layer.ShortConvKernel
		switch {
		case k == nil || k.Data == nil || k.R != hidden || k.C < 1:
			return false
		case layer.ShortConvInProj == nil || layer.ShortConvInProj.R != 3*hidden || layer.ShortConvInProj.C != hidden:
			return false
		case layer.ShortConvOutProj == nil || layer.ShortConvOutProj.R != hidden || layer.ShortConvOutProj.C != hidden:
			return false
		case len(layer.ShortConvState.Buf) != (k.C-1)*hidden:
			return false
		}
		for _, w := range []*instance.Mat{layer.ShortConvInProj, layer.ShortConvOutProj} {
			if !batchedWeightDense(o, w) {
				return false
			}
		}
	} else if layer.HeadDim != headDim || layer.HeadKV <= 0 {
		return false
	}
	if layer.NoRoPE {
		return false
	}
	if len(layer.AttnNorm) != hidden || len(layer.FfnNorm) != hidden {
		return false
	}
	// FFN shape checks apply to every layer, recurrent included.
	if layer.FfnUp == nil || layer.FfnGate == nil || layer.FfnDown == nil {
		return false
	}
	interm := layer.FfnUp.R
	if interm <= 0 || layer.FfnGate.R != interm || layer.FfnDown.C != interm {
		return false
	}
	if layer.FfnUp.C != hidden || layer.FfnGate.C != hidden || layer.FfnDown.R != hidden {
		return false
	}
	if layer.IsRecurrent {
		// Recurrent layers have no KV cache.
		if len(layer.PostAttnNorm) > 0 && len(layer.PostAttnNorm) != hidden {
			return false
		}
		if len(layer.PostFfnNorm) > 0 && len(layer.PostFfnNorm) != hidden {
			return false
		}
		return true
	}
	if layer.AttnCache.KvStride != kvStride || layer.AttnCache.CacheLen <= 0 {
		return false
	}
	if kvStride%int(mcf.QuantBlockSize) != 0 {
		return false
	}
	if layer.Wq == nil || layer.Wk == nil || layer.Wv == nil || layer.Wo == nil {
		return false
	}
	if layer.Wq.R != qDim || layer.Wq.C != hidden {
		return false
	}
	if layer.Wk.R != kvStride || layer.Wv.R != kvStride {
		return false
	}
	if layer.Wk.C != hidden || layer.Wv.C != hidden {
		return false
	}
	if layer.Wo.R != hidden || layer.Wo.C != qDim {
		return false
	}
	if n := len(layer.WqBias); n != 0 && n != qDim {
		return false
	}
	if n := len(layer.WkBias); n != 0 && n != kvStride {
		return false
	}
	if n := len(layer.WvBias); n != 0 && n != kvStride {
		return false
	}
	if n := len(layer.AttnQNorm); n != 0 && n != headDim {
		return false
	}
	if n := len(layer.AttnKNorm); n != 0 && n != headDim {
		return false
	}
	if layer.AttnGate != nil && (layer.AttnGate.R != qDim && layer.AttnGate.R != m.HeadCount || layer.AttnGate.C != hidden) {
		return false
	}
	if len(layer.PostAttnNorm) > 0 && len(layer.PostAttnNorm) != hidden {
		return false
	}
	if len(layer.PostFfnNorm) > 0 && len(layer.PostFfnNorm) != hidden {
		return false
	}
	apply, invFreq, _ := batchedLayerRoPE(m, layer)
	if apply && (len(invFreq) == 0 || len(invFreq)*2 > headDim) {
		return false
	}
	for _, w := range []*instance.Mat{layer.Wq, layer.Wk, layer.Wv, layer.Wo, layer.FfnUp, layer.FfnGate, layer.FfnDown, layer.AttnGate} {
		if w != nil && !batchedWeightDense(o, w) {
			return false
		}
	}
	return true
}

// batchedWeightDense reports whether a weight will be executed by the dense
// GEMM path in the batched implementation. Quantized, offloaded and
// non-F32/BF16 dense weights fall back to the sequential path — except in
// dequant weight mode, where quantized weights get a dense F16 device copy via
// Ops.deviceMat and the batched GEMM runs them as F16.
func batchedWeightDense(o *Ops, w *instance.Mat) bool {
	if w == nil || w.R == 0 || w.C == 0 {
		return false
	}
	if _, off := o.offloadedMats[w]; off {
		return false
	}
	mode := currentCUDAWeightMode()
	if mode != cudaWeightModeDequant && batchedQuantSupported(o, w) {
		// Quantized weight executed by the batched quant GEMM kernel.
		return true
	}
	if _, ok := o.qweights[w]; ok {
		return false
	}
	if useQuantKernel() && (shouldPreferQuantWeights(w, mode) || (w.Quant != nil && w.Quant.ValidFor(w))) {
		return false
	}
	if mode != cudaWeightModeDequant && shouldPreferQuantWeights(w, mode) {
		return false
	}
	if mcf.DTypeRequiresAligned64(w.DType) {
		// Quantized payload: only usable when the runtime dequantizes weights
		// to a dense device copy (dequant mode), never via the quant kernels.
		return mode == cudaWeightModeDequant && len(w.Raw) > 0
	}
	_, ok := denseWeightUploadType(w)
	return ok
}

// denseWeightUploadType mirrors weightUploadSpec but rejects encodings the
// batched path cannot feed to cublasGemmEx. Dense F16 weights are rejected
// because the sequential fast paths only ever use mixed F16/F32 GemmEx
// arguments, which cublas rejects (CUBLAS_STATUS_NOT_SUPPORTED).
func denseWeightUploadType(w *instance.Mat) (native.BlasDataType, bool) {
	if w == nil {
		return 0, false
	}
	if w.Raw == nil || w.DType == mcf.DTypeF32 {
		if len(w.Data) == 0 {
			return 0, false
		}
		return native.BlasF32, true
	}
	switch w.DType {
	case mcf.DTypeBF16:
		if len(w.Raw) == 0 {
			return 0, false
		}
		return native.BlasBF16, true
	default:
		return 0, false
	}
}

// batchedLayerRoPE returns the RoPE configuration for a layer, matching the
// selection logic in simd.Attention.
func batchedLayerRoPE(m *instance.Instance, layer *instance.Layer) (bool, []float64, float32) {
	apply := !layer.NoRoPE && (!m.RopeLocalOnly || layer.AttnType != "full_attention")
	invFreq := layer.RopeInvFreq
	scale := layer.RopeAttnScale
	if len(invFreq) == 0 {
		invFreq = m.RopeInvFreq
		scale = m.RopeAttnScale
		if layer.AttnType == "sliding_attention" && len(m.RopeInvFreqLocal) > 0 {
			invFreq = m.RopeInvFreqLocal
			scale = m.RopeAttnScaleLocal
		}
	}
	return apply, invFreq, scale
}

// batchedBuffers holds all [N,...] scratch owned by one batched prefill call.
type batchedBuffers struct {
	x      native.DeviceBuffer // [N, hidden] hidden state
	normed native.DeviceBuffer // [N, hidden] pre-norm output
	proj   native.DeviceBuffer // [N, hidden] attention/FFN block output
	q      native.DeviceBuffer // [N, qDim]
	attn   native.DeviceBuffer // [N, qDim] attention output
	k      native.DeviceBuffer // [N, maxKvStride]
	v      native.DeviceBuffer // [N, maxKvStride]
	up     native.DeviceBuffer // [N, maxInterm]
	gate   native.DeviceBuffer // [N, maxInterm]
	act    native.DeviceBuffer // [N, maxInterm]
	down   native.DeviceBuffer // [N, hidden]
	conv   native.DeviceBuffer // F16/BF16 input staging, >= N*max(hidden,maxInterm) elems
	last   native.DeviceBuffer // [hidden] final row
	lastN  native.DeviceBuffer // [hidden] final row normed
	logits native.DeviceBuffer // [vocab]

	// Hybrid-layer scratch: short-conv in_proj output [N, 3*hidden], gated conv
	// inputs [N, hidden], conv output [N, hidden], state-rotate staging
	// [(maxKlen-1), hidden], and the attention gate projection [N, qDim].
	scProj native.DeviceBuffer
	// int8 activation scratch for the dp4a quant GEMM: xq [N, maxK] int8,
	// xsc [N, maxBlocksPerRow] f32 per-block scales
	xq      native.DeviceBuffer
	xsc     native.DeviceBuffer
	scBx    native.DeviceBuffer
	scOut   native.DeviceBuffer
	scState native.DeviceBuffer
	gateBuf native.DeviceBuffer
}

func (b *batchedBuffers) all() []native.DeviceBuffer {
	return []native.DeviceBuffer{b.x, b.normed, b.proj, b.q, b.attn, b.k, b.v, b.up, b.gate, b.act, b.down, b.conv, b.last, b.lastN, b.logits, b.scProj, b.scBx, b.scOut, b.scState, b.gateBuf, b.xq, b.xsc}
}

type batchedLayerPlan struct {
	layer    *instance.Layer
	index    int
	kvStride int
	kvHeads  int
	interm   int

	// cacheLen is the kernel ring size (after effective-context bounding);
	// moduloLen is AttnCache.CacheLen, used for the store index exactly as the
	// sequential path does.
	cacheLen     int
	moduloLen    int
	useQ8K       bool
	useQ8V       bool
	blocksPerRow int
	cache        deviceAttnCache

	wq, wk, wv, wo deviceMat
	up, gate, down deviceMat

	attnNormDev native.DeviceBuffer
	ffnNormDev  native.DeviceBuffer
	qNormDev    native.DeviceBuffer
	kNormDev    native.DeviceBuffer

	wqBiasDev native.DeviceBuffer
	wkBiasDev native.DeviceBuffer
	wvBiasDev native.DeviceBuffer

	ropeDev   native.DeviceBuffer
	ropeHalf  int
	ropeScale float32
	applyRope bool

	scale   float32
	useGelu bool

	// Hybrid (short-conv + attention) layer support.
	isRecurrent bool
	convIn      deviceMat
	convOut     deviceMat
	convW       native.DeviceBuffer // f32 kernel [hidden, klen]
	convState   native.DeviceBuffer // [(klen-1), hidden], device-resident
	klen        int

	attnGate    deviceMat // zero when the layer has no output gate
	gatePerHead bool
	gateRows    int

	postAttnNormDev native.DeviceBuffer
	postFfnNormDev  native.DeviceBuffer

	window int // sliding attention window (0 = full)
}

type batchedPrefillPlan struct {
	gr   *GraphRuntime
	o    *Ops
	inst *instance.Instance

	// embeds, when non-nil, is parallel to the plan's tokens and overrides the
	// input embedding of positions with a non-nil entry (multimodal image
	// tokens). Applied during the host embedding gather, so the single H2D
	// upload covers the overrides too.
	embeds [][]float32

	n        int
	startPos int
	hidden   int
	nHead    int
	headDim  int
	qDim     int
	vocab    int
	eps      float32
	softcap  float32

	maxKvStride int
	maxInterm   int

	layers []batchedLayerPlan
	buf    batchedBuffers

	outNormDev native.DeviceBuffer
	outWeight  deviceMat

	// matByBuf maps a resolved device weight buffer back to its model matrix
	// so gemmW can route quantized weights to the batched quant kernel.
	matByBuf map[uintptr]*instance.Mat

	// tapRows is the number of hidden-tap rows this forward captures (0 when tap
	// capture is disabled or the buffer is too small). tapDev holds the compact
	// [taps, tapRows, hidden] device copy, laid out like core.HiddenTaps.Host.
	tapRows int
	tapDev  native.DeviceBuffer

	hostEmbed  []float32
	hostLogits []float32
}

func (p *batchedPrefillPlan) free() {
	if p == nil {
		return
	}
	for _, b := range p.buf.all() {
		if b.Ptr() != nil {
			_ = b.Free()
		}
	}
	if p.tapDev.Ptr() != nil {
		_ = p.tapDev.Free()
	}
	for i := range p.layers {
		for _, b := range []native.DeviceBuffer{p.layers[i].wqBiasDev, p.layers[i].wkBiasDev, p.layers[i].wvBiasDev} {
			if b.Ptr() != nil {
				_ = b.Free()
			}
		}
	}
	p.buf = batchedBuffers{}
	p.layers = nil
}

// maxInt is a local int max; the package already defines max(int64,int64)
// for the perf counters.
func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func allocF32Elems(elems int) (native.DeviceBuffer, error) {
	if elems <= 0 {
		return native.DeviceBuffer{}, nil
	}
	return native.AllocDevice(int64(elems) * int64(unsafe.Sizeof(float32(0))))
}

// newBatchedPrefillPlan performs every fallible allocation and validation
// before the compute phase. Nothing here writes to the KV cache, so returning
// an error is always safe to fall back from.
func (gr *GraphRuntime) newBatchedPrefillPlan(tokens []int) (*batchedPrefillPlan, error) {
	return gr.newBatchedPrefillPlanEmbeds(tokens, nil)
}

func (gr *GraphRuntime) newBatchedPrefillPlanEmbeds(tokens []int, embeds [][]float32) (*batchedPrefillPlan, error) {
	if embeds != nil && len(embeds) != len(tokens) {
		return nil, fmt.Errorf("cuda batched prefill: embeds length %d != tokens %d", len(embeds), len(tokens))
	}
	o := gr.ops
	m := gr.inst

	hidden := m.Config.Config.EmbeddingLength
	nHead := m.HeadCount
	headDim := m.HeadDim
	qDim := nHead * headDim

	o.mu.Lock()
	defer o.mu.Unlock()

	// Ensure no stale device-resident scratch from a prior operation is pending.
	if err := o.flushLastResult(); err != nil {
		return nil, fmt.Errorf("cuda batched prefill: flush pending result: %w", err)
	}

	p := &batchedPrefillPlan{
		gr:       gr,
		o:        o,
		inst:     m,
		embeds:   embeds,
		n:        len(tokens),
		startPos: m.Pos,
		hidden:   hidden,
		nHead:    nHead,
		headDim:  headDim,
		qDim:     qDim,
		vocab:    m.Config.Config.VocabSize,
		eps:      m.RMSEpsilon,
	}
	if m.Config != nil {
		p.softcap = m.Config.Config.AttnLogitSoftcap
	}
	fail := func(err error) (*batchedPrefillPlan, error) {
		p.free()
		return nil, err
	}

	// Gather the N embeddings into one host [N,hidden] staging buffer.
	p.hostEmbed = make([]float32, p.n*hidden)
	var embScale float32
	if s := m.Config.Config.EmbeddingMultiplier; s != 0 && s != 1 {
		embScale = float32(s)
	}
	var mupScale float32
	if m.Config.Config.MuPEnabled && m.MuPScale != 1 {
		mupScale = m.MuPScale
	}
	for i, tok := range tokens {
		row := p.hostEmbed[i*hidden : (i+1)*hidden]
		if p.embeds != nil && p.embeds[i] != nil {
			if len(p.embeds[i]) != hidden {
				return fail(fmt.Errorf("cuda batched prefill: embedding override length %d, want %d", len(p.embeds[i]), hidden))
			}
			copy(row, p.embeds[i])
		} else {
			m.Embeddings.RowTo(row, tok)
		}
		if embScale != 0 {
			for j := range row {
				row[j] *= embScale
			}
		}
		if mupScale != 0 {
			for j := range row {
				row[j] *= mupScale
			}
		}
	}

	// Determine the maximum scratch dimensions across layers.
	for i := range m.Layers {
		l := &m.Layers[i]
		if s := l.HeadKV * headDim; s > p.maxKvStride {
			p.maxKvStride = s
		}
		if l.FfnUp.R > p.maxInterm {
			p.maxInterm = l.FfnUp.R
		}
	}
	if p.maxKvStride <= 0 || p.maxInterm <= 0 {
		return fail(fmt.Errorf("cuda batched prefill: invalid layer dimensions"))
	}

	// Allocate all scratch up front.
	allocInto := func(dst *native.DeviceBuffer, elems int, label string) error {
		b, err := allocF32Elems(elems)
		if err != nil {
			return fmt.Errorf("cuda batched prefill: alloc %s (%d elems): %w", label, elems, err)
		}
		*dst = b
		return nil
	}
	if err := allocInto(&p.buf.x, p.n*hidden, "x"); err != nil {
		return fail(err)
	}
	// dp4a activation scratch sized for the widest projection (maxInterm)
	maxK := p.maxInterm
	if hidden > maxK {
		maxK = hidden
	}
	if err := allocInto(&p.buf.xq, p.n*maxK, "xq"); err != nil {
		return fail(err)
	}
	if err := allocInto(&p.buf.xsc, p.n*maxK/32, "xsc"); err != nil {
		return fail(err)
	}
	if err := allocInto(&p.buf.normed, p.n*hidden, "normed"); err != nil {
		return fail(err)
	}
	if err := allocInto(&p.buf.proj, p.n*hidden, "proj"); err != nil {
		return fail(err)
	}
	if err := allocInto(&p.buf.down, p.n*hidden, "down"); err != nil {
		return fail(err)
	}
	if err := allocInto(&p.buf.q, p.n*qDim, "q"); err != nil {
		return fail(err)
	}
	if err := allocInto(&p.buf.attn, p.n*qDim, "attn"); err != nil {
		return fail(err)
	}
	if err := allocInto(&p.buf.k, p.n*p.maxKvStride, "k"); err != nil {
		return fail(err)
	}
	if err := allocInto(&p.buf.v, p.n*p.maxKvStride, "v"); err != nil {
		return fail(err)
	}
	if err := allocInto(&p.buf.up, p.n*p.maxInterm, "up"); err != nil {
		return fail(err)
	}
	if err := allocInto(&p.buf.gate, p.n*p.maxInterm, "gate"); err != nil {
		return fail(err)
	}
	if err := allocInto(&p.buf.act, p.n*p.maxInterm, "act"); err != nil {
		return fail(err)
	}
	if err := allocInto(&p.buf.last, hidden, "last"); err != nil {
		return fail(err)
	}
	if err := allocInto(&p.buf.lastN, hidden, "lastNorm"); err != nil {
		return fail(err)
	}
	if err := allocInto(&p.buf.logits, p.vocab, "logits"); err != nil {
		return fail(err)
	}
	// Hybrid-layer scratch. maxKlen is the widest short-conv kernel in the model.
	maxKlen := 1
	for i := range m.Layers {
		if m.Layers[i].IsRecurrent && m.Layers[i].ShortConvKernel != nil && m.Layers[i].ShortConvKernel.C > maxKlen {
			maxKlen = m.Layers[i].ShortConvKernel.C
		}
	}
	if err := allocInto(&p.buf.scProj, p.n*3*hidden, "scProj"); err != nil {
		return fail(err)
	}
	if err := allocInto(&p.buf.scBx, p.n*hidden, "scBx"); err != nil {
		return fail(err)
	}
	if err := allocInto(&p.buf.scOut, p.n*hidden, "scOut"); err != nil {
		return fail(err)
	}
	scStateElems := (maxKlen - 1) * hidden
	if scStateElems < 0 {
		scStateElems = 0
	}
	if err := allocInto(&p.buf.scState, scStateElems, "scState"); err != nil {
		return fail(err)
	}
	if err := allocInto(&p.buf.gateBuf, p.n*qDim, "gateBuf"); err != nil {
		return fail(err)
	}
	convElems := p.n * maxInt(hidden, p.maxInterm)
	if convElems > 0 {
		b, err := native.AllocDevice(int64(convElems) * 2)
		if err != nil {
			return fail(fmt.Errorf("cuda batched prefill: alloc conv (%d elems): %w", convElems, err))
		}
		p.buf.conv = b
	}

	// Hidden-tap capture, when the caller enabled it: a compact [taps, rows,
	// hidden] device buffer that each tapped layer copies into with a D2D copy.
	// One D2D per tapped layer, one host read at the end, no per-layer host sync.
	// A forward wider than the host buffer captures its first Capacity rows and
	// reports the short row count through HiddenTaps.Rows.
	if taps := &m.HiddenTaps; taps.Enabled() {
		p.tapRows = min(p.n, taps.Capacity)
		if p.tapRows > 0 {
			elems := len(taps.Layers) * p.tapRows * taps.Hidden
			b, err := allocF32Elems(elems)
			if err != nil {
				return fail(fmt.Errorf("cuda batched prefill: alloc hidden taps (%d elems): %w", elems, err))
			}
			p.tapDev = b
		}
	}
	p.hostLogits = make([]float32, p.vocab)

	// Upload the embeddings once.
	if err := native.MemcpyH2D(p.buf.x, unsafe.Pointer(&p.hostEmbed[0]), int64(len(p.hostEmbed))*int64(unsafe.Sizeof(float32(0)))); err != nil {
		return fail(fmt.Errorf("cuda batched prefill: upload embeddings: %w", err))
	}

	// Resolve every device weight, norm, RoPE table and KV cache up front.
	for i := range m.Layers {
		l := &m.Layers[i]
		lp := batchedLayerPlan{
			layer:     l,
			index:     i,
			kvStride:  l.HeadKV * headDim,
			kvHeads:   l.HeadKV,
			interm:    l.FfnUp.R,
			moduloLen: l.AttnCache.CacheLen,
		}
		cacheLen := l.AttnCache.CacheLen
		if o.effectiveContextLen > 0 && cacheLen > o.effectiveContextLen {
			cacheLen = o.effectiveContextLen
		}
		lp.cacheLen = cacheLen

		var err error
		lp.isRecurrent = l.IsRecurrent
		lp.window = l.AttnWindow
		if lp.isRecurrent {
			// Short-conv weights, kernel and the device-resident conv state.
			if lp.convIn, err = p.resolveWeight(l.ShortConvInProj); err != nil {
				return fail(fmt.Errorf("cuda batched prefill: layer %d conv in_proj: %w", i, err))
			}
			p.track(lp.convIn, l.ShortConvInProj)
			if lp.convOut, err = p.resolveWeight(l.ShortConvOutProj); err != nil {
				return fail(fmt.Errorf("cuda batched prefill: layer %d conv out_proj: %w", i, err))
			}
			p.track(lp.convOut, l.ShortConvOutProj)
			if lp.convW, err = o.ensureConvKernel(l.ShortConvKernel); err != nil {
				return fail(fmt.Errorf("cuda batched prefill: layer %d conv kernel: %w", i, err))
			}
			lp.klen = l.ShortConvKernel.C
			stateBytes := int64((lp.klen-1)*hidden) * 4
			cs, err := o.ensureConvState(l, int(stateBytes))
			if err != nil {
				return fail(fmt.Errorf("cuda batched prefill: layer %d conv state: %w", i, err))
			}
			if stateBytes > 0 && !o.convStateReady[l] {
				// First use of this layer: seed the device state from the host
				// buffer (zeroed on load/reset), exactly as the per-token path.
				if err := native.MemcpyH2DAsync(cs.buf, unsafe.Pointer(&l.ShortConvState.Buf[0]), stateBytes, o.stream); err != nil {
					return fail(fmt.Errorf("cuda batched prefill: layer %d conv state upload: %w", i, err))
				}
			}
			o.convStateReady[l] = true
			lp.convState = cs.buf
		} else {
			if lp.wq, err = p.resolveWeight(l.Wq); err != nil {
				return fail(fmt.Errorf("cuda batched prefill: layer %d Wq: %w", i, err))
			}
			p.track(lp.wq, l.Wq)
			if lp.wk, err = p.resolveWeight(l.Wk); err != nil {
				return fail(fmt.Errorf("cuda batched prefill: layer %d Wk: %w", i, err))
			}
			p.track(lp.wk, l.Wk)
			if lp.wv, err = p.resolveWeight(l.Wv); err != nil {
				return fail(fmt.Errorf("cuda batched prefill: layer %d Wv: %w", i, err))
			}
			p.track(lp.wv, l.Wv)
			if lp.wo, err = p.resolveWeight(l.Wo); err != nil {
				return fail(fmt.Errorf("cuda batched prefill: layer %d Wo: %w", i, err))
			}
			p.track(lp.wo, l.Wo)

			apply, invFreq, ropeScale := batchedLayerRoPE(m, l)
			lp.applyRope = apply
			lp.ropeScale = ropeScale
			if apply {
				if lp.ropeDev, lp.ropeHalf, err = o.ensureRoPEInvFreqDev(invFreq); err != nil {
					return fail(fmt.Errorf("cuda batched prefill: layer %d RoPE: %w", i, err))
				}
			}

			cache, err := o.ensureAttnCache(l, lp.kvStride, cacheLen)
			if err != nil {
				return fail(fmt.Errorf("cuda batched prefill: layer %d attn cache: %w", i, err))
			}
			if cache.kvStride != lp.kvStride || cache.cacheLen != cacheLen {
				return fail(fmt.Errorf("cuda batched prefill: layer %d attn cache mismatch", i))
			}
			lp.cache = cache
			lp.useQ8K = cache.useQ8K
			lp.useQ8V = cache.useQ8V
			lp.blocksPerRow = lp.kvStride / int(mcf.QuantBlockSize)
			if lp.blocksPerRow < 1 {
				lp.blocksPerRow = 1
			}

			lp.scale = l.AttnScale
			if lp.scale == 0 {
				lp.scale = float32(1.0 / math.Sqrt(float64(headDim)))
			}
		}
		if lp.up, err = p.resolveWeight(l.FfnUp); err != nil {
			return fail(fmt.Errorf("cuda batched prefill: layer %d FfnUp: %w", i, err))
		}
		p.track(lp.up, l.FfnUp)
		if lp.gate, err = p.resolveWeight(l.FfnGate); err != nil {
			return fail(fmt.Errorf("cuda batched prefill: layer %d FfnGate: %w", i, err))
		}
		p.track(lp.gate, l.FfnGate)
		if lp.down, err = p.resolveWeight(l.FfnDown); err != nil {
			return fail(fmt.Errorf("cuda batched prefill: layer %d FfnDown: %w", i, err))
		}
		p.track(lp.down, l.FfnDown)
		if lp.attnNormDev, err = o.deviceNormWeight(l.AttnNorm); err != nil {
			return fail(fmt.Errorf("cuda batched prefill: layer %d AttnNorm: %w", i, err))
		}
		if lp.ffnNormDev, err = o.deviceNormWeight(l.FfnNorm); err != nil {
			return fail(fmt.Errorf("cuda batched prefill: layer %d FfnNorm: %w", i, err))
		}
		if len(l.AttnQNorm) > 0 && !lp.isRecurrent {
			if lp.qNormDev, err = o.deviceNormWeight(l.AttnQNorm); err != nil {
				return fail(fmt.Errorf("cuda batched prefill: layer %d AttnQNorm: %w", i, err))
			}
		}
		if len(l.AttnKNorm) > 0 && !lp.isRecurrent {
			if lp.kNormDev, err = o.deviceNormWeight(l.AttnKNorm); err != nil {
				return fail(fmt.Errorf("cuda batched prefill: layer %d AttnKNorm: %w", i, err))
			}
		}
		if l.AttnGate != nil {
			if lp.attnGate, err = p.resolveWeight(l.AttnGate); err != nil {
				return fail(fmt.Errorf("cuda batched prefill: layer %d AttnGate: %w", i, err))
			}
			p.track(lp.attnGate, l.AttnGate)
			lp.gateRows = l.AttnGate.R
			lp.gatePerHead = l.AttnGate.R == m.HeadCount && m.HeadCount != qDim
		}
		if len(l.PostAttnNorm) > 0 {
			if lp.postAttnNormDev, err = o.deviceNormWeight(l.PostAttnNorm); err != nil {
				return fail(fmt.Errorf("cuda batched prefill: layer %d PostAttnNorm: %w", i, err))
			}
		}
		if len(l.PostFfnNorm) > 0 {
			if lp.postFfnNormDev, err = o.deviceNormWeight(l.PostFfnNorm); err != nil {
				return fail(fmt.Errorf("cuda batched prefill: layer %d PostFfnNorm: %w", i, err))
			}
		}
		lp.useGelu = strings.Contains(l.FFNActivation, "gelu")

		if len(l.WqBias) > 0 {
			if lp.wqBiasDev, err = p.uploadBroadcastBias(l.WqBias); err != nil {
				return fail(fmt.Errorf("cuda batched prefill: layer %d WqBias: %w", i, err))
			}
		}
		if len(l.WkBias) > 0 {
			if lp.wkBiasDev, err = p.uploadBroadcastBias(l.WkBias); err != nil {
				return fail(fmt.Errorf("cuda batched prefill: layer %d WkBias: %w", i, err))
			}
		}
		if len(l.WvBias) > 0 {
			if lp.wvBiasDev, err = p.uploadBroadcastBias(l.WvBias); err != nil {
				return fail(fmt.Errorf("cuda batched prefill: layer %d WvBias: %w", i, err))
			}
		}

		p.layers = append(p.layers, lp)
	}

	// Output head weight and norm, resolved up front so run() performs no
	// fallible allocations after the first KV write.
	outNormDev, err := o.deviceNormWeight(m.OutputNorm)
	if err != nil {
		return fail(fmt.Errorf("cuda batched prefill: output norm: %w", err))
	}
	p.outNormDev = outNormDev
	outWeight, err := p.resolveWeight(m.Output)
	if err != nil {
		return fail(fmt.Errorf("cuda batched prefill: output weight: %w", err))
	}
	p.outWeight = outWeight
	p.track(outWeight, m.Output)

	return p, nil
}

func (p *batchedPrefillPlan) uploadBroadcastBias(bias []float32) (native.DeviceBuffer, error) {
	if len(bias) == 0 {
		return native.DeviceBuffer{}, nil
	}
	host := make([]float32, p.n*len(bias))
	for i := range p.n {
		copy(host[i*len(bias):], bias)
	}
	buf, err := native.AllocDevice(int64(len(host)) * int64(unsafe.Sizeof(float32(0))))
	if err != nil {
		return native.DeviceBuffer{}, err
	}
	if err := native.MemcpyH2D(buf, unsafe.Pointer(&host[0]), int64(len(host))*int64(unsafe.Sizeof(float32(0)))); err != nil {
		_ = buf.Free()
		return native.DeviceBuffer{}, err
	}
	return buf, nil
}

// run executes the batched compute. Every buffer and weight has already been
// allocated and validated, so any error here is unexpected.
func (p *batchedPrefillPlan) run() ([]float32, error) {
	o := p.o
	o.mu.Lock()
	defer o.mu.Unlock()

	// The embedding output (layer -1) is still in p.buf.x before the first layer
	// overwrites it, so copy it now when that tap is configured.
	if p.tapRows > 0 {
		if slot := p.inst.HiddenTaps.SlotFor(-1); slot >= 0 {
			if err := p.copyTapSlot(slot, p.buf.x); err != nil {
				return nil, err
			}
		}
	}

	for i := range p.layers {
		if err := p.runLayer(&p.layers[i]); err != nil {
			return nil, err
		}
	}

	// Recurrent layers rotated their conv state on device during the batch.
	// Copy it back to the host buffer so the per-token CPU fallback (used when
	// the weights are not quantized) decodes from the same state; the device
	// copy stays valid for the quant fast path via convStateReady.
	for i := range p.layers {
		lp := &p.layers[i]
		if !lp.isRecurrent || lp.klen <= 1 {
			continue
		}
		stateBytes := int64((lp.klen-1)*p.hidden) * int64(unsafe.Sizeof(float32(0)))
		host := lp.layer.ShortConvState.Buf
		if len(host)*int(unsafe.Sizeof(float32(0))) < int(stateBytes) {
			return nil, fmt.Errorf("cuda batched prefill: layer %d conv state host buffer too small", i)
		}
		if err := native.MemcpyD2HAsync(unsafe.Pointer(&host[0]), lp.convState, stateBytes, o.stream); err != nil {
			return nil, fmt.Errorf("cuda batched prefill: layer %d conv state writeback to host: %w", i, err)
		}
	}

	// The whole tap set crosses to the host once, after the layer loop, so no
	// tapped layer pays a host sync.
	if p.tapRows > 0 {
		if err := p.readTaps(); err != nil {
			return nil, err
		}
	}

	// Output head for the last row only, matching the sequential path's final
	// ForwardToken: RMSNorm then a device matvec on m.Output.
	lastBytes := int64(p.hidden) * int64(unsafe.Sizeof(float32(0)))
	lastRow := devSub(p.buf.x, (p.n-1)*p.hidden*int(unsafe.Sizeof(float32(0))))
	if err := native.MemcpyD2DAsync(p.buf.last, lastRow, lastBytes, o.stream); err != nil {
		return nil, fmt.Errorf("cuda batched prefill: stage last row: %w", err)
	}
	if err := native.RMSNormF32(p.buf.lastN, p.buf.last, p.outNormDev, p.eps, p.hidden, o.stream); err != nil {
		return nil, fmt.Errorf("cuda batched prefill: output norm kernel: %w", err)
	}
	if err := p.gemmLastW(p.outWeight, p.buf.lastN, p.hidden, p.buf.logits, p.vocab); err != nil {
		return nil, err
	}
	if err := native.MemcpyD2H(unsafe.Pointer(&p.hostLogits[0]), p.buf.logits, int64(p.vocab)*int64(unsafe.Sizeof(float32(0)))); err != nil {
		return nil, fmt.Errorf("cuda batched prefill: read logits: %w", err)
	}
	batchedLogitPostprocess(p.hostLogits, p.inst)

	p.inst.Pos = p.startPos + p.n
	return p.hostLogits, nil
}

func (p *batchedPrefillPlan) runLayer(lp *batchedLayerPlan) error {
	o := p.o
	stream := o.stream

	// Pre-attention norm over all rows.
	if err := native.RMSNormBatchedF32(p.buf.normed, p.buf.x, lp.attnNormDev, p.eps, p.hidden, p.n, stream); err != nil {
		return fmt.Errorf("cuda batched prefill: attn norm: %w", err)
	}
	if lp.isRecurrent {
		if err := p.runShortConvBlock(lp); err != nil {
			return err
		}
	} else {
		if err := p.runAttentionBlock(lp); err != nil {
			return err
		}
	}
	// Sandwich (post-operator) norm, then the scaled residual add.
	if lp.postAttnNormDev.Ptr() != nil {
		if err := native.RMSNormBatchedF32(p.buf.proj, p.buf.proj, lp.postAttnNormDev, p.eps, p.hidden, p.n, stream); err != nil {
			return fmt.Errorf("cuda batched prefill: post-attn norm: %w", err)
		}
	}
	if err := p.residualAdd(p.buf.proj); err != nil {
		return err
	}

	// FFN.
	if err := native.RMSNormBatchedF32(p.buf.normed, p.buf.x, lp.ffnNormDev, p.eps, p.hidden, p.n, stream); err != nil {
		return fmt.Errorf("cuda batched prefill: ffn norm: %w", err)
	}
	if err := p.gemmW(lp.up, p.buf.normed, p.hidden, p.buf.up, lp.interm); err != nil {
		return err
	}
	if err := p.gemmW(lp.gate, p.buf.normed, p.hidden, p.buf.gate, lp.interm); err != nil {
		return err
	}
	if lp.useGelu {
		if err := native.GeluMulF32(p.buf.gate, p.buf.up, p.buf.act, p.n*lp.interm, stream); err != nil {
			return fmt.Errorf("cuda batched prefill: gelu mul: %w", err)
		}
	} else {
		if err := native.SiluMulF32(p.buf.gate, p.buf.up, p.buf.act, p.n*lp.interm, stream); err != nil {
			return fmt.Errorf("cuda batched prefill: silu mul: %w", err)
		}
	}
	if err := p.gemmW(lp.down, p.buf.act, lp.interm, p.buf.down, p.hidden); err != nil {
		return err
	}
	if lp.postFfnNormDev.Ptr() != nil {
		if err := native.RMSNormBatchedF32(p.buf.down, p.buf.down, lp.postFfnNormDev, p.eps, p.hidden, p.n, stream); err != nil {
			return fmt.Errorf("cuda batched prefill: post-ffn norm: %w", err)
		}
	}
	if err := p.residualAdd(p.buf.down); err != nil {
		return err
	}

	// p.buf.x is now this layer's post-FFN residual. Copy the captured rows on
	// device; the host read happens once, after the whole layer loop.
	if p.tapRows > 0 {
		if slot := p.inst.HiddenTaps.SlotFor(lp.index); slot >= 0 {
			if err := p.copyTapSlot(slot, p.buf.x); err != nil {
				return err
			}
		}
	}
	return nil
}

// residualAdd adds src ([N, hidden]) to the residual p.buf.x, scaled by the
// model's residual multiplier when set (Granite scales branch outputs).
func (p *batchedPrefillPlan) residualAdd(src native.DeviceBuffer) error {
	o := p.o
	scale := p.inst.Config.Config.ResidualScale()
	n := p.n * p.hidden
	if scale == 1 {
		if err := native.AddVectorsF32(p.buf.x, src, n, o.stream); err != nil {
			return fmt.Errorf("cuda batched prefill: residual: %w", err)
		}
		return nil
	}
	if err := native.ScaleAddF32(p.buf.x, src, scale, n, o.stream); err != nil {
		return fmt.Errorf("cuda batched prefill: scaled residual: %w", err)
	}
	return nil
}

// runShortConvBlock runs a recurrent (short-conv) layer over all N rows:
// in_proj GEMM -> batched causal conv (reading the chunk rows plus the carried
// device state) -> state rotation -> out_proj GEMM, leaving the block output in
// p.buf.proj. It mirrors simd's ShortConv for a whole chunk.
func (p *batchedPrefillPlan) runShortConvBlock(lp *batchedLayerPlan) error {
	o := p.o
	stream := o.stream
	hidden := p.hidden
	// in_proj: [N, hidden] -> [N, 3*hidden] (b|c|xg per row).
	if err := p.gemmW(lp.convIn, p.buf.normed, hidden, p.buf.scProj, 3*hidden); err != nil {
		return err
	}
	// Causal conv over the chunk; scBx receives the gated inputs for the
	// state rotation, scOut the c*convOut rows (out_proj input).
	if err := native.ShortConvDepthwiseBatch(p.buf.scProj, lp.convW, lp.convState, p.buf.scOut, p.buf.scBx, p.n, hidden, lp.klen, stream); err != nil {
		return fmt.Errorf("cuda batched prefill: short-conv (layer %d): %w", lp.index, err)
	}
	// Rotate the device state forward by N rows (via staging to avoid races).
	if lp.klen > 1 {
		stateBytes := int64((lp.klen-1)*hidden) * 4
		if err := native.ShortConvStateRotate(lp.convState, p.buf.scBx, p.buf.scState, p.n, hidden, lp.klen, stream); err != nil {
			return fmt.Errorf("cuda batched prefill: conv state rotate (layer %d): %w", lp.index, err)
		}
		if err := native.MemcpyD2DAsync(lp.convState, p.buf.scState, stateBytes, stream); err != nil {
			return fmt.Errorf("cuda batched prefill: conv state writeback (layer %d): %w", lp.index, err)
		}
	}
	// out_proj: [N, hidden] -> [N, hidden] into proj.
	if err := p.gemmW(lp.convOut, p.buf.scOut, hidden, p.buf.proj, hidden); err != nil {
		return err
	}
	return nil
}

// runAttentionBlock runs the attention block for all N rows, leaving the Wo
// output in p.buf.proj. Sliding windows clamp each row's start to
// max(pos-window+1, 0); the output gate (per-head or per-element) is applied
// before Wo.
func (p *batchedPrefillPlan) runAttentionBlock(lp *batchedLayerPlan) error {
	o := p.o
	stream := o.stream

	// Q/K/V projections.
	if err := p.gemmW(lp.wq, p.buf.normed, p.hidden, p.buf.q, p.qDim); err != nil {
		return err
	}
	if err := p.gemmW(lp.wk, p.buf.normed, p.hidden, p.buf.k, lp.kvStride); err != nil {
		return err
	}
	if err := p.gemmW(lp.wv, p.buf.normed, p.hidden, p.buf.v, lp.kvStride); err != nil {
		return err
	}
	if lp.wqBiasDev.Ptr() != nil {
		if err := native.AddVectorsF32(p.buf.q, lp.wqBiasDev, p.n*p.qDim, stream); err != nil {
			return fmt.Errorf("cuda batched prefill: q bias: %w", err)
		}
	}
	if lp.wkBiasDev.Ptr() != nil {
		if err := native.AddVectorsF32(p.buf.k, lp.wkBiasDev, p.n*lp.kvStride, stream); err != nil {
			return fmt.Errorf("cuda batched prefill: k bias: %w", err)
		}
	}
	if lp.wvBiasDev.Ptr() != nil {
		if err := native.AddVectorsF32(p.buf.v, lp.wvBiasDev, p.n*lp.kvStride, stream); err != nil {
			return fmt.Errorf("cuda batched prefill: v bias: %w", err)
		}
	}
	// Per-head Q/K norms.
	if lp.qNormDev.Ptr() != nil {
		if err := native.RMSNormBatchedF32(p.buf.q, p.buf.q, lp.qNormDev, p.eps, p.headDim, p.n*p.nHead, stream); err != nil {
			return fmt.Errorf("cuda batched prefill: q norm: %w", err)
		}
	}
	if lp.kNormDev.Ptr() != nil {
		if err := native.RMSNormBatchedF32(p.buf.k, p.buf.k, lp.kNormDev, p.eps, p.headDim, p.n*lp.kvHeads, stream); err != nil {
			return fmt.Errorf("cuda batched prefill: k norm: %w", err)
		}
	}

	if p.prefillAttentionOK(lp) {
		if err := p.runPrefillAttention(lp); err != nil {
			return err
		}
	} else {
		// Per-row RoPE and KV store.
		//
		// Ring constraint: the batched attention kernel needs every row resident
		// before it reads, so store-all-then-attend is only equivalent to the
		// sequential oracle when the batch does not wrap the cache. Once startPos+N
		// exceeds cacheLen, later stores overwrite slots that earlier queries still
		// read (position t lives at t%cacheLen), so those rows must interleave
		// store-then-attend exactly as the sequential path does.
		wrapInBatch := p.startPos+p.n > lp.cacheLen
		qRowBytes := p.qDim * int(unsafe.Sizeof(float32(0)))
		kRowBytes := lp.kvStride * int(unsafe.Sizeof(float32(0)))
		for i := range p.n {
			pos := p.startPos + i
			qRow := devSub(p.buf.q, i*qRowBytes)
			kRow := devSub(p.buf.k, i*kRowBytes)
			if lp.applyRope {
				if err := native.ApplyRoPEInplaceF32(qRow, lp.ropeDev, pos, lp.ropeScale, p.headDim, lp.ropeHalf, p.nHead, stream); err != nil {
					return fmt.Errorf("cuda batched prefill: q rope (pos=%d): %w", pos, err)
				}
				if err := native.ApplyRoPEInplaceF32(kRow, lp.ropeDev, pos, lp.ropeScale, p.headDim, lp.ropeHalf, lp.kvHeads, stream); err != nil {
					return fmt.Errorf("cuda batched prefill: k rope (pos=%d): %w", pos, err)
				}
			}
			cachePos := pos % lp.moduloLen
			if lp.useQ8K {
				if err := native.StoreKVQ8RowBroadcast(lp.cache.kQ8, lp.cache.kQ8Scales, kRow, cachePos, lp.kvStride, lp.blocksPerRow, stream); err != nil {
					return fmt.Errorf("cuda batched prefill: store K (pos=%d): %w", pos, err)
				}
			} else {
				if err := native.StoreKVF16Row(lp.cache.kF16, kRow, cachePos, lp.kvStride, stream); err != nil {
					return fmt.Errorf("cuda batched prefill: store K (pos=%d): %w", pos, err)
				}
			}
			vRow := devSub(p.buf.v, i*kRowBytes)
			if lp.useQ8V {
				if err := native.StoreKVQ8RowBroadcast(lp.cache.vQ8, lp.cache.vQ8Scales, vRow, cachePos, lp.kvStride, lp.blocksPerRow, stream); err != nil {
					return fmt.Errorf("cuda batched prefill: store V (pos=%d): %w", pos, err)
				}
			} else {
				if err := native.StoreKVF16Row(lp.cache.vF16, vRow, cachePos, lp.kvStride, stream); err != nil {
					return fmt.Errorf("cuda batched prefill: store V (pos=%d): %w", pos, err)
				}
			}

			if wrapInBatch {
				attnRow := devSub(p.buf.attn, i*qRowBytes)
				start := 0
				if lp.window > 0 && pos-lp.window+1 > 0 {
					start = pos - lp.window + 1
				}
				if lp.useQ8K || lp.useQ8V {
					if err := native.AttentionInnerMixedCacheF32(
						qRow, lp.cache.kF16, lp.cache.vF16,
						lp.cache.kQ8, lp.cache.vQ8, lp.cache.kQ8Scales, lp.cache.vQ8Scales,
						attnRow, lp.useQ8K, lp.useQ8V,
						pos, start, lp.kvStride, p.headDim, p.nHead, lp.kvHeads,
						lp.cacheLen, lp.scale, p.softcap, stream,
					); err != nil {
						return fmt.Errorf("cuda batched prefill: attention (pos=%d): %w", pos, err)
					}
				} else {
					if err := native.AttentionInnerF16CacheF32(
						qRow, lp.cache.kF16, lp.cache.vF16, attnRow,
						pos, start, lp.kvStride, p.headDim, p.nHead, lp.kvHeads,
						lp.cacheLen, lp.scale, p.softcap, stream,
					); err != nil {
						return fmt.Errorf("cuda batched prefill: attention (pos=%d): %w", pos, err)
					}
				}
			}
		}

		// Attention for all N rows in a single launch, valid because no row was
		// overwritten during the stores above.
		if !wrapInBatch {
			if lp.useQ8K || lp.useQ8V {
				if err := native.AttentionInnerMixedCacheF32Batch(
					p.buf.q, lp.cache.kF16, lp.cache.vF16,
					lp.cache.kQ8, lp.cache.vQ8, lp.cache.kQ8Scales, lp.cache.vQ8Scales,
					p.buf.attn, lp.useQ8K, lp.useQ8V,
					p.n, p.startPos, lp.kvStride, p.headDim, p.nHead, lp.kvHeads,
					lp.cacheLen, lp.window, lp.scale, p.softcap, stream,
				); err != nil {
					return fmt.Errorf("cuda batched prefill: batched attention: %w", err)
				}
			} else {
				if err := native.AttentionInnerF16CacheF32Batch(
					p.buf.q, lp.cache.kF16, lp.cache.vF16, p.buf.attn,
					p.n, p.startPos, lp.kvStride, p.headDim, p.nHead, lp.kvHeads,
					lp.cacheLen, lp.window, lp.scale, p.softcap, stream,
				); err != nil {
					return fmt.Errorf("cuda batched prefill: batched attention: %w", err)
				}
			}
		}
	}

	// Attention output gate (LFM3-style): gate projection over the normed
	// input per row, sigmoid-scaled onto the attention output before Wo.
	if lp.attnGate.buf.Ptr() != nil {
		if err := p.gemmW(lp.attnGate, p.buf.normed, p.hidden, p.buf.gateBuf, lp.gateRows); err != nil {
			return err
		}
		if err := native.AttnGateScaleF32(p.buf.attn, p.buf.gateBuf, p.n, p.qDim, p.nHead, p.headDim, lp.gatePerHead, stream); err != nil {
			return fmt.Errorf("cuda batched prefill: attn gate (layer %d): %w", lp.index, err)
		}
	}

	// Attention output projection (the residual add happens in runLayer).
	if err := p.gemmW(lp.wo, p.buf.attn, p.qDim, p.buf.proj, p.hidden); err != nil {
		return err
	}

	// Record the last position written to this layer's device KV cache,
	// exactly as the device fast paths do. The batched path always writes the
	// whole batch, so this is startPos+N-1.
	o.lastDevKVPos[lp.layer] = p.startPos + p.n - 1
	return nil
}

// prefillAttentionOK reports whether a layer's attention can run through
// AttentionPrefillF16, which reads positions before the chunk from the cache
// (slot t % cacheLen) and the chunk's own positions from its K/V rows, then
// stores the rows (slot t % moduloLen). Every position a row attends to must
// still be resident at the slot it is read from. That holds when the chunk
// ends within both lengths, so no slot wraps, or when a sliding window fits a
// ring whose read and store lengths agree.
func (p *batchedPrefillPlan) prefillAttentionOK(lp *batchedLayerPlan) bool {
	if lp.useQ8K || lp.useQ8V || lp.cache.kF16.Ptr() == nil || lp.cache.vF16.Ptr() == nil {
		return false
	}
	noWrap := p.startPos+p.n <= min(lp.cacheLen, lp.moduloLen)
	windowRing := lp.window > 0 && lp.window <= lp.moduloLen && lp.cacheLen == lp.moduloLen
	if !noWrap && !windowRing {
		return false
	}
	return native.AttentionPrefillF16Supported(p.headDim, p.nHead, lp.kvHeads)
}

// runPrefillAttention applies RoPE to every Q and K row, attends all rows in
// one launch, and only then stores the chunk's K/V rows into the cache, so a
// chunk longer than a sliding-window ring needs no per-row interleaving. Only
// the last moduloLen rows are stored; earlier ones would be overwritten
// within the chunk anyway.
func (p *batchedPrefillPlan) runPrefillAttention(lp *batchedLayerPlan) error {
	stream := p.o.stream
	if lp.applyRope {
		if err := native.ApplyRoPERowsF32(p.buf.q, lp.ropeDev, p.startPos, lp.ropeScale, p.headDim, lp.ropeHalf, p.nHead, p.n, stream); err != nil {
			return fmt.Errorf("cuda batched prefill: q rope (layer %d): %w", lp.index, err)
		}
		if err := native.ApplyRoPERowsF32(p.buf.k, lp.ropeDev, p.startPos, lp.ropeScale, p.headDim, lp.ropeHalf, lp.kvHeads, p.n, stream); err != nil {
			return fmt.Errorf("cuda batched prefill: k rope (layer %d): %w", lp.index, err)
		}
	}
	if err := native.AttentionPrefillF16(
		p.buf.q, p.buf.k, p.buf.v, lp.cache.kF16, lp.cache.vF16, p.buf.attn,
		p.n, p.startPos, lp.kvStride, p.headDim, p.nHead, lp.kvHeads,
		lp.cacheLen, lp.window, lp.scale, p.softcap, stream,
	); err != nil {
		return fmt.Errorf("cuda batched prefill: prefill attention (layer %d): %w", lp.index, err)
	}
	row0 := maxInt(0, p.n-lp.moduloLen)
	if err := native.StoreKVF16Rows(lp.cache.kF16, lp.cache.vF16, p.buf.k, p.buf.v, p.startPos, row0, p.n, lp.kvStride, lp.moduloLen, stream); err != nil {
		return fmt.Errorf("cuda batched prefill: store KV rows (layer %d): %w", lp.index, err)
	}
	return nil
}

// copyTapSlot issues the device-to-device copy of the first tapRows rows of src
// into the compact tap buffer's slot region. It performs no host sync.
func (p *batchedPrefillPlan) copyTapSlot(slot int, src native.DeviceBuffer) error {
	taps := &p.inst.HiddenTaps
	rowBytes := taps.Hidden * int(unsafe.Sizeof(float32(0)))
	bytes := int64(p.tapRows * taps.Hidden * int(unsafe.Sizeof(float32(0))))
	dst := devSub(p.tapDev, slot*p.tapRows*rowBytes)
	if err := native.MemcpyD2DAsync(dst, src, bytes, p.o.stream); err != nil {
		return fmt.Errorf("cuda batched prefill: copy hidden tap slot %d: %w", slot, err)
	}
	return nil
}

// readTaps copies the captured rows from device into core.HiddenTaps.Host and
// publishes the captured row count. It is one D2H per tap in the common case
// where the forward fits the buffer, and one per tap only when the forward was
// wider than Capacity.
func (p *batchedPrefillPlan) readTaps() error {
	taps := &p.inst.HiddenTaps
	elem := int(unsafe.Sizeof(float32(0)))
	if len(taps.Host) == 0 {
		return nil
	}
	if p.tapRows == taps.Capacity {
		// The compact device layout matches the host layout exactly.
		bytes := int64(len(taps.Layers) * p.tapRows * taps.Hidden * elem)
		if err := native.MemcpyD2H(unsafe.Pointer(&taps.Host[0]), p.tapDev, bytes); err != nil {
			return fmt.Errorf("cuda batched prefill: read hidden taps: %w", err)
		}
		taps.Rows = p.tapRows
		return nil
	}
	rowBytes := taps.Hidden * elem
	for slot := range taps.Layers {
		dst := unsafe.Pointer(&taps.Host[slot*taps.Capacity*taps.Hidden])
		src := devSub(p.tapDev, slot*p.tapRows*rowBytes)
		bytes := int64(p.tapRows * rowBytes)
		if err := native.MemcpyD2H(dst, src, bytes); err != nil {
			return fmt.Errorf("cuda batched prefill: read hidden tap slot %d: %w", slot, err)
		}
	}
	taps.Rows = p.tapRows
	return nil
}

// gemm runs rows = W @ xBatch for all N rows. xF32 is [N,k] row-major and dst
// is [N,rows] row-major; both are laid out such that cublasGemmEx with
// transA=T, transB=N sees the intended shapes.
func (p *batchedPrefillPlan) gemm(devW deviceMat, xF32 native.DeviceBuffer, k int, dst native.DeviceBuffer, rows int) error {
	o := p.o
	x := xF32
	xt := native.BlasF32
	switch devW.dtype {
	case native.BlasF32:
	case native.BlasBF16:
		if err := native.ConvertF32ToBF16(xF32, p.buf.conv, p.n*k, o.stream); err != nil {
			return fmt.Errorf("cuda batched prefill: convert input to bf16 (n=%d): %w", p.n*k, err)
		}
		x = p.buf.conv
		xt = native.BlasBF16
	case native.BlasF16:
		// Dequantized quant weights: F16 weight x F16 input, F32 compute.
		if err := native.ConvertF32ToF16(xF32, p.buf.conv, p.n*k, o.stream); err != nil {
			return fmt.Errorf("cuda batched prefill: convert input to f16 (n=%d): %w", p.n*k, err)
		}
		x = p.buf.conv
		xt = native.BlasF16
	default:
		return fmt.Errorf("cuda batched prefill: unsupported weight dtype %d", devW.dtype)
	}
	if err := native.GemmEx(o.blas, native.BlasOpT, native.BlasOpN, rows, p.n, k, 1.0,
		devW.buf, devW.dtype, k, x, xt, k, 0.0, dst, native.BlasF32, rows,
		native.BlasComputeF32, native.BlasGemmDefault); err != nil {
		return fmt.Errorf("cuda batched prefill: gemm (m=%d n=%d k=%d): %w", rows, p.n, k, err)
	}
	return nil
}

// gemmLast is the n=1 variant used for the output head.
func (p *batchedPrefillPlan) gemmLast(devW deviceMat, xF32 native.DeviceBuffer, k int, dst native.DeviceBuffer, rows int) error {
	o := p.o
	x := xF32
	xt := native.BlasF32
	switch devW.dtype {
	case native.BlasF32:
	case native.BlasBF16:
		if err := native.ConvertF32ToBF16(xF32, p.buf.conv, k, o.stream); err != nil {
			return fmt.Errorf("cuda batched prefill: convert output input to bf16 (n=%d): %w", k, err)
		}
		x = p.buf.conv
		xt = native.BlasBF16
	case native.BlasF16:
		if err := native.ConvertF32ToF16(xF32, p.buf.conv, k, o.stream); err != nil {
			return fmt.Errorf("cuda batched prefill: convert output input to f16 (n=%d): %w", k, err)
		}
		x = p.buf.conv
		xt = native.BlasF16
	default:
		return fmt.Errorf("cuda batched prefill: unsupported output weight dtype %d", devW.dtype)
	}
	if err := native.GemmEx(o.blas, native.BlasOpT, native.BlasOpN, rows, 1, k, 1.0,
		devW.buf, devW.dtype, k, x, xt, k, 0.0, dst, native.BlasF32, rows,
		native.BlasComputeF32, native.BlasGemmDefault); err != nil {
		return fmt.Errorf("cuda batched prefill: output gemm (m=%d k=%d): %w", rows, k, err)
	}
	return nil
}

// track records which model matrix a resolved device weight buffer came from.
func (p *batchedPrefillPlan) track(devW deviceMat, w *instance.Mat) {
	if p.matByBuf == nil {
		p.matByBuf = make(map[uintptr]*instance.Mat)
	}
	if devW.buf.Ptr() != nil && w != nil {
		p.matByBuf[uintptr(devW.buf.Ptr())] = w
	}
}

// resolveWeight resolves a model weight for the batched plan: supported
// quantized weights get their cached-int8 device copy (no dequant), dense
// weights the usual deviceMat upload.
func (p *batchedPrefillPlan) resolveWeight(w *instance.Mat) (deviceMat, error) {
	if batchedQuantSupported(p.o, w) {
		qw, err := p.o.ensureBatchQuantMat(w)
		if err != nil {
			return deviceMat{}, err
		}
		devW := deviceMat{buf: qw.q, dtype: native.BlasF32, rows: w.R, cols: w.C}
		p.track(devW, w)
		return devW, nil
	}
	devW, err := p.o.deviceMat(w)
	if err != nil {
		return deviceMat{}, err
	}
	p.track(devW, w)
	return devW, nil
}

// gemmW runs the batched projection for a weight, using the quant GEMM kernel
// when the source matrix is a supported quantized weight.
func (p *batchedPrefillPlan) gemmW(devW deviceMat, xF32 native.DeviceBuffer, k int, dst native.DeviceBuffer, rows int) error {
	if w, ok := p.matByBuf[uintptr(devW.buf.Ptr())]; ok && batchedQuantSupported(p.o, w) {
		qw, err := p.o.ensureBatchQuantMat(w)
		if err != nil {
			return fmt.Errorf("cuda batched prefill: quant weight upload: %w", err)
		}
		if err := p.quantGemm(qw, xF32, dst, rows, p.n, k); err != nil {
			return fmt.Errorf("cuda batched prefill: quant batch gemm (m=%d n=%d k=%d): %w", rows, p.n, k, err)
		}
		return nil
	}
	return p.gemm(devW, xF32, k, dst, rows)
}

// gemmLastW is the n=1 output-head variant of gemmW.
func (p *batchedPrefillPlan) gemmLastW(devW deviceMat, xF32 native.DeviceBuffer, k int, dst native.DeviceBuffer, rows int) error {
	if w, ok := p.matByBuf[uintptr(devW.buf.Ptr())]; ok && batchedQuantSupported(p.o, w) {
		qw, err := p.o.ensureBatchQuantMat(w)
		if err != nil {
			return fmt.Errorf("cuda batched prefill: quant output upload: %w", err)
		}
		if err := p.quantGemm(qw, xF32, dst, rows, 1, k); err != nil {
			return fmt.Errorf("cuda batched prefill: quant output gemm (m=%d k=%d): %w", rows, k, err)
		}
		return nil
	}
	return p.gemmLast(devW, xF32, k, dst, rows)
}

// quantMMAMinRows is the smallest token count sent to the tensor-core quant
// GEMM. Below it most of each 64-token tile would be padding, and the dp4a
// kernel, whose output is identical, is cheaper.
const quantMMAMinRows = 16

var quantMMASupported = sync.OnceValue(native.QuantBatchMMASupported)

// quantGemm computes dst[n, rows] = x[n, k] * qwᵀ for a cached-int8 quant
// weight, on tensor cores when the device and shape allow it and with the
// dp4a kernel otherwise. Both produce the same bits.
func (p *batchedPrefillPlan) quantGemm(qw deviceQuantMat, x, dst native.DeviceBuffer, rows, n, k int) error {
	if n >= quantMMAMinRows && k == qw.blocksPerRow*32 && quantMMASupported() &&
		native.QuantBatchMMAAligned(qw.q, p.buf.xq) {
		return native.QuantMatVecCachedBatchMMA(qw.q, qw.scales, x, p.buf.xq, p.buf.xsc, dst, rows, n, k, qw.blocksPerRow, p.o.stream)
	}
	return native.QuantMatVecCachedBatchDp4a(qw.q, qw.scales, x, p.buf.xq, p.buf.xsc, dst, rows, n, k, qw.blocksPerRow, p.o.stream)
}

func devSub(buf native.DeviceBuffer, byteOffset int) native.DeviceBuffer {
	if byteOffset == 0 {
		return buf
	}
	return native.DeviceBufferFromRaw(unsafe.Add(buf.Ptr(), byteOffset))
}

// batchedLogitPostprocess applies the same host-side output-head postprocessing
// the sequential ForwardToken path applies after its device matvec, so the
// batched output head matches the sequential oracle for models that set
// FinalLogitSoftcap or LMHeadMultiplier.
func batchedLogitPostprocess(logits []float32, m *instance.Instance) {
	if scale := m.Config.Config.LMHeadMultiplier; scale != 0 && scale != 1 {
		s := float32(scale)
		for i := range logits {
			logits[i] *= s
		}
	}
	if softcap := m.Config.Config.FinalLogitSoftcap; softcap > 0 {
		for i := range logits {
			logits[i] = float32(math.Tanh(float64(logits[i]/softcap))) * softcap
		}
	}
}
