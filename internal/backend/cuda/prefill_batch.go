//go:build cuda

package cuda

import (
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
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
	wantKvStride := m.Layers[0].HeadKV * headDim
	wantInterm := 0
	if m.Layers[0].FfnUp != nil {
		wantInterm = m.Layers[0].FfnUp.R
	}
	for i := range m.Layers {
		l := &m.Layers[i]
		if l.HeadKV*headDim != wantKvStride || l.FfnUp == nil || l.FfnUp.R != wantInterm {
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

	if layer.Mamba != nil || layer.DeltaNet != nil || layer.IsRecurrent {
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
	if layer.FusedQGate || layer.AttnGate != nil {
		return false
	}
	if layer.ValueFromKey || layer.ApplyVNorm {
		return false
	}
	if layer.AttnWindow > 0 {
		return false
	}
	if layer.RoundActivationsBF16 {
		return false
	}
	if len(layer.PostAttnNorm) > 0 || len(layer.PostFfnNorm) > 0 {
		return false
	}
	if layer.HeadDim != headDim || layer.HeadKV <= 0 {
		return false
	}
	if layer.NoRoPE {
		return false
	}
	if len(layer.AttnNorm) != hidden || len(layer.FfnNorm) != hidden {
		return false
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
	apply, invFreq, _ := batchedLayerRoPE(m, layer)
	if apply && (len(invFreq) == 0 || len(invFreq)*2 > headDim) {
		return false
	}
	for _, w := range []*instance.Mat{layer.Wq, layer.Wk, layer.Wv, layer.Wo, layer.FfnUp, layer.FfnGate, layer.FfnDown} {
		if !batchedWeightDense(o, w) {
			return false
		}
	}
	return true
}

// batchedWeightDense reports whether a weight will be executed by the dense
// GEMM path in the batched implementation. Quantized, offloaded and
// non-F32/BF16 dense weights fall back to the sequential path.
func batchedWeightDense(o *Ops, w *instance.Mat) bool {
	if w == nil || w.R == 0 || w.C == 0 {
		return false
	}
	if _, off := o.offloadedMats[w]; off {
		return false
	}
	if _, ok := o.qweights[w]; ok {
		return false
	}
	mode := currentCUDAWeightMode()
	if useQuantKernel() && (shouldPreferQuantWeights(w, mode) || (w.Quant != nil && w.Quant.ValidFor(w))) {
		return false
	}
	if mode != cudaWeightModeDequant && shouldPreferQuantWeights(w, mode) {
		return false
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
}

func (b *batchedBuffers) all() []native.DeviceBuffer {
	return []native.DeviceBuffer{b.x, b.normed, b.proj, b.q, b.attn, b.k, b.v, b.up, b.gate, b.act, b.down, b.conv, b.last, b.lastN, b.logits}
}

type batchedLayerPlan struct {
	layer    *instance.Layer
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
}

type batchedPrefillPlan struct {
	gr   *GraphRuntime
	o    *Ops
	inst *instance.Instance

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
		m.Embeddings.RowTo(row, tok)
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
	convElems := p.n * maxInt(hidden, p.maxInterm)
	if convElems > 0 {
		b, err := native.AllocDevice(int64(convElems) * 2)
		if err != nil {
			return fail(fmt.Errorf("cuda batched prefill: alloc conv (%d elems): %w", convElems, err))
		}
		p.buf.conv = b
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
		if lp.wq, err = o.deviceMat(l.Wq); err != nil {
			return fail(fmt.Errorf("cuda batched prefill: layer %d Wq: %w", i, err))
		}
		if lp.wk, err = o.deviceMat(l.Wk); err != nil {
			return fail(fmt.Errorf("cuda batched prefill: layer %d Wk: %w", i, err))
		}
		if lp.wv, err = o.deviceMat(l.Wv); err != nil {
			return fail(fmt.Errorf("cuda batched prefill: layer %d Wv: %w", i, err))
		}
		if lp.wo, err = o.deviceMat(l.Wo); err != nil {
			return fail(fmt.Errorf("cuda batched prefill: layer %d Wo: %w", i, err))
		}
		if lp.up, err = o.deviceMat(l.FfnUp); err != nil {
			return fail(fmt.Errorf("cuda batched prefill: layer %d FfnUp: %w", i, err))
		}
		if lp.gate, err = o.deviceMat(l.FfnGate); err != nil {
			return fail(fmt.Errorf("cuda batched prefill: layer %d FfnGate: %w", i, err))
		}
		if lp.down, err = o.deviceMat(l.FfnDown); err != nil {
			return fail(fmt.Errorf("cuda batched prefill: layer %d FfnDown: %w", i, err))
		}
		if lp.attnNormDev, err = o.deviceNormWeight(l.AttnNorm); err != nil {
			return fail(fmt.Errorf("cuda batched prefill: layer %d AttnNorm: %w", i, err))
		}
		if lp.ffnNormDev, err = o.deviceNormWeight(l.FfnNorm); err != nil {
			return fail(fmt.Errorf("cuda batched prefill: layer %d FfnNorm: %w", i, err))
		}
		if len(l.AttnQNorm) > 0 {
			if lp.qNormDev, err = o.deviceNormWeight(l.AttnQNorm); err != nil {
				return fail(fmt.Errorf("cuda batched prefill: layer %d AttnQNorm: %w", i, err))
			}
		}
		if len(l.AttnKNorm) > 0 {
			if lp.kNormDev, err = o.deviceNormWeight(l.AttnKNorm); err != nil {
				return fail(fmt.Errorf("cuda batched prefill: layer %d AttnKNorm: %w", i, err))
			}
		}

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
	outWeight, err := o.deviceMat(m.Output)
	if err != nil {
		return fail(fmt.Errorf("cuda batched prefill: output weight: %w", err))
	}
	p.outWeight = outWeight

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

	for i := range p.layers {
		if err := p.runLayer(&p.layers[i]); err != nil {
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
	if err := p.gemmLast(p.outWeight, p.buf.lastN, p.hidden, p.buf.logits, p.vocab); err != nil {
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
	// Q/K/V projections.
	if err := p.gemm(lp.wq, p.buf.normed, p.hidden, p.buf.q, p.qDim); err != nil {
		return err
	}
	if err := p.gemm(lp.wk, p.buf.normed, p.hidden, p.buf.k, lp.kvStride); err != nil {
		return err
	}
	if err := p.gemm(lp.wv, p.buf.normed, p.hidden, p.buf.v, lp.kvStride); err != nil {
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
			if lp.useQ8K || lp.useQ8V {
				if err := native.AttentionInnerMixedCacheF32(
					qRow, lp.cache.kF16, lp.cache.vF16,
					lp.cache.kQ8, lp.cache.vQ8, lp.cache.kQ8Scales, lp.cache.vQ8Scales,
					attnRow, lp.useQ8K, lp.useQ8V,
					pos, 0, lp.kvStride, p.headDim, p.nHead, lp.kvHeads,
					lp.cacheLen, lp.scale, p.softcap, stream,
				); err != nil {
					return fmt.Errorf("cuda batched prefill: attention (pos=%d): %w", pos, err)
				}
			} else {
				if err := native.AttentionInnerF16CacheF32(
					qRow, lp.cache.kF16, lp.cache.vF16, attnRow,
					pos, 0, lp.kvStride, p.headDim, p.nHead, lp.kvHeads,
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
				lp.cacheLen, lp.scale, p.softcap, stream,
			); err != nil {
				return fmt.Errorf("cuda batched prefill: batched attention: %w", err)
			}
		} else {
			if err := native.AttentionInnerF16CacheF32Batch(
				p.buf.q, lp.cache.kF16, lp.cache.vF16, p.buf.attn,
				p.n, p.startPos, lp.kvStride, p.headDim, p.nHead, lp.kvHeads,
				lp.cacheLen, lp.scale, p.softcap, stream,
			); err != nil {
				return fmt.Errorf("cuda batched prefill: batched attention: %w", err)
			}
		}
	}

	// Attention output projection and residual.
	if err := p.gemm(lp.wo, p.buf.attn, p.qDim, p.buf.proj, p.hidden); err != nil {
		return err
	}
	if err := native.AddVectorsF32(p.buf.x, p.buf.proj, p.n*p.hidden, stream); err != nil {
		return fmt.Errorf("cuda batched prefill: attention residual: %w", err)
	}

	// FFN.
	if err := native.RMSNormBatchedF32(p.buf.normed, p.buf.x, lp.ffnNormDev, p.eps, p.hidden, p.n, stream); err != nil {
		return fmt.Errorf("cuda batched prefill: ffn norm: %w", err)
	}
	if err := p.gemm(lp.up, p.buf.normed, p.hidden, p.buf.up, lp.interm); err != nil {
		return err
	}
	if err := p.gemm(lp.gate, p.buf.normed, p.hidden, p.buf.gate, lp.interm); err != nil {
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
	if err := p.gemm(lp.down, p.buf.act, lp.interm, p.buf.down, p.hidden); err != nil {
		return err
	}
	if err := native.AddVectorsF32(p.buf.x, p.buf.down, p.n*p.hidden, stream); err != nil {
		return fmt.Errorf("cuda batched prefill: ffn residual: %w", err)
	}

	// Record the last position written to this layer's device KV cache,
	// exactly as the device fast paths do. The batched path always writes the
	// whole batch, so this is startPos+N-1.
	o.lastDevKVPos[lp.layer] = p.startPos + p.n - 1
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
