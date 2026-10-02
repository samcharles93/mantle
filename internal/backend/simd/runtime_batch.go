package simd

import (
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/samcharles93/mantle/pkg/mcf"
)

// This file gives GemmParWT (gemm_batch.go) its production caller: a CPU
// batched prefill that runs a whole prompt as one [n,hidden] block instead of
// one ForwardToken per position. The per-token path in runtime.go,
// runtime_decode.go, attn.go and ffn.go stays untouched as the correctness
// oracle and the fallback.
//
// Two invariants hold throughout:
//
//   - batchEligible only accepts plain dense attention+FFN layers whose
//     projections GemmParWT can decode. Anything else falls back per prompt.
//   - Every condition is checked and every KV slot the batch will write is
//     allocated before the first StoreKV, so the decision to take the batched
//     path is always reversible.
const (
	cpuBatchedPrefillEnv = "MANTLE_CPU_BATCHED_PREFILL"

	// defaultMaxBatch is the number of prompt positions one batched chunk
	// processes. sizeBatchScratch sizes the [chunk,width] scratch buffers from
	// it, so cache and scratch stay bounded while a prompt is split into as many
	// chunks as it needs.
	defaultMaxBatch = 32
)

// errBatchIneligible marks a pre-flight refusal from the batched prefill gate.
// No KV cache slot has been written when it is returned, so the caller can
// safely retry the prompt on the sequential per-token path.
var errBatchIneligible = errors.New("batched prefill not used")

// cpuBatchedPrefillEnabled reports whether the batched prefill path may run. It
// defaults to enabled and can be forced off with MANTLE_CPU_BATCHED_PREFILL=0
// ("0"/"false"/"off"/"no"), which leaves ForwardTokens on the sequential path.
func cpuBatchedPrefillEnabled() bool {
	v, ok := os.LookupEnv(cpuBatchedPrefillEnv)
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

// sizeBatchScratch allocates the [MaxBatch, width] row buffers the batched
// prefill path slices per prompt. It runs once per instance from
// initInstanceScratch; leaving MaxBatch at 0 keeps the path unavailable.
func sizeBatchScratch(m *Instance) {
	if m == nil || m.Config == nil {
		return
	}
	hidden := m.Config.Config.EmbeddingLength
	if hidden <= 0 {
		return
	}
	qDim := m.MaxQDim
	if qDim < 1 {
		qDim = hidden
	}
	kvStride := m.MaxKVStride
	if kvStride < 1 {
		kvStride = hidden
	}
	ffn := m.Config.Config.FFNLength
	for i := range m.Layers {
		for _, w := range []*Mat{m.Layers[i].FfnUp, m.Layers[i].FfnGate} {
			if w != nil && w.R > ffn {
				ffn = w.R
			}
		}
		if w := m.Layers[i].FfnDown; w != nil && w.C > ffn {
			ffn = w.C
		}
	}
	if ffn < 1 {
		ffn = hidden
	}

	n := defaultMaxBatch
	m.MaxBatch = n
	scratch := &m.Scratch
	scratch.BatchX = make([]float32, n*hidden)
	scratch.BatchNorm = make([]float32, n*hidden)
	scratch.BatchProj = make([]float32, n*hidden)
	scratch.BatchQ = make([]float32, n*qDim)
	scratch.BatchAttnOut = make([]float32, n*qDim)
	scratch.BatchK = make([]float32, n*kvStride)
	scratch.BatchV = make([]float32, n*kvStride)
	scratch.BatchFfnUp = make([]float32, n*ffn)
	scratch.BatchFfnGate = make([]float32, n*ffn)
	scratch.BatchFfnAct = make([]float32, n*ffn)
}

// batchEligible reports whether tokens can run through the batched prefill
// path. It is a pure predicate: it never mutates state and returns false for
// every condition the batched path does not reproduce exactly, so callers fall
// back to the per-token path.
func (m *Instance) batchEligible(tokens []int) bool {
	return m.batchIneligibleReason(tokens) == ""
}

// batchIneligibleReason names the first failed gate condition, or returns ""
// when the batch is eligible. The reason is carried in the fallback error so an
// unexpectedly unbatched model is diagnosable.
func (m *Instance) batchIneligibleReason(tokens []int) string {
	if m == nil || m.Config == nil {
		return "no model config"
	}
	cfg := &m.Config.Config
	hidden := cfg.EmbeddingLength
	n := len(tokens)
	switch {
	case n < 2:
		return fmt.Sprintf("prompt has %d token(s); a batch needs at least 2", n)
	case hidden <= 0 || cfg.VocabSize <= 0:
		return "config is missing embedding length or vocab size"
	case m.HeadCount <= 0 || m.HeadDim <= 0:
		return "config is missing head count or head dim"
	case m.Gemma4PerLayer != nil:
		return "gemma4 per-layer input is not reproduced"
	case cfg.FlashAttention:
		// The CPU flash-attention branch in Attention returns the raw attention
		// output without the Wo projection, so the sequential path's own result
		// depends on this switch. Leave those models on the sequential path.
		return "flash attention config selects a different sequential path"
	case m.MaxBatch <= 0:
		return "batch scratch is not allocated"
	case m.Pos < 0 || m.Pos+n > m.MaxContext:
		return fmt.Sprintf("batch of %d at position %d exceeds context %d", n, m.Pos, m.MaxContext)
	case len(m.Scratch.Scores) < m.Pos+n:
		return "attention scores scratch is smaller than the prompt window"
	}
	if _, isDevice := m.Ops().(DeviceStateOps); isDevice {
		return "device-state ops require the per-token path"
	}
	if m.Embeddings == nil || m.Embeddings.C != hidden || m.Embeddings.R != cfg.VocabSize {
		return "embeddings are not [vocab,hidden]"
	}
	if m.Output == nil || m.Output.C != hidden || m.Output.R != cfg.VocabSize {
		return "output projection is not [vocab,hidden]"
	}
	if !supportedGemmWeight(m.Output) {
		return "output projection is quantised"
	}
	for _, tok := range tokens {
		if tok < 0 || tok >= cfg.VocabSize {
			return fmt.Sprintf("token id %d is out of range", tok)
		}
	}
	if len(m.Layers) == 0 {
		return "model has no layers"
	}
	if m.Layers[0].FfnUp == nil || m.Layers[0].FfnUp.R <= 0 {
		return "layer 0 has no dense FFN weights"
	}
	// Every layer must share one FFN width: the batch buffers are allocated as
	// [n, max] rows and indexed with a per-layer stride.
	ffn := m.Layers[0].FfnUp.R
	for i := range m.Layers {
		if why := m.batchLayerIneligible(i, ffn); why != "" {
			return why
		}
	}

	qDim := m.HeadCount * m.HeadDim
	kvStride := m.MaxKVStride
	chunk := min(n, m.MaxBatch)
	scratch := &m.Scratch
	if len(scratch.BatchX) < chunk*hidden || len(scratch.BatchNorm) < chunk*hidden ||
		len(scratch.BatchProj) < chunk*hidden || len(scratch.BatchQ) < chunk*qDim ||
		len(scratch.BatchAttnOut) < chunk*qDim || len(scratch.BatchK) < chunk*kvStride ||
		len(scratch.BatchV) < chunk*kvStride || len(scratch.BatchFfnUp) < chunk*ffn ||
		len(scratch.BatchFfnGate) < chunk*ffn || len(scratch.BatchFfnAct) < chunk*ffn {
		return "batch scratch buffers are too small for this prompt"
	}
	return ""
}

// batchLayerIneligible reports why layer i cannot be run as part of a dense
// batch, or "" when it can. ffnWidth is the model-wide dense FFN width.
func (m *Instance) batchLayerIneligible(i, ffnWidth int) string {
	layer := &m.Layers[i]
	why := fmt.Sprintf("layer %d: ", i)
	hidden := m.Config.Config.EmbeddingLength
	headDim := m.HeadDim
	qDim := m.HeadCount * headDim
	kvStride := m.MaxKVStride

	switch {
	case layer.Mamba != nil:
		return why + "mamba block"
	case layer.DeltaNet != nil:
		return why + "deltanet block"
	case layer.IsRecurrent:
		return why + "recurrent (short-conv) block"
	case layer.MoE != nil:
		return why + "mixture-of-experts FFN"
	case layer.Gemma4MoE != nil:
		return why + "gemma4 MoE FFN"
	case layer.Gemma4PLE != nil:
		return why + "gemma4 per-layer input"
	case layer.LayerScale != 1:
		// runDecoderLayers routes any non-unit LayerScale through the gemma4
		// BF16-rounding FFN block, which the batched path does not reproduce.
		return why + "non-unit layer scale"
	case layer.SharedKVSource >= 0:
		return why + "shared KV source"
	case layer.FusedQGate || layer.AttnGate != nil:
		return why + "fused or gated attention"
	case layer.ValueFromKey:
		return why + "value-from-key projection"
	case layer.ApplyVNorm:
		return why + "value norm"
	case layer.AttnWindow > 0:
		return why + "sliding attention window"
	case layer.AttnType != "" && layer.AttnType != "full_attention":
		return why + "non-full attention type"
	case layer.RoundActivationsBF16:
		return why + "bf16 activation rounding"
	case layer.NoRoPE:
		return why + "rope disabled"
	case len(layer.PostAttnNorm) > 0:
		return why + "post-attention norm"
	case len(layer.PostFfnNorm) > 0:
		return why + "post-ffn norm"
	case len(layer.AttnQNorm) > 0 || len(layer.AttnKNorm) > 0:
		return why + "per-head q/k norm"
	case len(layer.WqBias) > 0 || len(layer.WkBias) > 0 || len(layer.WvBias) > 0:
		return why + "qkv bias"
	case layer.HeadDim != headDim || layer.HeadKV <= 0:
		return why + "head dim or kv head count differs from the instance"
	case layer.HeadKV*headDim != kvStride:
		return why + "kv stride differs from the instance maximum"
	case len(layer.AttnNorm) != hidden:
		return why + "attention norm is not [hidden]"
	case len(layer.FfnNorm) != hidden:
		return why + "ffn norm is not [hidden]"
	case layer.AttnCache.CacheLen <= 0 || layer.AttnCache.KvStride != kvStride:
		return why + "attention cache is not a full-size ring of the instance kv stride"
	case layer.Wq == nil || layer.Wq.R != qDim || layer.Wq.C != hidden:
		return why + "query projection is not [qDim,hidden]"
	case layer.Wk == nil || layer.Wk.R != kvStride || layer.Wk.C != hidden:
		return why + "key projection is not [kvDim,hidden]"
	case layer.Wv == nil || layer.Wv.R != kvStride || layer.Wv.C != hidden:
		return why + "value projection is not [kvDim,hidden]"
	case layer.Wo == nil || layer.Wo.R != hidden || layer.Wo.C != qDim:
		return why + "output projection is not [hidden,qDim]"
	case layer.FfnUp == nil || layer.FfnUp.R != ffnWidth || layer.FfnUp.C != hidden:
		return why + "ffn up projection shape differs"
	case layer.FfnGate == nil || layer.FfnGate.R != ffnWidth || layer.FfnGate.C != hidden:
		return why + "ffn gate projection shape differs"
	case layer.FfnDown == nil || layer.FfnDown.R != hidden || layer.FfnDown.C != ffnWidth:
		return why + "ffn down projection shape differs"
	}
	for _, w := range []*Mat{layer.Wq, layer.Wk, layer.Wv, layer.Wo, layer.FfnUp, layer.FfnGate, layer.FfnDown} {
		if !supportedGemmWeight(w) {
			return why + "quantised projection weight"
		}
	}
	return ""
}

// batchPrefillChunk owns the [n,width] row buffers and per-row positions of one
// chunk of a batched prefill (n <= MaxBatch). forEachRow cannot fail and never
// needs a mid-prompt fallback: batchIneligibleReason validates the whole prompt
// before any chunk runs, and newBatchPrefillChunk only slices the scratch.
type batchPrefillChunk struct {
	m        *Instance
	ops      Ops
	tokens   []int
	n        int
	startPos int
	hidden   int
	ffn      int
	qDim     int
	kvStride int
	useGelu  bool

	x    []float32 // [n,hidden] hidden state
	norm []float32 // [n,hidden] pre-norm scratch
	proj []float32 // [n,hidden] block output, reused by the attention and FFN blocks
	q    []float32 // [n,qDim]
	k    []float32 // [n,kvStride]
	v    []float32 // [n,kvStride]
	attn []float32 // [n,qDim]
	up   []float32 // [n,ffn]
	gate []float32 // [n,ffn]
	act  []float32 // [n,ffn]

	// tapBase is the tap row of this chunk's first position. Hidden-tap capture
	// writes row tapBase+r; a prompt longer than the tap buffer's capacity
	// therefore leaves HiddenTaps.Rows short of the prompt length.
	tapBase int
}

// forEachBatchLogits runs a whole prompt through the batched path, splitting it
// into chunks of at most MaxBatch positions, and calls keep once per position
// with that position's logits (the model-owned logits buffer, reused for every
// row, so keep must copy anything it retains).
//
// The caller may fall back to the per-token path only when this returns
// errBatchIneligible. That is guaranteed to mean nothing has been mutated
// because of one invariant: every gate predicate is evaluated against the FULL
// prompt, every KV slot the prompt will touch is grown, and every scratch
// buffer is capacity-checked against the largest chunk BEFORE the first chunk
// issues its first StoreKV. A chunk cannot refuse after that point, so no
// partially prefilled state can ever be handed back.
func (m *Instance) forEachBatchLogits(tokens []int, keep func(logits []float32)) error {
	if !cpuBatchedPrefillEnabled() {
		return errBatchIneligible
	}
	if why := m.batchIneligibleReason(tokens); why != "" {
		return fmt.Errorf("%w: %s", errBatchIneligible, why)
	}

	startPos := m.Pos
	n := len(tokens)

	// Taps describe the forward just performed, so the row counter restarts
	// here; HiddenTaps.Rows ends up holding this prompt's captured row count.
	m.beginForwardCapture()

	// Pre-flight, before any chunk can store a KV row: EnsurePos only allocates
	// backing storage and never writes cache contents.
	for i := range m.Layers {
		cache := &m.Layers[i].AttnCache
		for r := range n {
			cache.EnsurePos((startPos + r) % cache.CacheLen)
		}
	}

	for start := 0; start < n; start += m.MaxBatch {
		end := min(start+m.MaxBatch, n)
		// Each chunk plans against its own start position, so the ring-wrap
		// store/attend interleaving is decided per chunk by the same rule the
		// single-chunk path uses.
		chunk := newBatchPrefillChunk(m, tokens[start:end], m.Pos)
		chunk.tapBase = start
		chunk.forEachRow(keep)
	}
	return nil
}

// newBatchPrefillChunk slices the instance's batch scratch for one chunk of at
// most MaxBatch positions. Callers must have passed batchIneligibleReason for
// the whole prompt first; this constructor cannot fail and must never be given
// a chunk larger than MaxBatch.
func newBatchPrefillChunk(m *Instance, tokens []int, startPos int) *batchPrefillChunk {
	n := len(tokens)
	hidden := m.Config.Config.EmbeddingLength
	qDim := m.HeadCount * m.HeadDim
	kvStride := m.MaxKVStride
	ffn := m.Layers[0].FfnUp.R
	scratch := &m.Scratch

	return &batchPrefillChunk{
		m:        m,
		ops:      m.Ops(),
		tokens:   tokens,
		n:        n,
		startPos: startPos,
		hidden:   hidden,
		ffn:      ffn,
		qDim:     qDim,
		kvStride: kvStride,
		useGelu:  strings.Contains(m.Config.Config.HiddenAct, "gelu"),
		x:        scratch.BatchX[:n*hidden],
		norm:     scratch.BatchNorm[:n*hidden],
		proj:     scratch.BatchProj[:n*hidden],
		q:        scratch.BatchQ[:n*qDim],
		k:        scratch.BatchK[:n*kvStride],
		v:        scratch.BatchV[:n*kvStride],
		attn:     scratch.BatchAttnOut[:n*qDim],
		up:       scratch.BatchFfnUp[:n*ffn],
		gate:     scratch.BatchFfnGate[:n*ffn],
		act:      scratch.BatchFfnAct[:n*ffn],
	}
}

// gemmRows computes dst[n, w.R] = src[n, w.C] * wᵀ over the whole chunk block.
// dst and src are contiguous [n,width] chunk buffers whose widths the gate sized
// to w.R and w.C, so GemmParWT's dimension check cannot fire.
func (p *batchPrefillChunk) gemmRows(dst, src []float32, w *Mat) {
	n := p.n
	rows := Mat{R: n, C: w.R, Stride: w.R, DType: mcf.DTypeF32, Data: dst[:n*w.R]}
	block := Mat{R: n, C: w.C, Stride: w.C, DType: mcf.DTypeF32, Data: src[:n*w.C]}
	GemmParWT(SelectGemmConfig(n, w.C, w.R), &rows, &block, w, 1, 0, 0)
}

// forEachRow executes the chunk and calls keep once per position with that
// position's logits. keep receives the model-owned logits buffer, which is
// reused for every row, so it must copy anything it retains. forEachRow
// allocates nothing on the model and cannot fail: everything it needs was
// validated and allocated by batchIneligibleReason and newBatchPrefillChunk.
func (p *batchPrefillChunk) forEachRow(keep func(logits []float32)) {
	m := p.m
	n := p.n
	hidden := p.hidden
	cfg := &m.Config.Config

	for r, tok := range p.tokens {
		m.Embeddings.RowTo(p.x[r*hidden:(r+1)*hidden], tok)
	}
	if scale := cfg.EmbeddingMultiplier; scale != 0 && scale != 1 {
		s := float32(scale)
		for i := range p.x {
			p.x[i] *= s
		}
	}
	if cfg.MuPEnabled && m.MuPScale != 1 {
		for i := range p.x {
			p.x[i] *= m.MuPScale
		}
	}

	// Hidden taps: the embedding output (layer -1) then each layer's post-FFN
	// residual, written row-by-row for the whole chunk. Both are read-only.
	p.m.captureTapBatch(-1, p.tapBase, n, p.x)

	for i := range m.Layers {
		layer := &m.Layers[i]
		for r := range n {
			p.ops.RMSNorm(p.norm[r*hidden:(r+1)*hidden], p.x[r*hidden:(r+1)*hidden], layer.AttnNorm, m.RMSEpsilon)
		}
		p.attentionBlock(layer)
		for r := range n {
			Add(p.x[r*hidden:(r+1)*hidden], p.proj[r*hidden:(r+1)*hidden])
		}

		for r := range n {
			p.ops.RMSNorm(p.norm[r*hidden:(r+1)*hidden], p.x[r*hidden:(r+1)*hidden], layer.FfnNorm, m.RMSEpsilon)
		}
		p.ffnBlock(layer)
		for r := range n {
			Add(p.x[r*hidden:(r+1)*hidden], p.proj[r*hidden:(r+1)*hidden])
		}
		p.m.captureTapBatch(i, p.tapBase, n, p.x)
	}

	logits := m.Scratch.Logits
	for r := range n {
		FusedRMSNormMatVec(p.ops, logits, m.Output, p.x[r*hidden:(r+1)*hidden], m.OutputNorm, m.RMSEpsilon, m.Scratch.Tmp)
		if scale := cfg.LMHeadMultiplier; scale != 0 && scale != 1 {
			s := float32(scale)
			for i := range logits {
				logits[i] *= s
			}
		}
		if softcap := cfg.FinalLogitSoftcap; softcap > 0 {
			for i := range logits {
				logits[i] = fastTanh(logits[i]/softcap) * softcap
			}
		}
		keep(logits)
	}
	m.Pos += n
}
