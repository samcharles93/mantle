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
//   - batchEligible only accepts layers the batched kernels reproduce exactly:
//     dense attention (full or sliding window, with per-head q/k norms, an
//     output gate and sandwich norms) or recurrent short-conv blocks, all with
//     projections GemmParWT can decode. Anything else falls back per prompt.
//   - Every condition is checked and every KV slot the batch will write is
//     allocated before the first StoreKV, so the decision to take the batched
//     path is always reversible.
const (
	cpuBatchedPrefillEnv = "MANTLE_CPU_BATCHED_PREFILL"

	// defaultMaxBatch is the number of prompt positions one batched chunk
	// processes when MANTLE_CPU_MAX_BATCH does not override it. sizeBatchScratch
	// sizes the [chunk,width] scratch buffers from it, so cache and scratch stay
	// bounded while a prompt is split into as many chunks as it needs.
	defaultMaxBatch = 64

	// cpuMaxBatchEnv overrides the chunk size, for tuning prefill throughput
	// against scratch memory on a given host.
	cpuMaxBatchEnv = "MANTLE_CPU_MAX_BATCH"
)

// cpuMaxBatch reports the chunk size the batched prefill path runs with:
// MANTLE_CPU_MAX_BATCH when it holds a positive integer, otherwise
// defaultMaxBatch. The value is clamped to [1, 4096] so a stray setting cannot
// make the [chunk,width] scratch buffers absurdly large.
func cpuMaxBatch() int {
	if v, ok := os.LookupEnv(cpuMaxBatchEnv); ok {
		if n, err := strconv.Atoi(strings.TrimSpace(v)); err == nil {
			return min(max(n, 1), 4096)
		}
	}
	return defaultMaxBatch
}

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

	// The short-conv state staging buffer is sized from the widest recurrent
	// kernel; the attention gate rows from the widest gate (never above qDim,
	// which the gate enforces per layer).
	convState := 0
	for i := range m.Layers {
		if m.Layers[i].IsRecurrent && m.Layers[i].ShortConvKernel != nil {
			if k := m.Layers[i].ShortConvKernel.C - 1; k > convState {
				convState = k
			}
		}
	}

	n := cpuMaxBatch()
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
	scratch.BatchScProj = make([]float32, n*3*hidden)
	scratch.BatchScBx = make([]float32, n*hidden)
	scratch.BatchScConv = make([]float32, n*hidden)
	scratch.BatchAttnGate = make([]float32, n*qDim)
	scratch.BatchScState = make([]float32, convState*hidden)
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
	return m.batchIneligibleReasonMode(tokens, false)
}

// batchIneligibleReasonMode is batchIneligibleReason for a specific caller shape.
// nonCausal blocks (ForwardBlockNonCausal) additionally refuse the features the
// non-causal batch does not reproduce: recurrent layers, gated attention,
// sliding windows, sandwich norms and sliding attention types all keep the
// plain dense-only contract there.
func (m *Instance) batchIneligibleReasonMode(tokens []int, nonCausal bool) string {
	if nonCausal {
		if why := m.nonCausalIneligible(); why != "" {
			return why
		}
	}
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
	if !batchProjectable(m.Output) {
		return "output projection dtype is not supported"
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
		len(scratch.BatchFfnGate) < chunk*ffn || len(scratch.BatchFfnAct) < chunk*ffn ||
		len(scratch.BatchScProj) < chunk*3*hidden || len(scratch.BatchScBx) < chunk*hidden ||
		len(scratch.BatchScConv) < chunk*hidden || len(scratch.BatchAttnGate) < chunk*qDim {
		return "batch scratch buffers are too small for this prompt"
	}
	return ""
}

// nonCausalIneligible names the first layer feature a non-causal block forward
// (speculative draft blocks) does not reproduce, or "" when all layers are
// plain dense full-attention layers.
func (m *Instance) nonCausalIneligible() string {
	if m == nil {
		return "no instance"
	}
	for i := range m.Layers {
		layer := &m.Layers[i]
		why := fmt.Sprintf("layer %d: ", i)
		switch {
		case layer.IsRecurrent:
			return why + "recurrent (short-conv) block in a non-causal block"
		case layer.AttnGate != nil || layer.FusedQGate:
			return why + "gated attention in a non-causal block"
		case layer.AttnWindow > 0 || layer.AttnType == "sliding_attention":
			return why + "sliding window in a non-causal block"
		case len(layer.PostAttnNorm) > 0 || len(layer.PostFfnNorm) > 0:
			return why + "sandwich norm in a non-causal block"
		}
	}
	return ""
}

// batchLayerIneligible reports why layer i cannot be run as part of a batch,
// or "" when it can. ffnWidth is the model-wide dense FFN width. Recurrent
// (short-conv) layers and attention layers are checked against their own
// weight sets: a hybrid layer runs one block or the other, mirroring
// computeAttentionBranch in runtime_decode.go.
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
	case layer.FusedQGate:
		return why + "fused q gate"
	case layer.RoundActivationsBF16:
		return why + "bf16 activation rounding"
	case layer.NoRoPE:
		return why + "rope disabled"
	case len(layer.AttnQNorm) > 0 && len(layer.AttnQNorm) != headDim,
		len(layer.AttnKNorm) > 0 && len(layer.AttnKNorm) != headDim:
		return why + "per-head q/k norm is not [headDim]"
	case len(layer.WqBias) > 0 || len(layer.WkBias) > 0 || len(layer.WvBias) > 0:
		return why + "qkv bias"
	case len(layer.AttnNorm) != hidden:
		return why + "attention norm is not [hidden]"
	case len(layer.FfnNorm) != hidden:
		return why + "ffn norm is not [hidden]"
	case len(layer.PostAttnNorm) > 0 && len(layer.PostAttnNorm) != hidden:
		return why + "post-attention norm is not [hidden]"
	case len(layer.PostFfnNorm) > 0 && len(layer.PostFfnNorm) != hidden:
		return why + "post-ffn norm is not [hidden]"
	case layer.FfnUp == nil || layer.FfnUp.R != ffnWidth || layer.FfnUp.C != hidden:
		return why + "ffn up projection shape differs"
	case layer.FfnGate == nil || layer.FfnGate.R != ffnWidth || layer.FfnGate.C != hidden:
		return why + "ffn gate projection shape differs"
	case layer.FfnDown == nil || layer.FfnDown.R != hidden || layer.FfnDown.C != ffnWidth:
		return why + "ffn down projection shape differs"
	}
	for _, w := range []*Mat{layer.FfnUp, layer.FfnGate, layer.FfnDown} {
		if !batchProjectable(w) {
			return why + "unsupported FFN weight dtype"
		}
	}

	if layer.IsRecurrent {
		return m.batchShortConvIneligible(layer, why, hidden)
	}

	switch {
	case layer.SharedKVSource >= 0:
		return why + "shared KV source"
	case layer.ValueFromKey:
		return why + "value-from-key projection"
	case layer.ApplyVNorm:
		return why + "value norm"
	case layer.AttnType != "" && layer.AttnType != "full_attention" && layer.AttnType != "sliding_attention":
		return why + "non-full attention type"
	case layer.AttnGate != nil &&
		(layer.AttnGate.R != qDim && layer.AttnGate.R != m.HeadCount || layer.AttnGate.C != hidden):
		return why + "attention gate is not [qDim|heads,hidden]"
	case layer.HeadDim != headDim || layer.HeadKV <= 0:
		return why + "head dim or kv head count differs from the instance"
	case layer.HeadKV*headDim != kvStride:
		return why + "kv stride differs from the instance maximum"
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
	}
	for _, w := range []*Mat{layer.Wq, layer.Wk, layer.Wv, layer.Wo, layer.AttnGate} {
		if w != nil && !batchProjectable(w) {
			return why + "unsupported projection weight dtype"
		}
	}
	return ""
}

// batchShortConvIneligible reports why a recurrent (short-conv) layer cannot run
// as part of a batch, or "" when its weights and state fit the batched kernels.
func (m *Instance) batchShortConvIneligible(layer *Layer, why string, hidden int) string {
	kernel := layer.ShortConvKernel
	switch {
	case kernel == nil || kernel.Data == nil || kernel.R != hidden:
		// The batched conv reads kernel rows directly, so the kernel must be
		// f32 in Data (the loader's readTensorF32 layout), not raw 2-byte.
		return why + "short-conv kernel is not f32 [hidden,kernelLen]"
	case kernel.C < 1:
		return why + "short-conv kernel length is zero"
	case layer.ShortConvInProj == nil || layer.ShortConvInProj.R != 3*hidden || layer.ShortConvInProj.C != hidden:
		return why + "short-conv in projection is not [3*hidden,hidden]"
	case layer.ShortConvOutProj == nil || layer.ShortConvOutProj.R != hidden || layer.ShortConvOutProj.C != hidden:
		return why + "short-conv out projection is not [hidden,hidden]"
	case len(layer.ShortConvState.Buf) != (kernel.C-1)*hidden:
		return why + "short-conv state size does not match the kernel"
	case len(m.Scratch.BatchScState) < (kernel.C-1)*hidden:
		return why + "batch short-conv state scratch is too small"
	}
	for _, w := range []*Mat{layer.ShortConvInProj, layer.ShortConvOutProj} {
		if !batchProjectable(w) {
			return why + "unsupported short-conv projection weight dtype"
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

	x        []float32 // [n,hidden] hidden state
	norm     []float32 // [n,hidden] pre-norm scratch, then post-block norm
	proj     []float32 // [n,hidden] block output, reused by the attention/conv and FFN blocks
	q        []float32 // [n,qDim]
	k        []float32 // [n,kvStride]
	v        []float32 // [n,kvStride]
	attn     []float32 // [n,qDim]
	up       []float32 // [n,ffn]
	gate     []float32 // [n,ffn]
	act      []float32 // [n,ffn]
	scProj   []float32 // [n,3*hidden] short-conv in_proj output
	scBx     []float32 // [n,hidden] gated conv input (the future state rows)
	scConv   []float32 // [n,hidden] conv output, then out_proj input
	attnGate []float32 // [n,gateRows] attention gate rows
	scState  []float32 // [(maxKernelLen-1), hidden] state staging copy

	// rowEmbeds, when non-nil, is parallel to tokens and overrides the input
	// embedding of positions whose entry is non-nil (multimodal image tokens).
	rowEmbeds [][]float32

	// nonCausal makes every row attend to all n rows of the chunk instead of
	// only the rows at or before its own position. The chunk must not wrap the
	// cache ring, since all rows are stored before any row attends.
	nonCausal bool
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
func (m *Instance) forEachBatchLogits(tokens []int, emit logitEmit, keep func(logits []float32)) error {
	return m.forEachBatchLogitsEmbeds(tokens, nil, emit, keep)
}

// forEachBatchLogitsEmbeds is forEachBatchLogits with optional per-position
// input embedding overrides (embeds parallel to tokens, nil entries use the
// token embedding). The override only changes how p.x rows are initialized;
// every downstream gate and buffer rule is unchanged.
func (m *Instance) forEachBatchLogitsEmbeds(tokens []int, embeds [][]float32, emit logitEmit, keep func(logits []float32)) error {
	if embeds != nil && len(embeds) != len(tokens) {
		return fmt.Errorf("%w: embeds length %d != tokens %d", errBatchIneligible, len(embeds), len(tokens))
	}
	if !cpuBatchedPrefillEnabled() {
		return errBatchIneligible
	}
	if why := m.batchIneligibleReason(tokens); why != "" {
		return fmt.Errorf("%w: %s", errBatchIneligible, why)
	}

	startPos := m.Pos
	n := len(tokens)
	if emit == emitAllRows && batchHeadRows(m.Output, min(n, m.MaxBatch)) {
		if need := min(n, m.MaxBatch) * m.Output.R; len(m.Scratch.BatchLogits) < need {
			m.Scratch.BatchLogits = make([]float32, need)
		}
	}

	// Pre-flight, before any chunk can store a KV row: EnsurePos only allocates
	// backing storage and never writes cache contents. Recurrent (short-conv)
	// layers carry no KV cache (CacheLen 0) and are skipped.
	for i := range m.Layers {
		cache := &m.Layers[i].AttnCache
		if cache.CacheLen <= 0 {
			continue
		}
		for r := range n {
			cache.EnsurePos((startPos + r) % cache.CacheLen)
		}
	}

	for start := 0; start < n; start += m.MaxBatch {
		end := min(start+m.MaxBatch, n)
		// Taps describe the forward just performed, so the row counter restarts
		// per chunk: a consumer draining each chunk gets a self-contained
		// capture and never needs more than one chunk of rows, however long the
		// prompt is.
		m.beginForwardCapture()
		// Each chunk plans against its own start position, so the ring-wrap
		// store/attend interleaving is decided per chunk by the same rule the
		// single-chunk path uses.
		chunk := newBatchPrefillChunk(m, tokens[start:end], m.Pos)
		if embeds != nil {
			chunk.rowEmbeds = embeds[start:end]
		}
		chunkEmit := emit
		if emit == emitLastRow && end < n {
			chunkEmit = emitNoRows
		}
		chunk.forEachRow(chunkEmit, keep)
		if err := m.drainTaps(startPos + start); err != nil {
			return err
		}
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
		scProj:   scratch.BatchScProj[:n*3*hidden],
		scBx:     scratch.BatchScBx[:n*hidden],
		scConv:   scratch.BatchScConv[:n*hidden],
		attnGate: scratch.BatchAttnGate[:n*qDim],
		scState:  scratch.BatchScState,
	}
}

// gemmRows computes dst[n, w.R] = src[n, w.C] * wᵀ over the whole chunk block.
// dst and src are contiguous [n,width] chunk buffers whose widths the gate sized
// to w.R and w.C, so GemmParWT's dimension check cannot fire.
func (p *batchPrefillChunk) gemmRows(dst, src []float32, w *Mat) {
	projectRows(dst, w, src, p.n)
}

// projectRows computes dst[n, w.R] = src[n, w.C] * wᵀ for contiguous row
// blocks, picking the small-batch BF16 kernel, the cached int8 quant kernel,
// the packed GEMM, or one matvec per row for weights none of those take.
func projectRows(dst []float32, w *Mat, src []float32, n int) {
	switch {
	case useSmallBatchBF16(w, n):
		MatMatBF16(dst, w, src, n)
	case useBatchQuant(w, n):
		MatMatQuant(dst, w, src, n)
	case n > 1 && supportedGemmWeight(w):
		rows := Mat{R: n, C: w.R, Stride: w.R, DType: mcf.DTypeF32, Data: dst[:n*w.R]}
		block := Mat{R: n, C: w.C, Stride: w.C, DType: mcf.DTypeF32, Data: src[:n*w.C]}
		GemmParWT(SelectGemmConfig(n, w.C, w.R), &rows, &block, w, 1, 0, 0)
	default:
		for r := range n {
			MatVec(dst[r*w.R:(r+1)*w.R], w, src[r*w.C:(r+1)*w.C])
		}
	}
}

// batchProjectable reports whether projectRows accepts w: anything the GEMM
// takes, plus quantised weights, which run through MatMatQuant or fall back to
// one MatVec per row.
func batchProjectable(w *Mat) bool {
	return supportedGemmWeight(w) || (w.Raw != nil && mcf.DTypeRequiresAligned64(w.DType))
}

// batchHeadRows reports whether an all-rows output head for n rows should be
// computed as one block into BatchLogits rather than one fused matvec per row.
func batchHeadRows(w *Mat, n int) bool {
	return useSmallBatchBF16(w, n) || useBatchQuant(w, n)
}

// logitEmit selects which rows of a batched forward get logits.
type logitEmit uint8

const (
	emitAllRows logitEmit = iota // every position, e.g. speculative verification
	emitLastRow                  // the final position only, e.g. prompt prefill
	emitNoRows                   // none: an intermediate prefill chunk
)

// forEachRow executes the chunk and calls keep for the positions emit selects
// with that position's logits. keep receives the model-owned logits buffer, which is
// reused for every row, so it must copy anything it retains. forEachRow
// allocates nothing on the model and cannot fail: everything it needs was
// validated and allocated by batchIneligibleReason and newBatchPrefillChunk.
func (p *batchPrefillChunk) forEachRow(emit logitEmit, keep func(logits []float32)) {
	m := p.m
	n := p.n
	hidden := p.hidden
	cfg := &m.Config.Config

	for r, tok := range p.tokens {
		if p.rowEmbeds != nil && p.rowEmbeds[r] != nil {
			copy(p.x[r*hidden:(r+1)*hidden], p.rowEmbeds[r])
		} else {
			m.Embeddings.RowTo(p.x[r*hidden:(r+1)*hidden], tok)
		}
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
	p.m.captureTapBatch(-1, 0, n, p.x)

	for i := range m.Layers {
		layer := &m.Layers[i]
		for r := range n {
			p.ops.RMSNorm(p.norm[r*hidden:(r+1)*hidden], p.x[r*hidden:(r+1)*hidden], layer.AttnNorm, m.RMSEpsilon)
		}
		if layer.IsRecurrent {
			p.shortConvBlock(layer)
		} else {
			p.attentionBlock(layer)
		}
		p.addBlockResidual(layer.PostAttnNorm)

		for r := range n {
			p.ops.RMSNorm(p.norm[r*hidden:(r+1)*hidden], p.x[r*hidden:(r+1)*hidden], layer.FfnNorm, m.RMSEpsilon)
		}
		p.ffnBlock(layer)
		p.addBlockResidual(layer.PostFfnNorm)
		p.m.captureTapBatch(i, 0, n, p.x)
	}

	switch emit {
	case emitNoRows:
		m.Pos += n
		return
	case emitLastRow:
		logits := m.Scratch.Logits
		FusedRMSNormMatVec(p.ops, logits, m.Output, p.x[(n-1)*hidden:n*hidden], m.OutputNorm, m.RMSEpsilon, m.Scratch.Tmp)
		p.finishLogits(logits)
		keep(logits)
		m.Pos += n
		return
	}

	vocab := m.Output.R
	if batchHeadRows(m.Output, n) && len(m.Scratch.BatchLogits) >= n*vocab {
		for r := range n {
			p.ops.RMSNorm(p.norm[r*hidden:(r+1)*hidden], p.x[r*hidden:(r+1)*hidden], m.OutputNorm, m.RMSEpsilon)
		}
		block := m.Scratch.BatchLogits[:n*vocab]
		projectRows(block, m.Output, p.norm[:n*hidden], n)
		for r := range n {
			row := block[r*vocab : (r+1)*vocab]
			p.finishLogits(row)
			keep(row)
		}
		m.Pos += n
		return
	}

	logits := m.Scratch.Logits
	for r := range n {
		FusedRMSNormMatVec(p.ops, logits, m.Output, p.x[r*hidden:(r+1)*hidden], m.OutputNorm, m.RMSEpsilon, m.Scratch.Tmp)
		p.finishLogits(logits)
		keep(logits)
	}
	m.Pos += n
}

// finishLogits applies the configured LM-head multiplier and final softcap.
func (p *batchPrefillChunk) finishLogits(logits []float32) {
	cfg := &p.m.Config.Config
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
}

// ForwardBlockNonCausal runs tokens as one block at positions Pos..Pos+n-1
// where every row attends to the cached context and to all n block rows. It
// calls keep with each row's logits and final residual (before the output
// norm); both are model-owned and reused, so keep must copy what it retains.
// The block's K/V rows are written to the cache but Pos is left unchanged, so
// the next store at those positions replaces them.
//
// It serves draft models that predict a whole block at once. The block needs
// 2 to MaxBatch tokens and must not wrap the cache ring.
func (m *Instance) ForwardBlockNonCausal(tokens []int, keep func(row int, logits, hidden []float32)) error {
	n := len(tokens)
	if n < 2 || n > m.MaxBatch {
		return fmt.Errorf("non-causal block: %d tokens outside [2,%d]", n, m.MaxBatch)
	}
	for i := range m.Layers {
		if cl := m.Layers[i].AttnCache.CacheLen; m.Pos+n > cl {
			return fmt.Errorf("non-causal block: layer %d cache of %d cannot hold positions up to %d", i, cl, m.Pos+n)
		}
	}
	if why := m.batchIneligibleReasonMode(tokens, true); why != "" {
		return fmt.Errorf("non-causal block: %s", why)
	}
	if batchHeadRows(m.Output, n) {
		if need := n * m.Output.R; len(m.Scratch.BatchLogits) < need {
			m.Scratch.BatchLogits = make([]float32, need)
		}
	}
	start := m.Pos
	for i := range m.Layers {
		cache := &m.Layers[i].AttnCache
		for r := range n {
			cache.EnsurePos(start + r)
		}
	}
	chunk := newBatchPrefillChunk(m, tokens, start)
	chunk.nonCausal = true
	hidden := chunk.hidden
	row := 0
	chunk.forEachRow(emitAllRows, func(logits []float32) {
		keep(row, logits, chunk.x[row*hidden:(row+1)*hidden])
		row++
	})
	m.Pos = start
	return nil
}
