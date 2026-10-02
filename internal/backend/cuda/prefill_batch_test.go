//go:build cuda

package cuda

import (
	"math"
	"testing"

	"github.com/samcharles93/mantle/internal/backend/core"
	"github.com/samcharles93/mantle/internal/backend/cuda/native"
	"github.com/samcharles93/mantle/internal/backend/simd"
	"github.com/samcharles93/mantle/pkg/mcf"
)

// The batched-prefill tests build a small, fully dense, RoPE-enabled
// transformer that satisfies the batched pre-flight gate, then compare the
// batched logits against the untouched sequential CUDA path used as the
// correctness oracle.
const (
	batchTestVocab   = 16
	batchTestHidden  = 64
	batchTestNHead   = 8
	batchTestHeadDim = 8
	batchTestKVHeads = 4
	batchTestFFN     = 96
	batchTestMaxCtx  = 64
)

// batchTestDims describes one synthetic model.
type batchTestDims struct {
	vocab    int
	hidden   int
	nHead    int
	headDim  int
	kvHeads  int
	ffnDim   int
	maxCtx   int
	cacheLen int
	layers   int
	noRoPE   bool
	seed     uint32

	finalSoftcap float32
	lmHeadMult   float64

	q8Cache bool
	bf16    bool
}

func defaultBatchTestDims() batchTestDims {
	return batchTestDims{
		vocab:    batchTestVocab,
		hidden:   batchTestHidden,
		nHead:    batchTestNHead,
		headDim:  batchTestHeadDim,
		kvHeads:  batchTestKVHeads,
		ffnDim:   batchTestFFN,
		maxCtx:   batchTestMaxCtx,
		cacheLen: batchTestMaxCtx,
		layers:   2,
		seed:     0x1234abcd,
	}
}

func batchTestFill(data []float32, seed uint32) {
	s := seed
	for i := range data {
		s = s*1664525 + 1013904223
		data[i] = (float32(s>>8)/float32(1<<24) - 0.5) * 0.2
	}
}

// batchTestMat builds a dense F32 or BF16 weight matrix from the same
// deterministic pattern.
func batchTestMat(r, c int, seed uint32, bf16 bool) *core.Mat {
	data := make([]float32, r*c)
	batchTestFill(data, seed)
	if !bf16 {
		return &core.Mat{R: r, C: c, Stride: c, DType: mcf.DTypeF32, Data: data}
	}
	raw := make([]byte, r*c*2)
	for i, v := range data {
		u := bf16FromF32(v)
		raw[i*2] = byte(u)
		raw[i*2+1] = byte(u >> 8)
	}
	return &core.Mat{R: r, C: c, Stride: c, DType: mcf.DTypeBF16, Raw: raw}
}

// batchTestModel builds a plain dense transformer instance. RoPE is enabled
// with an instance-level inverse-frequency table (the gate rejects NoRoPE).
func batchTestModel(d batchTestDims) *core.Instance {
	headDim := d.headDim
	qDim := d.nHead * headDim
	kvStride := d.kvHeads * headDim

	m := &core.Instance{
		Config: &core.ModelConfig{
			Config: core.Config{
				VocabSize:           d.vocab,
				EmbeddingLength:     d.hidden,
				HeadCount:           d.nHead,
				HeadDim:             headDim,
				BlockCount:          d.layers,
				FFNLength:           d.ffnDim,
				HiddenAct:           "silu",
				FinalLogitSoftcap:   d.finalSoftcap,
				LMHeadMultiplier:    d.lmHeadMult,
				EmbeddingMultiplier: 0,
			},
		},
		Embeddings:  batchTestMat(d.vocab, d.hidden, d.seed, d.bf16),
		OutputNorm:  make([]float32, d.hidden),
		Output:      batchTestMat(d.vocab, d.hidden, d.seed+1, d.bf16),
		Layers:      make([]core.Layer, d.layers),
		HeadDim:     headDim,
		HeadCount:   d.nHead,
		MaxKVStride: kvStride,
		MaxContext:  d.maxCtx,
		RMSEpsilon:  1e-5,
		RopeInvFreq: func() []float64 {
			half := headDim / 2
			tbl := make([]float64, half)
			for i := range tbl {
				tbl[i] = math.Pow(10000, -2*float64(i)/float64(headDim))
			}
			return tbl
		}(),
		RopeAttnScale: 1.0,
		Scratch: core.ScratchBuffers{
			X:        make([]float32, d.hidden),
			Tmp:      make([]float32, d.hidden),
			Tmp2:     make([]float32, maxInt(d.hidden, d.ffnDim)),
			Q:        make([]float32, qDim),
			K:        make([]float32, kvStride),
			V:        make([]float32, kvStride),
			AttnOut:  make([]float32, qDim),
			AttnProj: make([]float32, d.hidden),
			Scores:   make([]float32, d.nHead*d.maxCtx),
			FfnUp:    make([]float32, d.ffnDim),
			FfnGate:  make([]float32, d.ffnDim),
			FfnAct:   make([]float32, d.ffnDim),
			Logits:   make([]float32, d.vocab),
		},
	}
	for i := range m.OutputNorm {
		m.OutputNorm[i] = 1.0
	}

	for li := range m.Layers {
		layer := &m.Layers[li]
		s := d.seed + uint32(li+2)*7919
		layer.HeadKV = d.kvHeads
		layer.HeadDim = headDim
		layer.NoRoPE = d.noRoPE
		layer.SharedKVSource = -1
		layer.LayerScale = 1
		layer.FFNActivation = "silu"
		layer.AttnNorm = make([]float32, d.hidden)
		layer.FfnNorm = make([]float32, d.hidden)
		for i := range layer.AttnNorm {
			layer.AttnNorm[i] = 1.0
			layer.FfnNorm[i] = 1.0
		}
		// Exercise the optional per-head Q/K norm and QKV biases.
		layer.AttnQNorm = make([]float32, headDim)
		layer.AttnKNorm = make([]float32, headDim)
		for i := range layer.AttnQNorm {
			layer.AttnQNorm[i] = 1.0
			layer.AttnKNorm[i] = 1.0
		}
		layer.WqBias = make([]float32, qDim)
		layer.WkBias = make([]float32, kvStride)
		layer.WvBias = make([]float32, kvStride)
		batchTestFill(layer.WqBias, s+7)
		batchTestFill(layer.WkBias, s+8)
		batchTestFill(layer.WvBias, s+9)
		layer.Wq = batchTestMat(qDim, d.hidden, s, d.bf16)
		layer.Wk = batchTestMat(kvStride, d.hidden, s+1, d.bf16)
		layer.Wv = batchTestMat(kvStride, d.hidden, s+2, d.bf16)
		layer.Wo = batchTestMat(d.hidden, qDim, s+3, d.bf16)
		layer.FfnUp = batchTestMat(d.ffnDim, d.hidden, s+4, d.bf16)
		layer.FfnGate = batchTestMat(d.ffnDim, d.hidden, s+5, d.bf16)
		layer.FfnDown = batchTestMat(d.hidden, d.ffnDim, s+6, d.bf16)
		layer.AttnCache = core.AttnCache{
			K:        make([]float32, 0),
			V:        make([]float32, 0),
			KvStride: kvStride,
			CacheLen: d.cacheLen,
		}
		if d.q8Cache {
			layer.AttnCache.KQ8 = make([]int8, 0)
			layer.AttnCache.KQ8S = make([]float32, 0)
			layer.AttnCache.VQ8 = make([]int8, 0)
			layer.AttnCache.VQ8S = make([]float32, 0)
		}
	}
	return m
}

// newBatchTestRuntime binds CUDA ops to an instance and wraps it in a
// GraphRuntime. The returned cleanup releases the stream/blas/ops.
func newBatchTestRuntime(t testing.TB, inst *core.Instance) (*GraphRuntime, func()) {
	t.Helper()
	stream, err := native.NewStream()
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	blas, err := native.NewBlasHandle(stream)
	if err != nil {
		_ = stream.Destroy()
		t.Fatalf("NewBlasHandle: %v", err)
	}
	ops := NewOps(stream, blas)
	inst.SetOps(ops)
	rt := &cudaRuntime{model: (*simd.Instance)(inst), ops: ops, stream: stream, blas: blas}
	gr := NewGraphRuntime(rt, inst)
	cleanup := func() {
		_ = ops.Close()
		_ = blas.Destroy()
		_ = stream.Destroy()
	}
	return gr, cleanup
}

// sequentialOracle runs the untouched sequential CUDA prefill on its own
// instance so it cannot share device state with the batched run.
func sequentialOracle(t testing.TB, d batchTestDims, tokens []int) []float32 {
	t.Helper()
	gr, cleanup := newBatchTestRuntime(t, batchTestModel(d))
	defer cleanup()
	logits, err := gr.cudaRuntime.PrefillTokens(tokens)
	if err != nil {
		t.Fatalf("sequential PrefillTokens: %v", err)
	}
	return append([]float32(nil), logits...)
}

func compareBatchLogits(t *testing.T, name string, got, want []float32, atol, rtol float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: logit length got %d want %d", name, len(got), len(want))
	}
	var worst float32
	for i := range got {
		diff := float32(math.Abs(float64(got[i] - want[i])))
		allowed := atol + rtol*float32(math.Abs(float64(want[i])))
		if diff > allowed {
			t.Fatalf("%s: logit[%d] got %v want %v (diff %v > allowed %v)", name, i, got[i], want[i], diff, allowed)
		}
		if diff > worst {
			worst = diff
		}
	}
	t.Logf("%s: max abs logit diff = %v", name, worst)
}

// TestBatchedPrefillParityN2 checks the batched path matches the sequential
// oracle for a two-token prompt that fits entirely in the KV cache.
func TestBatchedPrefillParityN2(t *testing.T) {
	if !cudaAvailable(t) {
		return
	}
	d := defaultBatchTestDims()
	tokens := []int{1, 3}

	// The public method must select the batched path.
	grBatched, cleanup := newBatchTestRuntime(t, batchTestModel(d))
	defer cleanup()
	if !grBatched.batchedPrefillEligible(tokens) {
		t.Fatalf("expected two-token prompt to be eligible for batched prefill")
	}
	got, err := grBatched.PrefillTokens(tokens)
	if err != nil {
		t.Fatalf("batched PrefillTokens: %v", err)
	}
	got = append([]float32(nil), got...)

	want := sequentialOracle(t, d, tokens)
	compareBatchLogits(t, "N2", got, want, 1e-4, 1e-3)
}

// TestBatchedPrefillPlanParityN2 drives newBatchedPrefillPlan directly to prove
// the device batch path itself (not a silent fallback) matches the oracle.
func TestBatchedPrefillPlanParityN2(t *testing.T) {
	if !cudaAvailable(t) {
		return
	}
	d := defaultBatchTestDims()
	tokens := []int{2, 5}

	gr, cleanup := newBatchTestRuntime(t, batchTestModel(d))
	defer cleanup()
	plan, err := gr.newBatchedPrefillPlan(tokens)
	if err != nil {
		t.Fatalf("newBatchedPrefillPlan: %v", err)
	}
	if plan == nil {
		t.Fatal("expected a batched plan, got nil (path would have fallen back)")
	}
	got, err := plan.run()
	if err != nil {
		plan.free()
		t.Fatalf("plan.run: %v", err)
	}
	got = append([]float32(nil), got...)
	plan.free()

	if gr.inst.Pos != len(tokens) {
		t.Fatalf("after batched prefill Pos = %d, want %d", gr.inst.Pos, len(tokens))
	}
	want := sequentialOracle(t, d, tokens)
	compareBatchLogits(t, "plan-N2", got, want, 1e-4, 1e-3)
}

// TestBatchedPrefillRingWrap runs a prompt longer than the KV cache length so
// the ring store/attention index wraps, and checks parity with the oracle.
func TestBatchedPrefillRingWrap(t *testing.T) {
	if !cudaAvailable(t) {
		return
	}
	d := defaultBatchTestDims()
	d.cacheLen = 8 // smaller than the prompt -> ring wrap
	tokens := make([]int, 12)
	for i := range tokens {
		tokens[i] = (i*3 + 1) % d.vocab
	}

	gr, cleanup := newBatchTestRuntime(t, batchTestModel(d))
	defer cleanup()
	if !gr.batchedPrefillEligible(tokens) {
		t.Fatal("expected ring-wrap prompt to be eligible")
	}
	got, err := gr.PrefillTokens(tokens)
	if err != nil {
		t.Fatalf("batched PrefillTokens: %v", err)
	}
	got = append([]float32(nil), got...)

	want := sequentialOracle(t, d, tokens)
	compareBatchLogits(t, "ring", got, want, 2e-4, 2e-3)
}

// TestBatchedPrefillParityQ8Cache exercises the Q8 KV cache variant, which
// routes through StoreKVQ8RowBroadcast and the mixed attention kernel.
func TestBatchedPrefillParityQ8Cache(t *testing.T) {
	if !cudaAvailable(t) {
		return
	}
	d := defaultBatchTestDims()
	d.q8Cache = true
	tokens := []int{1, 3, 6}

	gr, cleanup := newBatchTestRuntime(t, batchTestModel(d))
	defer cleanup()
	if !gr.batchedPrefillEligible(tokens) {
		t.Fatal("expected Q8-cache prompt to be eligible")
	}
	got, err := gr.PrefillTokens(tokens)
	if err != nil {
		t.Fatalf("batched PrefillTokens: %v", err)
	}
	got = append([]float32(nil), got...)

	want := sequentialOracle(t, d, tokens)
	compareBatchLogits(t, "q8", got, want, 5e-4, 5e-3)
}

// TestBatchedPrefillParityBF16 exercises BF16 dense weights (the encoding of
// the target on-disk models). The sequential oracle falls back to its host FFN
// for BF16 because cublas rejects the mixed BF16/F32 arguments FFNBlock uses,
// so this parity check also covers that fallback.
func TestBatchedPrefillParityBF16(t *testing.T) {
	if !cudaAvailable(t) {
		return
	}
	d := defaultBatchTestDims()
	d.bf16 = true
	tokens := []int{1, 3, 6}

	gr, cleanup := newBatchTestRuntime(t, batchTestModel(d))
	defer cleanup()
	if !gr.batchedPrefillEligible(tokens) {
		t.Fatal("expected BF16 model to be eligible")
	}
	got, err := gr.PrefillTokens(tokens)
	if err != nil {
		t.Fatalf("batched PrefillTokens: %v", err)
	}
	got = append([]float32(nil), got...)

	want := sequentialOracle(t, d, tokens)
	compareBatchLogits(t, "bf16", got, want, 2e-3, 2e-2)
}

// TestBatchedPrefillSingleTokenFallback verifies a single-token prompt is
// rejected by the gate (it falls back to the sequential path) and still
// produces correct logits.
func TestBatchedPrefillSingleTokenFallback(t *testing.T) {
	if !cudaAvailable(t) {
		return
	}
	d := defaultBatchTestDims()
	tokens := []int{7}

	gr, cleanup := newBatchTestRuntime(t, batchTestModel(d))
	defer cleanup()
	if gr.batchedPrefillEligible(tokens) {
		t.Fatal("single-token prompt must not be eligible for batched prefill")
	}
	got, err := gr.PrefillTokens(tokens)
	if err != nil {
		t.Fatalf("fallback PrefillTokens: %v", err)
	}
	want := sequentialOracle(t, d, tokens)
	compareBatchLogits(t, "N1-fallback", got, want, 1e-5, 1e-5)
}

// TestBatchedPrefillGateRejectsRoPELess verifies the gate rejects a model whose
// layers disable RoPE and that PrefillTokens then still matches the oracle.
func TestBatchedPrefillGateRejectsRoPELess(t *testing.T) {
	if !cudaAvailable(t) {
		return
	}
	d := defaultBatchTestDims()
	d.noRoPE = true
	tokens := []int{1, 4}

	gr, cleanup := newBatchTestRuntime(t, batchTestModel(d))
	defer cleanup()
	if gr.batchedPrefillEligible(tokens) {
		t.Fatal("RoPE-less model must be rejected by the batched gate")
	}
	got, err := gr.PrefillTokens(tokens)
	if err != nil {
		t.Fatalf("fallback PrefillTokens: %v", err)
	}
	want := sequentialOracle(t, d, tokens)
	compareBatchLogits(t, "gate-reject", got, want, 1e-5, 1e-5)
}

// TestBatchedPrefillAppliesLogitPostprocess verifies the batched output head
// matches the sequential oracle for models that set FinalLogitSoftcap or
// LMHeadMultiplier. These models used to be rejected by the gate, because the
// sequential CUDA path's flushLastResult clobbered its own postprocessing; now
// that the oracle applies the knobs, the batched path must match it rather than
// decline. TestForwardTokenAppliesLogitPostprocessCUDA is the counterpart that
// proves the sequential path really applies them, so this comparison cannot
// pass vacuously.
func TestBatchedPrefillAppliesLogitPostprocess(t *testing.T) {
	if !cudaAvailable(t) {
		return
	}

	cases := []struct {
		name   string
		apply  func(*batchTestDims)
		tokens []int
	}{
		{"softcap", func(d *batchTestDims) { d.finalSoftcap = 10 }, []int{1, 6}},
		{"lm_head_multiplier", func(d *batchTestDims) { d.lmHeadMult = 1.5 }, []int{2, 5}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			d := defaultBatchTestDims()
			tc.apply(&d)

			gr, cleanup := newBatchTestRuntime(t, batchTestModel(d))
			defer cleanup()
			if !gr.batchedPrefillEligible(tc.tokens) {
				t.Fatal("model with non-trivial logit postprocessing should be eligible for batched prefill")
			}
			got, err := gr.PrefillTokens(tc.tokens)
			if err != nil {
				t.Fatalf("batched PrefillTokens: %v", err)
			}
			got = append([]float32(nil), got...)
			want := sequentialOracle(t, d, tc.tokens)
			compareBatchLogits(t, "postprocess-"+tc.name, got, want, 1e-5, 1e-5)
		})
	}
}

// TestBatchedPrefillKillSwitch verifies MANTLE_CUDA_BATCHED_PREFILL=0 forces
// the sequential path even for an otherwise-eligible prompt.
func TestBatchedPrefillKillSwitch(t *testing.T) {
	if !cudaAvailable(t) {
		return
	}
	d := defaultBatchTestDims()
	tokens := []int{1, 3}
	t.Setenv("MANTLE_CUDA_BATCHED_PREFILL", "0")

	gr, cleanup := newBatchTestRuntime(t, batchTestModel(d))
	defer cleanup()
	if !gr.batchedPrefillEligible(tokens) {
		t.Fatal("expected prompt to satisfy the batched gate (kill switch is checked separately)")
	}
	if batchedPrefillEnabled() {
		t.Fatal("kill switch should disable batched prefill")
	}
	got, err := gr.PrefillTokens(tokens)
	if err != nil {
		t.Fatalf("kill-switch PrefillTokens: %v", err)
	}
	want := sequentialOracle(t, d, tokens)
	compareBatchLogits(t, "kill-switch", got, want, 1e-5, 1e-5)
}

func cudaAvailable(t testing.TB) bool {
	t.Helper()
	count, err := native.DeviceCount()
	if err != nil {
		t.Skipf("cannot query CUDA devices: %v", err)
		return false
	}
	if count < 1 {
		t.Skip("no CUDA device available")
		return false
	}
	return true
}

func benchPrefillDims() batchTestDims {
	d := defaultBatchTestDims()
	d.layers = 4
	d.hidden = 128
	d.nHead = 8
	d.headDim = 16
	d.kvHeads = 4
	d.ffnDim = 256
	d.maxCtx = 64
	d.cacheLen = 64
	return d
}

func benchPrefillTokens(d batchTestDims, n int) []int {
	tokens := make([]int, n)
	for i := range tokens {
		tokens[i] = (i * 7) % d.vocab
	}
	return tokens
}

// BenchmarkBatchedPrefill measures the device-resident batched prefill path,
// including its per-call scratch allocation.
func BenchmarkBatchedPrefill(b *testing.B) {
	if !cudaAvailable(b) {
		return
	}
	d := benchPrefillDims()
	tokens := benchPrefillTokens(d, 32)
	gr, cleanup := newBatchTestRuntime(b, batchTestModel(d))
	defer cleanup()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		(*simd.Instance)(gr.inst).Reset()
		if _, err := gr.PrefillTokens(tokens); err != nil {
			b.Fatalf("batched prefill: %v", err)
		}
	}
}

// BenchmarkSequentialPrefill measures the untouched per-token sequential path
// for the same prompt, as a timing reference.
func BenchmarkSequentialPrefill(b *testing.B) {
	if !cudaAvailable(b) {
		return
	}
	d := benchPrefillDims()
	tokens := benchPrefillTokens(d, 32)
	gr, cleanup := newBatchTestRuntime(b, batchTestModel(d))
	defer cleanup()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		(*simd.Instance)(gr.inst).Reset()
		if _, err := gr.cudaRuntime.PrefillTokens(tokens); err != nil {
			b.Fatalf("sequential prefill: %v", err)
		}
	}
}
