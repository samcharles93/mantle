//go:build cuda

package cuda

import (
	"fmt"
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
	// q8 builds layer/output weights as DTypeQ8 quant mats so the batched
	// quant GEMM path is exercised.
	q8 bool

	// hybrid builds an LFM3-shaped model: layer 0 a short-conv layer, layer 1
	// sliding-window attention with a per-head output gate and sandwich norms,
	// the rest plain full attention.
	hybrid bool
	window int
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

// batchTestMatQ8 builds a DTypeQ8 quant mat (f16 block scales, align64, int8
// data) from the same deterministic pattern.
func batchTestMatQ8(r, c int, seed uint32) *core.Mat {
	data := make([]float32, r*c)
	batchTestFill(data, seed)
	bpr := (c + 31) / 32
	totalBlocks := r * bpr
	scales := make([]byte, totalBlocks*2)
	quant := make([]byte, totalBlocks*32)
	for b := 0; b < totalBlocks; b++ {
		maxAbs := float32(0)
		for i := 0; i < 32; i++ {
			idx := b*32 + i
			v := float32(0)
			if idx < r*c {
				v = data[idx]
			}
			if a := float32(math.Abs(float64(v))); a > maxAbs {
				maxAbs = a
			}
		}
		scale := float32(1e-9)
		if maxAbs > 0 {
			scale = maxAbs / 127
		}
		u := core.Float32ToFloat16(scale)
		scales[b*2] = byte(u)
		scales[b*2+1] = byte(u >> 8)
		for i := 0; i < 32; i++ {
			idx := b*32 + i
			v := float32(0)
			if idx < r*c {
				v = data[idx]
			}
			q := int32(math.Round(float64(v / scale)))
			if q > 127 {
				q = 127
			} else if q < -127 {
				q = -127
			}
			quant[b*32+i] = byte(int8(q))
		}
	}
	buf := append([]byte{}, scales...)
	for len(buf)%64 != 0 {
		buf = append(buf, 0)
	}
	buf = append(buf, quant...)
	return &core.Mat{R: r, C: c, Stride: c, DType: mcf.DTypeQ8, Raw: buf}
}

// batchTestModel builds a plain dense transformer instance. RoPE is enabled
// with an instance-level inverse-frequency table (the gate rejects NoRoPE).
func batchTestModel(d batchTestDims) *core.Instance {
	headDim := d.headDim
	qDim := d.nHead * headDim
	kvStride := d.kvHeads * headDim

	wmatOut := batchTestMat(d.vocab, d.hidden, d.seed+1, d.bf16)
	if d.q8 {
		wmatOut = batchTestMatQ8(d.vocab, d.hidden, d.seed+1)
	}
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
		Output:      wmatOut,
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
			Tmp2:     make([]float32, d.hidden),
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
			ScProj:   make([]float32, 3*d.hidden),
			ScBx:     make([]float32, d.hidden),
			ScConv:   make([]float32, d.hidden),
			AttnGate: make([]float32, 2*qDim),
		},
	}
	for i := range m.OutputNorm {
		m.OutputNorm[i] = 1.0
	}

	wmat := func(r, c int, seed uint32) *core.Mat {
		if d.q8 {
			return batchTestMatQ8(r, c, seed)
		}
		return batchTestMat(r, c, seed, d.bf16)
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
		if d.hybrid && li%2 == 1 {
			// Recurrent short-conv layer (LFM3 interleaves them).
			layer.IsRecurrent = true
			layer.HeadKV = 0
			const klen = 4
			kernel := core.NewMat(d.hidden, klen)
			batchTestFill(kernel.Data, s+20)
			layer.ShortConvKernel = &kernel
			layer.ShortConvInProj = wmat(3*d.hidden, d.hidden, s+21)
			layer.ShortConvOutProj = wmat(d.hidden, d.hidden, s+22)
			layer.ShortConvState = core.ShortConvState{Buf: make([]float32, (klen-1)*d.hidden), KernelLen: klen}
		} else if d.hybrid && li == 2 {
			layer.AttnType = "sliding_attention"
			layer.AttnWindow = d.window
			// LFM3-style per-layer zero-tail RoPE: half the frequencies zero.
			half := headDim / 2
			zt := make([]float64, half)
			for i := range zt {
				if 2*i < headDim/2 {
					zt[i] = math.Pow(10000, -2*float64(i)/float64(headDim))
				}
			}
			layer.RopeInvFreq = zt
			layer.RopeAttnScale = 1
			gate := wmat(d.nHead, d.hidden, s+23) // per-head gate
			layer.AttnGate = gate
			layer.PostAttnNorm = make([]float32, d.hidden)
			layer.PostFfnNorm = make([]float32, d.hidden)
			for i := range layer.PostAttnNorm {
				layer.PostAttnNorm[i] = 1.0 + float32(i%3)*0.1
				layer.PostFfnNorm[i] = 1.0 - float32(i%3)*0.05
			}
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
		if !layer.IsRecurrent {
			layer.Wq = wmat(qDim, d.hidden, s)
			layer.Wk = wmat(kvStride, d.hidden, s+1)
			layer.Wv = wmat(kvStride, d.hidden, s+2)
			layer.Wo = wmat(d.hidden, qDim, s+3)
			layer.AttnCache = core.AttnCache{
				K:        make([]float32, 0),
				V:        make([]float32, 0),
				KvStride: kvStride,
				CacheLen: d.cacheLen,
			}
		}
		layer.FfnUp = wmat(d.ffnDim, d.hidden, s+4)
		layer.FfnGate = wmat(d.ffnDim, d.hidden, s+5)
		layer.FfnDown = wmat(d.hidden, d.ffnDim, s+6)
		if d.q8Cache && !layer.IsRecurrent {
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

// TestBatchedPrefillGateQuantWeightsDequantMode pins the weight gate: quantized
// payloads are only eligible when the runtime dequantizes weights to dense
// device copies (dequant mode), where the batched GEMM runs them as F16.
// In auto/quant mode they stay on the per-token quant-kernel path.
func TestBatchedPrefillGateQuantWeightsDequantMode(t *testing.T) {
	quantMat := &core.Mat{R: 8, C: 64, Stride: 64, DType: mcf.DTypeQ8, Raw: make([]byte, 8*64+8*64/32*2)}
	denseMat := &core.Mat{R: 8, C: 64, Stride: 64, DType: mcf.DTypeBF16, Raw: make([]byte, 8*64*2)}

	newOps := func() *Ops {
		return &Ops{
			weights:       map[*core.Mat]deviceMat{},
			qweights:      map[*core.Mat]deviceQuantMat{},
			offloadedMats: map[*core.Mat]struct{}{},
		}
	}

	t.Setenv("MANTLE_CUDA_WEIGHT_MODE", "dequant")
	o := newOps()
	if !batchedWeightDense(o, quantMat) {
		t.Fatal("dequant mode must accept quantized weights (deviceMat dequantizes to F16)")
	}
	if !batchedWeightDense(o, denseMat) {
		t.Fatal("dense bf16 weights must stay eligible in dequant mode")
	}

	t.Setenv("MANTLE_CUDA_WEIGHT_MODE", "auto")
	o = newOps()
	if !batchedWeightDense(o, quantMat) {
		t.Fatal("auto mode must accept quantized weights (batched quant GEMM kernel)")
	}
	if !batchedWeightDense(o, denseMat) {
		t.Fatal("dense bf16 weights must stay eligible in auto mode")
	}

	t.Setenv("MANTLE_CUDA_WEIGHT_MODE", "dequant")
	o = newOps()
	o.qweights[quantMat] = deviceQuantMat{}
	if batchedWeightDense(o, quantMat) {
		t.Fatal("a weight with pre-loaded quant data must never take the dense GEMM path")
	}
}

// TestBatchedPrefillParityHybrid checks the batched path against the sequential
// oracle on an LFM3-shaped model: a short-conv layer, a sliding-window layer
// with a per-head output gate and sandwich norms, and plain attention layers.
// The small cache forces the ring-wrap interleaved path on the windowed layer.
func TestBatchedPrefillParityHybrid(t *testing.T) {
	if !cudaAvailable(t) {
		return
	}
	d := defaultBatchTestDims()
	d.hybrid = true
	d.window = 6
	d.cacheLen = 8 // smaller than the prompt: ring wrap + window clamping
	d.layers = 8
	d.hidden = 256
	d.ffnDim = 512
	d.nHead = 8
	d.headDim = 32
	d.kvHeads = 4
	d.maxCtx = 128
	d.vocab = 512
	for _, n := range []int{2, 6, 13, 40} {
		t.Run(fmt.Sprintf("n%d", n), func(t *testing.T) {
			tokens := make([]int, n)
			for i := range tokens {
				tokens[i] = (i*3 + 1) % d.vocab
			}
			gr, cleanup := newBatchTestRuntime(t, batchTestModel(d))
			defer cleanup()
			if !gr.batchedPrefillEligible(tokens) {
				t.Fatal("hybrid model should be eligible for the batched prefill")
			}
			got, err := gr.PrefillTokens(tokens)
			if err != nil {
				t.Fatalf("batched PrefillTokens: %v", err)
			}
			got = append([]float32(nil), got...)
			// The CPU simd path is the semantic oracle (the synthetic instance
			// is not wired for the CUDA per-token conv fast path); force the
			// per-token CPU path so no batch scratch is needed.
			t.Setenv("MANTLE_CPU_BATCHED_PREFILL", "0")
			oracle := batchTestModel(d)
			(*simd.Instance)(oracle).BindDefaultOps()
			want, err := (*simd.Instance)(oracle).PrefillTokens(tokens)
			if err != nil {
				t.Fatalf("oracle PrefillTokens: %v", err)
			}
			compareBatchLogits(t, fmt.Sprintf("hybrid/n%d", n), got, append([]float32(nil), want...), 2e-3, 2e-2)

			// Decode continuity: run a few greedy tokens on the batched
			// instance and on a fresh sequential oracle instance (both after
			// prefill) and compare the argmax sequence — this exercises the
			// conv-state rotation and KV handoff.
			seqRT, seqCleanup := newBatchTestRuntime(t, batchTestModel(d))
			defer seqCleanup()
			if _, err := seqRT.cudaRuntime.PrefillTokens(tokens); err != nil {
				t.Fatalf("sequential prefill: %v", err)
			}
			tokG := batchArgmax(got)
			tokW := tokG
			for i := 0; i < 4; i++ {
				lg, err := gr.ForwardToken(tokG)
				if err != nil {
					t.Fatalf("batched ForwardToken: %v", err)
				}
				lw, err := seqRT.ForwardToken(tokW)
				if err != nil {
					t.Fatalf("sequential ForwardToken: %v", err)
				}
				ng, nw := batchArgmax(lg), batchArgmax(lw)
				if ng != nw {
					t.Fatalf("hybrid/n%d: greedy token %d differs: got %d want %d", n, i, ng, nw)
				}
				tokG, tokW = ng, nw
			}
		})
	}
}

func batchArgmax(x []float32) int {
	best := 0
	for i := range x {
		if x[i] > x[best] {
			best = i
		}
	}
	return best
}

// TestBatchedPrefillParityQ8 checks the batched quant GEMM path (DTypeQ8
// weights in auto mode) against the sequential oracle, and that the public
// PrefillTokens really selects the batched path for quantized weights.
func TestBatchedPrefillParityQ8(t *testing.T) {
	if !cudaAvailable(t) {
		return
	}
	d := defaultBatchTestDims()
	d.q8 = true
	d.layers = 2
	tokens := []int{1, 3, 7, 11, 2}

	grBatched, cleanup := newBatchTestRuntime(t, batchTestModel(d))
	defer cleanup()
	if !grBatched.batchedPrefillEligible(tokens) {
		t.Fatal("q8 model should be eligible for the batched quant path")
	}
	got, err := grBatched.PrefillTokens(tokens)
	if err != nil {
		t.Fatalf("batched PrefillTokens: %v", err)
	}
	got = append([]float32(nil), got...)

	want := sequentialOracle(t, d, tokens)
	// Quant kernels accumulate in a different order than the f32 oracle.
	compareBatchLogits(t, "Q8", got, want, 2e-3, 2e-2)

	// Decode continuity: greedy tokens from the batched instance vs a fresh
	// sequential instance must agree.
	seqRT, seqCleanup := newBatchTestRuntime(t, batchTestModel(d))
	defer seqCleanup()
	if _, err := seqRT.cudaRuntime.PrefillTokens(tokens); err != nil {
		t.Fatalf("sequential prefill: %v", err)
	}
	tokG := batchArgmax(got)
	tokW := tokG
	for i := 0; i < 4; i++ {
		lg, err := grBatched.ForwardToken(tokG)
		if err != nil {
			t.Fatalf("batched ForwardToken: %v", err)
		}
		lw, err := seqRT.ForwardToken(tokW)
		if err != nil {
			t.Fatalf("sequential ForwardToken: %v", err)
		}
		ng, nw := batchArgmax(lg), batchArgmax(lw)
		if ng != nw {
			t.Fatalf("Q8: greedy token %d differs: got %d want %d", i, ng, nw)
		}
		tokG, tokW = ng, nw
	}
}
