package simd

import (
	"fmt"
	"testing"

	core "github.com/samcharles93/mantle/internal/backend/core"
	"github.com/samcharles93/mantle/pkg/mcf"
)

// These tests pin the batched prefill path (runtime_batch.go / attn_batch.go) to
// the untouched per-token path: every logits row must match, the KV cache must
// be left in the sequential-equivalent state, and ineligible prompts must fall
// back instead of half-running a batch.

const (
	batchParityVocab   = 16
	batchParityContext = 64
	batchParityTol     = 1e-5
)

// newBatchParityInstance builds a synthetic dense instance: newTestInstance plus
// the pieces the loader normally supplies (RoPE tables, attention type, MaxQDim,
// a KV cache of cacheLen slots, and the batch scratch).
func newBatchParityInstance(tb testing.TB, cacheLen int, dtype mcf.TensorDType) *Instance {
	tb.Helper()
	m := newTestInstance(batchParityVocab, 8, 2)
	m.BindDefaultOps()
	m.MaxContext = batchParityContext
	m.MaxQDim = m.HeadCount * m.HeadDim
	m.Scratch.Scores = make([]float32, batchParityContext)
	m.RopeInvFreq = []float64{1, 0.01}
	m.RopeAttnScale = 1
	kvStride := m.Layers[0].AttnCache.KvStride
	for i := range m.Layers {
		layer := &m.Layers[i]
		layer.NoRoPE = false
		layer.AttnType = "full_attention"
		layer.SharedKVSource = -1
		layer.LayerScale = 1 // the loader's dense default; without it the sequential path takes the gemma4 block
		layer.AttnCache = core.AttnCache{
			K:        make([]float32, kvStride*cacheLen),
			V:        make([]float32, kvStride*cacheLen),
			KvStride: kvStride,
			CacheLen: cacheLen,
			Cap:      cacheLen,
		}
	}
	if dtype != mcf.DTypeF32 {
		reencodeLayerWeights(m, dtype)
	}
	sizeBatchScratch((*Instance)(m))
	return (*Instance)(m)
}

// reencodeLayerWeights moves every dense projection into 2-byte raw storage, the
// way an f16/bf16 MCF load leaves it (Data nil, Raw set). MatVec and GemmParWT
// then take different decode paths, which is exactly what the parity tests must
// cover.
func reencodeLayerWeights(m *core.Instance, dtype mcf.TensorDType) {
	for i := range m.Layers {
		layer := &m.Layers[i]
		for _, w := range []*core.Mat{layer.Wq, layer.Wk, layer.Wv, layer.Wo, layer.FfnUp, layer.FfnGate, layer.FfnDown} {
			var raw []byte
			if dtype == mcf.DTypeF16 {
				raw = f16Raw(w.Data)
			} else {
				raw = bf16Raw(w.Data)
			}
			w.Data = nil
			w.Raw = raw
			w.DType = dtype
		}
	}
}

// sequentialLogits runs the per-token path, the correctness oracle for these
// tests.
func sequentialLogits(tb testing.TB, m *Instance, tokens []int) [][]float32 {
	tb.Helper()
	out := make([][]float32, 0, len(tokens))
	for _, tok := range tokens {
		logits, err := m.ForwardToken(tok)
		if err != nil {
			tb.Fatalf("ForwardToken(%d): %v", tok, err)
		}
		out = append(out, append([]float32(nil), logits...))
	}
	return out
}

func compareLogitRows(tb testing.TB, label string, got, want [][]float32, tol float32) {
	tb.Helper()
	if len(got) != len(want) {
		tb.Fatalf("%s: %d logits rows, want %d", label, len(got), len(want))
	}
	for i := range got {
		if !floatsEqual(got[i], want[i], tol) {
			tb.Fatalf("%s: logits row %d differ: got %v, want %v", label, i, got[i], want[i])
		}
	}
}

func compareKVCache(tb testing.TB, got, want *Instance, tol float32) {
	tb.Helper()
	for i := range got.Layers {
		g := got.Layers[i].AttnCache
		w := want.Layers[i].AttnCache
		if !floatsEqual(g.K, w.K, tol) {
			tb.Fatalf("layer %d K cache differs: got %v, want %v", i, g.K, w.K)
		}
		if !floatsEqual(g.V, w.V, tol) {
			tb.Fatalf("layer %d V cache differs: got %v, want %v", i, g.V, w.V)
		}
	}
}

// TestBatchPrefillParitySynthetic checks every logits row of a batched prompt
// against the same prompt run one token at a time, for f32, bf16 and f16
// weights and for batch sizes on both sides of the single-token gate.
func TestBatchPrefillParitySynthetic(t *testing.T) {
	dtypes := []struct {
		name  string
		dtype mcf.TensorDType
	}{
		{"f32", mcf.DTypeF32},
		{"bf16", mcf.DTypeBF16},
		{"f16", mcf.DTypeF16},
	}
	for _, dt := range dtypes {
		for _, n := range []int{1, 2, 3, 8} {
			t.Run(fmt.Sprintf("%s/n%d", dt.name, n), func(t *testing.T) {
				tokens := make([]int, n)
				for i := range tokens {
					tokens[i] = (i*3 + 1) % batchParityVocab
				}
				batched := newBatchParityInstance(t, batchParityContext, dt.dtype)
				sequential := newBatchParityInstance(t, batchParityContext, dt.dtype)

				switch {
				case n < 2 && batched.batchEligible(tokens):
					t.Fatal("single-token prompt must not be eligible for a batch")
				case n >= 2 && !batched.batchEligible(tokens):
					t.Fatalf("dense prompt should be eligible: %s", batched.batchIneligibleReason(tokens))
				}

				want := sequentialLogits(t, sequential, tokens)
				got, err := (*Instance)(batched).ForwardTokens(tokens)
				if err != nil {
					t.Fatalf("ForwardTokens: %v", err)
				}
				compareLogitRows(t, "logits", got, want, batchParityTol)
				if batched.Pos != sequential.Pos {
					t.Fatalf("position = %d, want %d", batched.Pos, sequential.Pos)
				}
			})
		}
	}
}

// TestBatchPrefillParityConfigVariants covers the config switches that route
// the sequential path elsewhere or postprocess the logits, all of which the
// batched path reproduces.
func TestBatchPrefillParityConfigVariants(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(m *Instance)
	}{
		{"attention softcap", func(m *Instance) { m.Config.Config.AttnLogitSoftcap = 8 }},
		{"logit postprocess", func(m *Instance) {
			m.Config.Config.LMHeadMultiplier = 1.5
			m.Config.Config.FinalLogitSoftcap = 10
		}},
		{"embedding multiplier", func(m *Instance) { m.Config.Config.EmbeddingMultiplier = 0.9 }},
		{"muP", func(m *Instance) {
			m.Config.Config.MuPEnabled = true
			m.MuPScale = 2
		}},
	}
	tokens := []int{1, 4, 2, 7}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			batched := newBatchParityInstance(t, batchParityContext, mcf.DTypeF32)
			sequential := newBatchParityInstance(t, batchParityContext, mcf.DTypeF32)
			for _, m := range []*Instance{batched, sequential} {
				tc.mutate(m)
			}
			if !batched.batchEligible(tokens) {
				t.Fatalf("prompt should be eligible: %s", batched.batchIneligibleReason(tokens))
			}
			want := sequentialLogits(t, sequential, tokens)
			got, err := batched.ForwardTokens(tokens)
			if err != nil {
				t.Fatalf("ForwardTokens: %v", err)
			}
			compareLogitRows(t, "config", got, want, batchParityTol)
		})
	}
}

// TestBatchPrefillPlanMatchesOracle drives newBatchPrefillPlan directly, so a
// pass proves the batched machinery ran rather than silently falling back.
func TestBatchPrefillPlanMatchesOracle(t *testing.T) {
	tokens := []int{1, 4, 2, 7}
	batched := newBatchParityInstance(t, batchParityContext, mcf.DTypeF32)
	sequential := newBatchParityInstance(t, batchParityContext, mcf.DTypeF32)

	plan, err := (*Instance)(batched).newBatchPrefillPlan(tokens)
	if err != nil {
		t.Fatalf("newBatchPrefillPlan: %v (path would have fallen back)", err)
	}
	got := plan.run()
	if batched.Pos != len(tokens) {
		t.Fatalf("after batched prefill Pos = %d, want %d", batched.Pos, len(tokens))
	}
	compareLogitRows(t, "plan", got, sequentialLogits(t, sequential, tokens), batchParityTol)
}

// TestBatchPrefillRingWrapParity runs a prompt longer than the KV cache so the
// per-row store/attend interleaving is required, and then checks the wrapped
// cache still serves a following single token correctly.
func TestBatchPrefillRingWrapParity(t *testing.T) {
	const cacheLen = 4
	tokens := []int{1, 4, 2, 7, 3, 5} // len 6 > cacheLen
	batched := newBatchParityInstance(t, cacheLen, mcf.DTypeF32)
	sequential := newBatchParityInstance(t, cacheLen, mcf.DTypeF32)

	if !batched.batchEligible(tokens) {
		t.Fatalf("ring-wrap prompt should be eligible: %s", batched.batchIneligibleReason(tokens))
	}
	want := sequentialLogits(t, sequential, tokens)
	got, err := (*Instance)(batched).ForwardTokens(tokens)
	if err != nil {
		t.Fatalf("ForwardTokens: %v", err)
	}
	compareLogitRows(t, "ring-wrap", got, want, batchParityTol)
	compareKVCache(t, batched, sequential, batchParityTol)

	gotNext, err := (*Instance)(batched).ForwardToken(9)
	if err != nil {
		t.Fatalf("batched ForwardToken after wrap: %v", err)
	}
	wantNext, err := (*Instance)(sequential).ForwardToken(9)
	if err != nil {
		t.Fatalf("sequential ForwardToken after wrap: %v", err)
	}
	if !floatsEqual(gotNext, wantNext, batchParityTol) {
		t.Fatalf("logits after ring wrap differ: got %v, want %v", gotNext, wantNext)
	}
}

// TestBatchPrefillLeavesSequentialKVState proves the KV cache a batched prefill
// leaves behind is the one the sequential path would have written, so a later
// single-token step continues correctly.
func TestBatchPrefillLeavesSequentialKVState(t *testing.T) {
	tokens := []int{1, 3, 2, 6}
	batched := newBatchParityInstance(t, batchParityContext, mcf.DTypeF32)
	sequential := newBatchParityInstance(t, batchParityContext, mcf.DTypeF32)

	if _, err := (*Instance)(batched).ForwardTokens(tokens); err != nil {
		t.Fatalf("ForwardTokens: %v", err)
	}
	sequentialLogits(t, sequential, tokens)
	compareKVCache(t, batched, sequential, batchParityTol)

	got, err := (*Instance)(batched).ForwardToken(9)
	if err != nil {
		t.Fatalf("batched ForwardToken: %v", err)
	}
	got = append([]float32(nil), got...)
	want, err := (*Instance)(sequential).ForwardToken(9)
	if err != nil {
		t.Fatalf("sequential ForwardToken: %v", err)
	}
	if !floatsEqual(got, want, batchParityTol) {
		t.Fatalf("logits after batched prefill differ from sequential: got %v, want %v", got, want)
	}
}

// TestBatchPrefillGateRejects checks that every excluded feature makes the gate
// refuse a batch. A fresh instance passing the same prompt is the control.
func TestBatchPrefillGateRejects(t *testing.T) {
	tokens := []int{1, 2}
	kvStride := 8

	cases := []struct {
		name   string
		tokens []int
		mutate func(m *Instance)
	}{
		{name: "single token", tokens: []int{7}},
		{name: "batch over MaxBatch", mutate: func(m *Instance) { m.MaxBatch = 1 }},
		{name: "beyond context", mutate: func(m *Instance) { m.Pos = batchParityContext - 1 }},
		{name: "batch scratch missing", mutate: func(m *Instance) { m.Scratch.BatchX = nil }},
		{name: "gemma4 per-layer input", mutate: func(m *Instance) {
			m.Gemma4PerLayer = &core.Gemma4PerLayerInputModel{}
		}},
		{name: "flash attention", mutate: func(m *Instance) { m.Config.Config.FlashAttention = true }},
		{name: "mamba", mutate: func(m *Instance) { m.Layers[0].Mamba = &core.MambaLayer{} }},
		{name: "deltanet", mutate: func(m *Instance) { m.Layers[0].DeltaNet = &core.DeltaNetLayer{} }},
		{name: "recurrent", mutate: func(m *Instance) { m.Layers[0].IsRecurrent = true }},
		{name: "moe", mutate: func(m *Instance) { m.Layers[0].MoE = &core.MoELayer{} }},
		{name: "gemma4 moe", mutate: func(m *Instance) { m.Layers[0].Gemma4MoE = &core.Gemma4MoELayer{} }},
		{name: "gemma4 ple", mutate: func(m *Instance) { m.Layers[0].Gemma4PLE = &core.Gemma4PLELayer{} }},
		{name: "layer scale", mutate: func(m *Instance) { m.Layers[1].LayerScale = 0.5 }},
		{name: "shared kv source", mutate: func(m *Instance) { m.Layers[1].SharedKVSource = 0 }},
		{name: "fused q gate", mutate: func(m *Instance) { m.Layers[0].FusedQGate = true }},
		{name: "attention gate", mutate: func(m *Instance) { m.Layers[0].AttnGate = &core.Mat{R: 8, C: 8, Stride: 8} }},
		{name: "value from key", mutate: func(m *Instance) { m.Layers[0].ValueFromKey = true }},
		{name: "value norm", mutate: func(m *Instance) { m.Layers[0].ApplyVNorm = true }},
		{name: "sliding window", mutate: func(m *Instance) { m.Layers[0].AttnWindow = 4 }},
		{name: "sliding attention type", mutate: func(m *Instance) { m.Layers[0].AttnType = "sliding_attention" }},
		{name: "round activations bf16", mutate: func(m *Instance) { m.Layers[0].RoundActivationsBF16 = true }},
		{name: "no rope", mutate: func(m *Instance) { m.Layers[0].NoRoPE = true }},
		{name: "post-attention norm", mutate: func(m *Instance) { m.Layers[0].PostAttnNorm = make([]float32, 8) }},
		{name: "post-ffn norm", mutate: func(m *Instance) { m.Layers[0].PostFfnNorm = make([]float32, 8) }},
		{name: "q/k norm", mutate: func(m *Instance) { m.Layers[0].AttnQNorm = make([]float32, 4) }},
		{name: "qkv bias", mutate: func(m *Instance) { m.Layers[0].WqBias = make([]float32, 8) }},
		{name: "head dim", mutate: func(m *Instance) { m.Layers[0].HeadDim = 8 }},
		{name: "attention norm width", mutate: func(m *Instance) { m.Layers[0].AttnNorm = make([]float32, 4) }},
		{name: "ffn width", mutate: func(m *Instance) { m.Layers[1].FfnUp = &core.Mat{R: 13, C: 8, Stride: 8} }},
		{name: "quantised projection", mutate: func(m *Instance) {
			m.Layers[0].Wk = &core.Mat{R: kvStride, C: 8, Stride: 8, DType: mcf.DTypeQ8, Raw: make([]byte, kvStride*8)}
		}},
		{name: "quantised output", mutate: func(m *Instance) {
			m.Output = &core.Mat{R: batchParityVocab, C: 8, Stride: 8, DType: mcf.DTypeQ8, Raw: make([]byte, batchParityVocab*8)}
		}},
	}

	control := newBatchParityInstance(t, batchParityContext, mcf.DTypeF32)
	if !control.batchEligible(tokens) {
		t.Fatalf("control instance should be eligible: %s", control.batchIneligibleReason(tokens))
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m := newBatchParityInstance(t, batchParityContext, mcf.DTypeF32)
			toks := tc.tokens
			if toks == nil {
				toks = tokens
			}
			if tc.mutate != nil {
				tc.mutate(m)
			}
			if m.batchEligible(toks) {
				t.Fatalf("gate accepted a model the batched path cannot reproduce; reason=%q",
					m.batchIneligibleReason(toks))
			}
		})
	}
}

// TestBatchPrefillGateFallbackMatchesOracle takes a rejected model end to end:
// ForwardTokens must fall back to the per-token path and still compute the same
// logits.
func TestBatchPrefillGateFallbackMatchesOracle(t *testing.T) {
	tokens := []int{1, 3, 2}
	batched := newBatchParityInstance(t, batchParityContext, mcf.DTypeF32)
	sequential := newBatchParityInstance(t, batchParityContext, mcf.DTypeF32)
	for _, m := range []*Instance{batched, sequential} {
		m.Layers[1].AttnWindow = 2 // sliding window: rejected by the gate
	}
	if batched.batchEligible(tokens) {
		t.Fatal("sliding-window model must not be eligible for a batch")
	}

	want := sequentialLogits(t, sequential, tokens)
	got, err := (*Instance)(batched).ForwardTokens(tokens)
	if err != nil {
		t.Fatalf("fallback ForwardTokens: %v", err)
	}
	compareLogitRows(t, "fallback", got, want, batchParityTol)
}

// TestBatchPrefillKillSwitch checks MANTLE_CPU_BATCHED_PREFILL=0 disables the
// batched path without changing the logits.
func TestBatchPrefillKillSwitch(t *testing.T) {
	t.Setenv(cpuBatchedPrefillEnv, "0")
	tokens := []int{1, 3, 2}

	batched := newBatchParityInstance(t, batchParityContext, mcf.DTypeF32)
	sequential := newBatchParityInstance(t, batchParityContext, mcf.DTypeF32)
	if cpuBatchedPrefillEnabled() {
		t.Fatal("kill switch should disable batched prefill")
	}
	if !batched.batchEligible(tokens) {
		t.Fatal("the gate itself must be independent of the kill switch")
	}
	if _, err := (*Instance)(batched).newBatchPrefillPlan(tokens); err == nil {
		t.Fatal("kill switch must refuse to build a batch plan")
	}

	want := sequentialLogits(t, sequential, tokens)
	got, err := (*Instance)(batched).ForwardTokens(tokens)
	if err != nil {
		t.Fatalf("kill-switch ForwardTokens: %v", err)
	}
	compareLogitRows(t, "kill-switch", got, want, batchParityTol)
}

// newBatchBenchInstance builds a wider synthetic instance so the benchmark
// measures realistic GEMM shapes rather than the parity tests' toy dimensions.
func newBatchBenchInstance(tb testing.TB) *Instance {
	tb.Helper()
	const maxContext = 512
	m := newTestInstance(256, 128, 4)
	m.BindDefaultOps()
	m.MaxContext = maxContext
	m.MaxQDim = m.HeadCount * m.HeadDim
	m.Scratch.Scores = make([]float32, maxContext)
	m.RopeInvFreq = []float64{1, 0.01}
	m.RopeAttnScale = 1
	kvStride := m.Layers[0].AttnCache.KvStride
	for i := range m.Layers {
		layer := &m.Layers[i]
		layer.NoRoPE = false
		layer.AttnType = "full_attention"
		layer.SharedKVSource = -1
		layer.LayerScale = 1 // the loader's dense default; without it the sequential path takes the gemma4 block
		layer.AttnCache = core.AttnCache{
			K:        make([]float32, kvStride*maxContext),
			V:        make([]float32, kvStride*maxContext),
			KvStride: kvStride,
			CacheLen: maxContext,
			Cap:      maxContext,
		}
	}
	sizeBatchScratch((*Instance)(m))
	return (*Instance)(m)
}

// BenchmarkBatchPrefillVsSequential measures the whole-prompt batched path
// against the per-token loop it replaces, on an identical workload.
func BenchmarkBatchPrefillVsSequential(b *testing.B) {
	tokens := make([]int, 16)
	for i := range tokens {
		tokens[i] = (i*7 + 3) % 256
	}

	b.Run("batched", func(b *testing.B) {
		m := newBatchBenchInstance(b)
		if !m.batchEligible(tokens) {
			b.Fatalf("benchmark instance should be eligible: %s", m.batchIneligibleReason(tokens))
		}
		if _, err := (*Instance)(m).ForwardTokens(tokens); err != nil { // warm the caches
			b.Fatal(err)
		}
		m.Reset()
		b.ResetTimer()
		for range b.N {
			m.Reset()
			if _, err := (*Instance)(m).ForwardTokens(tokens); err != nil {
				b.Fatal(err)
			}
		}
	})

	b.Run("sequential", func(b *testing.B) {
		m := newBatchBenchInstance(b)
		run := func() {
			for _, tok := range tokens {
				if _, err := (*Instance)(m).ForwardToken(tok); err != nil {
					b.Fatal(err)
				}
			}
		}
		run() // warm the caches
		m.Reset()
		b.ResetTimer()
		for range b.N {
			m.Reset()
			run()
		}
	})
}
