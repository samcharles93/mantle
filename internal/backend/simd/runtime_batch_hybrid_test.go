package simd

import (
	"fmt"
	"testing"

	core "github.com/samcharles93/mantle/internal/backend/core"
	"github.com/samcharles93/mantle/pkg/mcf"
)

// These tests pin the batched prefill path for hybrid short-conv + attention
// models (LFM2/LFM3 shape) to the per-token oracle: conv layers, sliding
// windows, per-head and per-element attention gates, per-head q/k norms,
// sandwich norms and per-layer (zero-tail) RoPE must all batch identically.

const (
	hybridVocab     = 16
	hybridEmbd      = 8
	hybridHeadDim   = 4
	hybridHeads     = 2 // qDim = 8
	hybridKVStride  = 8
	hybridSlidingLn = 4 // sliding window / ring size of the sliding layer
	hybridFullCache = 64
	hybridMaxCtx    = 128
	hybridTol       = 1e-5
)

// newHybridParityInstance builds a synthetic LFM3-shaped instance:
//
//	layer 0: full attention, per-head gate, q/k norms, sandwich norms,
//	         zero-tail per-layer RoPE
//	layer 1: recurrent short-conv block (kernel length 4) with sandwich norms
//	layer 2: sliding attention (window 4, ring cache of 4), per-element gate,
//	         q/k norms, per-layer RoPE
//	layer 3: plain full attention
func newHybridParityInstance(tb testing.TB) *Instance {
	tb.Helper()
	m := newTestInstance(hybridVocab, hybridEmbd, 4)
	m.BindDefaultOps()
	m.MaxContext = hybridMaxCtx
	m.MaxQDim = hybridHeads * hybridHeadDim
	m.MaxKVStride = hybridKVStride
	m.HeadCount = hybridHeads
	m.HeadDim = hybridHeadDim
	m.Scratch.Scores = make([]float32, hybridMaxCtx)
	// The real loader sizes Tmp2 to embd; newTestInstance uses max(embd, ffn),
	// which the sequential post-norm path (RMSNorm(Tmp2, opOut, ...)) rejects.
	m.Scratch.Tmp2 = make([]float32, hybridEmbd)
	// Single-token scratch the sequential conv/gate paths need.
	m.Scratch.ScProj = make([]float32, 3*hybridEmbd)
	m.Scratch.ScBx = make([]float32, hybridEmbd)
	m.Scratch.ScConv = make([]float32, hybridEmbd)
	m.Scratch.AttnGate = make([]float32, 2*hybridHeads*hybridHeadDim)
	m.RopeInvFreq = []float64{1, 0.01}
	m.RopeAttnScale = 1

	fill := func(n int, scale float32) []float32 {
		d := make([]float32, n)
		for i := range d {
			d[i] = scale * float32((i*7+3)%13-6) / 6
		}
		return d
	}
	mat := func(r, c int, scale float32) *core.Mat {
		return &core.Mat{R: r, C: c, Stride: c, DType: mcf.DTypeF32, Data: fill(r*c, scale)}
	}
	vec := func(n int, scale float32) []float32 {
		v := fill(n, scale)
		for i := range v {
			v[i] += 1
		}
		return v
	}
	fullCache := func() core.AttnCache {
		return core.AttnCache{
			K:        make([]float32, hybridKVStride*hybridFullCache),
			V:        make([]float32, hybridKVStride*hybridFullCache),
			KvStride: hybridKVStride,
			CacheLen: hybridFullCache,
			Cap:      hybridFullCache,
		}
	}

	for i := range m.Layers {
		layer := &m.Layers[i]
		layer.NoRoPE = false
		layer.AttnType = "full_attention"
		layer.SharedKVSource = -1
		layer.LayerScale = 1
		layer.AttnCache = fullCache()
		// Sandwich norms on every layer, as LFM3 has.
		layer.PostAttnNorm = vec(hybridEmbd, 0.05)
		layer.PostFfnNorm = vec(hybridEmbd, -0.04)
	}

	// Layer 0: per-head gate + q/k norms + zero-tail per-layer RoPE.
	l0 := &m.Layers[0]
	l0.AttnGate = mat(hybridHeads, hybridEmbd, 0.11)
	l0.AttnQNorm = vec(hybridHeadDim, 0.07)
	l0.AttnKNorm = vec(hybridHeadDim, -0.06)
	l0.RopeInvFreq = []float64{1, 0} // zero-tail: second frequency passes through
	l0.RopeAttnScale = 1

	// Layer 1: recurrent short-conv.
	l1 := &m.Layers[1]
	l1.IsRecurrent = true
	l1.HeadKV = 0
	l1.AttnType = "conv"
	l1.ShortConvKernel = mat(hybridEmbd, 4, 0.13)
	l1.ShortConvInProj = mat(3*hybridEmbd, hybridEmbd, 0.09)
	l1.ShortConvOutProj = mat(hybridEmbd, hybridEmbd, 0.08)
	l1.ShortConvState = core.ShortConvState{Buf: make([]float32, 3*hybridEmbd), KernelLen: 4}
	l1.AttnCache = core.AttnCache{KvStride: 0, CacheLen: hybridFullCache}

	// Layer 2: sliding attention with its own ring cache, per-element gate,
	// q/k norms and a per-layer RoPE table.
	l2 := &m.Layers[2]
	l2.AttnType = "sliding_attention"
	l2.AttnWindow = hybridSlidingLn
	l2.AttnGate = mat(hybridHeads*hybridHeadDim, hybridEmbd, -0.1)
	l2.AttnQNorm = vec(hybridHeadDim, 0.05)
	l2.AttnKNorm = vec(hybridHeadDim, 0.04)
	l2.RopeInvFreq = []float64{0.5, 0.25}
	l2.RopeAttnScale = 1
	l2.AttnCache = core.AttnCache{
		K:        make([]float32, hybridKVStride*hybridSlidingLn),
		V:        make([]float32, hybridKVStride*hybridSlidingLn),
		KvStride: hybridKVStride,
		CacheLen: hybridSlidingLn,
		Cap:      hybridSlidingLn,
	}

	sizeBatchScratch((*Instance)(m))
	return (*Instance)(m)
}

// compareConvState checks every recurrent layer's carried state matches the
// sequential oracle after a batched forward.
func compareConvState(tb testing.TB, got, want *Instance, tol float32) {
	tb.Helper()
	for i := range got.Layers {
		if !got.Layers[i].IsRecurrent {
			continue
		}
		if !floatsEqual(got.Layers[i].ShortConvState.Buf, want.Layers[i].ShortConvState.Buf, tol) {
			tb.Fatalf("layer %d short-conv state differs:\n got %v\nwant %v",
				i, got.Layers[i].ShortConvState.Buf, want.Layers[i].ShortConvState.Buf)
		}
	}
}

// TestBatchPrefillHybridParity runs hybrid prompts of several lengths through
// the batched path and the per-token oracle: every logits row, the KV caches,
// the short-conv states and a following decode token must all match. Lengths
// cover one chunk, a chunk boundary with a single-token tail chunk (n=33) and
// prompts that wrap the sliding layer's ring many times over.
func TestBatchPrefillHybridParity(t *testing.T) {
	t.Setenv(cpuMaxBatchEnv, "32") // n=33 is a 32-row chunk plus a 1-token tail
	for _, n := range []int{2, 5, 33, 40} {
		t.Run(fmt.Sprintf("n%d", n), func(t *testing.T) {
			tokens := make([]int, n)
			for i := range tokens {
				tokens[i] = (i*5 + 1) % hybridVocab
			}
			batched := newHybridParityInstance(t)
			sequential := newHybridParityInstance(t)
			if why := batched.batchIneligibleReason(tokens); why != "" {
				t.Fatalf("hybrid prompt should be eligible: %s", why)
			}

			want := sequentialLogits(t, sequential, tokens)
			got, err := batched.ForwardTokens(tokens)
			if err != nil {
				t.Fatalf("ForwardTokens: %v", err)
			}
			if !batchPathRan(batched) {
				t.Fatal("expected the batched path to run")
			}
			compareLogitRows(t, "hybrid", got, want, hybridTol)
			if batched.Pos != sequential.Pos {
				t.Fatalf("position = %d, want %d", batched.Pos, sequential.Pos)
			}
			compareKVCache(t, batched, sequential, hybridTol)
			compareConvState(t, batched, sequential, hybridTol)

			gotNext, err := batched.ForwardToken(9)
			if err != nil {
				t.Fatalf("batched ForwardToken: %v", err)
			}
			gotNext = append([]float32(nil), gotNext...)
			wantNext, err := sequential.ForwardToken(9)
			if err != nil {
				t.Fatalf("sequential ForwardToken: %v", err)
			}
			if !floatsEqual(gotNext, wantNext, hybridTol) {
				t.Fatal("logits after the hybrid batched prefill differ from the sequential path")
			}
			compareConvState(t, batched, sequential, hybridTol)
		})
	}
}

// TestPrefillTokensHybridParity checks PrefillTokens on the hybrid model: only
// the final row's logits come back, the batch path ran, and the state (KV,
// conv) matches the per-token oracle so the next token continues correctly.
func TestPrefillTokensHybridParity(t *testing.T) {
	t.Setenv(cpuMaxBatchEnv, "32")
	tokens := longTokens(40) // two chunks
	batched := newHybridParityInstance(t)
	sequential := newHybridParityInstance(t)
	if why := batched.batchIneligibleReason(tokens); why != "" {
		t.Fatalf("hybrid prompt should be eligible: %s", why)
	}

	want := sequentialPrefillLogits(t, sequential, tokens)
	got, err := batched.PrefillTokens(tokens)
	if err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	if !batchPathRan(batched) {
		t.Fatal("expected the batched path to prefill")
	}
	if !floatsEqual(got, want, hybridTol) {
		t.Fatal("final logits differ from the sequential oracle")
	}
	compareKVCache(t, batched, sequential, hybridTol)
	compareConvState(t, batched, sequential, hybridTol)

	gotNext, err := batched.ForwardToken(9)
	if err != nil {
		t.Fatalf("batched ForwardToken: %v", err)
	}
	gotNext = append([]float32(nil), gotNext...)
	wantNext, err := sequential.ForwardToken(9)
	if err != nil {
		t.Fatalf("sequential ForwardToken: %v", err)
	}
	if !floatsEqual(gotNext, wantNext, hybridTol) {
		t.Fatal("logits after the hybrid prefill differ from the sequential path")
	}
}

// TestBatchPrefillHybridGateNonCausalRejects checks a non-causal draft block
// still refuses hybrid layers: ForwardBlockNonCausal keeps the dense-only
// contract, so a hybrid model must get an error, not a wrong block.
func TestBatchPrefillHybridGateNonCausalRejects(t *testing.T) {
	m := newHybridParityInstance(t)
	tokens := []int{1, 2, 3, 4}
	err := m.ForwardBlockNonCausal(tokens, func(row int, logits, hidden []float32) {})
	if err == nil {
		t.Fatal("non-causal block must reject a hybrid model")
	}
}

// TestBatchPrefillHybridDenseNonCausalStillWorks is the control for the
// non-causal gate: a plain dense instance keeps the block path.
func TestBatchPrefillHybridDenseNonCausalStillWorks(t *testing.T) {
	m := newBatchParityInstance(t, batchParityContext, mcf.DTypeF32)
	tokens := []int{1, 2, 3, 4}
	rows := 0
	if err := m.ForwardBlockNonCausal(tokens, func(row int, logits, hidden []float32) { rows++ }); err != nil {
		t.Fatalf("dense non-causal block: %v", err)
	}
	if rows != len(tokens) {
		t.Fatalf("non-causal block kept %d rows, want %d", rows, len(tokens))
	}
}

// TestCPUMaxBatchEnv checks the MANTLE_CPU_MAX_BATCH override parsing and
// clamping used to size the batch scratch.
func TestCPUMaxBatchEnv(t *testing.T) {
	if got := cpuMaxBatch(); got != defaultMaxBatch {
		t.Fatalf("default cpuMaxBatch = %d, want %d", got, defaultMaxBatch)
	}
	for _, tc := range []struct {
		value string
		want  int
	}{
		{"32", 32},
		{" 16 ", 16},
		{"0", 1},
		{"-5", 1},
		{"999999", 4096},
		{"bogus", defaultMaxBatch},
	} {
		t.Setenv(cpuMaxBatchEnv, tc.value)
		if got := cpuMaxBatch(); got != tc.want {
			t.Errorf("MANTLE_CPU_MAX_BATCH=%q: cpuMaxBatch = %d, want %d", tc.value, got, tc.want)
		}
	}
}
