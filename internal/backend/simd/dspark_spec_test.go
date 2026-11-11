package simd

import (
	"slices"
	"testing"

	"github.com/samcharles93/mantle/internal/model"
	"github.com/samcharles93/mantle/pkg/mcf"
)

// newTestDSparkPair builds a synthetic target/draft pair: two identical
// deterministic instances, hidden taps enabled on the target, and a
// hand-filled DSparkDraft whose weights are arbitrary (acceptance rate is
// irrelevant here; the speculative loop must stay correct regardless). The
// drain is wired the way LoadDSparkDraftMCF wires it, so target forwards keep
// the draft in step and no test injects by hand.
func newTestDSparkPair(t *testing.T) (*Instance, *DSparkDraft, *model.DSparkConfig) {
	t.Helper()
	return newTestDSparkPairCtx(t, batchParityContext)
}

// newTestDSparkPairCtx is newTestDSparkPair with a context of cacheLen tokens,
// for tests that need a prompt longer than the default.
func newTestDSparkPairCtx(t *testing.T, cacheLen int) (*Instance, *DSparkDraft, *model.DSparkConfig) {
	t.Helper()
	target := withLongContext(newBatchParityInstance(t, cacheLen, mcf.DTypeBF16), cacheLen)
	h := target.Config.Config.EmbeddingLength
	taps := []int{0, 1}
	// One prefill chunk of rows, the same capacity the loader gives a tap buffer.
	if err := target.asCore().SetHiddenTapLayers(taps, target.MaxBatch); err != nil {
		t.Fatalf("SetHiddenTapLayers: %v", err)
	}
	inst := withLongContext(newBatchParityInstance(t, cacheLen, mcf.DTypeBF16), cacheLen)
	cfg := &model.DSparkConfig{
		HiddenSize:     h,
		VocabSize:      batchParityVocab,
		BlockSize:      4,
		MaskTokenID:    batchParityVocab - 1,
		TargetLayerIDs: taps,
		MarkovRank:     4,
		HasConfidence:  true,
		RMSNormEps:     1e-5,
	}
	rank := cfg.MarkovRank
	d := &DSparkDraft{Config: cfg, target: target.asCore(), inst: inst}
	d.fc = &Mat{R: h, C: len(taps) * h, Stride: len(taps) * h, DType: mcf.DTypeF32, Data: make([]float32, h*len(taps)*h)}
	for i := range d.fc.Data {
		d.fc.Data[i] = float32(i%7+1) * 0.01
	}
	d.hiddenNorm = make([]float32, h)
	for i := range d.hiddenNorm {
		d.hiddenNorm[i] = 1
	}
	d.markovW1 = &Mat{R: batchParityVocab, C: rank, Stride: rank, DType: mcf.DTypeF32, Data: make([]float32, batchParityVocab*rank)}
	d.markovW2 = &Mat{R: batchParityVocab, C: rank, Stride: rank, DType: mcf.DTypeF32, Data: make([]float32, batchParityVocab*rank)}
	for i := range d.markovW1.Data {
		d.markovW1.Data[i] = float32(i%5+1) * 0.02
	}
	for i := range d.markovW2.Data {
		d.markovW2.Data[i] = float32(i%3+1) * 0.03
	}
	d.confW = make([]float32, h+rank)
	for i := range d.confW {
		d.confW[i] = 0.01
	}
	d.confB = 0.5
	rows := model.DSparkMaxBlockSize
	d.feat = make([]float32, rows*len(taps)*h)
	d.fused = make([]float32, rows*h)
	d.keys = make([]float32, rows*inst.MaxKVStride)
	d.vals = make([]float32, rows*inst.MaxKVStride)
	d.logits = make([]float32, cfg.BlockSize*batchParityVocab)
	d.embd = make([]float32, h)
	d.bias = make([]float32, batchParityVocab)
	d.markov = make([]float32, rank)
	target.HiddenTaps.Drain = d.drainTarget
	return target, d, cfg
}

// TestDSparkSpeculatePrefillLongerThanTapBuffer covers the prompt every real
// user has: longer than one prefill chunk, so the target's tap buffer can only
// hold part of it at a time. The drain hands each chunk over as it finishes, so
// the draft must end up holding every prompt position and the first round must
// run on that state.
func TestDSparkSpeculatePrefillLongerThanTapBuffer(t *testing.T) {
	target, d, _ := newTestDSparkPairCtx(t, 128)
	prompt := longTokens(70) // 64 + 6: two chunks
	if len(prompt) <= target.MaxBatch {
		t.Fatalf("prompt of %d does not exceed the %d-token chunk", len(prompt), target.MaxBatch)
	}
	if _, err := target.ForwardTokens(prompt); err != nil {
		t.Fatalf("prefill: %v", err)
	}
	if !batchPathRan(target) {
		t.Fatal("expected the chunked batched path to run")
	}
	if target.Pos != len(prompt) {
		t.Fatalf("target Pos = %d, want %d", target.Pos, len(prompt))
	}
	if got := d.ContextLen(); got != len(prompt) {
		t.Fatalf("draft holds %d positions after a %d-token prompt: a chunk was not handed over", got, len(prompt))
	}
	accepted, next, err := d.Speculate(prompt[len(prompt)-1], 4, func(row []float32, drafted int, _ float32) bool {
		return argmaxHost(row) != drafted
	})
	if err != nil {
		t.Fatalf("speculate after a chunked prefill: %v", err)
	}
	if len(accepted) < 1 || accepted[0] != prompt[len(prompt)-1] {
		t.Fatalf("accepted %v does not start with the anchor", accepted)
	}
	if len(next) != batchParityVocab {
		t.Fatalf("next logits width %d, want %d", len(next), batchParityVocab)
	}
}

// TestDSparkSpeculateMatchesGreedy runs the speculative loop on a synthetic
// pair and checks it commits exactly the plain greedy continuation, keeps the
// draft cache in lockstep with the target, and rolls the KV cache back
// correctly on partial acceptance.
func TestDSparkSpeculateMatchesGreedy(t *testing.T) {
	target, d, _ := newTestDSparkPair(t)
	prompt := []int{1, 4, 2, 7, 3}

	logits, err := target.ForwardTokens(prompt)
	if err != nil {
		t.Fatalf("prefill: %v", err)
	}
	if target.HiddenTaps.Rows != len(prompt) {
		t.Fatalf("taps captured %d rows for %d prompt tokens", target.HiddenTaps.Rows, len(prompt))
	}

	const steps = 12
	greedyDisagree := func(row []float32, drafted int, _ float32) bool {
		return argmaxHost(row) != drafted
	}
	var got []int
	q := logits[len(logits)-1]
	rounds := 0
	for len(got) < steps {
		rounds++
		anchor := argmaxHost(q)
		posBefore := target.Pos
		accepted, next, err := d.Speculate(anchor, 4, greedyDisagree)
		if err != nil {
			t.Fatalf("round %d: speculate: %v", rounds, err)
		}
		if len(accepted) < 1 || accepted[0] != anchor {
			t.Fatalf("round %d: accepted %v does not start with anchor %d", rounds, accepted, anchor)
		}
		if target.Pos != posBefore+len(accepted) {
			t.Fatalf("round %d: target Pos %d, want %d (pos %d + %d accepted)", rounds, target.Pos, posBefore+len(accepted), posBefore, len(accepted))
		}
		if d.ContextLen() != target.Pos {
			t.Fatalf("round %d: draft holds %d, target %d", rounds, d.ContextLen(), target.Pos)
		}
		got = append(got, accepted...)
		q = next
	}

	// Plain greedy reference on a fresh identical instance.
	ref := newBatchParityInstance(t, batchParityContext, mcf.DTypeBF16)
	refLogits, err := ref.ForwardTokens(prompt)
	if err != nil {
		t.Fatalf("ref prefill: %v", err)
	}
	var want []int
	rq := refLogits[len(refLogits)-1]
	for len(want) < steps {
		tok := argmaxHost(rq)
		want = append(want, tok)
		rl, err := ref.ForwardToken(tok)
		if err != nil {
			t.Fatalf("ref step: %v", err)
		}
		rq = rl
	}

	got = got[:steps]
	if !slices.Equal(got, want) {
		t.Fatalf("speculative output %v != greedy %v", got, want)
	}
}

func TestDSparkSpeculatePartialAcceptanceRollsBack(t *testing.T) {
	for _, tc := range []struct {
		name     string
		accepted int
	}{
		{"reject first", 0},
		{"reject second", 1},
		{"reject last", 3},
	} {
		t.Run(tc.name, func(t *testing.T) {
			target, draft, _ := newTestDSparkPair(t)
			ref := withLongContext(newBatchParityInstance(t, batchParityContext, mcf.DTypeBF16), batchParityContext)
			prompt := []int{1, 4, 2, 7, 3}
			if _, err := target.ForwardTokens(prompt); err != nil {
				t.Fatalf("target prefill: %v", err)
			}
			if _, err := ref.ForwardTokens(prompt); err != nil {
				t.Fatalf("reference prefill: %v", err)
			}

			anchor := 5
			for round := range 2 {
				pos := target.Pos
				checked := 0
				accepted, next, err := draft.Speculate(anchor, 4, func([]float32, int, float32) bool {
					reject := checked == tc.accepted
					checked++
					return reject
				})
				if err != nil {
					t.Fatalf("round %d: speculate: %v", round, err)
				}
				if len(accepted) != tc.accepted+1 || accepted[0] != anchor {
					t.Fatalf("round %d: accepted %v, want anchor plus %d drafts", round, accepted, tc.accepted)
				}
				wantPos := pos + len(accepted)
				if target.Pos != wantPos || draft.ContextLen() != wantPos {
					t.Fatalf("round %d: target pos=%d draft pos=%d, want %d", round, target.Pos, draft.ContextLen(), wantPos)
				}
				rows := sequentialLogits(t, ref, accepted)
				if !floatsEqual(next, rows[len(rows)-1], batchParityTol) {
					t.Fatalf("round %d: next logits %v, want %v", round, next, rows[len(rows)-1])
				}
				anchor = argmaxHost(next)
			}
		})
	}
}

// TestDSparkSpeculateFullAcceptance checks that a draft the target always
// agrees with commits a whole block per round.
func TestDSparkSpeculateFullAcceptance(t *testing.T) {
	target, d, _ := newTestDSparkPair(t)
	prompt := []int{1, 4, 2, 7, 3}
	logits, err := target.ForwardTokens(prompt)
	if err != nil {
		t.Fatalf("prefill: %v", err)
	}
	q := logits[len(logits)-1]
	anchor := argmaxHost(q)
	posBefore := target.Pos
	// never disagree: every drafted token is accepted
	accepted, _, err := d.Speculate(anchor, 4, func([]float32, int, float32) bool { return false })
	if err != nil {
		t.Fatalf("speculate: %v", err)
	}
	if len(accepted) != 5 || accepted[0] != anchor {
		t.Fatalf("accepted %v, want anchor + 4 drafts", accepted)
	}
	if target.Pos != posBefore+5 || d.ContextLen() != target.Pos {
		t.Fatalf("Pos target=%d draft=%d, want %d", target.Pos, d.ContextLen(), posBefore+5)
	}
}

// TestDSparkSpeculateGuards covers the argument and context guards.
func TestDSparkSpeculateGuards(t *testing.T) {
	target, d, cfg := newTestDSparkPair(t)
	prompt := []int{1, 4, 2}
	if _, err := target.ForwardTokens(prompt); err != nil {
		t.Fatalf("prefill: %v", err)
	}
	noop := func([]float32, int, float32) bool { return true }
	if _, _, err := d.Speculate(0, 1, noop); err == nil {
		t.Fatal("n=1 must be rejected")
	}
	if _, _, err := d.Speculate(0, cfg.BlockSize+1, noop); err == nil {
		t.Fatal("n>block_size must be rejected")
	}
	if _, _, err := d.Speculate(batchParityVocab, 2, noop); err == nil {
		t.Fatal("out-of-vocab anchor must be rejected")
	}
	// Draft out of sync with the target.
	d.Truncate(target.Pos - 1)
	if _, _, err := d.Speculate(0, 2, noop); err == nil {
		t.Fatal("out-of-sync draft must be rejected")
	}
	d.Truncate(target.Pos)
	// Context edge: only one position left.
	target.Pos = target.MaxContext - 1
	if got := d.Room(); got != 0 {
		t.Fatalf("Room() = %d, want 0", got)
	}
	if _, _, err := d.Speculate(0, 2, noop); err == nil {
		t.Fatal("speculating past the context end must be rejected")
	}
}
