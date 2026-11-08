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
// irrelevant here; the speculative loop must stay correct regardless).
func newTestDSparkPair(t *testing.T) (*Instance, *DSparkDraft, *model.DSparkConfig) {
	t.Helper()
	target := newBatchParityInstance(t, batchParityContext, mcf.DTypeBF16)
	h := target.Config.Config.EmbeddingLength
	taps := []int{0, 1}
	if err := target.asCore().SetHiddenTapLayers(taps, batchParityContext); err != nil {
		t.Fatalf("SetHiddenTapLayers: %v", err)
	}
	inst := newBatchParityInstance(t, batchParityContext, mcf.DTypeBF16)
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
	return target, d, cfg
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
	if err := d.Inject(0, len(prompt)); err != nil {
		t.Fatalf("inject prompt: %v", err)
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

// TestDSparkSpeculateFullAcceptance checks that a draft the target always
// agrees with commits a whole block per round.
func TestDSparkSpeculateFullAcceptance(t *testing.T) {
	target, d, _ := newTestDSparkPair(t)
	prompt := []int{1, 4, 2, 7, 3}
	logits, err := target.ForwardTokens(prompt)
	if err != nil {
		t.Fatalf("prefill: %v", err)
	}
	if err := d.Inject(0, len(prompt)); err != nil {
		t.Fatalf("inject: %v", err)
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
	if err := d.Inject(0, len(prompt)); err != nil {
		t.Fatalf("inject: %v", err)
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
