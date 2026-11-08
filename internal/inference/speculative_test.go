package inference

import (
	"context"
	"slices"
	"testing"
)

// specFakeModel is a deterministic toy target: the greedy continuation of any
// token t is next(t), and both forward paths return one-hot logits for it.
type specFakeModel struct {
	vocab        int
	forwardCalls int
	batchCalls   int
	prefillCalls int
	seq          []int
}

func specNext(vocab, t int) int { return (t*3 + 1) % vocab }

func (m *specFakeModel) oneHot(t int) []float32 {
	l := make([]float32, m.vocab)
	l[t] = 10
	return l
}

func (m *specFakeModel) ForwardToken(id int) ([]float32, error) {
	m.forwardCalls++
	m.seq = append(m.seq, id)
	return m.oneHot(specNext(m.vocab, id)), nil
}

func (m *specFakeModel) PrefillTokens(tokens []int) ([]float32, error) {
	m.prefillCalls++
	m.seq = append(m.seq, tokens...)
	return m.oneHot(specNext(m.vocab, tokens[len(tokens)-1])), nil
}

func (m *specFakeModel) ForwardTokens(tokens []int) ([][]float32, error) {
	m.batchCalls++
	m.seq = append(m.seq, tokens...)
	rows := make([][]float32, len(tokens))
	for i, t := range tokens {
		rows[i] = m.oneHot(specNext(m.vocab, t))
	}
	return rows, nil
}

func (m *specFakeModel) Reset() {
	m.seq = m.seq[:0]
	m.forwardCalls, m.batchCalls, m.prefillCalls = 0, 0, 0
}

// specFakeDraft emulates a DSpark draft against specFakeModel. perfect drafts
// the target's own continuation; broken drafts something else.
type specFakeDraft struct {
	model   *specFakeModel
	perfect bool
	pos     int
	calls   int
}

func (d *specFakeDraft) ContextLen() int { return d.pos }
func (d *specFakeDraft) Room() int       { return 1000 }
func (d *specFakeDraft) Truncate(pos int) {
	if pos < d.pos {
		d.pos = pos
	}
}

func (d *specFakeDraft) Speculate(anchor, n int, disagree func(rowLogits []float32, drafted int, conf float32) bool) ([]int, []float32, error) {
	d.calls++
	drafted := make([]int, n)
	tok := anchor
	for i := range drafted {
		tok = specNext(d.model.vocab, tok)
		if !d.perfect {
			tok = (tok + 1) % d.model.vocab
		}
		drafted[i] = tok
	}
	batch := append([]int{anchor}, drafted...)
	rows, err := d.model.ForwardTokens(batch)
	if err != nil {
		return nil, nil, err
	}
	k := 0
	for k < n && !disagree(rows[k], drafted[k], 0.9) {
		k++
	}
	d.pos += k + 1
	return batch[:k+1], rows[k], nil
}

func runSpecGenerator(t *testing.T, model *specFakeModel, draft SpeculativeDraft, stop []int, steps int) ([]int, Stats, error) {
	t.Helper()
	g := &Generator{
		Model:      model,
		Sampler:    newGreedySampler(),
		StopTokens: stop,
	}
	if draft != nil {
		g.Speculator = &Speculator{Draft: draft, Tokens: 4}
	}
	return g.RunWithContext(context.Background(), []int{1, 2}, steps, nil)
}

func TestSpeculativeMatchesPlainGreedy(t *testing.T) {
	t.Parallel()

	plainModel := &specFakeModel{vocab: 16}
	plainToks, plainStats, err := runSpecGenerator(t, plainModel, nil, nil, 9)
	if err != nil {
		t.Fatalf("plain run: %v", err)
	}

	// A perfect draft: every round accepts the whole block in one batch.
	perfectModel := &specFakeModel{vocab: 16}
	perfectDraft := &specFakeDraft{model: perfectModel, perfect: true}
	specToks, specStats, err := runSpecGenerator(t, perfectModel, perfectDraft, nil, 9)
	if err != nil {
		t.Fatalf("speculative run: %v", err)
	}
	if !slices.Equal(specToks, plainToks) {
		t.Fatalf("speculative tokens %v != plain %v", specToks, plainToks)
	}
	if specStats.TokensGenerated != plainStats.TokensGenerated {
		t.Fatalf("generated %d vs plain %d", specStats.TokensGenerated, plainStats.TokensGenerated)
	}
	// 9 tokens with 5-token rounds: two verification batches, no per-token
	// forwards during generation.
	if perfectDraft.calls != 2 {
		t.Fatalf("draft rounds = %d, want 2", perfectDraft.calls)
	}
	if perfectModel.forwardCalls != 0 {
		t.Fatalf("per-token forwards during speculative generation = %d, want 0", perfectModel.forwardCalls)
	}
	if perfectModel.batchCalls != 2 {
		t.Fatalf("verification batches = %d, want 2", perfectModel.batchCalls)
	}

	// A broken draft: only the anchor is accepted every round, output must
	// still match plain greedy exactly.
	brokenModel := &specFakeModel{vocab: 16}
	brokenDraft := &specFakeDraft{model: brokenModel, perfect: false}
	brokenToks, _, err := runSpecGenerator(t, brokenModel, brokenDraft, nil, 9)
	if err != nil {
		t.Fatalf("broken-draft run: %v", err)
	}
	if !slices.Equal(brokenToks, plainToks) {
		t.Fatalf("broken-draft tokens %v != plain %v", brokenToks, plainToks)
	}
}

func TestSpeculativeStopsOnStopTokenMidBlock(t *testing.T) {
	t.Parallel()

	// The greedy chain from prompt [1,2]: compute the token at generated
	// position 3 and use it as the stop token.
	tok := specNext(16, 2)
	for i := 0; i < 3; i++ {
		tok = specNext(16, tok)
	}
	stop := []int{tok}

	plainModel := &specFakeModel{vocab: 16}
	plainToks, _, err := runSpecGenerator(t, plainModel, nil, stop, 20)
	if err != nil {
		t.Fatalf("plain run: %v", err)
	}

	specModel := &specFakeModel{vocab: 16}
	specDraft := &specFakeDraft{model: specModel, perfect: true}
	specToks, _, err := runSpecGenerator(t, specModel, specDraft, stop, 20)
	if err != nil {
		t.Fatalf("speculative run: %v", err)
	}
	if !slices.Equal(specToks, plainToks) {
		t.Fatalf("speculative tokens %v != plain %v (stop %v)", specToks, plainToks, stop)
	}
	if slices.Contains(specToks, tok) {
		t.Fatalf("stop token %d was committed: %v", tok, specToks)
	}
}
