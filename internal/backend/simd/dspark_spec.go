package simd

import (
	"fmt"
)

// targetInstance returns the paired target as a simd Instance, or nil when
// the draft is not paired.
func (d *DSparkDraft) targetInstance() *Instance {
	if d == nil || d.target == nil {
		return nil
	}
	return (*Instance)(d.target)
}

// Room reports how many drafted tokens a verification batch can still hold:
// the batch runs at positions Pos..Pos+n, so n <= MaxContext-Pos-1.
func (d *DSparkDraft) Room() int {
	ti := d.targetInstance()
	if ti == nil {
		return 0
	}
	return ti.MaxContext - ti.Pos - 1
}

// Speculate runs one speculative decoding round. It drafts n tokens following
// anchor (the last committed token), then verifies the whole block against the
// target in a single batched forward over [anchor, drafted...]. Row i of that
// batch is the target's distribution for the position of drafted[i], so the
// longest prefix of drafted that the target agrees with is accepted; the first
// disagreeing row becomes the next round's distribution. disagree decides
// acceptance per row: at temperature 0 it compares the argmax, when sampling it
// compares the sample, optionally overridden by the draft's confidence head.
//
// accepted always starts with anchor and holds the committed tokens for
// positions Pos..Pos+len(accepted)-1; nextLogits is the target's distribution
// for the position right after them. The target's KV cache is rolled back to
// the accepted prefix (later positions are overwritten by the next batch) and
// the accepted prefix's hidden taps are injected into the draft cache, so both
// stay in lockstep at Pos+len(accepted).
func (d *DSparkDraft) Speculate(anchor, n int, disagree func(rowLogits []float32, drafted int, conf float32) bool) ([]int, []float32, error) {
	ti := d.targetInstance()
	c := d.Config
	switch {
	case ti == nil:
		return nil, nil, fmt.Errorf("dspark: no target instance")
	case n < 2 || n > c.BlockSize:
		return nil, nil, fmt.Errorf("dspark: draft length %d outside [2,%d]", n, c.BlockSize)
	case anchor < 0 || anchor >= c.VocabSize:
		return nil, nil, fmt.Errorf("dspark: anchor token %d outside vocab", anchor)
	}
	if room := ti.MaxContext - ti.Pos - 1; n > room {
		n = room
	}
	if n < 2 {
		return nil, nil, fmt.Errorf("dspark: context has room for %d more position(s)", n+1)
	}
	pos := ti.Pos
	if d.inst.Pos < pos {
		// The target prefilled (or generated) positions the draft has not seen
		// yet — e.g. the prompt. The last forward's tap rows cover exactly
		// positions [pos-taps.Rows, pos), so inject them to catch up.
		missing := pos - d.inst.Pos
		if ti.HiddenTaps.Rows != missing {
			return nil, nil, fmt.Errorf("dspark: cannot sync draft: target captured %d tap rows for %d missing positions", ti.HiddenTaps.Rows, missing)
		}
		if err := d.Inject(pos-missing, missing); err != nil {
			return nil, nil, err
		}
	}
	if d.inst.Pos != pos {
		return nil, nil, fmt.Errorf("dspark: draft holds %d positions, target %d", d.inst.Pos, pos)
	}
	drafted, conf, err := d.DraftBlock(anchor, n)
	if err != nil {
		return nil, nil, err
	}
	batch := make([]int, n+1)
	batch[0] = anchor
	copy(batch[1:], drafted)
	rows, err := ti.ForwardTokens(batch)
	if err != nil {
		return nil, nil, fmt.Errorf("dspark: verify: %w", err)
	}
	if len(rows) != n+1 {
		return nil, nil, fmt.Errorf("dspark: verify returned %d rows, want %d", len(rows), n+1)
	}
	k := 0
	for k < n {
		var ck float32
		if conf != nil {
			ck = conf[k]
		}
		// disagree may mutate the row (samplers apply the repetition penalty
		// in place); hand it a copy so the returned distribution stays raw.
		row := append([]float32(nil), rows[k]...)
		if disagree(row, drafted[k], ck) {
			break
		}
		k++
	}
	// Roll back to the accepted prefix; the rejected suffix stays in the cache
	// but is past Pos, so the next forward overwrites it.
	ti.Pos = pos + k + 1
	if err := d.Inject(pos, k+1); err != nil {
		return nil, nil, err
	}
	// Return a copy of the next distribution: samplers apply the repetition
	// penalty in place, and the caller samples this row again next round, so
	// sharing the row would penalize it twice.
	next := append([]float32(nil), rows[k]...)
	return batch[:k+1], next, nil
}

// CanBatchForward reports whether a causal batch of n tokens runs through the
// batched prefill path. Speculative verification needs this on the target.
func (m *Instance) CanBatchForward(n int) bool {
	if n <= 0 {
		return false
	}
	return m.batchIneligibleReasonMode(make([]int, n), false) == ""
}
