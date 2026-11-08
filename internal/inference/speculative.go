package inference

import (
	"context"
	"fmt"

	"github.com/samcharles93/mantle/internal/graph"
)

// greedyStepper is a model that can advance one token and return the greedy
// next token without materializing logits.
type greedyStepper interface {
	ForwardTokenGreedy(id int) (next int, err error)
}

// SpeculativeDraft is the draft half of a speculative decoding pair. The
// simd DSpark draft implements it; tests may substitute fakes.
type SpeculativeDraft interface {
	// ContextLen is the number of target positions injected into the draft
	// cache so far; it must equal the target's position count.
	ContextLen() int
	// Room is how many drafted tokens a verification batch can still hold.
	Room() int
	// Truncate discards injected positions at and after pos (0 resets).
	Truncate(pos int)
	// Speculate drafts up to n tokens after anchor and verifies them against
	// the target in one batched forward. disagree decides per row whether the
	// target rejects drafted. It returns the accepted tokens (always starting
	// with anchor) and the target's distribution for the next position.
	Speculate(anchor, n int, disagree func(rowLogits []float32, drafted int, conf float32) bool) (accepted []int, nextLogits []float32, err error)
}

// Speculator drives speculative decoding rounds for a Generator. Tokens is
// the draft block length; Confidence is the acceptance-probability threshold
// the draft's confidence head must reach to override a target mismatch when
// sampling (0 disables the override, giving exact rejection sampling).
type Speculator struct {
	Draft      SpeculativeDraft
	Tokens     int
	Confidence float64
}

// Reset drops all injected draft state; called when the target context resets.
func (s *Speculator) Reset() {
	if s != nil && s.Draft != nil {
		s.Draft.Truncate(0)
	}
}

// generateSpeculative runs the generation loop with speculative decoding. It
// returns the target distribution for the next uncommitted position and
// whether generation stopped on a stop token; when the context can no longer
// hold a full draft block it returns early so the caller can finish with the
// plain per-token loop.
func (g *Generator) generateSpeculative(ctx context.Context, sb *streamBuffer, logitsVec []float32, toks []int, limit int, stats *Stats, gs greedyStepper, canDeviceGreedy bool) ([]float32, bool, error) {
	spec := g.Speculator
	q := logitsVec
	for stats.TokensGenerated < limit {
		if err := ctx.Err(); err != nil {
			return q, false, err
		}
		anchor, sampleErr := safeSample(g.Sampler, q, toks, g.StopTokens)
		logGenerationSampleDebug(anchor, g.StopTokens, q)
		if sampleErr != nil {
			return q, false, sampleErr
		}
		if g.isStopToken(anchor) {
			return q, true, nil
		}
		g.commitGenerated(anchor, sb, &toks, stats)

		n := spec.Tokens
		if rem := limit - stats.TokensGenerated; n > rem {
			n = rem
		}
		if room := spec.Draft.Room(); n > room {
			n = room
		}
		if n < 2 {
			// Context edge: commit the anchor plainly and let the plain loop
			// finish the tail.
			next, err := safeForwardToken(g.Model, anchor)
			if err != nil {
				return q, false, err
			}
			return next, false, nil
		}

		// The penalty window for row i must include the round's accepted
		// prefix (anchor + drafted[:i]), matching the per-token loop where
		// every committed token is in the window. Speculate calls disagree in
		// row order, so grow runToks as rows are accepted.
		runToks := append(append([]int(nil), toks...), anchor)
		disagree := func(row []float32, drafted int, conf float32) bool {
			s, err := safeSample(g.Sampler, row, runToks, g.StopTokens)
			if err != nil {
				return true
			}
			if s == drafted {
				runToks = append(runToks, drafted)
				return false
			}
			return !(spec.Confidence > 0 && conf >= float32(spec.Confidence))
		}
		accepted, nextQ, err := spec.Draft.Speculate(anchor, n, disagree)
		if err != nil {
			return q, false, fmt.Errorf("speculate: %w", err)
		}
		if len(accepted) < 1 || accepted[0] != anchor {
			return q, false, fmt.Errorf("speculate: accepted block %v does not start with anchor %d", accepted, anchor)
		}
		q = nextQ
		stopped := false
		for _, tok := range accepted[1:] {
			if stats.TokensGenerated >= limit {
				break
			}
			if !g.commitGenerated(tok, sb, &toks, stats) {
				stopped = true
				break
			}
		}
		if stopped {
			return q, true, nil
		}
	}
	return q, false, nil
}

// generatePlain is the per-token generation loop (graph, device-greedy and
// plain forward paths).
func (g *Generator) generatePlain(ctx context.Context, sb *streamBuffer, logitsVec []float32, toks []int, limit int, stats *Stats, gs greedyStepper, canDeviceGreedy bool) error {
	next, sampleErr := safeSample(g.Sampler, logitsVec, toks, g.StopTokens)
	logGenerationSampleDebug(next, g.StopTokens, logitsVec)
	if sampleErr != nil {
		return sampleErr
	}
	for stats.TokensGenerated < limit {
		if err := ctx.Err(); err != nil {
			return err
		}
		if !g.commitGenerated(next, sb, &toks, stats) {
			break
		}
		var err error
		switch {
		case g.useGraph:
			// Graph path: Compute → sample; no device-greedy shortcut.
			computeCtx := &graph.ComputeContext{
				Token: next,
				Pos:   len(g.ContextTokens) - 1,
				KVLen: len(g.ContextTokens) - 1,
			}
			logitsVec, err = g.gr.Compute(computeCtx, g.engineGraph)
			if err != nil {
				return err
			}
			next, sampleErr = safeSample(g.Sampler, logitsVec, toks, g.StopTokens)
			logGenerationSampleDebug(next, g.StopTokens, logitsVec)
			if sampleErr != nil {
				return sampleErr
			}
		case canDeviceGreedy:
			next, err = safeForwardTokenGreedy(gs, next)
			logGenerationSampleDebug(next, g.StopTokens, nil)
			if err != nil {
				return err
			}
		default:
			logitsVec, err = safeForwardToken(g.Model, next)
			if err != nil {
				return err
			}
			next, sampleErr = safeSample(g.Sampler, logitsVec, toks, g.StopTokens)
			logGenerationSampleDebug(next, g.StopTokens, logitsVec)
			if sampleErr != nil {
				return sampleErr
			}
		}
	}
	return nil
}

// commitGenerated appends a generated token to the context and the stream.
// It reports false (without committing) when tok is a stop token.
func (g *Generator) commitGenerated(tok int, sb *streamBuffer, toks *[]int, stats *Stats) bool {
	if g.isStopToken(tok) {
		return false
	}
	*toks = append(*toks, tok)
	g.ContextTokens = append(g.ContextTokens, tok)
	sb.add(tok)
	stats.TokensGenerated++
	return true
}
