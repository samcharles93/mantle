package inference

import (
	"context"
	"fmt"
	"slices"
	"time"

	"github.com/samcharles93/mantle/internal/backend/simd"
	"github.com/samcharles93/mantle/internal/logits"
)

type Stats struct {
	TokensGenerated int
	Duration        time.Duration
	TPS             float64
}

// Generate runs the inference loop for a fixed number of steps.
// It handles sampling and updates the simd state.
func Generate(m simd.Model, sampler *logits.Sampler, promptTokens []int, steps int, callback func(string)) (Stats, error) {
	var stats Stats

	// Prefill
	var logitsVec []float32
	var err error

	// Process prompt tokens
	for _, id := range promptTokens {
		logitsVec, err = m.ForwardToken(id)
		if err != nil {
			return stats, fmt.Errorf("forward error during prefill: %w", err)
		}
	}

	toks := append([]int(nil), promptTokens...)

	// Generation loop
	genStart := time.Now()
	for i := 0; i < steps; i++ {
		next := sampler.Sample(logitsVec, toks, nil)
		toks = append(toks, next)

		logitsVec, err = m.ForwardToken(next)
		if err != nil {
			return stats, fmt.Errorf("forward error during generation step %d: %w", i, err)
		}
		stats.TokensGenerated++
	}

	stats.Duration = time.Since(genStart)
	if stats.Duration.Seconds() > 0 {
		stats.TPS = float64(stats.TokensGenerated) / stats.Duration.Seconds()
	}

	return stats, nil
}

// Generator manages the state of a generation session
type Generator struct {
	Model     simd.Model
	Sampler   *logits.Sampler
	Tokenizer interface {
		Decode([]int) (string, error)
	}
	ContextTokens []int
	StopTokens    []int
}

func (g *Generator) Run(allTokens []int, steps int, stream func(string)) ([]int, Stats, error) {
	return g.RunWithContext(context.Background(), allTokens, steps, stream)
}

func (g *Generator) RunWithContext(ctx context.Context, allTokens []int, steps int, stream func(string)) ([]int, Stats, error) {
	var stats Stats
	start := time.Now()

	mismatch := false
	if len(allTokens) < len(g.ContextTokens) {
		mismatch = true
	} else {
		for i, id := range g.ContextTokens {
			if allTokens[i] != id {
				mismatch = true
				break
			}
		}
	}

	if mismatch {
		g.Model.Reset()
		g.ContextTokens = g.ContextTokens[:0]
	}

	newInTokens := allTokens[len(g.ContextTokens):]

	var logitsVec []float32
	var err error
	for _, id := range newInTokens {
		logitsVec, err = g.Model.ForwardToken(id)
		if err != nil {
			return nil, stats, err
		}
	}
	g.ContextTokens = append(g.ContextTokens, newInTokens...)

	toks := append([]int(nil), g.ContextTokens...)

	limit := steps
	if limit < 0 {
		limit = 1000000
	}

	for i := 0; i < limit; i++ {
		if err := ctx.Err(); err != nil {
			return g.ContextTokens, stats, err
		}
		next := g.Sampler.Sample(logitsVec, toks, g.StopTokens)

		stop := slices.Contains(g.StopTokens, next)
		if stop {
			break
		}

		toks = append(toks, next)
		g.ContextTokens = append(g.ContextTokens, next)

		if stream != nil {
			s, _ := g.Tokenizer.Decode([]int{next})
			stream(s)
		}

		logitsVec, err = g.Model.ForwardToken(next)
		if err != nil {
			break
		}
		stats.TokensGenerated++
	}

	stats.Duration = time.Since(start)
	if stats.Duration.Seconds() > 0 {
		stats.TPS = float64(stats.TokensGenerated) / stats.Duration.Seconds()
	}

	return g.ContextTokens, stats, nil
}
