package inference

import (
	"fmt"
	"slices"
	"time"

	"infer/internal/llm"
	"infer/internal/logits"
)

type Stats struct {
	TokensGenerated int
	Duration        time.Duration
	TPS             float64
}

// Generate runs the inference loop for a fixed number of steps.
// It handles sampling and updates the model state.
func Generate(model llm.Model, sampler *logits.Sampler, promptTokens []int, steps int, callback func(string)) (Stats, error) {
	var stats Stats

	// Prefill
	var logitsVec []float32
	var err error

	// Process prompt tokens
	for _, id := range promptTokens {
		logitsVec, err = model.ForwardToken(id)
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

		logitsVec, err = model.ForwardToken(next)
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
	Model     llm.Model
	Sampler   *logits.Sampler
	Tokenizer interface {
		Decode([]int) (string, error)
	}
	ContextTokens []int
	StopTokens    []int
}

func (g *Generator) Run(allTokens []int, steps int, stream func(string)) ([]int, Stats, error) {
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
