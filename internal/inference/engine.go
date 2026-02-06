package inference

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/samcharles93/mantle/internal/backend/simd"
	"github.com/samcharles93/mantle/internal/logits"
)

type Stats struct {
	TokensGenerated int
	Duration        time.Duration
	TPS             float64

	PromptTokens   int
	PromptDuration time.Duration
	PromptTPS      float64

	GenerationDuration time.Duration
	GenerationTPS      float64
}

// Generate runs the inference loop for a fixed number of steps.
// It handles sampling and updates the simd state.
func Generate(m simd.Model, sampler *logits.Sampler, promptTokens []int, steps int, callback func(string)) (Stats, error) {
	var stats Stats

	// Prefill
	var logitsVec []float32
	var err error

	// Process prompt tokens
	promptStart := time.Now()
	for _, id := range promptTokens {
		logitsVec, err = safeForwardToken(m, id)
		if err != nil {
			return stats, fmt.Errorf("forward error during prefill: %w", err)
		}
	}
	stats.PromptTokens = len(promptTokens)
	stats.PromptDuration = time.Since(promptStart)
	if stats.PromptDuration.Seconds() > 0 && stats.PromptTokens > 0 {
		stats.PromptTPS = float64(stats.PromptTokens) / stats.PromptDuration.Seconds()
	}

	toks := append([]int(nil), promptTokens...)

	// Generation loop
	genStart := time.Now()
	for i := 0; i < steps; i++ {
		next, sampleErr := safeSample(sampler, logitsVec, toks, nil)
		if sampleErr != nil {
			return stats, fmt.Errorf("sample error during generation step %d: %w", i, sampleErr)
		}
		toks = append(toks, next)

		logitsVec, err = safeForwardToken(m, next)
		if err != nil {
			return stats, fmt.Errorf("forward error during generation step %d: %w", i, err)
		}
		stats.TokensGenerated++
	}

	stats.Duration = time.Since(genStart)
	if stats.Duration.Seconds() > 0 {
		stats.TPS = float64(stats.TokensGenerated) / stats.Duration.Seconds()
	}
	stats.GenerationDuration = stats.Duration
	stats.GenerationTPS = stats.TPS

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

	tokenCache    []string
	tokenCacheOK  []bool
	tokenCacheMap map[int]string

	stopTokenSet   []bool
	stopTokenMap   map[int]struct{}
	stopTokenReady bool
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
	promptStart := time.Now()
	for _, id := range newInTokens {
		logitsVec, err = safeForwardToken(g.Model, id)
		if err != nil {
			return nil, stats, err
		}
	}
	g.ContextTokens = append(g.ContextTokens, newInTokens...)
	stats.PromptTokens = len(newInTokens)
	stats.PromptDuration = time.Since(promptStart)
	if stats.PromptDuration.Seconds() > 0 && stats.PromptTokens > 0 {
		stats.PromptTPS = float64(stats.PromptTokens) / stats.PromptDuration.Seconds()
	}

	toks := append([]int(nil), g.ContextTokens...)

	limit := steps
	if limit < 0 {
		limit = 1000000
	}

	const (
		streamMaxTokens  = 32
		streamMaxChars   = 128
		streamMinChars   = 16
		streamMaxLatency = 40 * time.Millisecond
	)
	var (
		pending       []int
		pendingChars  int
		lastFlushTime = time.Now()
	)
	flush := func() {
		if stream == nil || len(pending) == 0 {
			pending = pending[:0]
			pendingChars = 0
			return
		}
		s, _ := g.decodeTokens(pending)
		if s != "" {
			stream(s)
		}
		pending = pending[:0]
		pendingChars = 0
		lastFlushTime = time.Now()
	}

	genStart := time.Now()
	for i := 0; i < limit; i++ {
		if err := ctx.Err(); err != nil {
			return g.ContextTokens, stats, err
		}
		next, sampleErr := safeSample(g.Sampler, logitsVec, toks, g.StopTokens)
		if sampleErr != nil {
			flush()
			return g.ContextTokens, stats, sampleErr
		}

		stop := g.isStopToken(next)
		if stop {
			break
		}

		toks = append(toks, next)
		g.ContextTokens = append(g.ContextTokens, next)

		if stream != nil {
			pending = append(pending, next)
			if tokStr, err := g.decodeToken(next); err == nil {
				pendingChars += len(tokStr)
			}
			age := time.Since(lastFlushTime)
			if len(pending) >= streamMaxTokens ||
				pendingChars >= streamMaxChars ||
				(age >= streamMaxLatency && pendingChars >= streamMinChars) {
				flush()
			}
		}

		logitsVec, err = safeForwardToken(g.Model, next)
		if err != nil {
			flush()
			return g.ContextTokens, stats, err
		}
		stats.TokensGenerated++
	}

	stats.GenerationDuration = time.Since(genStart)
	if stats.GenerationDuration.Seconds() > 0 {
		stats.GenerationTPS = float64(stats.TokensGenerated) / stats.GenerationDuration.Seconds()
	}
	stats.Duration = time.Since(start)
	if stats.Duration.Seconds() > 0 {
		stats.TPS = float64(stats.TokensGenerated) / stats.Duration.Seconds()
	}

	flush()
	return g.ContextTokens, stats, nil
}

func safeForwardToken(m simd.Model, id int) (logits []float32, err error) {
	defer func() {
		if rec := recover(); rec != nil {
			err = fmt.Errorf("panic in ForwardToken(%d): %v", id, rec)
		}
	}()
	return m.ForwardToken(id)
}

func safeSample(sampler *logits.Sampler, logitsVec []float32, toks []int, excludePenalty []int) (next int, err error) {
	defer func() {
		if rec := recover(); rec != nil {
			err = fmt.Errorf("panic in Sample: %v", rec)
		}
	}()
	return sampler.Sample(logitsVec, toks, excludePenalty), nil
}

func (g *Generator) isStopToken(id int) bool {
	if !g.stopTokenReady {
		g.initStopTokenSet()
	}
	if len(g.stopTokenSet) > 0 {
		if id >= 0 && id < len(g.stopTokenSet) {
			return g.stopTokenSet[id]
		}
		return false
	}
	_, ok := g.stopTokenMap[id]
	return ok
}

func (g *Generator) initStopTokenSet() {
	g.stopTokenReady = true
	if len(g.StopTokens) == 0 {
		return
	}
	if t, ok := g.Tokenizer.(interface{ Decoder() []string }); ok {
		if dec := t.Decoder(); len(dec) > 0 {
			g.stopTokenSet = make([]bool, len(dec))
			for _, id := range g.StopTokens {
				if id >= 0 && id < len(g.stopTokenSet) {
					g.stopTokenSet[id] = true
				}
			}
			return
		}
	}
	g.stopTokenMap = make(map[int]struct{}, len(g.StopTokens))
	for _, id := range g.StopTokens {
		if id >= 0 {
			g.stopTokenMap[id] = struct{}{}
		}
	}
}

func (g *Generator) initTokenCache() {
	if g.tokenCache != nil || g.tokenCacheMap != nil {
		return
	}
	if t, ok := g.Tokenizer.(interface{ Decoder() []string }); ok {
		if dec := t.Decoder(); len(dec) > 0 {
			g.tokenCache = make([]string, len(dec))
			g.tokenCacheOK = make([]bool, len(dec))
			return
		}
	}
	g.tokenCacheMap = make(map[int]string)
}

func (g *Generator) decodeToken(id int) (string, error) {
	g.initTokenCache()
	if g.tokenCache != nil {
		if id >= 0 && id < len(g.tokenCache) && g.tokenCacheOK[id] {
			return g.tokenCache[id], nil
		}
		s, err := g.Tokenizer.Decode([]int{id})
		if err != nil {
			return "", err
		}
		if id >= 0 && id < len(g.tokenCache) {
			g.tokenCache[id] = s
			g.tokenCacheOK[id] = true
		}
		return s, nil
	}
	if s, ok := g.tokenCacheMap[id]; ok {
		return s, nil
	}
	s, err := g.Tokenizer.Decode([]int{id})
	if err != nil {
		return "", err
	}
	g.tokenCacheMap[id] = s
	return s, nil
}

func (g *Generator) decodeTokens(ids []int) (string, error) {
	if len(ids) == 0 {
		return "", nil
	}
	var b strings.Builder
	for _, id := range ids {
		s, err := g.decodeToken(id)
		if err != nil {
			return "", err
		}
		b.WriteString(s)
	}
	return b.String(), nil
}
