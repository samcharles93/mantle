package inference

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/samcharles93/mantle/internal/backend/core"
	"github.com/samcharles93/mantle/internal/logger"
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

type prefillForwarder interface {
	PrefillTokens(tokens []int) ([]float32, error)
}

// Generator manages the state of a generation session
type Generator struct {
	Model     core.Model
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

	// Set effective context length for dynamic KV cache bounding.
	// This prevents allocating cache for the model's full max context window
	// when we only need space for prompt + requested generation steps.
	effectiveCtx := len(allTokens) + steps
	if steps <= 0 {
		// If steps is infinite (-1) or zero, use a reasonable default upper bound
		effectiveCtx = len(allTokens) + 1_000_000
	}
	if setter, ok := g.Model.(interface{ SetEffectiveContextLength(int) }); ok {
		setter.SetEffectiveContextLength(effectiveCtx)
	}

	newInTokens := allTokens[len(g.ContextTokens):]

	var logitsVec []float32
	var err error
	promptStart := time.Now()

	// Use batched ForwardTokens for prompt prefill (uses GEMM with tiling)
	// This is much faster than token-by-token for longer prompts
	type batchForwarder interface {
		ForwardTokens(tokens []int) ([][]float32, error)
	}

	if len(newInTokens) > 1 {
		if pf, ok := g.Model.(prefillForwarder); ok {
			prefillLogits, prefillErr := safePrefillTokens(pf, newInTokens)
			if prefillErr == nil && len(prefillLogits) > 0 {
				logitsVec = prefillLogits
				g.ContextTokens = append(g.ContextTokens, newInTokens...)
				stats.PromptTokens = len(newInTokens)
				stats.PromptDuration = time.Since(promptStart)
				if stats.PromptDuration.Seconds() > 0 && stats.PromptTokens > 0 {
					stats.PromptTPS = float64(stats.PromptTokens) / stats.PromptDuration.Seconds()
				}
				goto afterPrompt
			}
			if prefillErr != nil {
				log := logger.FromContext(ctx)
				log.Debug("prefill fast path unavailable, falling back to token-by-token", "error", prefillErr)
			}
		}
		if bf, ok := g.Model.(batchForwarder); ok {
			allLogits, batchErr := bf.ForwardTokens(newInTokens)
			if batchErr == nil && len(allLogits) > 0 {
				// Use logits from the last token position
				logitsVec = allLogits[len(allLogits)-1]
				g.ContextTokens = append(g.ContextTokens, newInTokens...)
				stats.PromptTokens = len(newInTokens)
				stats.PromptDuration = time.Since(promptStart)
				if stats.PromptDuration.Seconds() > 0 && stats.PromptTokens > 0 {
					stats.PromptTPS = float64(stats.PromptTokens) / stats.PromptDuration.Seconds()
				}
				// Continue to generation phase with last logits already computed
				goto afterPrompt
			}
			if batchErr != nil {
				log := logger.FromContext(ctx)
				log.Debug("batch prefill unavailable, falling back to token-by-token", "error", batchErr)
			}
		}
	}

	// Process tokens one at a time (fallback or single token)
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

afterPrompt:
	toks := append([]int(nil), g.ContextTokens...)
	type greedyStepper interface {
		ForwardTokenGreedy(id int) (next int, err error)
	}
	gs, canDeviceGreedy := g.Model.(greedyStepper)
	canDeviceGreedy = canDeviceGreedy && g.Sampler.CanUseDeviceGreedy()

	limit := steps
	if limit < 0 {
		limit = 1000000
	}
	if limit == 0 {
		stats.Duration = time.Since(start)
		return g.ContextTokens, stats, nil
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
	var streamingOverhead time.Duration
	next, sampleErr := safeSample(g.Sampler, logitsVec, toks, g.StopTokens)
	if sampleErr != nil {
		flush()
		return g.ContextTokens, stats, sampleErr
	}
	for range limit {
		if err := ctx.Err(); err != nil {
			return g.ContextTokens, stats, err
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
				callbackStart := time.Now()
				flush()
				streamingOverhead += time.Since(callbackStart)
			}
		}

		if canDeviceGreedy {
			next, err = safeForwardTokenGreedy(gs, next)
			if err != nil {
				flush()
				return g.ContextTokens, stats, err
			}
		} else {
			logitsVec, err = safeForwardToken(g.Model, next)
			if err != nil {
				flush()
				return g.ContextTokens, stats, err
			}
			next, sampleErr = safeSample(g.Sampler, logitsVec, toks, g.StopTokens)
			if sampleErr != nil {
				flush()
				return g.ContextTokens, stats, sampleErr
			}
		}
		stats.TokensGenerated++
	}

	totalDuration := time.Since(genStart)
	stats.GenerationDuration = totalDuration - streamingOverhead
	if stats.GenerationDuration.Seconds() > 0 {
		stats.GenerationTPS = float64(stats.TokensGenerated) / stats.GenerationDuration.Seconds()
	}

	stats.Duration = stats.GenerationDuration

	if stats.Duration.Seconds() > 0 {
		stats.TPS = float64(stats.TokensGenerated) / stats.Duration.Seconds()
	}

	flush()
	return g.ContextTokens, stats, nil
}

func safeForwardToken(m core.Model, id int) (logits []float32, err error) {
	defer func() {
		if rec := recover(); rec != nil {
			err = fmt.Errorf("panic in ForwardToken(%d): %v", id, rec)
		}
	}()
	return m.ForwardToken(id)
}

func safePrefillTokens(m prefillForwarder, tokens []int) (logits []float32, err error) {
	defer func() {
		if rec := recover(); rec != nil {
			err = fmt.Errorf("panic in PrefillTokens(%d): %v", len(tokens), rec)
		}
	}()
	return m.PrefillTokens(tokens)
}

func safeSample(sampler *logits.Sampler, logitsVec []float32, toks []int, excludePenalty []int) (next int, err error) {
	defer func() {
		if rec := recover(); rec != nil {
			err = fmt.Errorf("panic in Sample: %v", rec)
		}
	}()
	return sampler.Sample(logitsVec, toks, excludePenalty), nil
}

func safeForwardTokenGreedy(m interface {
	ForwardTokenGreedy(id int) (next int, err error)
}, id int) (next int, err error) {
	defer func() {
		if rec := recover(); rec != nil {
			err = fmt.Errorf("panic in ForwardTokenGreedy(%d): %v", id, rec)
		}
	}()
	return m.ForwardTokenGreedy(id)
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
