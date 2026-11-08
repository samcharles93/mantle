package inference

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/samcharles93/mantle/internal/backend/core"
	"github.com/samcharles93/mantle/internal/graph"
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

// prefillEmbedForwarder is the prefill path with per-position input embedding
// overrides (multimodal image tokens).
type prefillEmbedForwarder interface {
	PrefillTokensEmbeds(tokens []int, embeds [][]float32) ([]float32, error)
}

// graphRunner is the subset of the SIMD Instance needed for graph-based token
// processing. Backends that support graph execution implement this interface.
type graphRunner interface {
	Compute(ctx *graph.ComputeContext, g *graph.Graph) ([]float32, error)
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

	// Embeds, when non-nil, is parallel to the token slice passed to the next
	// RunWithContext call and overrides the input embedding of positions with
	// a non-nil entry (vision embeddings at image-token positions). Consumed
	// (and cleared) by that call.
	Embeds [][]float32

	// Graph-based execution path (experimental, activated via --use-graph).
	useGraph    bool
	engineGraph *graph.Graph
	gr          graphRunner

	// Speculator, when non-nil, enables speculative decoding rounds instead
	// of the per-token loop (CPU draft models only; ignored with --use-graph).
	Speculator *Speculator // non-nil when useGraph is true

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
		if g.Speculator != nil {
			g.Speculator.Reset()
		}
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

	// Slice the pending embedding overrides to the tokens actually being
	// prefilled, then hand them to the embed-aware prefill path.
	newEmbeds := g.Embeds
	if newEmbeds != nil {
		if len(newEmbeds) != len(allTokens) {
			return nil, stats, fmt.Errorf("embeds length %d != tokens %d", len(newEmbeds), len(allTokens))
		}
		newEmbeds = newEmbeds[len(g.ContextTokens):]
	}
	g.Embeds = nil
	hasEmbeds := false
	for _, e := range newEmbeds {
		if e != nil {
			hasEmbeds = true
			break
		}
	}

	var logitsVec []float32
	var err error
	promptStart := time.Now()

	// Graph path: per-token processing through the graph (no batch prefill).
	if g.useGraph {
		pos := len(g.ContextTokens)
		for _, id := range newInTokens {
			computeCtx := &graph.ComputeContext{
				Token: id,
				Pos:   pos,
				KVLen: pos,
			}
			logitsVec, err = g.gr.Compute(computeCtx, g.engineGraph)
			if err != nil {
				return nil, stats, err
			}
			pos++
		}
		g.ContextTokens = append(g.ContextTokens, newInTokens...)
		stats.PromptTokens = len(newInTokens)
		stats.PromptDuration = time.Since(promptStart)
		if stats.PromptDuration.Seconds() > 0 && stats.PromptTokens > 0 {
			stats.PromptTPS = float64(stats.PromptTokens) / stats.PromptDuration.Seconds()
		}
		goto afterPrompt
	}

	// Use batched ForwardTokens for prompt prefill (uses GEMM with tiling)
	// This is much faster than token-by-token for longer prompts
	type batchForwarder interface {
		ForwardTokens(tokens []int) ([][]float32, error)
	}

	if len(newInTokens) > 1 {
		if hasEmbeds {
			if pef, ok := g.Model.(prefillEmbedForwarder); ok {
				prefillLogits, prefillErr := pef.PrefillTokensEmbeds(newInTokens, newEmbeds)
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
					log.Debug("embed prefill fast path unavailable, falling back to token-by-token", "error", prefillErr)
				}
			}
		}
		if pf, ok := g.Model.(prefillForwarder); ok && !hasEmbeds {
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
		if bf, ok := g.Model.(batchForwarder); ok && !hasEmbeds {
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
	for i, id := range newInTokens {
		if hasEmbeds && newEmbeds[i] != nil {
			if fe, ok := g.Model.(interface {
				ForwardTokenEmbed(id int, emb []float32) ([]float32, error)
			}); ok {
				logitsVec, err = fe.ForwardTokenEmbed(id, newEmbeds[i])
			} else {
				return nil, stats, fmt.Errorf("model does not support embedding overrides")
			}
		} else {
			logitsVec, err = safeForwardToken(g.Model, id)
		}
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

	sb := newStreamBuffer(g, stream)
	genStart := time.Now()
	var genErr error
	if g.Speculator != nil && !g.useGraph {
		var stopped bool
		logitsVec, stopped, genErr = g.generateSpeculative(ctx, sb, logitsVec, toks, limit, &stats, gs, canDeviceGreedy)
		if genErr == nil && !stopped && stats.TokensGenerated < limit {
			// Context edge: the draft block no longer fits; finish per-token.
			genErr = g.generatePlain(ctx, sb, logitsVec, toks, limit, &stats, gs, canDeviceGreedy)
		}
	} else {
		genErr = g.generatePlain(ctx, sb, logitsVec, toks, limit, &stats, gs, canDeviceGreedy)
	}
	if genErr != nil {
		sb.flush()
		return g.ContextTokens, stats, genErr
	}

	totalDuration := time.Since(genStart)
	stats.GenerationDuration = totalDuration - sb.overhead
	if stats.GenerationDuration.Seconds() > 0 {
		stats.GenerationTPS = float64(stats.TokensGenerated) / stats.GenerationDuration.Seconds()
	}

	stats.Duration = stats.GenerationDuration

	if stats.Duration.Seconds() > 0 {
		stats.TPS = float64(stats.TokensGenerated) / stats.Duration.Seconds()
	}

	sb.flush()
	return g.ContextTokens, stats, nil
}

const (
	streamMaxTokens  = 32
	streamMaxChars   = 128
	streamMinChars   = 16
	streamMaxLatency = 40 * time.Millisecond
)

// streamBuffer batches generated tokens into stream callbacks, throttled by
// count, decoded length and latency, and accounts for the callback overhead in
// the generation stats.
type streamBuffer struct {
	gen          *Generator
	stream       func(string)
	pending      []int
	pendingChars int
	lastFlush    time.Time
	overhead     time.Duration
}

func newStreamBuffer(g *Generator, stream func(string)) *streamBuffer {
	return &streamBuffer{gen: g, stream: stream, lastFlush: time.Now()}
}

func (s *streamBuffer) add(tok int) {
	if s.stream == nil {
		return
	}
	s.pending = append(s.pending, tok)
	if tokStr, err := s.gen.decodeToken(tok); err == nil {
		s.pendingChars += len(tokStr)
	}
	age := time.Since(s.lastFlush)
	if len(s.pending) >= streamMaxTokens ||
		s.pendingChars >= streamMaxChars ||
		(age >= streamMaxLatency && s.pendingChars >= streamMinChars) {
		s.flush()
	}
}

func (s *streamBuffer) flush() {
	if s.stream == nil || len(s.pending) == 0 {
		s.pending = s.pending[:0]
		s.pendingChars = 0
		return
	}
	str, _ := s.gen.decodeTokens(s.pending)
	callbackStart := time.Now()
	if str != "" {
		s.stream(str)
	}
	s.overhead += time.Since(callbackStart)
	s.pending = s.pending[:0]
	s.pendingChars = 0
	s.lastFlush = time.Now()
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
}, id int,
) (next int, err error) {
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
