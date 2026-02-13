package inference

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/samcharles93/mantle/internal/backend/simd"
	"github.com/samcharles93/mantle/internal/logits"
	"github.com/samcharles93/mantle/internal/mcfstore"
	"github.com/samcharles93/mantle/internal/reasoning"
	"github.com/samcharles93/mantle/internal/tokenizer"
)

type EngineImpl struct {
	mcfFile          *mcfstore.File
	model            simd.Model
	tokenizer        tokenizer.Tokenizer
	tokenizerConfig  tokenizer.TokenizerConfig
	arch             string
	hfConfigJSON     []byte
	chatTemplatePath string
	stopTokens       []int

	generator   *Generator
	lastPrompt  string // rendered prompt text from last Generate() call
	lastGenText string // text generated in last Generate() call
}

func (e *EngineImpl) Close() error {
	if e == nil {
		return nil
	}
	var errs []error
	if e.model != nil {
		if closer, ok := e.model.(interface{ Close() error }); ok {
			if err := closer.Close(); err != nil {
				errs = append(errs, err)
			}
		}
	}
	if e.mcfFile != nil {
		if err := e.mcfFile.Close(); err != nil {
			errs = append(errs, err)
		}
		e.mcfFile = nil
	}
	return errors.Join(errs...)
}

func (e *EngineImpl) Generate(ctx context.Context, req *Request, stream StreamFunc) (*Result, error) {
	if ctx == nil {
		return nil, fmt.Errorf("context is required")
	}
	if req == nil {
		return nil, fmt.Errorf("request is required")
	}

	if err := ctx.Err(); err != nil {
		return nil, err
	}

	prompt, err := e.renderPrompt(req)
	if err != nil {
		return nil, err
	}

	if req.EchoPrompt && stream != nil && prompt != "" {
		stream(prompt)
	}

	// Build token sequence, reusing cached tokens when possible.
	// On subsequent calls in the same conversation, the new prompt starts with
	// lastPrompt + lastGenText. We skip past both (their tokens are already in
	// ContextTokens) and encode only the truly new content (end-of-turn markup
	// + new user message + generation prompt). This avoids re-encoding
	// generated text, which can produce different BPE token splits.
	var ids []int
	if e.generator != nil && e.lastPrompt != "" && strings.HasPrefix(prompt, e.lastPrompt) {
		afterLastPrompt := prompt[len(e.lastPrompt):]
		if strings.HasPrefix(afterLastPrompt, e.lastGenText) {
			newContent := afterLastPrompt[len(e.lastGenText):]
			if newContent != "" {
				newIDs, encErr := safeEncode(e.tokenizer, newContent)
				if encErr == nil {
					ids = make([]int, len(e.generator.ContextTokens)+len(newIDs))
					copy(ids, e.generator.ContextTokens)
					copy(ids[len(e.generator.ContextTokens):], newIDs)
				}
			} else {
				ids = append([]int(nil), e.generator.ContextTokens...)
			}
		}
	}
	if ids == nil {
		ids, err = safeEncode(e.tokenizer, prompt)
		if err != nil {
			return nil, fmt.Errorf("encode prompt: %w", err)
		}
	}

	if err := ctx.Err(); err != nil {
		return nil, err
	}

	sampler := logits.NewSampler(logits.SamplerConfig{
		Seed:          req.Seed,
		Temperature:   float32(req.Temperature),
		TopK:          int(req.TopK),
		TopP:          float32(req.TopP),
		MinP:          float32(req.MinP),
		RepeatPenalty: float32(req.RepeatPenalty),
		RepeatLastN:   int(req.RepeatLastN),
	})

	if e.generator == nil {
		e.generator = &Generator{
			Model:         e.model,
			Sampler:       sampler,
			Tokenizer:     e.tokenizer,
			StopTokens:    append([]int(nil), e.stopTokens...),
			ContextTokens: make([]int, 0, len(ids)),
		}
	} else {
		e.generator.Sampler = sampler
	}

	gen := e.generator

	var sb strings.Builder
	streamWrapper := func(tok string) {
		sb.WriteString(tok)
		if stream != nil {
			stream(tok)
		}
	}

	_, stats, err := gen.RunWithContext(ctx, ids, req.Steps, func(s string) {
		streamWrapper(s)
	})
	if err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	rawText := sb.String()
	if len(gen.ContextTokens) >= len(ids) {
		genIDs := gen.ContextTokens[len(ids):]
		if len(genIDs) > 0 {
			decoded, decErr := safeDecode(e.tokenizer, genIDs)
			if decErr == nil {
				rawText = decoded
			}
		} else {
			rawText = ""
		}
	}
	contentText := rawText
	reasoningText := ""
	switch strings.ToLower(strings.TrimSpace(req.ReasoningFormat)) {
	case "none":
		// Keep raw output untouched.
	default:
		split := reasoning.SplitRaw(rawText)
		contentText = split.Content
		reasoningText = split.Reasoning
	}

	e.lastPrompt = prompt
	e.lastGenText = contentText

	return &Result{
		Text:          contentText,
		ReasoningText: reasoningText,
		RawText:       rawText,
		Stats:         stats,
	}, nil
}

func (e *EngineImpl) ResetContext() {
	if e.generator != nil {
		e.generator.ContextTokens = e.generator.ContextTokens[:0]
		e.model.Reset()
	}
	e.lastPrompt = ""
	e.lastGenText = ""
}

func safeEncode(tok tokenizer.Tokenizer, prompt string) (ids []int, err error) {
	defer func() {
		if rec := recover(); rec != nil {
			err = fmt.Errorf("panic in Encode: %v", rec)
		}
	}()
	return tok.Encode(prompt)
}

func safeDecode(tok tokenizer.Tokenizer, ids []int) (text string, err error) {
	defer func() {
		if rec := recover(); rec != nil {
			err = fmt.Errorf("panic in Decode: %v", rec)
		}
	}()
	return tok.Decode(ids)
}

func (e *EngineImpl) renderPrompt(req *Request) (string, error) {
	return RenderPrompt(PromptRenderInput{
		TemplateOverride:    e.chatTemplatePath,
		TokenizerConfig:     e.tokenizerConfig,
		Arch:                e.arch,
		HFConfigJSON:        e.hfConfigJSON,
		Messages:            req.Messages,
		Tools:               req.Tools,
		AddGenerationPrompt: true,
		NoTemplate:          req.NoTemplate,
	})
}
