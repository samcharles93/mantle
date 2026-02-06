package inference

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/samcharles93/mantle/internal/backend/simd"
	"github.com/samcharles93/mantle/internal/logits"
	"github.com/samcharles93/mantle/internal/mcfstore"
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

	ids, err := safeEncode(e.tokenizer, prompt)
	if err != nil {
		return nil, fmt.Errorf("encode prompt: %w", err)
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

	gen := &Generator{
		Model:         e.model,
		Sampler:       sampler,
		Tokenizer:     e.tokenizer,
		StopTokens:    append([]int(nil), e.stopTokens...),
		ContextTokens: make([]int, 0, len(ids)),
	}

	if err := safeReset(e.model); err != nil {
		return nil, err
	}

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

	return &Result{
		Text:  sb.String(),
		Stats: stats,
	}, nil
}

func safeReset(m simd.Model) (err error) {
	defer func() {
		if rec := recover(); rec != nil {
			err = fmt.Errorf("panic in Reset: %v", rec)
		}
	}()
	m.Reset()
	return nil
}

func safeEncode(tok tokenizer.Tokenizer, prompt string) (ids []int, err error) {
	defer func() {
		if rec := recover(); rec != nil {
			err = fmt.Errorf("panic in Encode: %v", rec)
		}
	}()
	return tok.Encode(prompt)
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
