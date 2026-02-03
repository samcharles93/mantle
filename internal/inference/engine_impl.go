package inference

import (
	"context"
	"fmt"
	"strings"

	"github.com/samcharles93/mantle/internal/logits"
	"github.com/samcharles93/mantle/internal/model"
	"github.com/samcharles93/mantle/internal/tokenizer"
)

type EngineImpl struct {
	model            *model.Instance
	tokenizer        tokenizer.Tokenizer
	tokenizerConfig  tokenizer.TokenizerConfig
	arch             string
	hfConfigJSON     []byte
	chatTemplatePath string
	stopTokens       []int
}

func (e *EngineImpl) Close() error {
	return nil
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

	ids, err := e.tokenizer.Encode(prompt)
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

	e.model.Reset()

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
