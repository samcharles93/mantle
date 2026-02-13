package api

import (
	"context"
	"fmt"
	"time"

	"github.com/samcharles93/mantle/internal/inference"
	"github.com/samcharles93/mantle/internal/tokenizer"
)

type InferenceService struct {
	provider               EngineProvider
	defaultReasoningFormat string
	defaultReasoningBudget int
}

func NewInferenceService(provider EngineProvider) *InferenceService {
	return &InferenceService{
		provider:               provider,
		defaultReasoningFormat: "auto",
		defaultReasoningBudget: -1,
	}
}

func (s *InferenceService) SetReasoningDefaults(format string, budget int) {
	if format != "" {
		s.defaultReasoningFormat = format
	}
	if budget == -1 || budget == 0 {
		s.defaultReasoningBudget = budget
	}
}

type StreamWriter interface {
	Begin(resp ResponsesResponse) error
	EmitToken(delta string) error
	Complete(resp ResponsesResponse, result *inference.Result) error
	Failed(resp ResponsesResponse, err error) error
	Incomplete(resp ResponsesResponse, err error) error
}

func (s *InferenceService) CreateResponse(ctx context.Context, req *ResponsesRequest, stream StreamWriter) (*ResponsesResponse, []ResponseItem, error) {
	inputItems, err := normalizeInputItems(req.Input)
	if err != nil {
		return nil, nil, newInvalidRequest(fmt.Sprintf("input: %v", err))
	}
	msgs, err := responseItemsToMessages(inputItems)
	if err != nil {
		return nil, inputItems, newInvalidRequest(err.Error())
	}

	returnResp := ResponsesResponse{
		ID:                   newResponseID(),
		Object:               "response",
		CreatedAt:            s.clockNow(),
		Status:               "in_progress",
		Background:           req.Background,
		Instructions:         req.Instructions,
		MaxOutputTokens:      req.MaxOutputTokens,
		MaxToolCalls:         req.MaxToolCalls,
		Metadata:             req.Metadata,
		Model:                req.Model,
		ParallelToolCalls:    req.ParallelToolCalls,
		PreviousResponseID:   req.PreviousResponseID,
		Prompt:               req.Prompt,
		PromptCacheKey:       req.PromptCacheKey,
		PromptCacheRetention: req.PromptCacheRetention,
		Reasoning:            req.Reasoning,
		SafetyIdentifier:     req.SafetyIdentifier,
		ServiceTier:          req.ServiceTier,
		Store:                req.Store,
		Temperature:          req.Temperature,
		Text:                 req.Text,
		ToolChoice:           req.ToolChoice,
		Tools:                req.Tools,
		TopP:                 req.TopP,
		Truncation:           req.Truncation,
		Output:               []ResponseItem{},
	}

	if stream != nil {
		if err := stream.Begin(returnResp); err != nil {
			return &returnResp, inputItems, err
		}
	}

	err = s.provider.WithEngine(ctx, req.Model, func(engine inference.Engine, defaults inference.GenDefaults) error {
		reqOpts := toInferenceRequest(req, msgs, defaults, s.defaultReasoningFormat, s.defaultReasoningBudget)
		result, genErr := engine.Generate(ctx, &reqOpts, func(tok string) {
			if stream != nil {
				_ = stream.EmitToken(tok)
			}
		})
		if genErr != nil {
			return genErr
		}
		returnResp.Status = "completed"
		now := s.clockNow()
		returnResp.CompletedAt = &now
		returnResp.Output = buildOutputMessage(result.Text)
		returnResp.OutputText = result.Text
		returnResp.ReasoningText = result.ReasoningText
		returnResp.Usage = &ResponseUsage{
			InputTokens:  approximateTokenCount(inputItems),
			OutputTokens: approximateTokenCount(returnResp.Output),
			TotalTokens:  approximateTokenCount(inputItems) + approximateTokenCount(returnResp.Output),
		}
		if returnResp.Usage.OutputTokensDetails == nil {
			returnResp.Usage.OutputTokensDetails = &ResponseTokenDetails{}
		}
		returnResp.Usage.OutputTokensDetails.ReasoningTokens = max(len([]rune(result.ReasoningText))/4, 0)
		if stream != nil {
			if err := stream.Complete(returnResp, result); err != nil {
				return err
			}
		}
		return nil
	})

	if err != nil {
		returnResp.Status = "failed"
		returnResp.Error = &ResponseError{
			Message: err.Error(),
			Type:    "server_error",
		}
		if stream != nil {
			if ctx.Err() != nil {
				_ = stream.Incomplete(returnResp, ctx.Err())
			} else {
				_ = stream.Failed(returnResp, err)
			}
		}
		return &returnResp, inputItems, err
	}

	return &returnResp, inputItems, nil
}

func (s *InferenceService) clockNow() int64 {
	return timeNow().Unix()
}

var timeNow = func() time.Time {
	return time.Now()
}

func toInferenceRequest(req *ResponsesRequest, msgs []tokenizer.Message, defaults inference.GenDefaults, defaultReasoningFormat string, defaultReasoningBudget int) inference.Request {
	var opts inference.RequestOptions
	opts.Messages = msgs
	opts.Tools = toolsToAny(req.Tools)
	opts.NoTemplate = boolPtr(false)
	opts.EchoPrompt = boolPtr(false)
	opts.ReasoningFormat = stringPtr(defaultReasoningFormat)
	opts.ReasoningBudget = intPtr(defaultReasoningBudget)

	if req.MaxOutputTokens != nil {
		steps := int(*req.MaxOutputTokens)
		opts.Steps = &steps
	}
	if req.Temperature != nil {
		opts.Temperature = req.Temperature
	}
	if req.TopP != nil {
		opts.TopP = req.TopP
	}
	if req.TopLogprobs != nil {
		_ = req.TopLogprobs
	}
	if req.ReasoningFormat != "" {
		opts.ReasoningFormat = stringPtr(req.ReasoningFormat)
	}
	if req.ReasoningBudget != nil {
		opts.ReasoningBudget = req.ReasoningBudget
	}

	return inference.ResolveRequest(opts, defaults)
}

func toolsToAny(tools []ResponseTool) []any {
	if len(tools) == 0 {
		return nil
	}
	out := make([]any, 0, len(tools))
	for _, tool := range tools {
		out = append(out, map[string]any(tool))
	}
	return out
}

func buildOutputMessage(text string) []ResponseItem {
	if text == "" {
		return nil
	}
	return []ResponseItem{{
		ID:   newOutputItemID(),
		Type: "message",
		Role: "assistant",
		Content: []ResponseContent{{
			Type: "output_text",
			Text: text,
		}},
	}}
}

func boolPtr(v bool) *bool {
	return &v
}

func intPtr(v int) *int {
	return &v
}

func stringPtr(v string) *string {
	return &v
}
