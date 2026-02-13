package api

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/google/uuid"
	"github.com/labstack/echo/v5"

	"github.com/samcharles93/mantle/internal/inference"
	"github.com/samcharles93/mantle/internal/tokenizer"
)

// ChatCompletionRequest represents an OpenAI-compatible chat completion request.
type ChatCompletionRequest struct {
	Model               string              `json:"model"`
	Messages            []ChatMessage       `json:"messages"`
	Temperature         *float64            `json:"temperature,omitempty"`
	TopP                *float64            `json:"top_p,omitempty"`
	N                   *int                `json:"n,omitempty"`
	Stream              *bool               `json:"stream,omitempty"`
	Stop                any                 `json:"stop,omitempty"`
	MaxTokens           *int                `json:"max_tokens,omitempty"`
	MaxCompletionTokens *int                `json:"max_completion_tokens,omitempty"`
	PresencePenalty     *float64            `json:"presence_penalty,omitempty"`
	FrequencyPenalty    *float64            `json:"frequency_penalty,omitempty"`
	RepeatPenalty       *float64            `json:"repeat_penalty,omitempty"`
	Seed                *int64              `json:"seed,omitempty"`
	User                string              `json:"user,omitempty"`
	Tools               []ChatTool          `json:"tools,omitempty"`
	ToolChoice          any                 `json:"tool_choice,omitempty"`
	ResponseFormat      *ChatResponseFormat `json:"response_format,omitempty"`
}

type ChatMessage struct {
	Role       string         `json:"role"`
	Content    any            `json:"content"`
	Name       string         `json:"name,omitempty"`
	ToolCalls  []ChatToolCall `json:"tool_calls,omitempty"`
	ToolCallID string         `json:"tool_call_id,omitempty"`
}

type ChatToolCall struct {
	ID       string           `json:"id"`
	Type     string           `json:"type"`
	Function ChatFunctionCall `json:"function"`
}

type ChatFunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type ChatTool struct {
	Type     string       `json:"type"`
	Function ChatFunction `json:"function"`
}

type ChatFunction struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Parameters  any    `json:"parameters,omitempty"`
}

type ChatResponseFormat struct {
	Type string `json:"type,omitempty"`
}

// ChatCompletionResponse is the response for non-streaming chat completions.
type ChatCompletionResponse struct {
	ID                string       `json:"id"`
	Object            string       `json:"object"`
	Created           int64        `json:"created"`
	Model             string       `json:"model"`
	Choices           []ChatChoice `json:"choices"`
	Usage             ChatUsage    `json:"usage"`
	SystemFingerprint string       `json:"system_fingerprint,omitempty"`
}

type ChatChoice struct {
	Index        int          `json:"index"`
	Message      *ChatMessage `json:"message,omitempty"`
	Delta        *ChatMessage `json:"delta,omitempty"`
	FinishReason *string      `json:"finish_reason"`
}

type ChatUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// ChatCompletionChunk is a streaming SSE chunk.
type ChatCompletionChunk struct {
	ID                string       `json:"id"`
	Object            string       `json:"object"`
	Created           int64        `json:"created"`
	Model             string       `json:"model"`
	Choices           []ChatChoice `json:"choices"`
	SystemFingerprint string       `json:"system_fingerprint,omitempty"`
}

func (s *Server) RegisterChatCompletions(e *echo.Echo) {
	e.POST("/v1/chat/completions", s.handleChatCompletions)
	e.GET("/v1/models", s.handleListModels)
}

func (s *Server) handleListModels(c *echo.Context) error {
	modelIDs := []string{"mantle"}
	if s.service != nil && s.service.provider != nil {
		if provider, ok := s.service.provider.(interface {
			ListModels() ([]string, error)
		}); ok {
			discovered, err := provider.ListModels()
			if err != nil {
				return writeError(c, http.StatusInternalServerError, "server_error", err.Error(), "", "")
			}
			if len(discovered) > 0 {
				modelIDs = discovered
			}
		}
	}

	data := make([]map[string]any, 0, len(modelIDs))
	for _, id := range modelIDs {
		data = append(data, map[string]any{
			"id":       id,
			"object":   "model",
			"created":  time.Now().Unix(),
			"owned_by": "local",
		})
	}

	return c.JSON(http.StatusOK, map[string]any{
		"object": "list",
		"data":   data,
	})
}

func (s *Server) handleChatCompletions(c *echo.Context) error {
	if s.service == nil {
		return writeError(c, http.StatusInternalServerError, "server_error", "inference service not configured", "", "")
	}

	req, err := decodeJSON[ChatCompletionRequest](c.Request().Body)
	if err != nil {
		return writeBadRequest(c, err.Error())
	}

	if len(req.Messages) == 0 {
		return writeBadRequest(c, "messages is required and must not be empty")
	}

	msgs, err := chatMessagesToTokenizerMessages(req.Messages)
	if err != nil {
		return writeBadRequest(c, err.Error())
	}

	isStream := req.Stream != nil && *req.Stream
	completionID := "chatcmpl-" + uuid.NewString()
	created := s.clock().Unix()
	model := req.Model
	if model == "" {
		model = "mantle"
	}

	if isStream {
		return s.handleChatCompletionsStream(c, req, msgs, completionID, created, model)
	}

	return s.handleChatCompletionsSync(c, req, msgs, completionID, created, model)
}

func (s *Server) handleChatCompletionsSync(c *echo.Context, req ChatCompletionRequest, msgs []tokenizer.Message, completionID string, created int64, model string) error {
	var resultText string
	var resultStats inference.Stats

	err := s.service.provider.WithEngine(c.Request().Context(), req.Model, func(engine inference.Engine, defaults inference.GenDefaults) error {
		inferReq := chatToInferenceRequest(&req, msgs, defaults)
		result, err := engine.Generate(c.Request().Context(), &inferReq, nil)
		if err != nil {
			return err
		}
		resultText = result.Text
		resultStats = result.Stats
		return nil
	})
	if err != nil {
		return writeError(c, http.StatusInternalServerError, "server_error", err.Error(), "", "")
	}

	finishReason := "stop"
	resp := ChatCompletionResponse{
		ID:      completionID,
		Object:  "chat.completion",
		Created: created,
		Model:   model,
		Choices: []ChatChoice{
			{
				Index: 0,
				Message: &ChatMessage{
					Role:    "assistant",
					Content: resultText,
				},
				FinishReason: &finishReason,
			},
		},
		Usage: ChatUsage{
			PromptTokens:     resultStats.PromptTokens,
			CompletionTokens: resultStats.TokensGenerated,
			TotalTokens:      resultStats.PromptTokens + resultStats.TokensGenerated,
		},
	}

	return c.JSON(http.StatusOK, resp)
}

func (s *Server) handleChatCompletionsStream(c *echo.Context, req ChatCompletionRequest, msgs []tokenizer.Message, completionID string, created int64, model string) error {
	res := c.Response()
	res.Header().Set(echo.HeaderContentType, "text/event-stream")
	res.Header().Set("Cache-Control", "no-cache")
	res.Header().Set("Connection", "keep-alive")

	flusher, ok := res.(interface{ Flush() })
	if !ok {
		return writeBadRequest(c, "streaming unsupported")
	}

	// Send initial chunk with role
	initialChunk := ChatCompletionChunk{
		ID:      completionID,
		Object:  "chat.completion.chunk",
		Created: created,
		Model:   model,
		Choices: []ChatChoice{
			{
				Index: 0,
				Delta: &ChatMessage{Role: "assistant"},
			},
		},
	}
	if err := sendSSEChunk(res, initialChunk); err != nil {
		return err
	}
	flusher.Flush()

	err := s.service.provider.WithEngine(c.Request().Context(), req.Model, func(engine inference.Engine, defaults inference.GenDefaults) error {
		inferReq := chatToInferenceRequest(&req, msgs, defaults)
		_, err := engine.Generate(c.Request().Context(), &inferReq, func(tok string) {
			chunk := ChatCompletionChunk{
				ID:      completionID,
				Object:  "chat.completion.chunk",
				Created: created,
				Model:   model,
				Choices: []ChatChoice{
					{
						Index: 0,
						Delta: &ChatMessage{Content: tok},
					},
				},
			}
			_ = sendSSEChunk(res, chunk)
			flusher.Flush()
		})
		return err
	})

	if err != nil {
		// Best effort error chunk
		_ = sendSSEChunk(res, map[string]any{"error": err.Error()})
		flusher.Flush()
	}

	// Send final chunk with finish_reason
	finishReason := "stop"
	finalChunk := ChatCompletionChunk{
		ID:      completionID,
		Object:  "chat.completion.chunk",
		Created: created,
		Model:   model,
		Choices: []ChatChoice{
			{
				Index:        0,
				Delta:        &ChatMessage{},
				FinishReason: &finishReason,
			},
		},
	}
	_ = sendSSEChunk(res, finalChunk)
	_, _ = fmt.Fprint(res, "data: [DONE]\n\n")
	flusher.Flush()

	return nil
}

func sendSSEChunk(w io.Writer, v any) error {
	b, err := json.Marshal(v)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(w, "data: %s\n\n", string(b))
	return err
}

func chatMessagesToTokenizerMessages(msgs []ChatMessage) ([]tokenizer.Message, error) {
	out := make([]tokenizer.Message, 0, len(msgs))
	for _, m := range msgs {
		msg := tokenizer.Message{
			Role: m.Role,
		}

		switch content := m.Content.(type) {
		case string:
			msg.Content = content
		case nil:
			msg.Content = ""
		case []any:
			// Multi-part content (text + images)
			var textParts []string
			for _, part := range content {
				pm, ok := part.(map[string]any)
				if !ok {
					continue
				}
				if typ, _ := pm["type"].(string); typ == "text" {
					if text, ok := pm["text"].(string); ok {
						textParts = append(textParts, text)
					}
				}
			}
			msg.Content = joinStrings(textParts, "\n")
		default:
			// Try JSON marshal/unmarshal as fallback
			b, err := json.Marshal(content)
			if err != nil {
				return nil, fmt.Errorf("message content: unsupported type")
			}
			msg.Content = string(b)
		}

		// Convert tool calls
		if len(m.ToolCalls) > 0 {
			tcs := make([]tokenizer.ToolCall, 0, len(m.ToolCalls))
			for _, tc := range m.ToolCalls {
				// Parse arguments string to any for the tokenizer
				var args any = tc.Function.Arguments
				if tc.Function.Arguments != "" {
					var parsed map[string]any
					if json.Unmarshal([]byte(tc.Function.Arguments), &parsed) == nil {
						args = parsed
					}
				}
				tcs = append(tcs, tokenizer.ToolCall{
					ID:   tc.ID,
					Type: tc.Type,
					Function: tokenizer.ToolCallFunction{
						Name:      tc.Function.Name,
						Arguments: args,
					},
				})
			}
			msg.ToolCalls = tcs
		}

		out = append(out, msg)
	}
	return out, nil
}

func joinStrings(parts []string, sep string) string {
	if len(parts) == 0 {
		return ""
	}
	result := parts[0]
	for _, p := range parts[1:] {
		result += sep + p
	}
	return result
}

func chatToInferenceRequest(req *ChatCompletionRequest, msgs []tokenizer.Message, defaults inference.GenDefaults) inference.Request {
	var opts inference.RequestOptions
	opts.Messages = msgs
	opts.NoTemplate = boolPtr(false)
	opts.EchoPrompt = boolPtr(false)

	maxToks := req.MaxTokens
	if req.MaxCompletionTokens != nil {
		maxToks = req.MaxCompletionTokens
	}
	if maxToks != nil {
		opts.Steps = maxToks
	}
	if req.Temperature != nil {
		opts.Temperature = req.Temperature
	}
	if req.TopP != nil {
		opts.TopP = req.TopP
	}
	if req.Seed != nil {
		opts.Seed = req.Seed
	}
	if req.RepeatPenalty != nil {
		opts.RepeatPenalty = req.RepeatPenalty
	}

	if len(req.Tools) > 0 {
		tools := make([]any, 0, len(req.Tools))
		for _, t := range req.Tools {
			tools = append(tools, map[string]any{
				"type": t.Type,
				"function": map[string]any{
					"name":        t.Function.Name,
					"description": t.Function.Description,
					"parameters":  t.Function.Parameters,
				},
			})
		}
		opts.Tools = tools
	}

	return inference.ResolveRequest(opts, defaults)
}
