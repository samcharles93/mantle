package api

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"testing"

	"github.com/labstack/echo/v5"
	"github.com/samcharles93/mantle/internal/inference"
	"github.com/samcharles93/mantle/internal/tokenizer"
)

func TestChatCompletionsBasic(t *testing.T) {
	t.Parallel()

	e := newTestEcho()
	body := `{"model":"mantle","messages":[{"role":"user","content":"hello"}]}`
	rec := doJSON(t, e, http.MethodPost, "/v1/chat/completions", body)
	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d body=%s", rec.Code, rec.Body.String())
	}

	var resp ChatCompletionResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}

	if resp.Object != "chat.completion" {
		t.Fatalf("unexpected object: %q", resp.Object)
	}
	if !strings.HasPrefix(resp.ID, "chatcmpl-") {
		t.Fatalf("unexpected id format: %q", resp.ID)
	}
	if len(resp.Choices) != 1 {
		t.Fatalf("expected 1 choice, got %d", len(resp.Choices))
	}
	if resp.Choices[0].Message == nil {
		t.Fatal("expected message in choice")
	}
	if resp.Choices[0].Message.Role != "assistant" {
		t.Fatalf("expected assistant role, got %q", resp.Choices[0].Message.Role)
	}
	if resp.Choices[0].Message.Content != "ok" {
		t.Fatalf("expected 'ok' content, got %q", resp.Choices[0].Message.Content)
	}
	if resp.Choices[0].FinishReason == nil || *resp.Choices[0].FinishReason != "stop" {
		t.Fatal("expected finish_reason 'stop'")
	}
}

func TestChatCompletionsEmptyMessages(t *testing.T) {
	t.Parallel()

	e := newTestEcho()
	body := `{"model":"mantle","messages":[]}`
	rec := doJSON(t, e, http.MethodPost, "/v1/chat/completions", body)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 for empty messages, got %d body=%s", rec.Code, rec.Body.String())
	}
}

func TestChatCompletionsWithSystemMessage(t *testing.T) {
	t.Parallel()

	e := newTestEcho()
	body := `{"model":"mantle","messages":[{"role":"system","content":"You are helpful."},{"role":"user","content":"hi"}]}`
	rec := doJSON(t, e, http.MethodPost, "/v1/chat/completions", body)
	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d body=%s", rec.Code, rec.Body.String())
	}

	var resp ChatCompletionResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if resp.Model != "mantle" {
		t.Fatalf("expected model 'mantle', got %q", resp.Model)
	}
}

func TestChatCompletionsStreaming(t *testing.T) {
	t.Parallel()

	e := newTestEcho()
	body := `{"model":"mantle","messages":[{"role":"user","content":"hello"}],"stream":true}`
	rec := doJSON(t, e, http.MethodPost, "/v1/chat/completions", body)
	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d body=%s", rec.Code, rec.Body.String())
	}

	respBody := rec.Body.String()
	if !strings.Contains(respBody, "data: ") {
		t.Fatal("expected SSE data events in streaming response")
	}
	if !strings.Contains(respBody, "data: [DONE]") {
		t.Fatal("expected [DONE] sentinel in streaming response")
	}
	if !strings.Contains(respBody, "chat.completion.chunk") {
		t.Fatal("expected chat.completion.chunk objects in streaming response")
	}
}

func TestChatCompletionsStreamingOmitsReasoningChunks(t *testing.T) {
	t.Parallel()

	provider := testProvider{
		engine: testEngine{
			chunks: []inference.StreamChunk{
				{Type: inference.StreamChunkReasoningDelta, Delta: "secret"},
				{Type: inference.StreamChunkTextDelta, Delta: "visible"},
			},
			result: &inference.Result{
				Text:          "visible",
				ReasoningText: "secret",
			},
		},
	}
	service := NewInferenceService(provider)
	server := NewServer(NewResponseStore(), service)
	e := echo.New()
	server.Register(e)

	body := `{"model":"mantle","messages":[{"role":"user","content":"hello"}],"stream":true}`
	rec := doJSON(t, e, http.MethodPost, "/v1/chat/completions", body)
	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d body=%s", rec.Code, rec.Body.String())
	}

	respBody := rec.Body.String()
	if !strings.Contains(respBody, "visible") {
		t.Fatalf("expected visible content in stream, got %s", respBody)
	}
	if strings.Contains(respBody, "secret") {
		t.Fatalf("expected reasoning to be omitted from chat stream, got %s", respBody)
	}
}

// TestChatCompletionsReasoningOutput pins where a reasoning model's generated
// text lands in a chat completion and how finish_reason reports truncation. A
// thinking model that spends its whole budget inside <think>...</think> has no
// visible content, so its reasoning text is the only generated text there is to
// return, and a run that used its whole step budget is a length truncation.
func TestChatCompletionsReasoningOutput(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		maxTokens   int
		result      inference.Result
		wantContent string
		wantFinish  string
	}{
		{
			name:        "reasoning only, truncated",
			maxTokens:   16,
			result:      inference.Result{ReasoningText: "the user wants a greeting", Stats: inference.Stats{PromptTokens: 11, TokensGenerated: 16}},
			wantContent: "the user wants a greeting",
			wantFinish:  "length",
		},
		{
			name:        "reasoning only, stopped early",
			maxTokens:   16,
			result:      inference.Result{ReasoningText: "hmm", Stats: inference.Stats{TokensGenerated: 3}},
			wantContent: "hmm",
			wantFinish:  "stop",
		},
		{
			name:        "content with reasoning",
			maxTokens:   16,
			result:      inference.Result{Text: "Hi!", ReasoningText: "thinking", Stats: inference.Stats{TokensGenerated: 8}},
			wantContent: "Hi!",
			wantFinish:  "stop",
		},
		{
			name:        "content, truncated",
			maxTokens:   4,
			result:      inference.Result{Text: "Hi", Stats: inference.Stats{TokensGenerated: 4}},
			wantContent: "Hi",
			wantFinish:  "length",
		},
		{
			name:        "no token limit, not truncated",
			maxTokens:   0,
			result:      inference.Result{Text: "Hi", Stats: inference.Stats{TokensGenerated: 64}},
			wantContent: "Hi",
			wantFinish:  "stop",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			service := NewInferenceService(testProvider{engine: testEngine{result: &tc.result}})
			e := echo.New()
			NewServer(NewResponseStore(), service).Register(e)

			body := `{"model":"mantle","messages":[{"role":"user","content":"hello"}]}`
			if tc.maxTokens > 0 {
				body = fmt.Sprintf(`{"model":"mantle","messages":[{"role":"user","content":"hello"}],"max_tokens":%d}`, tc.maxTokens)
			}
			rec := doJSON(t, e, http.MethodPost, "/v1/chat/completions", body)
			if rec.Code != http.StatusOK {
				t.Fatalf("expected 200, got %d body=%s", rec.Code, rec.Body.String())
			}

			var resp ChatCompletionResponse
			if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
				t.Fatalf("decode response: %v", err)
			}
			if len(resp.Choices) != 1 || resp.Choices[0].Message == nil {
				t.Fatalf("expected one choice with a message, got %s", rec.Body.String())
			}
			if got := resp.Choices[0].Message.Content; got != tc.wantContent {
				t.Fatalf("content = %v, want %q", got, tc.wantContent)
			}
			if resp.Choices[0].FinishReason == nil {
				t.Fatalf("missing finish_reason in %s", rec.Body.String())
			}
			if got := *resp.Choices[0].FinishReason; got != tc.wantFinish {
				t.Fatalf("finish_reason = %q, want %q", got, tc.wantFinish)
			}
		})
	}
}

// TestChatCompletionsStreamingReasoningOnly covers the streaming arm of the
// same rule: a reasoning-only completion sends its reasoning text once as
// content instead of an empty assistant message, and reports the truncation.
func TestChatCompletionsStreamingReasoningOnly(t *testing.T) {
	t.Parallel()

	engine := testEngine{
		chunks: []inference.StreamChunk{
			{Type: inference.StreamChunkReasoningDelta, Delta: "the user wants a greeting"},
		},
		result: &inference.Result{
			ReasoningText: "the user wants a greeting",
			Stats:         inference.Stats{TokensGenerated: 16},
		},
	}
	service := NewInferenceService(testProvider{engine: engine})
	e := echo.New()
	NewServer(NewResponseStore(), service).Register(e)

	body := `{"model":"mantle","messages":[{"role":"user","content":"hello"}],"stream":true,"max_tokens":16}`
	rec := doJSON(t, e, http.MethodPost, "/v1/chat/completions", body)
	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d body=%s", rec.Code, rec.Body.String())
	}

	respBody := rec.Body.String()
	if !strings.Contains(respBody, "the user wants a greeting") {
		t.Fatalf("expected reasoning text as content in stream, got %s", respBody)
	}
	if !strings.Contains(respBody, `"finish_reason":"length"`) {
		t.Fatalf("expected finish_reason length in stream, got %s", respBody)
	}
}

func TestChatCompletionsWithTemperature(t *testing.T) {
	t.Parallel()

	e := newTestEcho()
	body := `{"model":"mantle","messages":[{"role":"user","content":"hello"}],"temperature":0.5,"max_tokens":50}`
	rec := doJSON(t, e, http.MethodPost, "/v1/chat/completions", body)
	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d body=%s", rec.Code, rec.Body.String())
	}
}

func TestChatCompletionsDefaultModel(t *testing.T) {
	t.Parallel()

	e := newTestEcho()
	body := `{"messages":[{"role":"user","content":"hello"}]}`
	rec := doJSON(t, e, http.MethodPost, "/v1/chat/completions", body)
	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d body=%s", rec.Code, rec.Body.String())
	}

	var resp ChatCompletionResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if resp.Model != "mantle" {
		t.Fatalf("expected default model 'mantle', got %q", resp.Model)
	}
}

func TestListModels(t *testing.T) {
	t.Parallel()

	e := newTestEcho()
	rec := doJSON(t, e, http.MethodGet, "/v1/models", "")
	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d body=%s", rec.Code, rec.Body.String())
	}

	var resp map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if resp["object"] != "list" {
		t.Fatalf("expected object 'list', got %q", resp["object"])
	}
	data, ok := resp["data"].([]any)
	if !ok || len(data) == 0 {
		t.Fatal("expected non-empty data array")
	}
}

func TestChatMessageContentTypes(t *testing.T) {
	t.Parallel()

	// Test string content
	msgs := []ChatMessage{
		{Role: "user", Content: "hello"},
	}
	result, err := chatMessagesToTokenizerMessages(msgs)
	if err != nil {
		t.Fatalf("string content: %v", err)
	}
	if len(result) != 1 {
		t.Fatalf("expected 1 message, got %d", len(result))
	}
	if result[0].Content != "hello" {
		t.Fatalf("expected 'hello', got %v", result[0].Content)
	}

	// Test nil content
	msgs = []ChatMessage{
		{Role: "assistant", Content: nil},
	}
	result, err = chatMessagesToTokenizerMessages(msgs)
	if err != nil {
		t.Fatalf("nil content: %v", err)
	}
	if result[0].Content != "" {
		t.Fatalf("expected empty string for nil content, got %v", result[0].Content)
	}

	// Test multi-part content
	msgs = []ChatMessage{
		{
			Role: "user",
			Content: []any{
				map[string]any{"type": "text", "text": "hello"},
				map[string]any{"type": "text", "text": "world"},
			},
		},
	}
	result, err = chatMessagesToTokenizerMessages(msgs)
	if err != nil {
		t.Fatalf("multi-part content: %v", err)
	}
	contentArr, ok := result[0].Content.([]any)
	if !ok {
		t.Fatalf("expected []any content for multi-part, got %T", result[0].Content)
	}
	if len(contentArr) != 2 {
		t.Fatalf("expected 2 content parts, got %d", len(contentArr))
	}
}

// TestExtractMessageAudio pins the audio content-block extraction: "audio",
// "audio_url", and OpenAI-style "input_audio" blocks are collected in order
// (data: URLs and base64 payloads decode to the raw WAV bytes), other block
// types are ignored, and non-wav formats are rejected.
func TestExtractMessageAudio(t *testing.T) {
	t.Parallel()

	wav := []byte("RIFF....WAVE")
	b64 := base64.StdEncoding.EncodeToString(wav)
	msgs := []tokenizer.Message{
		{Role: "user", Content: []any{
			map[string]any{"type": "text", "text": "listen"},
			map[string]any{"type": "input_audio", "input_audio": map[string]any{"data": b64, "format": "wav"}},
			map[string]any{"type": "audio_url", "audio_url": "data:audio/wav;base64," + b64},
			map[string]any{"type": "audio", "audio": "data:audio/wav;base64," + b64},
			map[string]any{"type": "image", "image_url": "data:image/png;base64,AAAA"},
		}},
	}
	got, err := extractMessageAudio(msgs)
	if err != nil {
		t.Fatalf("extractMessageAudio: %v", err)
	}
	if len(got) != 3 {
		t.Fatalf("got %d audio blocks, want 3", len(got))
	}
	for i, b := range got {
		if string(b) != string(wav) {
			t.Fatalf("block %d = %q, want %q", i, b, wav)
		}
	}

	bad := []tokenizer.Message{{Role: "user", Content: []any{
		map[string]any{"type": "input_audio", "input_audio": map[string]any{"data": b64, "format": "mp3"}},
	}}}
	if _, err := extractMessageAudio(bad); err == nil {
		t.Fatalf("expected error for mp3 format")
	}
}
