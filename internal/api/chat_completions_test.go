package api

import (
	"encoding/json"
	"net/http"
	"strings"
	"testing"
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
	if result[0].Content != "hello\nworld" {
		t.Fatalf("expected 'hello\\nworld', got %v", result[0].Content)
	}
}
