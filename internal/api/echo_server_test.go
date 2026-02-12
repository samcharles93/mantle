package api

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/labstack/echo/v5"
	"github.com/samcharles93/mantle/internal/inference"
)

type testProvider struct {
	engine   inference.Engine
	defaults inference.GenDefaults
	err      error
}

func (p testProvider) WithEngine(ctx context.Context, modelID string, fn func(engine inference.Engine, defaults inference.GenDefaults) error) error {
	if p.err != nil {
		return p.err
	}
	return fn(p.engine, p.defaults)
}

type testEngine struct {
	text string
	err  error
}

func (e testEngine) Generate(ctx context.Context, req *inference.Request, stream inference.StreamFunc) (*inference.Result, error) {
	if e.err != nil {
		return nil, e.err
	}
	if stream != nil && e.text != "" {
		stream(e.text)
	}
	return &inference.Result{Text: e.text}, nil
}

func (e testEngine) ResetContext() {}
func (e testEngine) Close() error  { return nil }

func newTestEcho() *echo.Echo {
	provider := testProvider{
		engine: testEngine{text: "ok"},
	}
	service := NewInferenceService(provider)
	server := NewServer(NewResponseStore(), service)
	e := echo.New()
	server.Register(e)
	return e
}

func doJSON(t *testing.T, e *echo.Echo, method, path, body string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(method, path, strings.NewReader(body))
	req.Header.Set(echo.HeaderContentType, echo.MIMEApplicationJSON)
	rec := httptest.NewRecorder()
	e.ServeHTTP(rec, req)
	return rec
}

func TestCreateGetDeleteResponseLifecycle(t *testing.T) {
	t.Parallel()

	e := newTestEcho()
	createRec := doJSON(t, e, http.MethodPost, "/v1/responses", `{"input":"hello"}`)
	if createRec.Code != http.StatusOK {
		t.Fatalf("create status: got %d body=%s", createRec.Code, createRec.Body.String())
	}

	var created ResponsesResponse
	if err := json.Unmarshal(createRec.Body.Bytes(), &created); err != nil {
		t.Fatalf("decode create response: %v", err)
	}
	if created.ID == "" {
		t.Fatalf("expected response id")
	}
	if created.Status != "completed" {
		t.Fatalf("expected completed status, got %q", created.Status)
	}
	if created.OutputText != "ok" {
		t.Fatalf("unexpected output text: %q", created.OutputText)
	}

	getRec := doJSON(t, e, http.MethodGet, "/v1/responses/"+created.ID, "")
	if getRec.Code != http.StatusOK {
		t.Fatalf("get status: got %d body=%s", getRec.Code, getRec.Body.String())
	}

	delRec := doJSON(t, e, http.MethodDelete, "/v1/responses/"+created.ID, "")
	if delRec.Code != http.StatusOK {
		t.Fatalf("delete status: got %d body=%s", delRec.Code, delRec.Body.String())
	}
	if !strings.Contains(delRec.Body.String(), `"deleted":true`) {
		t.Fatalf("delete response missing deleted=true: %s", delRec.Body.String())
	}

	getDeletedRec := doJSON(t, e, http.MethodGet, "/v1/responses/"+created.ID, "")
	if getDeletedRec.Code != http.StatusNotFound {
		t.Fatalf("expected 404 after delete, got %d body=%s", getDeletedRec.Code, getDeletedRec.Body.String())
	}
}

func TestCreateValidationErrors(t *testing.T) {
	t.Parallel()

	e := newTestEcho()

	streamBackground := `{"input":"x","stream":true,"background":true}`
	rec := doJSON(t, e, http.MethodPost, "/v1/responses", streamBackground)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d body=%s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "streaming background responses is not supported") {
		t.Fatalf("unexpected error body: %s", rec.Body.String())
	}

	prevAndConv := `{"input":"x","previous_response_id":"r1","conversation":{"id":"c1"}}`
	rec = doJSON(t, e, http.MethodPost, "/v1/responses", prevAndConv)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d body=%s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "mutually exclusive") {
		t.Fatalf("unexpected error body: %s", rec.Body.String())
	}
}

func TestCancelNonBackgroundResponse(t *testing.T) {
	t.Parallel()

	e := newTestEcho()
	createRec := doJSON(t, e, http.MethodPost, "/v1/responses", `{"input":"hello"}`)
	if createRec.Code != http.StatusOK {
		t.Fatalf("create status: got %d body=%s", createRec.Code, createRec.Body.String())
	}
	var created ResponsesResponse
	if err := json.Unmarshal(createRec.Body.Bytes(), &created); err != nil {
		t.Fatalf("decode create response: %v", err)
	}

	cancelRec := doJSON(t, e, http.MethodPost, "/v1/responses/"+created.ID+"/cancel", `{}`)
	if cancelRec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400 for non-background cancel, got %d body=%s", cancelRec.Code, cancelRec.Body.String())
	}
	if !strings.Contains(cancelRec.Body.String(), "only background responses can be cancelled") {
		t.Fatalf("unexpected cancel response: %s", cancelRec.Body.String())
	}
}

func TestCompactAndInputTokensEndpoints(t *testing.T) {
	t.Parallel()

	e := newTestEcho()

	compactRec := doJSON(t, e, http.MethodPost, "/v1/responses/compact", `{"input":"hello world"}`)
	if compactRec.Code != http.StatusOK {
		t.Fatalf("compact status: got %d body=%s", compactRec.Code, compactRec.Body.String())
	}
	var compactResp ResponseCompaction
	if err := json.Unmarshal(compactRec.Body.Bytes(), &compactResp); err != nil {
		t.Fatalf("decode compact response: %v", err)
	}
	if compactResp.Object != "response.compaction" {
		t.Fatalf("unexpected compact object: %q", compactResp.Object)
	}
	if len(compactResp.Output) == 0 {
		t.Fatalf("expected compact output")
	}

	tokRec := doJSON(t, e, http.MethodPost, "/v1/responses/input_tokens", `{"input":"hello world"}`)
	if tokRec.Code != http.StatusOK {
		t.Fatalf("input_tokens status: got %d body=%s", tokRec.Code, tokRec.Body.String())
	}
	var tokResp ResponseInputTokensResponse
	if err := json.NewDecoder(bytes.NewReader(tokRec.Body.Bytes())).Decode(&tokResp); err != nil {
		t.Fatalf("decode input_tokens response: %v", err)
	}
	if tokResp.Object != "response.input_tokens" {
		t.Fatalf("unexpected object: %q", tokResp.Object)
	}
	if tokResp.InputTokens <= 0 {
		t.Fatalf("expected positive token count, got %d", tokResp.InputTokens)
	}
}
