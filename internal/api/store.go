package api

import (
	"crypto/rand"
	"encoding/hex"
	"sync"
	"time"
)

type responseRecord struct {
	Response   ResponsesResponse
	InputItems []ResponseItem
	Visible    bool
}

type ResponseStore struct {
	mu        sync.Mutex
	responses map[string]*responseRecord
}

func NewResponseStore() *ResponseStore {
	return &ResponseStore{
		responses: make(map[string]*responseRecord),
	}
}

func (s *ResponseStore) Create(req *ResponsesRequest, inputItems []ResponseItem, now time.Time) ResponsesResponse {
	id := newResponseID()
	resp := ResponsesResponse{
		ID:                   id,
		Object:               "response",
		CreatedAt:            now.Unix(),
		Status:               responseStatus(req),
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
	}

	if resp.Status == "completed" {
		completedAt := now.Unix()
		resp.CompletedAt = &completedAt
	}
	resp.Output = []ResponseItem{}

	visible := true
	if req.Store != nil && !*req.Store {
		visible = false
	}

	s.mu.Lock()
	s.responses[id] = &responseRecord{
		Response:   resp,
		InputItems: inputItems,
		Visible:    visible,
	}
	s.mu.Unlock()

	return resp
}

func (s *ResponseStore) Get(id string) (*responseRecord, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	rec, ok := s.responses[id]
	return rec, ok
}

func (s *ResponseStore) Delete(id string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.responses[id]; !ok {
		return false
	}
	delete(s.responses, id)
	return true
}

func (s *ResponseStore) Cancel(id string, now time.Time) (*ResponsesResponse, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	rec, ok := s.responses[id]
	if !ok {
		return nil, false
	}
	resp := rec.Response
	if resp.Background == nil || !*resp.Background {
		return &resp, true
	}
	status := resp.Status
	if status == "in_progress" || status == "queued" {
		resp.Status = "cancelled"
		completedAt := now.Unix()
		resp.CompletedAt = &completedAt
		rec.Response = resp
	}
	return &rec.Response, true
}

func newResponseID() string {
	b := make([]byte, 12)
	_, _ = rand.Read(b)
	return "resp_" + hex.EncodeToString(b)
}

func responseStatus(req *ResponsesRequest) string {
	if req.Background != nil && *req.Background {
		return "in_progress"
	}
	return "completed"
}
