package api

import (
	"sync"
	"time"

	"github.com/google/uuid"
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

func (s *ResponseStore) Save(resp ResponsesResponse, inputItems []ResponseItem) {
	visible := true
	if resp.Store != nil && !*resp.Store {
		visible = false
	}
	s.mu.Lock()
	s.responses[resp.ID] = &responseRecord{
		Response:   resp,
		InputItems: inputItems,
		Visible:    visible,
	}
	s.mu.Unlock()
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
	return "resp_" + uuid.NewString()
}

func newOutputItemID() string {
	return "msg_" + uuid.NewString()
}
