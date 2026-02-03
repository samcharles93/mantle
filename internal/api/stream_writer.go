package api

import (
	"encoding/json"
	"fmt"
	"io"

	"github.com/labstack/echo/v5"
	"github.com/samcharles93/mantle/internal/inference"
)

type SSEStreamWriter struct {
	w             io.Writer
	flusher       func()
	startingAfter int
	seq           int
	itemID        string
	outputIndex   int
	contentIndex  int
	startedItem   bool
	begun         bool
}

func NewSSEStreamWriter(c *echo.Context) (*SSEStreamWriter, error) {
	res := c.Response()
	res.Header().Set(echo.HeaderContentType, "text/event-stream")
	res.Header().Set("Cache-Control", "no-cache")
	res.Header().Set("Connection", "keep-alive")

	flusher, ok := res.(interface{ Flush() })
	if !ok {
		return nil, fmt.Errorf("streaming unsupported")
	}

	startingAfter := parseStartingAfter(c.QueryParam("starting_after"))

	return &SSEStreamWriter{
		w:             res,
		flusher:       flusher.Flush,
		startingAfter: startingAfter,
		seq:           1,
		outputIndex:   0,
		contentIndex:  0,
	}, nil
}

func (s *SSEStreamWriter) Begin(resp ResponsesResponse) error {
	s.begun = true
	resp.Status = "in_progress"
	resp.CompletedAt = nil
	if err := s.send(streamEvent{
		Type:           "response.created",
		Response:       &resp,
		SequenceNumber: s.seq,
	}); err != nil {
		return err
	}
	s.flush()
	s.seq++

	if err := s.send(streamEvent{
		Type:           "response.in_progress",
		Response:       &resp,
		SequenceNumber: s.seq,
	}); err != nil {
		return err
	}
	s.flush()
	s.seq++
	return nil
}

func (s *SSEStreamWriter) Started() bool {
	return s.begun
}

func (s *SSEStreamWriter) EmitToken(delta string) error {
	if !s.startedItem {
		s.itemID = newOutputItemID()
		s.startedItem = true
		added := ResponseItem{
			ID:      s.itemID,
			Type:    "message",
			Role:    "assistant",
			Status:  "in_progress",
			Content: []ResponseContent{},
		}
		if err := s.send(map[string]any{
			"type":            "response.output_item.added",
			"output_index":    s.outputIndex,
			"item":            added,
			"sequence_number": s.seq,
		}); err != nil {
			return err
		}
		s.flush()
		s.seq++

		part := ResponseContent{
			Type: "output_text",
			Text: "",
		}
		if err := s.send(map[string]any{
			"type":            "response.content_part.added",
			"item_id":         s.itemID,
			"output_index":    s.outputIndex,
			"content_index":   s.contentIndex,
			"part":            part,
			"sequence_number": s.seq,
		}); err != nil {
			return err
		}
		s.flush()
		s.seq++
	}

	if err := s.send(map[string]any{
		"type":            "response.output_text.delta",
		"item_id":         s.itemID,
		"output_index":    s.outputIndex,
		"content_index":   s.contentIndex,
		"delta":           delta,
		"sequence_number": s.seq,
	}); err != nil {
		return err
	}
	s.flush()
	s.seq++
	return nil
}

func (s *SSEStreamWriter) Complete(resp ResponsesResponse, result *inference.Result) error {
	if s.startedItem {
		if err := s.send(map[string]any{
			"type":            "response.output_text.done",
			"item_id":         s.itemID,
			"output_index":    s.outputIndex,
			"content_index":   s.contentIndex,
			"text":            result.Text,
			"sequence_number": s.seq,
		}); err != nil {
			return err
		}
		s.flush()
		s.seq++

		if err := s.send(map[string]any{
			"type":          "response.content_part.done",
			"item_id":       s.itemID,
			"output_index":  s.outputIndex,
			"content_index": s.contentIndex,
			"part": ResponseContent{
				Type: "output_text",
				Text: result.Text,
			},
			"sequence_number": s.seq,
		}); err != nil {
			return err
		}
		s.flush()
		s.seq++

		if err := s.send(map[string]any{
			"type":         "response.output_item.done",
			"output_index": s.outputIndex,
			"item": ResponseItem{
				ID:     s.itemID,
				Type:   "message",
				Role:   "assistant",
				Status: "completed",
				Content: []ResponseContent{{
					Type: "output_text",
					Text: result.Text,
				}},
			},
			"sequence_number": s.seq,
		}); err != nil {
			return err
		}
		s.flush()
		s.seq++
	}

	resp.Status = "completed"
	if err := s.send(streamEvent{
		Type:           "response.completed",
		Response:       &resp,
		SequenceNumber: s.seq,
	}); err != nil {
		return err
	}
	s.flush()
	s.seq++
	return nil
}

func (s *SSEStreamWriter) Failed(resp ResponsesResponse, err error) error {
	resp.Status = "failed"
	if resp.Error == nil {
		resp.Error = &ResponseError{
			Message: err.Error(),
			Type:    "server_error",
		}
	}
	if err := s.send(streamEvent{
		Type:           "response.failed",
		Response:       &resp,
		SequenceNumber: s.seq,
	}); err != nil {
		return err
	}
	s.flush()
	s.seq++
	return nil
}

func (s *SSEStreamWriter) Incomplete(resp ResponsesResponse, err error) error {
	resp.Status = "incomplete"
	if resp.IncompleteDetails == nil {
		resp.IncompleteDetails = &ResponseIncomplete{Reason: "cancelled"}
	}
	if err := s.send(streamEvent{
		Type:           "response.incomplete",
		Response:       &resp,
		SequenceNumber: s.seq,
	}); err != nil {
		return err
	}
	s.flush()
	s.seq++
	return nil
}

func (s *SSEStreamWriter) send(payload any) error {
	if s.startingAfter >= s.seq {
		return nil
	}
	b, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(s.w, "data: %s\n\n", string(b))
	return err
}

func (s *SSEStreamWriter) flush() {
	if s.flusher != nil {
		s.flusher()
	}
}

func parseStartingAfter(v string) int {
	if v == "" {
		return 0
	}
	n := 0
	for _, r := range v {
		if r < '0' || r > '9' {
			return 0
		}
		n = n*10 + int(r-'0')
	}
	return n
}
