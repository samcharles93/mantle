package api

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/labstack/echo/v5"
)

type Server struct {
	store   *ResponseStore
	service *InferenceService
	clock   func() time.Time
}

func NewServer(store *ResponseStore, service *InferenceService) *Server {
	if store == nil {
		store = NewResponseStore()
	}
	return &Server{
		store:   store,
		service: service,
		clock:   time.Now,
	}
}

func (s *Server) Register(e *echo.Echo) {
	// Responses API
	e.POST("/v1/responses", s.handleCreateResponse)
	e.GET("/v1/responses/:id", s.handleGetResponse)
	e.DELETE("/v1/responses/:id", s.handleDeleteResponse)
	e.POST("/v1/responses/:id/cancel", s.handleCancelResponse)
	e.GET("/v1/responses/:id/input_items", s.handleInputItems)
	e.POST("/v1/responses/compact", s.handleCompactResponse)
	e.POST("/v1/responses/input_tokens", s.handleInputTokens)

	// Chat Completions API (OpenAI-compatible)
	s.RegisterChatCompletions(e)
}

func (s *Server) handleCreateResponse(c *echo.Context) error {
	if s.service == nil {
		return writeError(c, http.StatusInternalServerError, "server_error", "inference service not configured", "", "")
	}
	req, err := decodeJSON[ResponsesRequest](c.Request().Body)
	if err != nil {
		return writeBadRequest(c, err.Error())
	}
	if req.Stream != nil && *req.Stream && req.Background != nil && *req.Background {
		return writeBadRequest(c, "streaming background responses is not supported")
	}
	if req.PreviousResponseID != "" && req.Conversation != nil {
		return writeBadRequest(c, "previous_response_id and conversation are mutually exclusive")
	}

	var (
		saveInputItems     []ResponseItem
		overrideInputItems bool
	)
	if req.PreviousResponseID != "" {
		historyItems, err := s.resolvePreviousInputItems(req.PreviousResponseID)
		if err != nil {
			return writeBadRequest(c, err.Error())
		}
		currentItems, err := normalizeInputItems(req.Input)
		if err != nil {
			return writeBadRequest(c, fmt.Sprintf("input: %v", err))
		}
		merged := make([]ResponseItem, 0, len(historyItems)+len(currentItems))
		merged = append(merged, historyItems...)
		merged = append(merged, currentItems...)
		req.Input = merged

		saveInputItems = currentItems
		overrideInputItems = true
	}

	var writer *SSEStreamWriter
	var streamWriter StreamWriter
	if req.Stream != nil && *req.Stream {
		w, err := NewSSEStreamWriter(c)
		if err != nil {
			return writeBadRequest(c, err.Error())
		}
		writer = w
		streamWriter = w
	}

	resp, inputItems, err := s.service.CreateResponse(c.Request().Context(), &req, streamWriter)
	if err != nil {
		if writer != nil && writer.Started() {
			return nil
		}
		if errors.Is(err, ErrInvalidRequest) {
			return writeBadRequest(c, err.Error())
		}
		return writeError(c, http.StatusInternalServerError, "server_error", err.Error(), "", "")
	}

	if resp != nil {
		if overrideInputItems {
			s.store.Save(*resp, saveInputItems)
		} else {
			s.store.Save(*resp, inputItems)
		}
	}

	if writer != nil {
		return nil
	}
	return c.JSON(http.StatusOK, resp)
}

func (s *Server) resolvePreviousInputItems(responseID string) ([]ResponseItem, error) {
	if strings.TrimSpace(responseID) == "" {
		return nil, fmt.Errorf("previous_response_id is required")
	}
	visited := make(map[string]struct{})
	chain := make([]*responseRecord, 0, 8)
	id := responseID
	for id != "" {
		if _, ok := visited[id]; ok {
			return nil, fmt.Errorf("previous_response_id chain contains a cycle")
		}
		visited[id] = struct{}{}

		rec, ok := s.store.Get(id)
		if !ok || !rec.Visible {
			return nil, fmt.Errorf("previous_response_id %q not found", id)
		}
		chain = append(chain, rec)
		id = rec.Response.PreviousResponseID
	}

	merged := make([]ResponseItem, 0, len(chain)*2)
	for i := len(chain) - 1; i >= 0; i-- {
		rec := chain[i]
		merged = append(merged, rec.InputItems...)
		merged = append(merged, rec.Response.Output...)
	}
	return merged, nil
}

func (s *Server) handleGetResponse(c *echo.Context) error {
	id := c.Param("id")
	if id == "" {
		return writeNotFound(c, "response not found")
	}
	rec, ok := s.store.Get(id)
	if !ok || !rec.Visible {
		return writeNotFound(c, "response not found")
	}
	if streamParam(c) {
		return s.writeSSE(c, rec.Response, rec.Response.Background != nil && *rec.Response.Background)
	}
	return c.JSON(http.StatusOK, rec.Response)
}

func (s *Server) handleDeleteResponse(c *echo.Context) error {
	id := c.Param("id")
	if id == "" {
		return writeNotFound(c, "response not found")
	}
	rec, ok := s.store.Get(id)
	if !ok || !rec.Visible {
		return writeNotFound(c, "response not found")
	}
	if !s.store.Delete(id) {
		return writeNotFound(c, "response not found")
	}
	return c.JSON(http.StatusOK, DeleteResponseResp{
		ID:      id,
		Object:  "response",
		Deleted: true,
	})
}

func (s *Server) handleCancelResponse(c *echo.Context) error {
	id := c.Param("id")
	if id == "" {
		return writeNotFound(c, "response not found")
	}
	resp, ok := s.store.Cancel(id, s.clock())
	if !ok || resp == nil {
		return writeNotFound(c, "response not found")
	}
	if resp.Background == nil || !*resp.Background {
		return writeBadRequest(c, "only background responses can be cancelled")
	}
	return c.JSON(http.StatusOK, resp)
}

func (s *Server) handleInputItems(c *echo.Context) error {
	id := c.Param("id")
	if id == "" {
		return writeNotFound(c, "response not found")
	}
	rec, ok := s.store.Get(id)
	if !ok || !rec.Visible {
		return writeNotFound(c, "response not found")
	}
	out := ResponseInputItemList{
		Object:  "list",
		Data:    rec.InputItems,
		FirstID: "",
		LastID:  "",
		HasMore: false,
	}
	if len(rec.InputItems) > 0 {
		out.FirstID = rec.InputItems[0].ID
		out.LastID = rec.InputItems[len(rec.InputItems)-1].ID
	}
	return c.JSON(http.StatusOK, out)
}

func (s *Server) handleCompactResponse(c *echo.Context) error {
	req, err := decodeJSON[CompactResponseReq](c.Request().Body)
	if err != nil {
		return writeBadRequest(c, err.Error())
	}
	if req.Input == nil {
		return writeBadRequest(c, "input is required")
	}

	var (
		inputItems []ResponseItem
	)
	if req.Input.String != nil {
		inputItems, err = normalizeInputItems(*req.Input.String)
	} else {
		inputItems, err = normalizeInputItems(req.Input.Items)
	}
	if err != nil {
		return writeBadRequest(c, fmt.Sprintf("input: %v", err))
	}

	output := compactToSingleMessage(inputItems)
	now := s.clock()
	inputTokens := approximateTokenCount(inputItems)
	outputTokens := approximateTokenCount(output)
	resp := ResponseCompaction{
		ID:        newResponseID(),
		Object:    "response.compaction",
		CreatedAt: now.Unix(),
		Output:    output,
	}
	if inputTokens > 0 || outputTokens > 0 {
		resp.Usage = &ResponseUsage{
			InputTokens:  inputTokens,
			OutputTokens: outputTokens,
			TotalTokens:  inputTokens + outputTokens,
		}
	}
	return c.JSON(http.StatusOK, resp)
}

func (s *Server) handleInputTokens(c *echo.Context) error {
	req, err := decodeJSON[CompactResponseReq](c.Request().Body)
	if err != nil {
		return writeBadRequest(c, err.Error())
	}
	if req.Input == nil {
		return writeBadRequest(c, "input is required")
	}
	var (
		inputItems []ResponseItem
	)
	if req.Input.String != nil {
		inputItems, err = normalizeInputItems(*req.Input.String)
	} else {
		inputItems, err = normalizeInputItems(req.Input.Items)
	}
	if err != nil {
		return writeBadRequest(c, fmt.Sprintf("input: %v", err))
	}
	tokenCount := approximateTokenCount(inputItems)
	resp := ResponseInputTokensResponse{
		Object:      "response.input_tokens",
		InputTokens: tokenCount,
	}
	return c.JSON(http.StatusOK, resp)
}

func streamParam(c *echo.Context) bool {
	q := c.QueryParam("stream")
	return q == "1" || strings.EqualFold(q, "true")
}

func (s *Server) writeSSE(c *echo.Context, resp ResponsesResponse, background bool) error {
	res := c.Response()
	res.Header().Set(echo.HeaderContentType, "text/event-stream")
	res.Header().Set("Cache-Control", "no-cache")
	res.Header().Set("Connection", "keep-alive")

	flusher, ok := res.(http.Flusher)
	if !ok {
		return writeBadRequest(c, "streaming unsupported")
	}

	seq := 1
	startingAfter := parseStartingAfter(c.QueryParam("starting_after"))

	createdResp := resp
	createdResp.Status = "in_progress"
	createdResp.CompletedAt = nil

	if err := sendStreamEvent(res, streamEvent{
		Type:           "response.created",
		Response:       &createdResp,
		SequenceNumber: seq,
	}, startingAfter); err != nil {
		return err
	}
	flusher.Flush()
	seq++

	if err := sendStreamEvent(res, streamEvent{
		Type:           "response.in_progress",
		Response:       &createdResp,
		SequenceNumber: seq,
	}, startingAfter); err != nil {
		return err
	}
	flusher.Flush()
	seq++

	if background {
		return nil
	}

	switch resp.Status {
	case "failed":
		failedResp := resp
		failedResp.Status = "failed"
		if err := sendStreamEvent(res, streamEvent{
			Type:           "response.failed",
			Response:       &failedResp,
			SequenceNumber: seq,
		}, startingAfter); err != nil {
			return err
		}
		flusher.Flush()
	case "incomplete":
		incompleteResp := resp
		incompleteResp.Status = "incomplete"
		if err := sendStreamEvent(res, streamEvent{
			Type:           "response.incomplete",
			Response:       &incompleteResp,
			SequenceNumber: seq,
		}, startingAfter); err != nil {
			return err
		}
		flusher.Flush()
	default:
		finalResp := resp
		finalResp.Status = "completed"
		now := s.clock().Unix()
		finalResp.CompletedAt = &now

		if err := sendStreamEvent(res, streamEvent{
			Type:           "response.completed",
			Response:       &finalResp,
			SequenceNumber: seq,
		}, startingAfter); err != nil {
			return err
		}
		flusher.Flush()
	}
	return nil
}

type streamEvent struct {
	Type           string             `json:"type"`
	Response       *ResponsesResponse `json:"response,omitempty"`
	SequenceNumber int                `json:"sequence_number"`
}

func sendStreamEvent(w io.Writer, event streamEvent, startingAfter int) error {
	if startingAfter >= event.SequenceNumber {
		return nil
	}
	b, err := json.Marshal(event)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(w, "data: %s\n\n", string(b))
	return err
}

func decodeJSON[T any](r io.Reader) (T, error) {
	var out T
	dec := json.NewDecoder(r)
	if err := dec.Decode(&out); err != nil {
		return out, err
	}
	return out, nil
}
