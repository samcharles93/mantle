package api

import "encoding/json"

type ResponsesRequest struct {
	Background           *bool              `json:"background,omitempty"`
	Conversation         any                `json:"conversation,omitempty"`
	Include              []string           `json:"include,omitempty"`
	Input                any                `json:"input,omitempty"`
	Instructions         any                `json:"instructions,omitempty"`
	MaxOutputTokens      *int               `json:"max_output_tokens,omitempty"`
	MaxToolCalls         *int               `json:"max_tool_calls,omitempty"`
	Metadata             map[string]string  `json:"metadata,omitempty"`
	Model                string             `json:"model,omitempty"`
	ParallelToolCalls    *bool              `json:"parallel_tool_calls,omitempty"`
	PreviousResponseID   string             `json:"previous_response_id,omitempty"`
	Prompt               map[string]any     `json:"prompt,omitempty"`
	PromptCacheKey       string             `json:"prompt_cache_key,omitempty"`
	PromptCacheRetention string             `json:"prompt_cache_retention,omitempty"`
	Reasoning            *ResponseReasoning `json:"reasoning,omitempty"`
	ReasoningFormat      string             `json:"reasoning_format,omitempty"`
	ReasoningBudget      *int               `json:"reasoning_budget,omitempty"`
	SafetyIdentifier     string             `json:"safety_identifier,omitempty"`
	ServiceTier          string             `json:"service_tier,omitempty"`
	Store                *bool              `json:"store,omitempty"`
	Stream               *bool              `json:"stream,omitempty"`
	StreamOptions        map[string]any     `json:"stream_options,omitempty"`
	Temperature          *float64           `json:"temperature,omitempty"`
	Text                 *ResponseText      `json:"text,omitempty"`
	ToolChoice           any                `json:"tool_choice,omitempty"`
	Tools                []ResponseTool     `json:"tools,omitempty"`
	TopLogprobs          *int               `json:"top_logprobs,omitempty"`
	TopP                 *float64           `json:"top_p,omitempty"`
	Truncation           string             `json:"truncation,omitempty"`
}

type ResponsesResponse struct {
	ID                   string              `json:"id"`
	Object               string              `json:"object"`
	CreatedAt            int64               `json:"created_at,omitempty"`
	Status               string              `json:"status,omitempty"`
	Background           *bool               `json:"background,omitempty"`
	CompletedAt          *int64              `json:"completed_at,omitempty"`
	Conversation         *Conversation       `json:"conversation,omitempty"`
	Error                *ResponseError      `json:"error,omitempty"`
	IncompleteDetails    *ResponseIncomplete `json:"incomplete_details,omitempty"`
	Instructions         any                 `json:"instructions,omitempty"`
	MaxOutputTokens      *int                `json:"max_output_tokens,omitempty"`
	MaxToolCalls         *int                `json:"max_tool_calls,omitempty"`
	Metadata             map[string]string   `json:"metadata,omitempty"`
	Model                string              `json:"model,omitempty"`
	Output               []ResponseItem      `json:"output,omitempty"`
	OutputText           string              `json:"output_text,omitempty"`
	ReasoningText        string              `json:"reasoning_text,omitempty"`
	ParallelToolCalls    *bool               `json:"parallel_tool_calls,omitempty"`
	PreviousResponseID   string              `json:"previous_response_id,omitempty"`
	Prompt               map[string]any      `json:"prompt,omitempty"`
	PromptCacheKey       string              `json:"prompt_cache_key,omitempty"`
	PromptCacheRetention string              `json:"prompt_cache_retention,omitempty"`
	Reasoning            *ResponseReasoning  `json:"reasoning,omitempty"`
	SafetyIdentifier     string              `json:"safety_identifier,omitempty"`
	ServiceTier          string              `json:"service_tier,omitempty"`
	Store                *bool               `json:"store,omitempty"`
	Temperature          *float64            `json:"temperature,omitempty"`
	Text                 *ResponseText       `json:"text,omitempty"`
	ToolChoice           any                 `json:"tool_choice,omitempty"`
	Tools                []ResponseTool      `json:"tools,omitempty"`
	TopP                 *float64            `json:"top_p,omitempty"`
	Truncation           string              `json:"truncation,omitempty"`
	Usage                *ResponseUsage      `json:"usage,omitempty"`
}

type Conversation struct {
	ID        string            `json:"id"`
	Object    string            `json:"object"`
	CreatedAt int64             `json:"created_at,omitempty"`
	Metadata  map[string]string `json:"metadata,omitempty"`
}

type ResponseItem struct {
	ID        string            `json:"id,omitempty"`
	Type      string            `json:"type,omitempty"`
	Role      string            `json:"role,omitempty"`
	Status    string            `json:"status,omitempty"`
	Content   []ResponseContent `json:"content,omitempty"`
	CallID    string            `json:"call_id,omitempty"`
	Name      string            `json:"name,omitempty"`
	Arguments json.RawMessage   `json:"arguments,omitempty"`
	Output    json.RawMessage   `json:"output,omitempty"`
	Error     *ResponseError    `json:"error,omitempty"`
}

type ResponseContent struct {
	Type        string               `json:"type,omitempty"`
	Text        string               `json:"text,omitempty"`
	Annotations []ResponseAnnotation `json:"annotations,omitempty"`
	ImageURL    string               `json:"image_url,omitempty"`
	FileID      string               `json:"file_id,omitempty"`
	FileURL     string               `json:"file_url,omitempty"`
	FileData    string               `json:"file_data,omitempty"`
	Filename    string               `json:"filename,omitempty"`
	Detail      string               `json:"detail,omitempty"`
}

type ResponseAnnotation struct {
	Type       string `json:"type,omitempty"`
	FileID     string `json:"file_id,omitempty"`
	Filename   string `json:"filename,omitempty"`
	Index      int    `json:"index,omitempty"`
	StartIndex int    `json:"start_index,omitempty"`
	EndIndex   int    `json:"end_index,omitempty"`
	Title      string `json:"title,omitempty"`
	URL        string `json:"url,omitempty"`
}

type ResponseTool map[string]any

type ResponseText struct {
	Format *ResponseTextFormat `json:"format,omitempty"`
}

type ResponseTextFormat struct {
	Type string `json:"type,omitempty"`
}

type ResponseReasoning struct {
	Effort           any    `json:"effort,omitempty"`
	Summary          any    `json:"summary,omitempty"`
	EncryptedContent string `json:"encrypted_content,omitempty"`
}

type ResponseUsage struct {
	InputTokens         int                   `json:"input_tokens,omitempty"`
	InputTokensDetails  *ResponseTokenDetails `json:"input_tokens_details,omitempty"`
	OutputTokens        int                   `json:"output_tokens,omitempty"`
	OutputTokensDetails *ResponseTokenDetails `json:"output_tokens_details,omitempty"`
	TotalTokens         int                   `json:"total_tokens,omitempty"`
}

type ResponseTokenDetails struct {
	CachedTokens    int `json:"cached_tokens,omitempty"`
	ReasoningTokens int `json:"reasoning_tokens,omitempty"`
}

type ResponseError struct {
	Message string `json:"message,omitempty"`
	Type    string `json:"type,omitempty"`
	Code    string `json:"code,omitempty"`
	Param   string `json:"param,omitempty"`
}

type ResponseIncomplete struct {
	Reason string `json:"reason,omitempty"`
}

type ResponseInputItemList struct {
	Object  string         `json:"object,omitempty"`
	Data    []ResponseItem `json:"data,omitempty"`
	FirstID string         `json:"first_id,omitempty"`
	LastID  string         `json:"last_id,omitempty"`
	HasMore bool           `json:"has_more,omitempty"`
}

type ResponseInputTokensResponse struct {
	Object      string `json:"object"`
	InputTokens int    `json:"input_tokens"`
}

type ResponseCompaction struct {
	ID        string         `json:"id"`
	Object    string         `json:"object"`
	CreatedAt int64          `json:"created_at"`
	Output    []ResponseItem `json:"output"`
	Usage     *ResponseUsage `json:"usage,omitempty"`
}
