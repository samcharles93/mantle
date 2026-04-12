package inference

import (
	"context"

	"github.com/samcharles93/mantle/internal/tokenizer"
)

type StreamChunkType string

const (
	StreamChunkTextDelta      StreamChunkType = "text_delta"
	StreamChunkReasoningDelta StreamChunkType = "reasoning_delta"
	StreamChunkPromptEcho     StreamChunkType = "prompt_echo"
)

type StreamChunk struct {
	Type  StreamChunkType
	Delta string
}

type StreamFunc func(chunk StreamChunk)

type EngineCapabilities struct {
	Arch           string
	MaxContext     int
	VocabSize      int
	SupportsVision bool
	SupportsTools  bool
	Quantization   string
}

type Engine interface {
	Generate(ctx context.Context, req *Request, stream StreamFunc) (*Result, error)
	Capabilities() EngineCapabilities
	ResetContext()
	Close() error
}

type Request struct {
	Messages []tokenizer.Message
	Tools    []any

	Steps int
	Seed  int64

	Temperature   float64
	TopK          int
	TopP          float64
	MinP          float64
	RepeatPenalty float64
	RepeatLastN   int

	NoTemplate bool
	EchoPrompt bool

	ReasoningFormat string
	ReasoningBudget int
}

type Result struct {
	Text          string
	ReasoningText string
	Stats         Stats
}
