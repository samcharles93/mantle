package inference

import (
	"context"

	"github.com/samcharles93/mantle/internal/tokenizer"
)

type StreamFunc func(token string)

type Engine interface {
	Generate(ctx context.Context, req *Request, stream StreamFunc) (*Result, error)
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
	RawText       string
	Stats         Stats
}
