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
	SupportsAudio  bool
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

	// Images holds raw encoded image bytes (PNG/JPEG), one per image content
	// block in the messages, in order. Only valid for models with a vision
	// tower (Capabilities.SupportsVision); the engine expands each image
	// placeholder token into the projected vision embeddings.
	Images [][]byte

	// Videos holds one video as an ordered list of encoded frame images
	// (PNG/JPEG), one per video content block in the messages. Frames are
	// treated as sampled at the processor's fps; the engine expands each
	// video placeholder into per-segment timestamp + vision tokens.
	Videos [][][]byte

	// Audio holds raw WAV bytes (16 kHz mono/stereo PCM16 or float32), one per
	// audio content block in the messages, in order. Only valid for models
	// with an audio tower (Capabilities.SupportsAudio); the engine expands
	// each <|audio|> placeholder token into ceil(melFrames/stride) tokens,
	// one per projected audio embedding (processing_lfm3_omni.py semantics).
	Audio [][]byte

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
