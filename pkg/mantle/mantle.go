// Package mantle provides the public API for loading and running inference
// with Mantle models. See the doc.go file for package-level documentation.
package mantle

import (
	"context"
	"fmt"
	"time"

	"github.com/samcharles93/mantle/internal/backend/core"
	"github.com/samcharles93/mantle/internal/backend/simd"
	"github.com/samcharles93/mantle/internal/hostcaps"
	"github.com/samcharles93/mantle/internal/inference"
	"github.com/samcharles93/mantle/internal/tokenizer"
	"github.com/samcharles93/mantle/pkg/mcf"
)

// Engine wraps a loaded model runtime and provides text generation.
// Create one via NewEngine and call Close when done to release resources.
type Engine struct {
	engine  inference.Engine
	runtime core.Runtime
	tok     tokenizer.Tokenizer
	tokCfg  tokenizer.TokenizerConfig
	cfgJSON []byte
	def     inference.GenDefaults
	arch    string
}

// Request describes a text generation request.
// Zero values use sensible defaults (temperature=0.8, top-k=40, etc.).
type Request struct {
	// Prompt is a simple text prompt. If set alongside Messages,
	// Messages takes precedence.
	Prompt string

	// System is an optional system prompt prepended to the conversation.
	System string

	// Messages provides structured chat messages. If set, Prompt
	// and System are ignored.
	Messages []Message

	// Steps is the maximum number of tokens to generate.
	// -1 means generate until a stop token is produced (unlimited).
	// 0 means prompt-only (no generation).
	Steps int

	// Temperature controls randomness (0 = greedy, higher = more random).
	Temperature float64

	// TopK limits sampling to the K most likely tokens (0 = disabled).
	TopK int

	// TopP nucleus sampling threshold (0-1, 1 = disabled).
	TopP float64

	// MinP minimum probability threshold (0 = disabled).
	MinP float64

	// RepeatPenalty penalizes repeated tokens (1.0 = disabled).
	RepeatPenalty float64

	// RepeatLastN sets the window size for repetition penalty.
	RepeatLastN int

	// Seed for the random number generator. -1 uses current time.
	Seed int64
}

// Message represents a single turn in a chat conversation.
type Message struct {
	Role    string // "system", "user", "assistant"
	Content string
}

// Result contains the generation output and statistics.
type Result = inference.Result

// Stats contains performance statistics for a generation run.
type Stats = inference.Stats

// StreamChunk is a single chunk of streaming output.
type StreamChunk = inference.StreamChunk

// StreamChunkType constants describe the kind of streaming output.
const (
	StreamChunkTextDelta      = inference.StreamChunkTextDelta
	StreamChunkReasoningDelta = inference.StreamChunkReasoningDelta
	StreamChunkPromptEcho     = inference.StreamChunkPromptEcho
)

// StreamFunc is a callback that receives streaming output chunks.
type StreamFunc = inference.StreamFunc

// EngineCapabilities describes the loaded model's capabilities.
type EngineCapabilities = inference.EngineCapabilities

// NewEngine loads a model from the given MCF file and returns a ready-to-use Engine.
//
// The model path must point to a valid .mcf file. Optional EngineOption values
// control backend selection, caching, and other runtime behaviour.
func NewEngine(ctx context.Context, modelPath string, opts ...EngineOption) (*Engine, error) {
	cfg := engineOptions{
		backend:    "auto",
		maxContext: 0,
		cacheTypeK: "f16",
		cacheTypeV: "f16",
		gpuLayers:  -1,
		tileM:      32,
		tileN:      32,
		tileK:      16,
		noSWA:      false,
	}

	for _, opt := range opts {
		if err := opt(&cfg); err != nil {
			return nil, fmt.Errorf("engine option: %w", err)
		}
	}

	// Ensure a logger is in context.
	// Detect host capabilities.
	caps := hostcaps.Detect()
	ctx = hostcaps.WithContext(ctx, caps)

	// Build a Loader similar to what the CLI uses.
	tiling := simd.TilingConfig{
		TileM: cfg.tileM,
		TileN: cfg.tileN,
		TileK: cfg.tileK,
	}

	loader := inference.Loader{
		TokenizerJSONPath:   cfg.tokenizerJSON,
		TokenizerConfigPath: cfg.tokenizerConfig,
		ChatTemplatePath:    cfg.chatTemplate,
		HFConfigPath:        cfg.hfConfigFile,
		Backend:             cfg.backend,
		DisableSWA:          cfg.noSWA,
		TilingConfig:        tiling,
	}
	loader.LoadOptions.CacheTypeK = cfg.cacheTypeK
	loader.LoadOptions.CacheTypeV = cfg.cacheTypeV
	loader.LoadOptions.GpuLayers = cfg.gpuLayers
	loader.LoadOptions.HostCaps = caps

	loadResult, err := loader.Load(ctx, modelPath, cfg.maxContext)
	if err != nil {
		return nil, fmt.Errorf("load model: %w", err)
	}

	return &Engine{
		engine:  loadResult.Engine,
		runtime: loadResult.Runtime,
		tok:     loadResult.Tokenizer,
		tokCfg:  loadResult.TokenizerConfig,
		cfgJSON: loadResult.HFConfigJSON,
		def:     loadResult.GenerationDefaults,
		arch:    loadResult.Arch,
	}, nil
}

// toMessages converts our public Message type to the internal tokenizer.Message type.
func toMessages(msgs []Message) []tokenizer.Message {
	if msgs == nil {
		return nil
	}
	out := make([]tokenizer.Message, len(msgs))
	for i, m := range msgs {
		out[i] = tokenizer.Message{
			Role:    m.Role,
			Content: m.Content,
		}
	}
	return out
}

// Generate runs inference and returns the result.
//
// If stream is non-nil, it receives streaming chunks as tokens are generated.
// Streaming is progressive: the callback may receive zero, one, or many calls
// per token depending on the output (reasoning chunks, text deltas, etc.).
//
// The returned Result always contains the full generated text and statistics.
func (e *Engine) Generate(ctx context.Context, req Request, stream StreamFunc) (*Result, error) {
	var msgs []tokenizer.Message

	if len(req.Messages) > 0 {
		msgs = toMessages(req.Messages)
	} else {
		msgs = make([]tokenizer.Message, 0, 2)
		if req.System != "" {
			msgs = append(msgs, tokenizer.Message{Role: "system", Content: req.System})
		}
		if req.Prompt != "" {
			msgs = append(msgs, tokenizer.Message{Role: "user", Content: req.Prompt})
		}
	}

	steps := req.Steps
	if steps == 0 {
		steps = -1 // default to unlimited
	}

	seed := req.Seed
	if seed == 0 {
		seed = time.Now().UnixNano()
	}

	opts := inference.RequestOptions{
		Messages:        msgs,
		Steps:           &steps,
		Seed:            &seed,
		Temperature:     &req.Temperature,
		TopK:            &req.TopK,
		TopP:            &req.TopP,
		MinP:            &req.MinP,
		RepeatPenalty:   &req.RepeatPenalty,
		RepeatLastN:     &req.RepeatLastN,
		NoTemplate:      new(false),
		EchoPrompt:      new(false),
		ReasoningFormat: new("auto"),
		ReasoningBudget: new(-1),
	}

	infReq := inference.ResolveRequest(opts, e.def)

	result, err := e.engine.Generate(ctx, &infReq, stream)
	if err != nil {
		return nil, err
	}
	return result, nil
}

// Capabilities returns information about the loaded model.
func (e *Engine) Capabilities() EngineCapabilities {
	return e.engine.Capabilities()
}

// ResetContext clears the model's internal KV cache and conversation state.
// Call this between unrelated conversations to avoid memory waste.
func (e *Engine) ResetContext() {
	e.engine.ResetContext()
}

// Close releases all resources held by the engine (MCF file, model runtime, etc.).
func (e *Engine) Close() error {
	return e.engine.Close()
}

// SectionBytes returns the raw bytes of a named MCF section, if present.
// This is useful for inspecting model metadata, HF config, or tokenizer data.
// Returns nil if the section does not exist.
func (e *Engine) SectionBytes(section mcf.SectionType) []byte {
	if e.runtime == nil {
		return nil
	}
	if acc, ok := e.runtime.(interface{ SectionData(mcf.SectionType) []byte }); ok {
		return acc.SectionData(section)
	}
	return nil
}
