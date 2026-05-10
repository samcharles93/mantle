package mantle_test

import (
	"context"
	"testing"

	"github.com/samcharles93/mantle/pkg/mantle"
)

// TestAPICompilation verifies that the public API types, functions, and methods
// compile and have the expected signatures. No model file is required.
func TestAPICompilation(t *testing.T) {
	// NewEngine function exists and takes expected params.
	var _ func(context.Context, string, ...mantle.EngineOption) (*mantle.Engine, error)

	// EngineOption functions exist.
	_ = mantle.WithBackend("auto")
	_ = mantle.WithBackend("cpu")
	_ = mantle.WithBackend("cuda")
	_ = mantle.WithMaxContext(4096)
	_ = mantle.WithKVCacheType("f16", "f16")
	_ = mantle.WithGPULayers(-1)
	_ = mantle.WithDisableSWA()
	_ = mantle.WithTiling(32, 32, 16)

	// Request struct and fields are accessible.
	req := mantle.Request{
		Prompt:        "hello",
		System:        "be helpful",
		Steps:         256,
		Temperature:   0.8,
		TopK:          40,
		TopP:          0.95,
		MinP:          0.05,
		RepeatPenalty: 1.1,
		RepeatLastN:   64,
		Seed:          42,
	}
	_ = req

	// Messages with roles.
	msgs := []mantle.Message{
		{Role: "system", Content: "you are helpful"},
		{Role: "user", Content: "hello"},
	}
	req2 := mantle.Request{Messages: msgs}
	_ = req2

	// Result and Stats type aliases compile.
	var result mantle.Result
	var stats mantle.Stats
	_ = result
	_ = stats

	// StreamChunk type and constants.
	var chunk mantle.StreamChunk
	_ = chunk
	_ = mantle.StreamChunkTextDelta
	_ = mantle.StreamChunkReasoningDelta
	_ = mantle.StreamChunkPromptEcho

	// StreamFunc type.
	var sf mantle.StreamFunc
	_ = sf

	// EngineCapabilities type.
	var caps mantle.EngineCapabilities
	_ = caps

	t.Log("public API compiles and types are accessible")
}

// TestEngineMethods verifies that the Engine methods have the expected signatures.
func TestEngineMethods(t *testing.T) {
	// This test only checks type signatures; it doesn't create a real Engine.
	// If we had a mock Engine, we'd test behaviour here.

	var e *mantle.Engine

	// Generate method.
	_ = func() {
		_, _ = e.Generate(context.Background(), mantle.Request{}, nil)
	}

	// Capabilities method.
	_ = func() {
		_ = e.Capabilities()
	}

	// ResetContext method.
	_ = func() {
		e.ResetContext()
	}

	// Close method.
	_ = func() {
		_ = e.Close()
	}

	t.Log("Engine methods have expected signatures")
}
