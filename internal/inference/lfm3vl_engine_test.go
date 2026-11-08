package inference

import (
	"context"
	"encoding/json"
	"os"
	"testing"

	"github.com/samcharles93/mantle/internal/tokenizer"
)

// TestLFM3VLEngineImageGeneration runs the full engine path (chat template
// rendering, image placeholder expansion, vision tower, embedding splice,
// prefill, greedy decode) for LFM3-VL-2B with an image and compares the
// generated tokens against the HF reference fixture (testdata in the simd
// package, captured from HF transformers with a manual greedy loop).
func TestLFM3VLEngineImageGeneration(t *testing.T) {
	path := os.Getenv("MANTLE_LFM3VL_MCF")
	if path == "" {
		t.Skip("set MANTLE_LFM3VL_MCF to the LFM3-VL-2B container to run the reference comparison")
	}

	raw, err := os.ReadFile("../backend/simd/testdata/lfm3vl_reference.json")
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var fx struct {
		Cases []struct {
			Name   string `json:"name"`
			Tokens []int  `json:"tokens"`
			Greedy []int  `json:"greedy"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(raw, &fx); err != nil {
		t.Fatalf("parse fixture: %v", err)
	}
	var imageCase, textCase int
	for i, c := range fx.Cases {
		if c.Name == "image_256" {
			imageCase = i
		}
		if c.Name == "text_only" {
			textCase = i
		}
	}
	if imageCase == textCase {
		t.Fatalf("fixture missing cases")
	}

	img, err := os.ReadFile("../backend/simd/testdata/lfm3vl_test_256.png")
	if err != nil {
		t.Fatalf("read test image: %v", err)
	}

	loader := Loader{Backend: "cpu"}
	res, err := loader.Load(context.Background(), path, 4096)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer func() { _ = res.Engine.Close() }()

	if !res.Engine.Capabilities().SupportsVision {
		t.Fatalf("expected SupportsVision=true for lfm3_vl")
	}

	run := func(t *testing.T, opts RequestOptions, wantPrompt, wantGreedy []int) {
		t.Helper()
		steps := len(wantGreedy)
		zero := 0.0
		opts.Steps = &steps
		opts.Temperature = &zero
		req := ResolveRequest(opts, res.GenerationDefaults)
		result, err := res.Engine.Generate(context.Background(), &req, nil)
		if err != nil {
			t.Fatalf("Generate: %v", err)
		}
		eng, ok := res.Engine.(*EngineImpl)
		if !ok {
			t.Fatalf("engine is %T, want *EngineImpl", res.Engine)
		}
		toks := eng.generator.ContextTokens
		if len(toks) < len(wantPrompt)+len(wantGreedy) {
			t.Fatalf("context too short: %d tokens", len(toks))
		}
		prompt := toks[:len(wantPrompt)]
		for i, id := range wantPrompt {
			if prompt[i] != id {
				t.Fatalf("prompt token %d: got %d, want %d", i, prompt[i], id)
			}
		}
		gen := toks[len(wantPrompt):]
		for i, id := range wantGreedy {
			if i >= len(gen) {
				break
			}
			if gen[i] != id {
				t.Fatalf("generated token %d: got %d, reference %d (text %q)", i, gen[i], id, result.Text)
			}
		}
	}

	textPrompt := fx.Cases[textCase]
	t.Run("text_only", func(t *testing.T) {
		res.Engine.ResetContext()
		run(t, RequestOptions{
			Messages: []tokenizer.Message{{
				Role:    "user",
				Content: "What is 2+2? Answer with just the number.",
			}},
		}, textPrompt.Tokens, textPrompt.Greedy)
	})

	imgCase := fx.Cases[imageCase]
	t.Run("image_256", func(t *testing.T) {
		res.Engine.ResetContext()
		run(t, RequestOptions{
			Messages: []tokenizer.Message{{
				Role: "user",
				Content: []any{
					map[string]any{"type": "image"},
					map[string]any{"type": "text", "text": "What color is the top-left block? Answer with one word."},
				},
			}},
			Images: [][]byte{img},
		}, imgCase.Tokens, imgCase.Greedy)
	})
}
