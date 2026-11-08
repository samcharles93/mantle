package inference

import (
	"context"
	"encoding/json"
	"os"
	"testing"

	"github.com/samcharles93/mantle/internal/tokenizer"
)

// TestLFM3OmniEngineAudioGeneration runs the full engine path (chat template
// rendering with the audio marker, <|audio|> placeholder expansion, WAV ->
// log-mel -> audio tower -> projector, embedding splice, prefill, greedy
// decode) for LFM3-Omni-2B with an audio clip and compares the prompt and
// generated tokens against the HF reference fixture (captured from HF
// transformers with the repo's own modeling_lfm3_omni.py on CPU float32).
func TestLFM3OmniEngineAudioGeneration(t *testing.T) {
	path := os.Getenv("MANTLE_LFM3OMNI_MCF")
	if path == "" {
		t.Skip("set MANTLE_LFM3OMNI_MCF to the LFM3-Omni-2B container to run the reference comparison")
	}

	raw, err := os.ReadFile("../backend/simd/testdata/lfm3omni_reference.json")
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var fx struct {
		WavFile string `json:"wav_file"`
		Cases   []struct {
			Name   string `json:"name"`
			Tokens []int  `json:"tokens"`
			Greedy []int  `json:"greedy"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(raw, &fx); err != nil {
		t.Fatalf("parse fixture: %v", err)
	}
	var audioCase, textCase int
	found := 0
	for i, c := range fx.Cases {
		switch c.Name {
		case "audio_3s":
			audioCase, found = i, found|1
		case "text_only":
			textCase, found = i, found|2
		}
	}
	if found != 3 {
		t.Fatalf("fixture missing cases")
	}

	wav, err := os.ReadFile("../backend/simd/testdata/" + fx.WavFile)
	if err != nil {
		t.Fatalf("read test wav: %v", err)
	}

	loader := Loader{Backend: "cpu"}
	res, err := loader.Load(context.Background(), path, 4096)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	defer func() { _ = res.Engine.Close() }()

	if !res.Engine.Capabilities().SupportsAudio {
		t.Fatalf("expected SupportsAudio=true for lfm3_omni")
	}

	run := func(t *testing.T, opts RequestOptions, wantPrompt, wantGreedy []int) {
		t.Helper()
		steps := len(wantGreedy)
		zero := 0.0
		one := 1.0
		opts.Steps = &steps
		opts.Temperature = &zero
		opts.RepeatPenalty = &one // the fixture is pure argmax (HF default)
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
		// Generation stops at the first stop token without appending it, so the
		// context may be shorter than prompt+len(greedy) when the reference
		// continuation ends in <|im_end|>.
		if len(toks) < len(wantPrompt) {
			t.Fatalf("context too short: %d tokens, prompt alone is %d", len(toks), len(wantPrompt))
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

	audCase := fx.Cases[audioCase]
	t.Run("audio_3s", func(t *testing.T) {
		res.Engine.ResetContext()
		run(t, RequestOptions{
			Messages: []tokenizer.Message{{
				Role: "user",
				Content: []any{
					map[string]any{"type": "audio"},
					map[string]any{"type": "text", "text": "What do you hear in this audio? Answer in one short sentence."},
				},
			}},
			Audio: [][]byte{wav},
		}, audCase.Tokens, audCase.Greedy)
	})
}
