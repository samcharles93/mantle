package simd

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"testing"

	"github.com/samcharles93/mantle/internal/mcfstore"
	"github.com/samcharles93/mantle/pkg/mcf"
)

// lfm3omniReferenceCase mirrors one case of testdata/lfm3omni_reference.json,
// captured from HF transformers (LiquidAI/LFM3-Omni-2B, float32 CPU) with the
// downloaded repo's own modeling_lfm3_omni.py: expanded prompt token ids,
// last-position top-10 logits, an 8-token greedy continuation with per-step
// top-10. The audio case also carries the reference log-mel payload (f32,
// [128][frames] bin-major) and projected audio features (f32,
// [numAudioTokens][2048]).
type lfm3omniReferenceCase struct {
	Name           string    `json:"name"`
	Tokens         []int     `json:"tokens"`
	TopIDs         []int     `json:"top_ids"`
	TopLogits      []float64 `json:"top_logits"`
	Greedy         []int     `json:"greedy"`
	NumAudioTokens int       `json:"num_audio_tokens,omitempty"`
	MelShape       []int     `json:"mel_shape,omitempty"`
	FeatsShape     []int     `json:"feats_shape,omitempty"`
	StepTop        []struct {
		IDs    []int     `json:"ids"`
		Logits []float64 `json:"logits"`
	} `json:"step_top,omitempty"`
}

type lfm3omniReference struct {
	Model     string                  `json:"model"`
	DType     string                  `json:"dtype"`
	WavFile   string                  `json:"wav_file"`
	MelFile   string                  `json:"mel_file"`
	FeatsFile string                  `json:"feats_file"`
	Cases     []lfm3omniReferenceCase `json:"cases"`
}

func loadLFM3OmniReference(t *testing.T) *lfm3omniReference {
	t.Helper()
	raw, err := os.ReadFile("testdata/lfm3omni_reference.json")
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var fx lfm3omniReference
	if err := json.Unmarshal(raw, &fx); err != nil {
		t.Fatalf("parse fixture: %v", err)
	}
	return &fx
}

func readF32File(t *testing.T, path string) []float32 {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	if len(raw)%4 != 0 {
		t.Fatalf("%s: %d bytes not a whole number of f32", path, len(raw))
	}
	out := make([]float32, len(raw)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	return out
}

// TestLFM3OmniReferenceLogits compares LFM3-Omni-2B against the HF reference
// fixture: a text-only prompt and an audio prompt (WAV -> log-mel -> audio
// tower -> projector -> embedding splice -> prefill), checking the reference
// mel payload, the projected audio features, last-position top-10 logits, and
// an 8-token greedy continuation. The audio fixture clip is 3.0 s -> 299 mel
// frames -> 38 audio tokens, deliberately > 32 so the encoder's
// sliding_window=32 causal attention window actually truncates (tokens past
// position 31 attend to a bounded window) in this test.
func TestLFM3OmniReferenceLogits(t *testing.T) {
	path := os.Getenv("MANTLE_LFM3OMNI_MCF")
	if path == "" {
		t.Skip("set MANTLE_LFM3OMNI_MCF to the LFM3-Omni-2B container to run the reference comparison")
	}
	fx := loadLFM3OmniReference(t)
	file, err := mcfstore.Open(path)
	if err != nil {
		t.Fatalf("open %s: %v", path, err)
	}
	t.Cleanup(func() { _ = file.Close() })

	// The packed audio codec (section 0x0107) must be present and decodable.
	atCfg, atST, err := file.AudioTokenizerPayload()
	if err != nil {
		t.Fatalf("AudioTokenizerPayload: %v", err)
	}
	if len(atCfg) == 0 || len(atST) == 0 {
		t.Fatalf("section 0x0107 missing from %s", path)
	}

	const audioTokenID = 125031

	for _, tc := range fx.Cases {
		t.Run(tc.Name, func(t *testing.T) {
			m, err := LoadModelMCF(file, file.SectionData(mcf.SectionHFConfigJSON), len(tc.Tokens)+len(tc.Greedy)+1, LoadModelOptions{})
			if err != nil {
				t.Fatalf("LoadModelMCF: %v", err)
			}
			if tc.Name == "audio_3s" && m.Audio == nil {
				t.Fatalf("model has no audio tower loaded")
			}

			tokens := tc.Tokens
			var embeds [][]float32
			if tc.NumAudioTokens > 0 {
				ac := m.Audio.Config()
				if ac.AudioTokenID != 0 && ac.AudioTokenID != audioTokenID {
					t.Fatalf("audio token id %d, want %d", ac.AudioTokenID, audioTokenID)
				}
				wav, err := os.ReadFile("testdata/" + fx.WavFile)
				if err != nil {
					t.Fatalf("read wav: %v", err)
				}
				mels, err := ProcessAudioClips([][]byte{wav}, ac)
				if err != nil {
					t.Fatalf("ProcessAudioClips: %v", err)
				}
				// Reference log-mel payload: [128][frames] bin-major f32.
				if tc.MelShape != nil {
					ref := readF32File(t, "testdata/"+fx.MelFile)
					bins, frames := tc.MelShape[0], tc.MelShape[1]
					if len(ref) != bins*frames {
						t.Fatalf("mel payload %d floats, want %d", len(ref), bins*frames)
					}
					if len(mels[0]) != bins || len(mels[0][0]) != frames {
						t.Fatalf("mel shape [%d][%d], want [%d][%d]", len(mels[0]), len(mels[0][0]), bins, frames)
					}
					// Go computes the STFT in f64, torch in f32; low-energy
					// bins differ by a few e-3 in the log domain.
					const melTol = 5e-2 // log10 amplifies f32-vs-f64 STFT noise at near-floor bins
					maxDiff := 0.0
					for b := range bins {
						for f := range frames {
							d := math.Abs(mels[0][b][f] - float64(ref[b*frames+f]))
							if d > maxDiff {
								maxDiff = d
							}
							if d > melTol {
								t.Fatalf("mel[%d][%d]: got %g, reference %g (diff %g)", b, f, mels[0][b][f], ref[b*frames+f], d)
							}
						}
					}
					t.Logf("max |mel diff| = %g", maxDiff)
				}
				feats, err := m.Audio.Encode(mels)
				if err != nil {
					t.Fatalf("audio Encode: %v", err)
				}
				if len(feats) != 1 || len(feats[0]) != tc.NumAudioTokens {
					t.Fatalf("audio encode produced %d clips x %d tokens, want 1 x %d", len(feats), len(feats[0]), tc.NumAudioTokens)
				}
				// Reference projected audio features.
				if tc.FeatsShape != nil {
					ref := readF32File(t, "testdata/"+fx.FeatsFile)
					n := tc.FeatsShape[0] * tc.FeatsShape[1]
					if len(ref) != n {
						t.Fatalf("feats payload %d floats, want %d", len(ref), n)
					}
					// Hard assert. Observed max diff 3.7e-4 (f32 accumulation over
					// the 17-layer tower + downsample convs with a different
					// summation order than torch, fed by mel inputs that differ up
					// to 2.5e-2 at near-floor bins). 5e-3 is ~13x above that noise
					// floor yet ~100x below the diffs structural bugs produced during
					// bring-up (projector arg swap ~0.4-0.9, shadowed feats ~0.07+),
					// so it still catches real regressions. VL uses 1e-4 on vision
					// feats, but the audio tower runs 17 layers + conv stages, hence
					// the looser bound.
					const featTol = 5e-3
					maxDiff := 0.0
					for i := range n {
						g := feats[0][i/tc.FeatsShape[1]][i%tc.FeatsShape[1]]
						if d := math.Abs(float64(g - ref[i])); d > featTol {
							t.Fatalf("audio feature %d: got %g, reference %g (diff %g)", i, g, ref[i], d)
						} else if d > maxDiff {
							maxDiff = d
						}
					}
					t.Logf("max |audio feature diff| = %g", maxDiff)
				}
				// The fixture tokens already contain the expanded audio
				// placeholders; splice one feature row per audio token.
				embeds = make([][]float32, len(tokens))
				row := 0
				for i, id := range tokens {
					if id == audioTokenID {
						embeds[i] = feats[0][row]
						row++
					}
				}
				if row != len(feats[0]) {
					t.Fatalf("spliced %d audio embeddings, tower produced %d", row, len(feats[0]))
				}
			}

			var logits []float32
			if embeds != nil {
				logits, err = m.PrefillTokensEmbeds(tokens, embeds)
			} else {
				logits, err = m.PrefillTokens(tokens)
			}
			if err != nil {
				t.Fatalf("prefill: %v", err)
			}
			// bf16 weights computed in f32 on both sides. Text-only prompts see
			// diffs ~2e-5; audio prompts add the 17-layer tower's f32 drift to
			// the spliced embeddings (observed logit diffs up to ~1.2e-2 with
			// 38 audio tokens), so allow 2e-2 there — still >1000x below the
			// ~15-magnitude logit errors structural bugs produced. Greedy tokens
			// and argmax below are exact assertions.
			tol := 2e-3
			if embeds != nil {
				tol = 2e-2
			}
			for k, id := range tc.TopIDs {
				if d := math.Abs(float64(logits[id]) - tc.TopLogits[k]); d > tol {
					t.Errorf("token %d: logit %g, reference %g (diff %g)", id, logits[id], tc.TopLogits[k], d)
				}
			}
			if got := argmaxHost(logits); got != tc.TopIDs[0] {
				t.Fatalf("argmax %d, reference %d", got, tc.TopIDs[0])
			}
			tok := tc.TopIDs[0]
			for i, want := range tc.Greedy {
				if tok != want {
					t.Fatalf("greedy token %d: got %d, reference %d", i, tok, want)
				}
				if i+1 == len(tc.Greedy) {
					break
				}
				l, err := m.ForwardToken(tok)
				if err != nil {
					t.Fatalf("ForwardToken: %v", err)
				}
				if i+1 < len(tc.StepTop) {
					for k, id := range tc.StepTop[i+1].IDs {
						if d := math.Abs(float64(l[id]) - tc.StepTop[i+1].Logits[k]); d > tol {
							t.Errorf("step %d token %d: logit %g, reference %g (diff %g)", i+1, id, l[id], tc.StepTop[i+1].Logits[k], d)
						}
					}
				}
				tok = argmaxHost(l)
			}
		})
	}
}
