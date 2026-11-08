package simd

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"testing"

	"github.com/samcharles93/mantle/internal/backend/core"
	"github.com/samcharles93/mantle/internal/mcfstore"
	"github.com/samcharles93/mantle/pkg/mcf"
)


// lfm3vlReferenceCase mirrors one case of testdata/lfm3vl_reference.json,
// captured from HF transformers (LiquidAI/LFM3-VL-2B, float32) via the
// Lfm3VlProcessor: prompt token ids (with image placeholders already expanded
// to one image token per projected vision token), last-position top-10 logits,
// and an 8-token greedy continuation. Image cases also carry the reference
// pixel payload (f32) and patch grid.
type lfm3vlReferenceCase struct {
	Name        string    `json:"name"`
	Tokens      []int     `json:"tokens"`
	TopIDs      []int     `json:"top_ids"`
	TopLogits   []float64 `json:"top_logits"`
	Greedy      []int     `json:"greedy"`
	PixelsFile  string    `json:"pixels_file,omitempty"`
	PixelsShape []int     `json:"pixels_shape,omitempty"`
	GridTHW     [][]int   `json:"grid_thw,omitempty"`
	ImageFile   string    `json:"image_file,omitempty"`
	FeatsFile   string    `json:"feats_file,omitempty"`
	// StepTop holds the reference top-10 (ids, logits) at every greedy step,
	// captured with a manual greedy loop in HF transformers.
	StepTop []struct {
		IDs    []int     `json:"ids"`
		Logits []float64 `json:"logits"`
	} `json:"step_top,omitempty"`
}

type lfm3vlReference struct {
	Cases []lfm3vlReferenceCase `json:"cases"`
}

func loadLFM3VLReference(t *testing.T) *lfm3vlReference {
	t.Helper()
	raw, err := os.ReadFile("testdata/lfm3vl_reference.json")
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var fx lfm3vlReference
	if err := json.Unmarshal(raw, &fx); err != nil {
		t.Fatalf("parse fixture: %v", err)
	}
	return &fx
}

// TestLFM3VLImageProcessing checks the Go image pipeline (decode, smart
// resize, normalize, patchify) against the HF processor's pixel_values.
// HF resizes uint8 through torchvision's fixed-point kernel, so a small
// per-pixel tolerance is allowed; the payload must otherwise match exactly in
// shape and layout.
func TestLFM3VLImageProcessing(t *testing.T) {
	fx := loadLFM3VLReference(t)
	for _, tc := range fx.Cases {
		if tc.PixelsFile == "" {
			continue
		}
		t.Run(tc.Name, func(t *testing.T) {
			imgPath := "testdata/lfm3vl_test_256.png"
			if tc.Name == "image_100x70" {
				imgPath = "testdata/lfm3vl_test_100x70.png"
			}
			raw, err := os.ReadFile(imgPath)
			if err != nil {
				t.Fatalf("read image: %v", err)
			}
			vc := coreVisionConfigForTest(t, tc)
			pix, grids, err := ProcessImages([][]byte{raw}, vc)
			if err != nil {
				t.Fatalf("ProcessImages: %v", err)
			}
			if len(grids) != 1 || grids[0][0] != tc.GridTHW[0][0] || grids[0][1] != tc.GridTHW[0][1] || grids[0][2] != tc.GridTHW[0][2] {
				t.Fatalf("grid = %v, want %v", grids, tc.GridTHW)
			}
			wantN := tc.PixelsShape[0] * tc.PixelsShape[1]
			if len(pix) != wantN {
				t.Fatalf("pixel payload %d, want %d", len(pix), wantN)
			}
			refRaw, err := os.ReadFile("testdata/" + tc.PixelsFile)
			if err != nil {
				t.Fatalf("read pixels: %v", err)
			}
			if len(refRaw) != wantN*4 {
				t.Fatalf("reference pixels size %d, want %d", len(refRaw), wantN*4)
			}
			maxDiff := 0.0
			for i := range wantN {
				ref := math.Float32frombits(binary.LittleEndian.Uint32(refRaw[i*4:]))
				d := math.Abs(float64(pix[i] - ref))
				if d > maxDiff {
					maxDiff = d
				}
			}
			if maxDiff > 1e-6 {
				t.Errorf("pixel max abs diff %g exceeds exact-match tolerance", maxDiff)
			}
		})
	}
}

func coreVisionConfigForTest(t *testing.T, tc lfm3vlReferenceCase) *core.VisionConfig {
	t.Helper()
	// minimal config matching LFM3-VL-2B's processor: patch 16, merge 2,
	// temporal 2, mean/std 0.5, rescale 1/255, min/max pixels from the packed
	// processor_config.json
	return &core.VisionConfig{
		InChannels:    3,
		PatchSize:     16,
		MergeSize:     2,
		TemporalPatch: 2,
		ImageMean:     []float64{0.5, 0.5, 0.5},
		ImageStd:      []float64{0.5, 0.5, 0.5},
		RescaleFactor: 1.0 / 255.0,
		MinPixels:     65536,
		MaxPixels:     16777216,
	}
}

// TestLFM3VLReferenceLogits compares LFM3-VL-2B against the HF reference
// fixture: text-only and image prompts, last-position top-10 logits plus
// 8-token greedy continuations. Image cases splice the vision tower's
// projected embeddings at the image-token positions via PrefillTokensEmbeds.
func TestLFM3VLReferenceLogits(t *testing.T) {
	path := os.Getenv("MANTLE_LFM3VL_MCF")
	if path == "" {
		t.Skip("set MANTLE_LFM3VL_MCF to the LFM3-VL-2B container to run the reference comparison")
	}
	fx := loadLFM3VLReference(t)
	file, err := mcfstore.Open(path)
	if err != nil {
		t.Fatalf("open %s: %v", path, err)
	}
	t.Cleanup(func() { _ = file.Close() })

	const imageTokenID = 125030

	for _, tc := range fx.Cases {
		t.Run(tc.Name, func(t *testing.T) {
			m, err := LoadModelMCF(file, file.SectionData(mcf.SectionHFConfigJSON), len(tc.Tokens)+len(tc.Greedy)+1, LoadModelOptions{})
			if err != nil {
				t.Fatalf("LoadModelMCF: %v", err)
			}
			if m.Vision == nil {
				t.Fatalf("model has no vision tower loaded")
			}
			vc := m.Vision.Config()
			if vc.ImageTokenID != 0 && vc.ImageTokenID != imageTokenID {
				t.Fatalf("image token id %d, want %d", vc.ImageTokenID, imageTokenID)
			}

			tokens := tc.Tokens
			var embeds [][]float32
			if tc.PixelsFile != "" {
				imgPath := "testdata/lfm3vl_test_256.png"
				if tc.Name == "image_100x70" {
					imgPath = "testdata/lfm3vl_test_100x70.png"
				}
				raw, err := os.ReadFile(imgPath)
				if err != nil {
					t.Fatalf("read image: %v", err)
				}
				pix, grids, err := ProcessImages([][]byte{raw}, vc)
				if err != nil {
					t.Fatalf("ProcessImages: %v", err)
				}
				feats, err := m.Vision.Encode(pix, grids)
				if err != nil {
					t.Fatalf("Encode: %v", err)
				}
				// The projected image features must match HF's
				// get_image_features output closely (observed ~2e-6).
				if tc.FeatsFile != "" {
					refRaw, err := os.ReadFile("testdata/" + tc.FeatsFile)
					if err != nil {
						t.Fatalf("read feats: %v", err)
					}
					n := len(refRaw) / 4
					if n != len(feats[0])*len(feats[0][0]) {
						t.Fatalf("feats size mismatch: got %d, ref %d", len(feats[0])*len(feats[0][0]), n)
					}
					for i := range n {
						ref := math.Float32frombits(binary.LittleEndian.Uint32(refRaw[i*4:]))
						g := feats[0][i/len(feats[0][0])][i%len(feats[0][0])]
						if d := math.Abs(float64(g - ref)); d > 1e-4 {
							t.Fatalf("image feature %d: got %g, reference %g (diff %g)", i, g, ref, d)
						}
					}
				}
				// The fixture tokens already contain the expanded image
				// placeholders; splice one feature row per image token.
				embeds = make([][]float32, len(tokens))
				row := 0
				for i, id := range tokens {
					if id == imageTokenID {
						embeds[i] = feats[0][row]
						row++
					}
				}
				if row != len(feats[0]) {
					t.Fatalf("spliced %d image embeddings, vision produced %d", row, len(feats[0]))
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
			// bf16 weights computed in f32 on both sides; observed diffs are
			// ~2e-5, so 2e-3 gives ample headroom while still catching real bugs.
			const tol = 2e-3
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
