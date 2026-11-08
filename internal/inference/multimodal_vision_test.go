package inference

import (
	"bytes"
	"context"
	"image"
	"image/png"
	"testing"

	"github.com/samcharles93/mantle/internal/backend/core"
)

// fakeVisionModel implements core.Runtime + core.VisionEncoderProvider and
// records the (tokens, embeds) pair its PrefillTokensEmbeds receives, so the
// test can assert the placeholder expansion and embed splice end to end
// without any MCF/env var.
type fakeVisionModel struct {
	cfg *core.ModelConfig
	enc core.VisionEncoder

	gotTokens []int
	gotEmbeds [][]float32
}

func (f *fakeVisionModel) ForwardToken(id int) ([]float32, error) { return []float32{1, 0}, nil }
func (f *fakeVisionModel) Reset()                                 {}
func (f *fakeVisionModel) ModelConfig() *core.ModelConfig         { return f.cfg }
func (f *fakeVisionModel) UpdateRoPE()                            {}
func (f *fakeVisionModel) VisionEncoder() core.VisionEncoder      { return f.enc }
func (f *fakeVisionModel) PrefillTokensEmbeds(tokens []int, embeds [][]float32) ([]float32, error) {
	f.gotTokens = append([]int(nil), tokens...)
	f.gotEmbeds = embeds
	return []float32{1, 0}, nil
}

// fakeVisionEncoder returns fixed marker rows: one row per image token, row
// k of image gi filled with 100*(gi+1)+k.
type fakeVisionEncoder struct {
	cfg *core.VisionConfig
}

func (f *fakeVisionEncoder) Config() *core.VisionConfig { return f.cfg }
func (f *fakeVisionEncoder) Encode(pixelValues []float32, grids [][3]int) ([][][]float32, error) {
	out := make([][][]float32, len(grids))
	for gi, g := range grids {
		n := f.cfg.TokensPerImage(g)
		rows := make([][]float32, n)
		for k := range rows {
			row := make([]float32, f.cfg.TextHiddenSize)
			for i := range row {
				row[i] = float32(100*(gi+1) + k)
			}
			rows[k] = row
		}
		out[gi] = rows
	}
	return out, nil
}

func testVisionConfig() *core.VisionConfig {
	return &core.VisionConfig{
		HiddenSize:         32,
		InChannels:         3,
		PatchSize:          16,
		MergeSize:          2,
		TemporalPatch:      1,
		TextHiddenSize:     8,
		ImageTokenID:       125030,
		VideoTokenID:       125029,
		VisionStartTokenID: 125009,
		VisionEndTokenID:   125010,
		ImageMean:          []float64{0.5, 0.5, 0.5},
		ImageStd:           []float64{0.5, 0.5, 0.5},
		RescaleFactor:      1.0 / 255.0,
		MinPixels:          256,
		MaxPixels:          1 << 24,
	}
}

// pngBytes renders a small solid-ish RGBA image as PNG.
func pngBytes(t *testing.T, w, h int) []byte {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for i := range img.Pix {
		img.Pix[i] = uint8(i * 7)
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatalf("png encode: %v", err)
	}
	return buf.Bytes()
}

// TestPrepareMultimodalVisionExpansion is a NON-env-gated regression test for
// the vision placeholder path: it drives prepareMultimodal with a fake vision
// encoder and a real (in-test PNG) image request, asserting no panic and
// correct expansion (one image token per projected row, embeds only at image
// token positions), then drives Generator.RunWithContext with the expanded
// ids + Embeds and asserts the splice reaches PrefillTokensEmbeds intact.
// It fails (panics) if the vision Encode result is shadowed and never reaches
// the expansion loop.
func TestPrepareMultimodalVisionExpansion(t *testing.T) {
	t.Parallel()

	vc := testVisionConfig()
	model := &fakeVisionModel{
		cfg: &core.ModelConfig{Arch: "lfm3_vl", Vision: vc},
		enc: &fakeVisionEncoder{cfg: vc},
	}
	e := &EngineImpl{model: model, tokenizer: okTokenizer{}}

	// 64x64 image, patch 16, merge 2 -> grid [1,4,4] -> 16/4 = 4 image tokens.
	img := pngBytes(t, 64, 64)
	const (
		vs = 125009
		ve = 125010
		im = 125030
	)
	ids := []int{11, vs, im, ve, 12}
	req := &Request{Images: [][]byte{img}}

	newIDs, embeds, err := prepareMultimodal(e, req, ids)
	if err != nil {
		t.Fatalf("prepareMultimodal: %v", err)
	}
	const wantTok = 4 // TokensPerImage([1,4,4])
	wantLen := 1 + 1 + wantTok + 1 + 1
	if len(newIDs) != wantLen || len(embeds) != wantLen {
		t.Fatalf("expanded len %d/%d, want %d", len(newIDs), len(embeds), wantLen)
	}
	imgCount := 0
	for i, id := range newIDs {
		if id == im {
			if embeds[i] == nil {
				t.Fatalf("image token at %d has nil embed (shadowed feats bug)", i)
			}
			want := float32(100 + imgCount)
			if embeds[i][0] != want {
				t.Fatalf("image embed %d marker %g, want %g", imgCount, embeds[i][0], want)
			}
			imgCount++
		} else if embeds[i] != nil {
			t.Fatalf("non-image token %d at %d carries an embed", id, i)
		}
	}
	if imgCount != wantTok {
		t.Fatalf("expanded %d image tokens, want %d", imgCount, wantTok)
	}

	// Drive the Generator embed-splice path: Embeds must reach
	// PrefillTokensEmbeds parallel to the expanded token ids.
	gen := &Generator{
		Model:         model,
		Sampler:       newGreedySampler(),
		Tokenizer:     okTokenizer{},
		StopTokens:    []int{124900},
		ContextTokens: make([]int, 0, len(newIDs)+1),
		Embeds:        embeds,
	}
	if _, _, err := gen.RunWithContext(context.Background(), newIDs, 1, nil); err != nil {
		t.Fatalf("RunWithContext: %v", err)
	}
	if len(model.gotTokens) != len(newIDs) {
		t.Fatalf("prefill got %d tokens, want %d", len(model.gotTokens), len(newIDs))
	}
	for i := range newIDs {
		if model.gotTokens[i] != newIDs[i] {
			t.Fatalf("prefill token %d: got %d, want %d", i, model.gotTokens[i], newIDs[i])
		}
	}
	if len(model.gotEmbeds) != len(newIDs) {
		t.Fatalf("prefill embeds len %d, want %d", len(model.gotEmbeds), len(newIDs))
	}
	for i, id := range newIDs {
		if id == im {
			if model.gotEmbeds[i] == nil {
				t.Fatalf("spliced embed nil at image token %d", i)
			}
		} else if model.gotEmbeds[i] != nil {
			t.Fatalf("spliced embed non-nil at non-image token %d", i)
		}
	}
}
