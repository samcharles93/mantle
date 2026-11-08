package inference

import (
	"encoding/binary"
	"math"
	"slices"
	"strings"
	"testing"

	"github.com/samcharles93/mantle/internal/audio"
	"github.com/samcharles93/mantle/internal/backend/core"
)

// fakeAudioModel implements core.Runtime + core.AudioEncoderProvider with a
// stub encoder that returns one marker row per audio token.
type fakeAudioModel struct {
	cfg *core.ModelConfig
	enc core.AudioEncoder
}

func (f *fakeAudioModel) ForwardToken(id int) ([]float32, error) { return []float32{1, 0}, nil }
func (f *fakeAudioModel) Reset()                                 {}
func (f *fakeAudioModel) ModelConfig() *core.ModelConfig         { return f.cfg }
func (f *fakeAudioModel) UpdateRoPE()                            {}
func (f *fakeAudioModel) AudioEncoder() core.AudioEncoder        { return f.enc }

type fakeAudioEncoder struct {
	cfg *core.AudioConfig
}

func (f *fakeAudioEncoder) Config() *core.AudioConfig { return f.cfg }

// Encode returns, per clip, NumEncoderTokens(frames) rows of TextHiddenSize
// filled with a per-(clip, token) marker so the test can verify placement.
func (f *fakeAudioEncoder) Encode(mels [][][]float64) ([][][]float32, error) {
	out := make([][][]float32, len(mels))
	for ci, mel := range mels {
		frames := len(mel[0])
		n := f.cfg.NumEncoderTokens(frames)
		rows := make([][]float32, n)
		for t := range rows {
			row := make([]float32, f.cfg.TextHiddenSize)
			for i := range row {
				row[i] = float32(1000*(ci+1) + t)
			}
			rows[t] = row
		}
		out[ci] = rows
	}
	return out, nil
}

// sineWAV builds a 16 kHz 16-bit PCM WAV of the given duration.
func sineWAV(t *testing.T, seconds float64) []byte {
	t.Helper()
	n := int(float64(audio.SampleRate) * seconds)
	data := make([]byte, n*2)
	for i := range n {
		s := 0.5 * math.Sin(2*math.Pi*440*float64(i)/float64(audio.SampleRate))
		v := int16(math.Max(-1, math.Min(1, s)) * 32767)
		binary.LittleEndian.PutUint16(data[i*2:], uint16(v))
	}
	var buf []byte
	u32 := func(v uint32) { var b [4]byte; binary.LittleEndian.PutUint32(b[:], v); buf = append(buf, b[:]...) }
	u16 := func(v uint16) { var b [2]byte; binary.LittleEndian.PutUint16(b[:], v); buf = append(buf, b[:]...) }
	buf = append(buf, "RIFF"...)
	u32(uint32(36 + len(data)))
	buf = append(buf, "WAVE"...)
	buf = append(buf, "fmt "...)
	u32(16)
	u16(1)
	u16(1)
	u32(uint32(audio.SampleRate))
	u32(uint32(audio.SampleRate * 2))
	u16(2)
	u16(16)
	buf = append(buf, "data"...)
	u32(uint32(len(data)))
	buf = append(buf, data...)
	return buf
}

// TestPrepareMultimodalAudioExpansion checks the <|audio|> placeholder
// expansion: each placeholder expands to ceil(melFrames/8) tokens carrying
// the clip's projected rows, in clip order, with all other tokens passed
// through untouched (processing_lfm3_omni.py replace_audio_token semantics).
func TestPrepareMultimodalAudioExpansion(t *testing.T) {
	t.Parallel()

	ac := &core.AudioConfig{
		FeatureExtractor:   audio.DefaultFeatureExtractorConfig(),
		HiddenSize:         8,
		NumMelBins:         audio.NumMelBins,
		TextHiddenSize:     6,
		AudioTokenID:       audio.AudioTokenID, // 125031
		VisionStartTokenID: 125009,
		VisionEndTokenID:   125010,
	}
	model := &fakeAudioModel{
		cfg: &core.ModelConfig{Arch: "lfm3_omni", Audio: ac},
		enc: &fakeAudioEncoder{cfg: ac},
	}
	e := &EngineImpl{model: model, tokenizer: okTokenizer{}}

	// 0.5 s clip: PadLength(8000)=8080 -> 49 frames -> ceil(49/8) = 7 tokens.
	wavA := sineWAV(t, 0.5)
	wavB := sineWAV(t, 1.0) // 99 frames -> 13 tokens
	const (
		vs = 125009
		ve = 125010
		au = audio.AudioTokenID
	)
	ids := []int{11, vs, au, ve, 12, vs, au, ve, 13}
	req := &Request{Audio: [][]byte{wavA, wavB}}

	newIDs, embeds, err := prepareMultimodal(e, req, ids)
	if err != nil {
		t.Fatalf("prepareMultimodal: %v", err)
	}
	// expansion: 11, vs, au x7, ve, 12, vs, au x13, ve, 13
	wantLen := 1 + 1 + 7 + 1 + 1 + 1 + 13 + 1 + 1
	if len(newIDs) != wantLen || len(embeds) != wantLen {
		t.Fatalf("expanded len %d/%d, want %d", len(newIDs), len(embeds), wantLen)
	}
	// audio token positions carry embeds; everything else nil
	audCount := 0
	for i, id := range newIDs {
		if id == au {
			if embeds[i] == nil {
				t.Fatalf("audio token at %d has nil embed", i)
			}
			audCount++
		} else if embeds[i] != nil {
			t.Fatalf("non-audio token %d at %d has embed", id, i)
		}
	}
	if audCount != 7+13 {
		t.Fatalf("audio token count %d, want 20", audCount)
	}
	// marker values: clip 0 rows are 1000+t, clip 1 rows 2000+t
	first := -1
	for i, id := range newIDs {
		if id == au {
			first = i
			break
		}
	}
	if first < 0 || embeds[first][0] != 1000 {
		t.Fatalf("first audio embed marker %v, want 1000", embeds[first])
	}
	last := -1
	for i, newID := range slices.Backward(newIDs) {
		if newID == au {
			last = i
			break
		}
	}
	if embeds[last][0] != 2000+12 {
		t.Fatalf("last audio embed marker %v, want %d", embeds[last], 2000+12)
	}

	// count mismatch: two placeholders but one clip
	if _, _, err := prepareMultimodal(e, &Request{Audio: [][]byte{wavA}}, ids); err == nil ||
		!strings.Contains(err.Error(), "audio placeholder") {
		t.Fatalf("want audio placeholder count error, got %v", err)
	}
	// media on a text-only model errors
	eText := &EngineImpl{model: &fakeAudioModel{cfg: &core.ModelConfig{Arch: "lfm3"}}, tokenizer: okTokenizer{}}
	if _, _, err := prepareMultimodal(eText, &Request{Audio: [][]byte{wavA}}, []int{1, au}); err == nil ||
		!strings.Contains(err.Error(), "no vision or audio tower") {
		t.Fatalf("want no-tower error, got %v", err)
	}
}
