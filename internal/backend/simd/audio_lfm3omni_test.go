package simd

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"math"
	"math/rand"
	"os"
	"testing"

	"github.com/samcharles93/mantle/internal/audio"
	"github.com/samcharles93/mantle/internal/backend/core"
)

// newTestAudioTower builds a tiny randomly-initialized AudioTower with the
// same block structure as LFM3-Omni's (downsample convs + LFM2 stack +
// projector) and returns it with its config.
func newTestAudioTower(rng *rand.Rand) (*AudioTower, *core.AudioConfig) {
	const (
		melBins = 16
		C       = 8
		hidden  = 32
		heads   = 4
		kvHeads = 2
		headDim = hidden / heads
		ffn     = 48
		layers  = 4
		textH   = 24
		window  = 4
	)
	ac := &core.AudioConfig{
		FeatureExtractor:   audio.DefaultFeatureExtractorConfig(),
		ConvChannels:       C,
		HiddenSize:         hidden,
		NumMelBins:         melBins,
		SlidingWindow:      window,
		NumLayers:          layers,
		NumHeads:           heads,
		NumKVHeads:         kvHeads,
		HeadDim:            headDim,
		FFNDim:             ffn,
		LayerTypes:         []string{"conv", "full_attention", "conv", "full_attention"},
		NormEps:            1e-5,
		ConvKernelSize:     3,
		RopeTheta:          1e6,
		TextHiddenSize:     textH,
		AudioTokenID:       audio.AudioTokenID,
		AudioOutTokenID:    audio.AudioOutTokenID,
		VisionStartTokenID: 125009,
		VisionEndTokenID:   125010,
	}
	ac.FeatureExtractor.FeatureSize = melBins // keep the tower dims tiny; front end overridden per test
	randMat := func(r, c int) *core.Mat {
		m := core.NewMat(r, c)
		for i := range m.Data {
			m.Data[i] = float32(rng.NormFloat64() * 0.2)
		}
		return &m
	}
	randVec := func(n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(1 + rng.NormFloat64()*0.1)
		}
		return v
	}
	randConv := func(c int) []float32 {
		v := make([]float32, c*9)
		for i := range v {
			v[i] = float32(rng.NormFloat64() * 0.2)
		}
		return v
	}
	// after three stride-2 convs the mel dim is melBins/8 = 2
	linOutIn := C * (melBins / 8)
	at := &AudioTower{
		cfg:       ac,
		convInW:   randConv(C),
		convInB:   randVec(C),
		convMidW:  randConv(C),
		convMidB:  randVec(C),
		linMid:    randMat(C, C),
		linMidB:   randVec(C),
		convEndW:  randConv(C),
		convEndB:  randVec(C),
		linEnd:    randMat(C, C),
		linEndB:   randVec(C),
		linOut:    randMat(hidden, linOutIn),
		linOutB:   randVec(hidden),
		embedNorm: randVec(hidden),
		projNormW: randVec(hidden),
		projNormB: randVec(hidden),
		proj1:     randMat(textH, hidden),
		proj1B:    randVec(textH),
		proj2:     randMat(textH, textH),
		proj2B:    randVec(textH),
	}
	at.layers = make([]audioLayer, layers)
	for i := range at.layers {
		l := &at.layers[i]
		l.isAttn = ac.LayerTypes[i] == "full_attention"
		l.opNorm = randVec(hidden)
		l.ffnNorm = randVec(hidden)
		l.w1 = randMat(ffn, hidden)
		l.w2 = randMat(hidden, ffn)
		l.w3 = randMat(ffn, hidden)
		if l.isAttn {
			l.q = randMat(heads*headDim, hidden)
			l.k = randMat(kvHeads*headDim, hidden)
			l.v = randMat(kvHeads*headDim, hidden)
			l.o = randMat(hidden, heads*headDim)
			l.qNorm = randVec(headDim)
			l.kNorm = randVec(headDim)
		} else {
			l.convIn = randMat(3*hidden, hidden)
			l.convW = make([]float32, hidden*3)
			for j := range l.convW {
				l.convW[j] = float32(rng.NormFloat64() * 0.2)
			}
			l.convOut = randMat(hidden, hidden)
		}
	}
	at.invFreq = make([]float64, headDim/2)
	for i := range at.invFreq {
		at.invFreq[i] = 1.0 / math.Pow(ac.RopeTheta, float64(2*i)/float64(headDim))
	}
	return at, ac
}

// TestAudioTowerSynthetic runs a tiny randomly-initialized audio tower end to
// end and checks the output contract: one row per audio token
// (ceil(frames/8)), TextHiddenSize wide, finite, and deterministic.
func TestAudioTowerSynthetic(t *testing.T) {
	t.Parallel()
	rng := rand.New(rand.NewSource(7))
	at, ac := newTestAudioTower(rng)

	const frames = 40 // -> ceil(40/8) = 5 tokens
	mel := make([][]float64, ac.NumMelBins)
	for m := range mel {
		mel[m] = make([]float64, frames)
		for fr := range mel[m] {
			mel[m][fr] = rng.NormFloat64() * 0.5
		}
	}
	out, err := at.Encode([][][]float64{mel})
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	if len(out) != 1 {
		t.Fatalf("got %d clips, want 1", len(out))
	}
	wantTok := ac.NumEncoderTokens(frames)
	if len(out[0]) != wantTok {
		t.Fatalf("got %d audio tokens, want %d", len(out[0]), wantTok)
	}
	for k, row := range out[0] {
		if len(row) != ac.TextHiddenSize {
			t.Fatalf("row %d has %d floats, want %d", k, len(row), ac.TextHiddenSize)
		}
		for _, v := range row {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("row %d has non-finite value %g", k, v)
			}
		}
	}
	// determinism
	out2, err := at.Encode([][][]float64{mel})
	if err != nil {
		t.Fatalf("Encode(2): %v", err)
	}
	for k := range out[0] {
		for i := range out[0][k] {
			if out[0][k][i] != out2[0][k][i] {
				t.Fatalf("non-deterministic at row %d idx %d: %g vs %g", k, i, out[0][k][i], out2[0][k][i])
			}
		}
	}
	// wrong mel-bin count must error
	bad := make([][]float64, ac.NumMelBins-1)
	for i := range bad {
		bad[i] = make([]float64, frames)
	}
	if _, err := at.Encode([][][]float64{bad}); err == nil {
		t.Fatalf("expected error for wrong mel bin count")
	}
}

// encodeTestWAV builds a 16-bit PCM WAV at the given sample rate.
func encodeTestWAV(t *testing.T, sr int, samples []float64) []byte {
	t.Helper()
	data := make([]byte, len(samples)*2)
	for i, s := range samples {
		v := int16(math.Max(-1, math.Min(1, s)) * 32767)
		binary.LittleEndian.PutUint16(data[i*2:], uint16(v))
	}
	buf := &bytes.Buffer{}
	_ = binary.Write(buf, binary.LittleEndian, []byte("RIFF"))
	_ = binary.Write(buf, binary.LittleEndian, uint32(36+len(data)))
	_ = binary.Write(buf, binary.LittleEndian, []byte("WAVE"))
	_ = binary.Write(buf, binary.LittleEndian, []byte("fmt "))
	_ = binary.Write(buf, binary.LittleEndian, uint32(16))
	_ = binary.Write(buf, binary.LittleEndian, uint16(1)) // PCM
	_ = binary.Write(buf, binary.LittleEndian, uint16(1)) // mono
	_ = binary.Write(buf, binary.LittleEndian, uint32(sr))
	_ = binary.Write(buf, binary.LittleEndian, uint32(sr*2))
	_ = binary.Write(buf, binary.LittleEndian, uint16(2))
	_ = binary.Write(buf, binary.LittleEndian, uint16(16))
	_ = binary.Write(buf, binary.LittleEndian, []byte("data"))
	_ = binary.Write(buf, binary.LittleEndian, uint32(len(data)))
	_, _ = buf.Write(data)
	return buf.Bytes()
}

// TestProcessAudioClips checks the WAV -> log-mel -> token-count pipeline:
// a 1.0 s 16 kHz clip yields 99 mel frames and ceil(99/8) = 13 audio tokens,
// and a non-16 kHz clip is rejected like HF's feature extractor __call__.
func TestProcessAudioClips(t *testing.T) {
	t.Parallel()
	ac := &core.AudioConfig{
		FeatureExtractor: audio.DefaultFeatureExtractorConfig(),
		NumMelBins:       audio.NumMelBins,
		AudioTokenID:     audio.AudioTokenID,
	}
	sr := audio.SampleRate
	samples := make([]float64, sr) // 1.0 s sine
	for i := range samples {
		samples[i] = 0.5 * math.Sin(2*math.Pi*440*float64(i)/float64(sr))
	}
	wav := encodeTestWAV(t, sr, samples)

	mels, err := ProcessAudioClips([][]byte{wav}, ac)
	if err != nil {
		t.Fatalf("ProcessAudioClips: %v", err)
	}
	if len(mels) != 1 || len(mels[0]) != ac.NumMelBins {
		t.Fatalf("mel shape: %d clips x %d bins", len(mels), len(mels[0]))
	}
	frames := len(mels[0][0])
	wantFrames := ac.FeatureExtractor.NumFrames(sr)
	if frames != wantFrames {
		t.Fatalf("frames %d, want %d", frames, wantFrames)
	}
	if wantFrames != 99 {
		t.Fatalf("1.0 s clip should produce 99 frames, got %d", wantFrames)
	}
	wantTok := ac.NumEncoderTokens(frames)
	if wantTok != 13 {
		t.Fatalf("1.0 s clip should produce 13 audio tokens, got %d", wantTok)
	}
	// spot-check the log-mel normalization range: (log10(mel) clamped at -12
	// plus 4) / 4 lies in [-2, ...]
	for m := range mels[0] {
		for fr := range mels[0][m] {
			v := mels[0][m][fr]
			if v < -2.0-1e-9 {
				t.Fatalf("mel[%d][%d] = %g below floor-derived minimum -2", m, fr, v)
			}
		}
	}

	// non-16 kHz input is rejected (HF __call__ semantics)
	wav8k := encodeTestWAV(t, 8000, samples[:8000])
	if _, err := ProcessAudioClips([][]byte{wav8k}, ac); err == nil {
		t.Fatalf("expected error for 8 kHz input")
	}
	// garbage is rejected
	if _, err := ProcessAudioClips([][]byte{[]byte("not a wav")}, ac); err == nil {
		t.Fatalf("expected error for garbage input")
	}
}

// TestAudioDownsampleStagesVsTorch checks the downsample primitives against
// torch.nn.functional reference fixtures (testdata/audiods_*.f32, generated
// by F.conv2d/F.conv1d with odd dims to exercise padding): conv2d k3 s2 p1
// (plain and depthwise), 1x1 conv, and the flatten(1,2)+conv1d lin_out with
// its [w][hidden] transpose.
func TestAudioDownsampleStagesVsTorch(t *testing.T) {
	t.Parallel()

	var m struct {
		ConvIn map[string]json.RawMessage `json:"conv_in"`
		ConvDw map[string]json.RawMessage `json:"conv_dw"`
		Lin1x1 map[string]json.RawMessage `json:"lin1x1"`
		LinOut map[string]json.RawMessage `json:"linout"`
	}
	raw, err := os.ReadFile("testdata/audiods_meta.json")
	if err != nil {
		t.Fatalf("read meta: %v", err)
	}
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatalf("parse meta: %v", err)
	}
	num := func(msg json.RawMessage) int {
		var v int
		if err := json.Unmarshal(msg, &v); err != nil {
			t.Fatalf("num: %v", err)
		}
		return v
	}
	shape := func(msg json.RawMessage) []int {
		var v []int
		if err := json.Unmarshal(msg, &v); err != nil {
			t.Fatalf("shape: %v", err)
		}
		return v
	}
	cmp := func(name string, got, want []float32, tol float64) {
		t.Helper()
		if len(got) != len(want) {
			t.Fatalf("%s: len %d, want %d", name, len(got), len(want))
		}
		for i := range got {
			if d := math.Abs(float64(got[i] - want[i])); d > tol {
				t.Fatalf("%s[%d]: got %g, want %g (diff %g)", name, i, got[i], want[i], d)
			}
		}
	}

	const tol = 1e-5

	// conv_in: groups=1
	{
		cIn, H, W := num(m.ConvIn["cIn"]), num(m.ConvIn["H"]), num(m.ConvIn["W"])
		cOut, groups := num(m.ConvIn["cOut"]), num(m.ConvIn["groups"])
		in := readF32File(t, "testdata/audiods_conv_in.in.f32")
		w := readF32File(t, "testdata/audiods_conv_in.w.f32")
		b := readF32File(t, "testdata/audiods_conv_in.b.f32")
		want := readF32File(t, "testdata/audiods_conv_in.out.f32")
		got := conv2dS2P1K3(in, cIn, H, W, w, groups, cOut, b)
		cmp("conv_in", got, want, tol)
		_ = shape(m.ConvIn["out"]) // payload length is checked in cmp
	}
	// conv_dw: depthwise groups=C
	{
		cIn, H, W := num(m.ConvDw["cIn"]), num(m.ConvDw["H"]), num(m.ConvDw["W"])
		cOut, groups := num(m.ConvDw["cOut"]), num(m.ConvDw["groups"])
		in := readF32File(t, "testdata/audiods_conv_dw.in.f32")
		w := readF32File(t, "testdata/audiods_conv_dw.w.f32")
		b := readF32File(t, "testdata/audiods_conv_dw.b.f32")
		want := readF32File(t, "testdata/audiods_conv_dw.out.f32")
		got := conv2dS2P1K3(in, cIn, H, W, w, groups, cOut, b)
		cmp("conv_dw", got, want, tol)
	}
	// lin1x1
	{
		C, H, W := num(m.Lin1x1["C"]), num(m.Lin1x1["H"]), num(m.Lin1x1["W"])
		in := readF32File(t, "testdata/audiods_lin1x1.in.f32")
		w := readF32File(t, "testdata/audiods_lin1x1.w.f32")
		b := readF32File(t, "testdata/audiods_lin1x1.b.f32")
		want := readF32File(t, "testdata/audiods_lin1x1.out.f32")
		wm := core.NewMatFromData(C, C, w)
		got := lin1x1(in, C, H, W, &wm, b)
		cmp("lin1x1", got, want, tol)
	}
	// lin_out: flatten(1,2) [C][h][W] -> [C*h][W] (hh fastest), conv1d k=1, out [W][hidden]
	{
		C, h, W, hidden := num(m.LinOut["C"]), num(m.LinOut["h"]), num(m.LinOut["W"]), num(m.LinOut["hidden"])
		in := readF32File(t, "testdata/audiods_linout.in.f32") // [C][h][W] channel-major
		w := readF32File(t, "testdata/audiods_linout.w.f32")   // [hidden][C*h]
		b := readF32File(t, "testdata/audiods_linout.b.f32")
		want := readF32File(t, "testdata/audiods_linout.out.f32") // [W][hidden]
		wm := core.NewMatFromData(hidden, C*h, w)
		got := make([]float32, W*hidden)
		vec := make([]float32, C*h)
		tmp := make([]float32, hidden)
		for tt := range W {
			for c := range C {
				base := c * h * W
				for hh := range h {
					vec[c*h+hh] = in[base+hh*W+tt]
				}
			}
			MatVec(tmp, &wm, vec)
			for o := range hidden {
				got[tt*hidden+o] = tmp[o] + b[o]
			}
		}
		cmp("linout", got, want, tol)
	}
}
