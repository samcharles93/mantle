package audio

import (
	"math"
	"testing"
)

type fsqReference struct {
	Levels       []int     `json:"levels"`
	Basis        []int     `json:"basis"`
	HalfWidth    []int     `json:"halfWidth"`
	BoundZero    []float64 `json:"boundZero"`
	CodeZero     int       `json:"codeZero"`
	CodeMax      int       `json:"codeMax"`
	IndicesMax   []int     `json:"indicesMax"`
	CodeOfMax    int       `json:"codeOfMax"`
	BoundSamples []struct {
		X float64   `json:"x"`
		B []float64 `json:"b"`
	} `json:"boundSamples"`
}

func loadFSQRef(t *testing.T) fsqReference {
	t.Helper()
	var ref fsqReference
	loadJSON(t, "testdata/fsq_reference.json", &ref)
	return ref
}

func TestFSQBasis(t *testing.T) {
	// basis = [prod(levels[:k])] = [1, 8, 56, 336] for levels [8,7,6,6]
	got := FSQBasis([]int{8, 7, 6, 6})
	want := []int{1, 8, 56, 336}
	if len(got) != len(want) {
		t.Fatalf("basis = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("basis[%d] = %d, want %d", i, got[i], want[i])
		}
	}
}

func TestFSQBound(t *testing.T) {
	ref := loadFSQRef(t)
	q := NewFSQQuantizer(ref.Levels, NumCodebooks, QuantizerDim)
	for dim := range ref.Levels {
		if got := q.Bound(0, dim); math.Abs(got-ref.BoundZero[dim]) > 1e-15 {
			t.Fatalf("Bound(0, dim %d) = %v, want %v", dim, got, ref.BoundZero[dim])
		}
	}
	for _, s := range ref.BoundSamples {
		for dim := range ref.Levels {
			got := q.Bound(s.X, dim)
			if math.Abs(got-s.B[dim]) > 1e-12 {
				t.Fatalf("Bound(%v, dim %d) = %v, want %v", s.X, dim, got, s.B[dim])
			}
		}
	}
	// bound saturates inside the rounding range: bound in
	// [-(halfLevel+offset), halfLevel-offset] with halfLevel=(level-1)*(1+eps)/2
	for dim, level := range ref.Levels {
		halfLevel := float64(level-1) * 1.001 / 2
		offset := 0.0
		if level%2 == 0 {
			offset = 0.5
		}
		max := halfLevel + offset
		for _, x := range []float64{-1e6, -100, 0, 100, 1e6} {
			if b := q.Bound(x, dim); math.Abs(b) > max+1e-9 {
				t.Fatalf("Bound(%v, dim %d) = %v exceeds saturation %v", x, dim, b, max)
			}
		}
	}
}

func TestFSQCodeFromIndicesRoundTrip(t *testing.T) {
	ref := loadFSQRef(t)
	q := NewFSQQuantizer(ref.Levels, NumCodebooks, QuantizerDim)
	prod := 1
	for _, l := range ref.Levels {
		prod *= l
	}
	if q.MaxCode() != prod-1 || prod-1 != MaxFSQCode {
		t.Fatalf("MaxCode = %d, want %d", q.MaxCode(), MaxFSQCode)
	}
	// exhaustive bijection: every code in [0, 2015] maps to indices and back
	for code := 0; code < prod; code++ {
		idx := q.IndicesFromCode(code)
		if len(idx) != len(ref.Levels) {
			t.Fatalf("code %d -> %d indices", code, len(idx))
		}
		for k, iv := range idx {
			if iv < 0 || iv >= ref.Levels[k] {
				t.Fatalf("code %d index %d = %d out of [0, %d)", code, k, iv, ref.Levels[k])
			}
		}
		if got := q.CodeFromIndices(idx); got != code {
			t.Fatalf("CodeFromIndices(IndicesFromCode(%d)) = %d", code, got)
		}
	}
	if got := q.IndicesFromCode(MaxFSQCode); !equalInts(got, ref.IndicesMax) {
		t.Fatalf("IndicesFromCode(2015) = %v, want %v", got, ref.IndicesMax)
	}
	if ref.CodeOfMax != 2015 {
		t.Fatalf("code of [7,6,5,5] = %d, want 2015", ref.CodeOfMax)
	}
}

func equalInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func TestFSQEncodeZeroAndExtremes(t *testing.T) {
	ref := loadFSQRef(t)
	q := NewFSQQuantizer(ref.Levels, NumCodebooks, QuantizerDim)
	// bound(0) = 0 for every level, so indices = halfWidth = [4,3,3,3] and the
	// code of a zero 4-vector is 4*1 + 3*8 + 3*56 + 3*336 = 1204.
	if got := q.EncodeIndices([]float64{0, 0, 0, 0}); got != ref.CodeZero || got != 1204 {
		t.Fatalf("EncodeIndices(zeros) = %d, want %d", got, ref.CodeZero)
	}
	if got := q.EncodeIndices([]float64{1e6, 1e6, 1e6, 1e6}); got != ref.CodeMax {
		t.Fatalf("EncodeIndices(+inf-ish) = %d, want %d", got, ref.CodeMax)
	}
	if got := q.EncodeIndices([]float64{-1e6, -1e6, -1e6, -1e6}); got != 0 {
		t.Fatalf("EncodeIndices(-inf-ish) = %d, want 0", got)
	}
}

// randomWeights builds deterministic projection weights for the linear-bound
// quantizer tests.
func randomWeights(rows, cols int, seed uint32) ([][]float32, []float32) {
	l := &lcg{state: seed}
	w := make([][]float32, rows)
	for i := range w {
		w[i] = make([]float32, cols)
		for j := range w[i] {
			w[i][j] = float32(l.next() * 0.1)
		}
	}
	b := make([]float32, rows)
	for i := range b {
		b[i] = float32(l.next() * 0.1)
	}
	return w, b
}

func TestFSQQuantizerEncodeDecode(t *testing.T) {
	ref := loadFSQRef(t)
	q := NewFSQQuantizer(ref.Levels, NumCodebooks, QuantizerDim)
	eff := NumCodebooks * FSQCodebookDim // 48
	inW, inB := randomWeights(eff, QuantizerDim, 99)
	outW, outB := randomWeights(QuantizerDim, eff, 1234)
	if err := q.SetProjectIn(inW, inB); err != nil {
		t.Fatalf("SetProjectIn: %v", err)
	}
	if err := q.SetProjectOut(outW, outB); err != nil {
		t.Fatalf("SetProjectOut: %v", err)
	}
	l := &lcg{state: 7}
	for trial := range 200 {
		latents := make([]float32, QuantizerDim)
		for i := range latents {
			latents[i] = float32(l.next() * 4) // include values beyond [-1,1]
		}
		codes := q.Encode(latents)
		if len(codes) != NumCodebooks {
			t.Fatalf("got %d codes, want %d", len(codes), NumCodebooks)
		}
		for c, code := range codes {
			if code < 0 || code > MaxFSQCode {
				t.Fatalf("trial %d codebook %d code %d out of [0, %d]", trial, c, code, MaxFSQCode)
			}
			// decode is the exact inverse of the index mapping
			idx := q.IndicesFromCode(code)
			if got := q.CodeFromIndices(idx); got != code {
				t.Fatalf("code %d round trip failed: %d", code, got)
			}
		}
		dec := q.Decode(codes)
		if len(dec) != QuantizerDim {
			t.Fatalf("Decode returned %d latents, want %d", len(dec), QuantizerDim)
		}
		for _, v := range dec {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				t.Fatalf("Decode produced non-finite latent %v", v)
			}
		}
	}
}

func TestFSQQuantizerShapeErrors(t *testing.T) {
	q := NewFSQQuantizer([]int{8, 7, 6, 6}, NumCodebooks, QuantizerDim)
	if err := q.SetProjectIn(make([][]float32, 48), nil); err == nil {
		t.Fatalf("SetProjectIn with wrong inner dim must fail")
	}
	if err := q.SetProjectIn(make([][]float32, 48), make([]float32, 48)); err == nil {
		t.Fatalf("SetProjectIn with 48 cols must fail (want 256)")
	}
	w := make([][]float32, 48)
	for i := range w {
		w[i] = make([]float32, QuantizerDim)
	}
	if err := q.SetProjectIn(w, make([]float32, 48)); err != nil {
		t.Fatalf("SetProjectIn valid: %v", err)
	}
	if err := q.SetProjectOut(make([][]float32, QuantizerDim), make([]float32, QuantizerDim)); err == nil {
		t.Fatalf("SetProjectOut with wrong inner dim must fail")
	}
}

func TestFSQEOSCode(t *testing.T) {
	// audio_eos_code = prod(fsq_levels) = 2016 sits just past the max FSQ code.
	q := NewFSQQuantizer([]int{8, 7, 6, 6}, NumCodebooks, QuantizerDim)
	if q.MaxCode() != 2015 {
		t.Fatalf("MaxCode = %d, want 2015", q.MaxCode())
	}
	if AudioEOSCode != 2016 || CodebookSize != 2017 {
		t.Fatalf("EOS/codebook constants wrong: %d %d", AudioEOSCode, CodebookSize)
	}
}

func BenchmarkFSQEncodePerSecondAudio(b *testing.B) {
	q := NewFSQQuantizer([]int{8, 7, 6, 6}, NumCodebooks, QuantizerDim)
	w := make([][]float32, NumCodebooks*FSQCodebookDim)
	for i := range w {
		w[i] = make([]float32, QuantizerDim)
	}
	if err := q.SetProjectIn(w, make([]float32, len(w))); err != nil {
		b.Fatalf("SetProjectIn: %v", err)
	}
	l := &lcg{state: 3}
	frames := make([][]float32, DefaultCodecConfig().NumCodecFrames(24000)) // 13 frames = 1s at 24 kHz
	for i := range frames {
		frames[i] = make([]float32, QuantizerDim)
		for j := range frames[i] {
			frames[i][j] = float32(l.next())
		}
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, f := range frames {
			_ = q.Encode(f)
		}
	}
}

func BenchmarkFSQDecodePerSecondAudio(b *testing.B) {
	q := NewFSQQuantizer([]int{8, 7, 6, 6}, NumCodebooks, QuantizerDim)
	w := make([][]float32, QuantizerDim)
	for i := range w {
		w[i] = make([]float32, NumCodebooks*FSQCodebookDim)
	}
	if err := q.SetProjectOut(w, make([]float32, QuantizerDim)); err != nil {
		b.Fatalf("SetProjectOut: %v", err)
	}
	codes := make([]int, NumCodebooks)
	for i := range codes {
		codes[i] = i * 100 % 2016
	}
	frames := DefaultCodecConfig().NumCodecFrames(24000)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for range frames {
			_ = q.Decode(codes)
		}
	}
}
