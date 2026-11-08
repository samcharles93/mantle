package audio

import (
	"encoding/json"
	"math"
	"os"
	"strconv"
	"testing"
)

type melOmniFixture struct {
	SR        int                  `json:"sr"`
	NFFT      int                  `json:"nFFT"`
	NumBins   int                  `json:"numBins"`
	NumMels   int                  `json:"numMels"`
	Fmin      float64              `json:"fmin"`
	Fmax      float64              `json:"fmax"`
	MelBandHz []float64            `json:"melBandHz"`
	FFTBinHz  []float64            `json:"fftBinHz"`
	RowSums   []float64            `json:"rowSums"`
	Rows      map[string][]float64 `json:"rows"`
}

type melSmallFixture struct {
	SR        int         `json:"sr"`
	NFFT      int         `json:"nFFT"`
	NumBins   int         `json:"numBins"`
	NumMels   int         `json:"numMels"`
	Fmin      float64     `json:"fmin"`
	Fmax      float64     `json:"fmax"`
	MelBandHz []float64   `json:"melBandHz"`
	FFTBinHz  []float64   `json:"fftBinHz"`
	Weights   [][]float64 `json:"weights"`
	RowSums   []float64   `json:"rowSums"`
}

func loadJSON(t *testing.T, path string, v any) {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	if err := json.Unmarshal(data, v); err != nil {
		t.Fatalf("parse %s: %v", path, err)
	}
}

func TestHzToMelSlaney(t *testing.T) {
	// Slaney scale: linear below 1 kHz with f_sp = 200/3, log above with
	// min_log_mel = 15 at 1000 Hz and logstep = ln(6.4)/27.
	if got := HzToMel(0); got != 0 {
		t.Fatalf("HzToMel(0) = %v, want 0", got)
	}
	if got := HzToMel(500); math.Abs(got-7.5) > 1e-12 {
		t.Fatalf("HzToMel(500) = %v, want 7.5 (500 / (200/3))", got)
	}
	if got := HzToMel(1000); math.Abs(got-15) > 1e-12 {
		t.Fatalf("HzToMel(1000) = %v, want 15", got)
	}
	logstep := math.Log(6.4) / 27.0
	if got, want := HzToMel(8000), 15+math.Log(8)/logstep; math.Abs(got-want) > 1e-12 {
		t.Fatalf("HzToMel(8000) = %v, want %v", got, want)
	}
	for _, f := range []float64{0, 1, 440, 999.9, 1000, 4000, 8000} {
		if got := MelToHz(HzToMel(f)); math.Abs(got-f) > 1e-9*f+1e-12 {
			t.Fatalf("MelToHz(HzToMel(%v)) = %v", f, got)
		}
	}
}

func TestMelFilterBankSmallHandCase(t *testing.T) {
	// sr 8000, n_fft 16 (9 bins at 500 Hz spacing), 3 mel filters over 0..4000 Hz.
	// Expected matrix computed from the auditory-toolbox Slaney formulas
	// (librosa.filters.mel with norm="slaney", htk=False) by an independent
	// implementation; stored in testdata/mel_filters_small.json.
	var fx melSmallFixture
	loadJSON(t, "testdata/mel_filters_small.json", &fx)
	if fx.SR != 8000 || fx.NFFT != 16 || fx.NumMels != 3 || fx.NumBins != 9 {
		t.Fatalf("fixture shape mismatch: %+v", fx)
	}
	W := MelFilterBank(8000, 16, 3, 0, 4000)
	if len(W) != 3 || len(W[0]) != 9 {
		t.Fatalf("got %dx%d, want 3x9", len(W), len(W[0]))
	}
	for i := range W {
		for j := range W[i] {
			if math.Abs(W[i][j]-fx.Weights[i][j]) > 1e-15 {
				t.Fatalf("W[%d][%d] = %v, want %v", i, j, W[i][j], fx.Weights[i][j])
			}
		}
		sum := 0.0
		for _, v := range W[i] {
			sum += v
		}
		if math.Abs(sum-fx.RowSums[i]) > 1e-15 {
			t.Fatalf("row %d sum = %v, want %v", i, sum, fx.RowSums[i])
		}
	}
	// hand-derived structural facts for this tiny case: mel band edges
	// [0, 586.06, 1194.24, 2185.62, 4000] Hz, fft bins every 500 Hz.
	if fx.MelBandHz[0] != 0 || fx.MelBandHz[4] != 4000 {
		t.Fatalf("mel band edges = %v", fx.MelBandHz)
	}
	if W[0][0] != 0 {
		t.Fatalf("filter 0 must be zero at DC (band edge), got %v", W[0][0])
	}
	if !(W[0][1] > W[0][2] && W[0][2] > 0 && W[0][3] == 0) {
		t.Fatalf("filter 0 triangle over bins 500/1000 Hz: %v", W[0])
	}
}

func TestMelFilterBankTriangularShapes(t *testing.T) {
	const sr, nFFT, numMels = 16000, 400, 128
	W := MelFilterBank(sr, nFFT, numMels, 0, 8000)
	bands := MelFrequencies(numMels+2, 0, 8000)
	freqs := FFTFrequencies(sr, nFFT)
	if len(W) != numMels || len(W[0]) != nFFT/2+1 {
		t.Fatalf("shape = %dx%d", len(W), len(W[0]))
	}
	binHz := float64(sr) / float64(nFFT)
	for i := range W {
		argmax := 0
		for j := range W[i] {
			w := W[i][j]
			if w < 0 {
				t.Fatalf("row %d bin %d negative weight %v", i, j, w)
			}
			// support is strictly inside (band[i], band[i+2])
			if w > 1e-15 && (freqs[j] <= bands[i] || freqs[j] >= bands[i+2]) {
				t.Fatalf("row %d bin %d (%.1f Hz) has weight %v outside (%.1f, %.1f)", i, j, freqs[j], w, bands[i], bands[i+2])
			}
			if w > W[i][argmax] {
				argmax = j
			}
		}
		if W[i][argmax] <= 0 {
			t.Fatalf("row %d has zero peak", i)
		}
		// peak near the band center, within one bin
		if math.Abs(freqs[argmax]-bands[i+1]) > binHz {
			t.Fatalf("row %d peak at %.1f Hz, center %.1f Hz", i, freqs[argmax], bands[i+1])
		}
		// unimodal: non-decreasing up to argmax, non-increasing after
		for j := 1; j <= argmax; j++ {
			if W[i][j] < W[i][j-1]-1e-18 {
				t.Fatalf("row %d not rising at bin %d: %v < %v", i, j, W[i][j], W[i][j-1])
			}
		}
		for j := argmax + 1; j < len(W[i]); j++ {
			if W[i][j] > W[i][j-1]+1e-18 {
				t.Fatalf("row %d not falling at bin %d: %v > %v", i, j, W[i][j], W[i][j-1])
			}
		}
	}
}

func TestMelFilterBankOmniFixture(t *testing.T) {
	// Full parity with the independent reference for the exact Omni config:
	// sr 16000, n_fft 400 (201 bins), 128 mel bins, fmin 0, fmax 8000, slaney
	// norm + scale.
	var fx melOmniFixture
	loadJSON(t, "testdata/mel_filters_omni.json", &fx)
	W := MelFilterBank(16000, 400, 128, 0, 8000)
	if len(W) != 128 || len(W[0]) != 201 {
		t.Fatalf("shape = %dx%d, want 128x201", len(W), len(W[0]))
	}
	bands := MelFrequencies(130, 0, 8000)
	for i := range bands {
		if math.Abs(bands[i]-fx.MelBandHz[i]) > 1e-9 {
			t.Fatalf("mel band %d = %v, want %v", i, bands[i], fx.MelBandHz[i])
		}
	}
	freqs := FFTFrequencies(16000, 400)
	for i := range fx.FFTBinHz {
		if math.Abs(freqs[i]-fx.FFTBinHz[i]) > 1e-9 {
			t.Fatalf("fft bin %d = %v, want %v", i, freqs[i], fx.FFTBinHz[i])
		}
	}
	for i := range W {
		sum := 0.0
		for _, v := range W[i] {
			sum += v
		}
		if math.Abs(sum-fx.RowSums[i]) > 1e-12*math.Max(1, math.Abs(fx.RowSums[i])) {
			t.Fatalf("row %d sum = %v, want %v", i, sum, fx.RowSums[i])
		}
	}
	for key, wantRow := range fx.Rows {
		i, err := strconv.Atoi(key)
		if err != nil {
			t.Fatalf("bad row key %q", key)
		}
		for j := range wantRow {
			if math.Abs(W[i][j]-wantRow[j]) > 1e-12*math.Max(1, math.Abs(wantRow[j])) {
				t.Fatalf("W[%d][%d] = %v, want %v", i, j, W[i][j], wantRow[j])
			}
		}
	}
}

type logmelSample struct {
	M int     `json:"m"`
	T int     `json:"t"`
	V float64 `json:"v"`
}

type logmelCase struct {
	Name       string         `json:"name"`
	NumSamples int            `json:"numSamples"`
	Bins       int            `json:"bins"`
	Frames     int            `json:"frames"`
	Values     []float64      `json:"values"`
	Row0       []float64      `json:"row0"`
	Row127     []float64      `json:"row127"`
	Sampled    []logmelSample `json:"sampled"`
}

func TestLogMelFixtures(t *testing.T) {
	var fx struct {
		Cases []logmelCase `json:"cases"`
	}
	loadJSON(t, "testdata/logmel_fixtures.json", &fx)
	if len(fx.Cases) != 3 {
		t.Fatalf("got %d cases, want 3", len(fx.Cases))
	}
	fe := NewFeatureExtractor(DefaultFeatureExtractorConfig())
	for _, c := range fx.Cases {
		t.Run(c.Name, func(t *testing.T) {
			wave := fixtureWave(c.NumSamples)
			got := fe.LogMel(wave)
			if len(got) != c.Bins || len(got[0]) != c.Frames {
				t.Fatalf("shape = %dx%d, want %dx%d", len(got), len(got[0]), c.Bins, c.Frames)
			}
			check := func(m, fr int, want float64) {
				t.Helper()
				if math.Abs(got[m][fr]-want) > 1e-7 {
					t.Fatalf("logmel[%d][%d] = %v, want %v", m, fr, got[m][fr], want)
				}
			}
			if c.Values != nil {
				for m := 0; m < c.Bins; m++ {
					for fr := 0; fr < c.Frames; fr++ {
						check(m, fr, c.Values[m*c.Frames+fr])
					}
				}
			}
			for fr := range c.Row0 {
				check(0, fr, c.Row0[fr])
				check(127, fr, c.Row127[fr])
			}
			for _, s := range c.Sampled {
				check(s.M, s.T, s.V)
			}
		})
	}
}

func TestLogMelSilenceIsFloor(t *testing.T) {
	fe := NewFeatureExtractor(DefaultFeatureExtractorConfig())
	got := fe.LogMel(make([]float32, 16000))
	if len(got) != 128 || len(got[0]) != 99 {
		t.Fatalf("shape = %dx%d, want 128x99", len(got), len(got[0]))
	}
	// log10(0) -> -Inf -> clamp(-12) -> (-12+4)/4 = -2 exactly
	for m := range got {
		for fr := range got[m] {
			if got[m][fr] != -2 {
				t.Fatalf("silence logmel[%d][%d] = %v, want -2", m, fr, got[m][fr])
			}
		}
	}
}

// fixtureWave regenerates the deterministic waveform used by the JS reference:
// LCG seed 0x12345678, u = (seed>>>8)/2^24, x[i] = fround(0.3*sin(2*pi*220*i/sr)
// + 0.2*sin(2*pi*997*i/sr) + 0.05*(2u-1)).
func fixtureWave(n int) []float32 {
	state := uint32(0x12345678)
	x := make([]float32, n)
	const sr = 16000
	for i := range n {
		state = state*1664525 + 1013904223
		u := float64(state>>8) / float64(1<<24)
		noise := 2*u - 1
		x[i] = float32(0.3*math.Sin(2*math.Pi*220*float64(i)/sr) + 0.2*math.Sin(2*math.Pi*997*float64(i)/sr) + 0.05*noise)
	}
	return x
}

func BenchmarkLogMelPerSecondAudio(b *testing.B) {
	fe := NewFeatureExtractor(DefaultFeatureExtractorConfig())
	wave := fixtureWave(16000) // exactly one second at 16 kHz
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = fe.LogMel(wave)
	}
}

func BenchmarkMelFilterBank(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = MelFilterBank(16000, 400, 128, 0, 8000)
	}
}
