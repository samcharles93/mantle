package audio

import (
	"math"
	"testing"
)

func TestHannWindowPeriodic(t *testing.T) {
	for _, n := range []int{400, 1280} {
		w := HannWindow(n)
		if len(w) != n {
			t.Fatalf("n=%d: len(w) = %d", n, len(w))
		}
		if w[0] != 0 {
			t.Fatalf("n=%d: w[0] = %v, want 0", n, w[0])
		}
		// periodic Hann: w[i] = 0.5 - 0.5*cos(2*pi*i/n), symmetric w[i] == w[n-i]
		for i := 1; i < n; i++ {
			want := 0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/float64(n))
			if math.Abs(w[i]-want) > 1e-15 {
				t.Fatalf("n=%d: w[%d] = %v, want %v", n, i, w[i], want)
			}
			if math.Abs(w[i]-w[n-i]) > 1e-15 {
				t.Fatalf("n=%d: w[%d]=%v != w[%d]=%v (periodic symmetry)", n, i, w[i], n-i, w[n-i])
			}
		}
		// periodic window does NOT end at zero (unlike symmetric)
		if w[n-1] == 0 {
			t.Fatalf("n=%d: periodic window must not be zero at n-1", n)
		}
	}
}

func TestSTFTFrameCount(t *testing.T) {
	x := make([]float64, 16000)
	w := HannWindow(400)
	spec := STFT(x, 400, 160, 400, w)
	if len(spec) != 201 {
		t.Fatalf("bins = %d, want 201", len(spec))
	}
	if len(spec[0]) != 98 {
		t.Fatalf("frames = %d, want 98 (1 + (16000-400)/160)", len(spec[0]))
	}
	// short inputs: no padding with center=False
	if got := len(STFT(make([]float64, 399), 400, 160, 400, w)[0]); got != 0 {
		t.Fatalf("len 399 -> %d frames, want 0", got)
	}
	if got := len(STFT(make([]float64, 400), 400, 160, 400, w)[0]); got != 1 {
		t.Fatalf("len 400 -> %d frames, want 1", got)
	}
	if got := len(STFT(make([]float64, 560), 400, 160, 400, w)[0]); got != 2 {
		t.Fatalf("len 560 -> %d frames, want 2", got)
	}
}

func TestSTFTMatchesNaiveDFT(t *testing.T) {
	// verifies torch.stft center=False semantics: frame t starts at sample t*hop,
	// windowed by the periodic Hann window, bins 0..n_fft/2.
	l := &lcg{state: 42}
	x := make([]float64, 1000)
	for i := range x {
		x[i] = l.next()
	}
	const nFFT, hop, win = 400, 160, 400
	w := HannWindow(win)
	spec := STFT(x, nFFT, hop, win, w)
	frames := len(spec[0])
	wantFrames := 1 + (len(x)-nFFT)/hop
	if frames != wantFrames {
		t.Fatalf("frames = %d, want %d", frames, wantFrames)
	}
	for fr := 0; fr < frames; fr++ {
		frame := make([]float64, nFFT)
		im := make([]float64, nFFT)
		for i := 0; i < nFFT; i++ {
			frame[i] = x[fr*hop+i] * w[i]
		}
		ore, oim := naiveDFT(frame, im)
		for k := 0; k <= nFFT/2; k++ {
			got := spec[k][fr]
			if math.Abs(real(got)-ore[k]) > 1e-9 || math.Abs(imag(got)-oim[k]) > 1e-9 {
				t.Fatalf("frame %d bin %d = %v, want %v%+vi", fr, k, got, ore[k], oim[k])
			}
		}
	}
}

func TestSTFTFrameStartOffset(t *testing.T) {
	// impulse at sample 160: frame 0 covers samples 0..399, so the impulse sits at
	// offset 160 of frame 0; frame 1 starts exactly at 160, so it sits at offset 0;
	// frame 2 starts at 320 and does not contain it.
	const nFFT, hop = 400, 160
	x := make([]float64, 1000)
	x[160] = 1
	w := HannWindow(nFFT)
	spec := STFT(x, nFFT, hop, nFFT, w)
	for k := 0; k <= nFFT/2; k++ {
		// impulse at frame offset 160 contributes w[160]*exp(-2*pi*i*k*160/400)
		ang := -2 * math.Pi * float64(k) * 160 / 400
		want0 := complex(w[160]*math.Cos(ang), w[160]*math.Sin(ang))
		if math.Abs(real(spec[k][0])-real(want0)) > 1e-12 || math.Abs(imag(spec[k][0])-imag(want0)) > 1e-12 {
			t.Fatalf("frame 0 bin %d = %v, want %v (w[160] rotated by frame offset)", k, spec[k][0], want0)
		}
		if math.Abs(real(spec[k][1])-w[0]) > 1e-12 {
			t.Fatalf("frame 1 bin %d = %v, want %v (w[0])", k, spec[k][1], w[0])
		}
		if spec[k][2] != 0 {
			t.Fatalf("frame 2 bin %d = %v, want 0 (frame 2 starts at sample 320)", k, spec[k][2])
		}
	}
}

func TestSTFTWindowShorterThanNFFT(t *testing.T) {
	// torch.stft pads a short window to n_fft centered: left pad (n_fft-win)/2.
	const nFFT, win = 400, 200
	x := make([]float64, 400)
	x[100] = 1 // first windowed sample (offset (400-200)/2 = 100)
	w := HannWindow(win)
	spec := STFT(x, nFFT, 160, win, w)
	for k := 0; k <= nFFT/2; k++ {
		if math.Abs(real(spec[k][0])-w[0]) > 1e-12 || math.Abs(imag(spec[k][0])) > 1e-12 {
			t.Fatalf("bin %d = %v, want %v+0i", k, spec[k][0], w[0])
		}
	}
}

func BenchmarkSTFT16k1s(b *testing.B) {
	l := &lcg{state: 7}
	x := make([]float64, 16000)
	for i := range x {
		x[i] = l.next()
	}
	w := HannWindow(400)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = STFT(x, 400, 160, 400, w)
	}
}
