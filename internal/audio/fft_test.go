package audio

import (
	"math"
	"math/cmplx"
	"testing"
)

// naiveDFT is the O(n^2) reference transform used to validate the mixed-radix FFT.
func naiveDFT(re, im []float64) (ore, oim []float64) {
	n := len(re)
	ore = make([]float64, n)
	oim = make([]float64, n)
	for k := range n {
		var sr, si float64
		for t := range n {
			ang := -2 * math.Pi * float64(k) * float64(t) / float64(n)
			sr += re[t]*math.Cos(ang) - im[t]*math.Sin(ang)
			si += re[t]*math.Sin(ang) + im[t]*math.Cos(ang)
		}
		ore[k], oim[k] = sr, si
	}
	return ore, oim
}

// lcg is a tiny deterministic PRNG shared by the DSP tests.
type lcg struct{ state uint32 }

func (l *lcg) next() float64 {
	l.state = l.state*1664525 + 1013904223
	return float64(l.state>>8)/float64(1<<24)*2 - 1 // [-1, 1)
}

func randomPair(n int, seed uint32) (re, im []float64) {
	l := &lcg{state: seed}
	re = make([]float64, n)
	im = make([]float64, n)
	for i := range re {
		re[i] = l.next()
		im[i] = l.next()
	}
	return re, im
}

func maxAbsDiff(a, b []float64) float64 {
	m := 0.0
	for i := range a {
		if d := math.Abs(a[i] - b[i]); d > m {
			m = d
		}
	}
	return m
}

func TestFFTVsNaiveDFT(t *testing.T) {
	// 400 = 2^4 * 5^2 and 1280 = 2^8 * 5 are not powers of two; cover both plus
	// small primes and composites on the way.
	for _, n := range []int{1, 2, 3, 4, 5, 6, 7, 8, 10, 25, 160, 400, 1280} {
		re, im := randomPair(n, uint32(0xC0FFEE+n))
		wantRe, wantIm := naiveDFT(re, im)
		FFT(re, im)
		scale := 1.0
		for _, v := range wantRe {
			if v > scale {
				scale = v
			}
		}
		if d := maxAbsDiff(re, wantRe); d > 1e-6*scale {
			t.Fatalf("n=%d: re max diff %g exceeds 1e-6 relative (scale %g)", n, d, scale)
		}
		if d := maxAbsDiff(im, wantIm); d > 1e-6*scale {
			t.Fatalf("n=%d: im max diff %g exceeds 1e-6 relative (scale %g)", n, d, scale)
		}
	}
}

func TestFFTKnownInputs(t *testing.T) {
	for _, n := range []int{400, 1280} {
		// impulse at 0 -> flat spectrum of ones
		re := make([]float64, n)
		im := make([]float64, n)
		re[0] = 1
		FFT(re, im)
		for k := range n {
			if math.Abs(re[k]-1) > 1e-9 || math.Abs(im[k]) > 1e-9 {
				t.Fatalf("n=%d impulse: bin %d = %v%+vi, want 1+0i", n, k, re[k], im[k])
			}
		}
		// constant -> all energy in bin 0
		for i := range re {
			re[i] = 1
			im[i] = 0
		}
		FFT(re, im)
		if math.Abs(re[0]-float64(n)) > 1e-9 || math.Abs(im[0]) > 1e-9 {
			t.Fatalf("n=%d constant: bin 0 = %v%+vi, want %d", n, re[0], im[0], n)
		}
		for k := 1; k < n; k++ {
			if math.Abs(re[k]) > 1e-9 || math.Abs(im[k]) > 1e-9 {
				t.Fatalf("n=%d constant: bin %d = %v%+vi, want 0", n, k, re[k], im[k])
			}
		}
	}
}

func TestIFFTRoundTrip(t *testing.T) {
	for _, n := range []int{5, 400, 1280} {
		re, im := randomPair(n, uint32(777+n))
		origRe := append([]float64(nil), re...)
		origIm := append([]float64(nil), im...)
		FFT(re, im)
		IFFT(re, im)
		if d := maxAbsDiff(re, origRe); d > 1e-9 {
			t.Fatalf("n=%d: IFFT(FFT(x)) re diff %g", n, d)
		}
		if d := maxAbsDiff(im, origIm); d > 1e-9 {
			t.Fatalf("n=%d: IFFT(FFT(x)) im diff %g", n, d)
		}
	}
}

func TestFFTMatchesNaiveDFTComplex(t *testing.T) {
	// pure tone: x[t] = exp(2*pi*i*f*t/n) lands entirely in bin f
	n, f := 1280, 37
	re := make([]float64, n)
	im := make([]float64, n)
	for tt := range n {
		ang := 2 * math.Pi * float64(f) * float64(tt) / float64(n)
		c := cmplx.Exp(complex(0, ang))
		re[tt], im[tt] = real(c), imag(c)
	}
	FFT(re, im)
	for k := range n {
		want := 0.0
		if k == f {
			want = float64(n)
		}
		if math.Abs(re[k]-want) > 1e-8 || math.Abs(im[k]) > 1e-8 {
			t.Fatalf("bin %d = %v%+vi, want %v", k, re[k], im[k], want)
		}
	}
}

func BenchmarkFFT400(b *testing.B) {
	re, im := randomPair(400, 1)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		FFT(re, im)
	}
}

func BenchmarkFFT1280(b *testing.B) {
	re, im := randomPair(1280, 1)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		FFT(re, im)
	}
}
