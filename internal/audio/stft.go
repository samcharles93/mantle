package audio

import "math"

// HannWindow returns the periodic Hann window of length n, matching
// torch.hann_window(n) with its default periodic=True:
//
//	w[i] = 0.5 - 0.5*cos(2*pi*i/n), i = 0..n-1
//
// (A symmetric window would divide by n-1; torch defaults to n.)
func HannWindow(n int) []float64 {
	w := make([]float64, n)
	for i := range n {
		w[i] = 0.5 - 0.5*math.Cos(2*math.Pi*float64(i)/float64(n))
	}
	return w
}

// STFT computes the short-time Fourier transform of x with torch.stft
// center=False semantics: frame t covers samples x[t*hop : t*hop+nFFT] with no
// padding or reflection, and is multiplied by window (length winLength, placed
// centered inside the nFFT frame when winLength < nFFT, as torch.stft pads it).
//
// The result has nFFT/2+1 rows (bins 0..nFFT/2, like rfft) and
// max(0, 1+(len(x)-nFFT)/hop) columns (frames). spec[bin][frame].
func STFT(x []float64, nFFT, hop, winLength int, window []float64) [][]complex128 {
	if nFFT <= 0 || hop <= 0 {
		panic("audio: STFT requires nFFT > 0 and hop > 0")
	}
	if len(window) != winLength || winLength > nFFT {
		panic("audio: STFT requires len(window) == winLength <= nFFT")
	}
	bins := nFFT/2 + 1
	frames := 0
	if len(x) >= nFFT {
		frames = 1 + (len(x)-nFFT)/hop
	}
	spec := make([][]complex128, bins)
	for k := range spec {
		spec[k] = make([]complex128, frames)
	}
	winOff := (nFFT - winLength) / 2 // torch pads a short window centered
	plan := NewPlan(nFFT)            // one twiddle table for all frames
	re := make([]float64, nFFT)
	im := make([]float64, nFFT)
	for t := 0; t < frames; t++ {
		start := t * hop
		for i := range nFFT {
			re[i] = 0
			im[i] = 0
			wi := i - winOff
			if wi >= 0 && wi < winLength {
				re[i] = x[start+i] * window[wi]
			}
		}
		plan.FFT(re, im)
		for k := range bins {
			spec[k][t] = complex(re[k], im[k])
		}
	}
	return spec
}
