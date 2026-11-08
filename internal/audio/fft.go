package audio

import "math"

// Plan is a precomputed FFT plan for one size: the twiddle table
// W_n^k = exp(-2*pi*i*k/n) makes every transform allocation-light and trig-free
// after the plan is built. Plans are safe for concurrent use.
type Plan struct {
	n        int
	twiddles []complex128 // twiddles[k] = exp(-2*pi*i*k/n), k = 0..n-1
}

// NewPlan builds the FFT plan for size n (any n >= 1).
func NewPlan(n int) *Plan {
	if n < 1 {
		panic("audio: FFT plan size must be >= 1")
	}
	p := &Plan{n: n, twiddles: make([]complex128, n)}
	for k := range n {
		ang := -2 * math.Pi * float64(k) / float64(n)
		p.twiddles[k] = complex(math.Cos(ang), math.Sin(ang))
	}
	return p
}

// FFT computes the forward discrete Fourier transform in place:
//
//	X[k] = sum_t x[t] * exp(-2*pi*i*k*t/n)
//
// (unnormalized, matching numpy/torch.fft.fft). It is equivalent to
// NewPlan(len(re)).FFT(re, im).
func FFT(re, im []float64) {
	NewPlan(len(re)).FFT(re, im)
}

// IFFT computes the inverse transform in place, including the 1/n scaling:
// x[t] = (1/n) * sum_k X[k] * exp(+2*pi*i*k*t/n).
func IFFT(re, im []float64) {
	NewPlan(len(re)).IFFT(re, im)
}

// FFT is the forward transform of the plan (see the package-level FFT).
func (p *Plan) FFT(re, im []float64) {
	if len(re) != p.n || len(im) != p.n {
		panic("audio: FFT input length does not match plan size")
	}
	p.fftRec(re, im, false, 1)
}

// IFFT is the inverse transform of the plan, including the 1/n scaling.
func (p *Plan) IFFT(re, im []float64) {
	if len(re) != p.n || len(im) != p.n {
		panic("audio: IFFT input length does not match plan size")
	}
	p.fftRec(re, im, true, 1)
	inv := 1.0 / float64(p.n)
	for i := range re {
		re[i] *= inv
		im[i] *= inv
	}
}

// fftRec is the mixed-radix decimation-in-time Cooley-Tukey transform:
// composite sizes split off their smallest factor recursively, prime sizes
// fall back to a direct DFT. This covers the sizes the Omni audio stack needs,
// 400 = 2^4*5^2 and 1280 = 2^8*5, which are not powers of two.
//
// twStep is the stride into the plan's twiddle table for the current level:
// a level of size m uses W_m^k = twiddles[k*(p.n/m)], so twStep = p.n/m and
// doubles... multiplies by the split factor at each recursion level.
func (p *Plan) fftRec(re, im []float64, inverse bool, twStep int) {
	n := len(re)
	if n <= 1 {
		return
	}
	f := smallestFactor(n)
	if f == n {
		p.dftNaive(re, im, inverse, twStep)
		return
	}
	n2 := n / f
	// decimated subsequences: sub_j[m] = x[j + f*m]
	subsRe := make([][]float64, f)
	subsIm := make([][]float64, f)
	subStep := twStep * f
	for j := range f {
		sr := make([]float64, n2)
		si := make([]float64, n2)
		for m := range n2 {
			sr[m] = re[j+f*m]
			si[m] = im[j+f*m]
		}
		p.fftRec(sr, si, inverse, subStep)
		subsRe[j] = sr
		subsIm[j] = si
	}
	// combine: X[k] = sum_j W_n^(j*k) * X_j[k mod n2]
	outRe := make([]float64, n)
	outIm := make([]float64, n)
	for k := range n {
		km := k % n2
		var sr, si float64
		for j := range f {
			idx := ((j * k) % n) * twStep
			w := p.twiddles[idx]
			if inverse {
				w = complex(real(w), -imag(w)) // conjugate = exp(+2*pi*i*idx/n)
			}
			xr, xi := subsRe[j][km], subsIm[j][km]
			sr += real(w)*xr - imag(w)*xi
			si += real(w)*xi + imag(w)*xr
		}
		outRe[k] = sr
		outIm[k] = si
	}
	copy(re, outRe)
	copy(im, outIm)
}

// smallestFactor returns the smallest divisor of n greater than 1, or n itself
// when n is prime.
func smallestFactor(n int) int {
	for i := 2; i*i <= n; i++ {
		if n%i == 0 {
			return i
		}
	}
	return n
}

// dftNaive is the direct O(n^2) DFT used as the base case for prime sizes.
func (p *Plan) dftNaive(re, im []float64, inverse bool, twStep int) {
	n := len(re)
	outRe := make([]float64, n)
	outIm := make([]float64, n)
	for k := range n {
		var sr, si float64
		for t := range n {
			idx := ((k * t) % n) * twStep
			w := p.twiddles[idx]
			if inverse {
				w = complex(real(w), -imag(w))
			}
			sr += real(w)*re[t] - imag(w)*im[t]
			si += real(w)*im[t] + imag(w)*re[t]
		}
		outRe[k] = sr
		outIm[k] = si
	}
	copy(re, outRe)
	copy(im, outIm)
}
