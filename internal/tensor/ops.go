package tensor

import (
	"math"
)

// Add adds src to dst element-wise.
func Add(dst, src []float32) {
	for i := range dst {
		dst[i] += src[i]
	}
}

// Dot computes the dot product of a and b.
func Dot(a, b []float32) float32 {
	var sum float32
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

// RMSNorm performs Root Mean Square Normalization.
func RMSNorm(dst, src, weight []float32, eps float32) {
	var sum float32
	for _, v := range src {
		sum += v * v
	}
	mean := sum / float32(len(src))
	scale := float32(1.0) / float32(math.Sqrt(float64(mean+eps)))
	for i := range src {
		dst[i] = src[i] * scale * weight[i]
	}
}

// Softmax applies the softmax function to x.
func Softmax(x []float32) {
	if len(x) == 0 {
		return
	}
	maxv := x[0]
	for i := 1; i < len(x); i++ {
		if x[i] > maxv {
			maxv = x[i]
		}
	}
	var sum float64
	for i := range x {
		v := math.Exp(float64(x[i] - maxv))
		x[i] = float32(v)
		sum += v
	}
	if sum == 0 {
		return
	}
	inv := float32(1.0 / sum)
	for i := range x {
		x[i] *= inv
	}
}

// Sigmoid computes the logistic sigmoid activation.
func Sigmoid(x float32) float32 {
	return float32(1.0 / (1.0 + math.Exp(float64(-x))))
}

// Silu computes the Sigmoid Linear Unit (SiLU) activation.
func Silu(x float32) float32 {
	return x * Sigmoid(x)
}

// SiluAndMul computes dst[i] = Silu(x[i]) * x[d+i] where d = len(x)/2.
// dst must have length d and x must have even length.
func SiluAndMul(dst, x []float32) {
	if len(x)%2 != 0 {
		panic("SiluAndMul requires even-length input")
	}
	d := len(x) / 2
	if len(dst) < d {
		panic("SiluAndMul dst too small")
	}
	for i := range d {
		dst[i] = Silu(x[i]) * x[d+i]
	}
}

// ApplyRoPE applies Rotary Positional Embeddings to x.
// headDim must be even.
func ApplyRoPE(x []float32, nHead, headDim, pos int, invFreq []float64) {
	if headDim%2 != 0 {
		panic("headDim must be even for RoPE")
	}
	for h := 0; h < nHead; h++ {
		base := h * headDim
		for i := 0; i < headDim/2; i++ {
			angle := float64(pos) * invFreq[i]
			c := float32(math.Cos(angle))
			s := float32(math.Sin(angle))
			i0 := base + 2*i
			i1 := i0 + 1
			x0 := x[i0]
			x1 := x[i1]
			x[i0] = x0*c - x1*s
			x[i1] = x0*s + x1*c
		}
	}
}
