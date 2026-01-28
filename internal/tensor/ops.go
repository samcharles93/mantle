package tensor

import (
	"math"

	"simd/archsimd"
)

// Add adds src to dst element-wise.
func Add(dst, src []float32) {
	if cpu.HasAVX2 {
		addSIMD(dst, src)
		return
	}
	addScalar(dst, src)
}

// addScalar adds src to dst element-wise using scalar operations.
func addScalar(dst, src []float32) {
	for i := range dst {
		dst[i] += src[i]
	}
}

// addSIMD adds src to dst element-wise using AVX2 SIMD.
func addSIMD(dst, src []float32) {
	n := len(dst)
	i := 0
	// Process 8 elements at a time
	for ; i+8 <= n; i += 8 {
		vd := archsimd.LoadFloat32x8Slice(dst[i:])
		vs := archsimd.LoadFloat32x8Slice(src[i:])
		vd = vd.Add(vs)
		vd.StoreSlice(dst[i:])
	}
	// Handle remaining elements
	for ; i < n; i++ {
		dst[i] += src[i]
	}
}

// Dot computes the dot product of a and b.
func Dot(a, b []float32) float32 {
	if cpu.HasAVX2 {
		return dotSIMD(a, b)
	}
	return dotScalar(a, b)
}

// dotScalar computes the dot product using scalar operations.
func dotScalar(a, b []float32) float32 {
	var sum float32
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

// dotSIMD computes the dot product using AVX2 SIMD.
// Uses a single accumulator to minimize register pressure.
func dotSIMD(a, b []float32) float32 {
	n := len(a)
	if n == 0 {
		return 0
	}

	// Single accumulator - reduces register pressure
	var acc archsimd.Float32x8

	i := 0
	// Process 8 elements at a time
	for ; i+8 <= n; i += 8 {
		va := archsimd.LoadFloat32x8Slice(a[i:])
		vb := archsimd.LoadFloat32x8Slice(b[i:])
		acc = acc.Add(va.Mul(vb))
	}

	// Horizontal reduction using AddPairsGrouped
	zero := archsimd.BroadcastFloat32x8(0)
	pairs := acc.AddPairsGrouped(zero) // [(a+b), (c+d), (e+f), (g+h), 0, 0, 0, 0]
	lo := pairs.GetLo() // [(a+b), (c+d), (e+f), (g+h)]
	sum := lo.GetElem(0) + lo.GetElem(1) + lo.GetElem(2) + lo.GetElem(3)

	// Handle remaining elements with scalar
	for ; i < n; i++ {
		sum += a[i] * b[i]
	}
	return sum
}

// RMSNorm performs Root Mean Square Normalization.
func RMSNorm(dst, src, weight []float32, eps float32) {
	if cpu.HasAVX2 {
		rmsNormSIMD(dst, src, weight, eps)
		return
	}
	rmsNormScalar(dst, src, weight, eps)
}

// rmsNormScalar performs Root Mean Square Normalization using scalar operations.
func rmsNormScalar(dst, src, weight []float32, eps float32) {
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

// rmsNormSIMD performs Root Mean Square Normalization using AVX2 SIMD.
func rmsNormSIMD(dst, src, weight []float32, eps float32) {
	n := len(src)
	if n == 0 {
		return
	}

	// Single accumulator for sum of squares
	var acc archsimd.Float32x8
	i := 0
	for ; i+8 <= n; i += 8 {
		v := archsimd.LoadFloat32x8Slice(src[i:])
		acc = acc.Add(v.Mul(v))
	}

	// Horizontal reduction: store to array and sum scalarly
	// This is faster than calling GetElem 4 times
	var tmp [8]float32
	acc.Store(&tmp)
	sum := tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7]

	// Handle remaining elements
	for ; i < n; i++ {
		sum += src[i] * src[i]
	}

	mean := sum / float32(n)
	scale := float32(1.0) / float32(math.Sqrt(float64(mean+eps)))

	// Apply scale to dst using SIMD
	vscale := archsimd.BroadcastFloat32x8(scale)
	i = 0
	for ; i+8 <= n; i += 8 {
		vsrc := archsimd.LoadFloat32x8Slice(src[i:])
		vw := archsimd.LoadFloat32x8Slice(weight[i:])
		v := vsrc.Mul(vscale)
		v = v.Mul(vw)
		v.StoreSlice(dst[i:])
	}
	for ; i < n; i++ {
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
// headDim must be even. attentionFactor scales the cos/sin components.
func ApplyRoPE(x []float32, nHead, headDim, pos int, invFreq []float64, attentionFactor float32) {
	if headDim%2 != 0 {
		panic("headDim must be even for RoPE")
	}
	if attentionFactor == 0 {
		attentionFactor = 1
	}
	for h := 0; h < nHead; h++ {
		base := h * headDim
		for i := 0; i < headDim/2; i++ {
			angle := float64(pos) * invFreq[i]
			c := float32(math.Cos(angle)) * attentionFactor
			s := float32(math.Sin(angle)) * attentionFactor
			i0 := base + 2*i
			i1 := i0 + 1
			x0 := x[i0]
			x1 := x[i1]
			x[i0] = x0*c - x1*s
			x[i1] = x0*s + x1*c
		}
	}
}