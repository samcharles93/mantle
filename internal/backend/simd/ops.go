package simd

import (
	"math"
)

// Add adds src to dst element-wise.
func Add(dst, src []float32) {
	if cpu.HasAVX512 {
		addAVX512(dst, src)
	} else if cpu.HasAVX2 {
		addSIMD(dst, src)
	} else {
		addScalar(dst, src)
	}
}

// addScalar adds src to dst element-wise using scalar operations.
func addScalar(dst, src []float32) {
	for i := range dst {
		dst[i] += src[i]
	}
}

// Dot computes the dot product of a and b.
func Dot(a, b []float32) float32 {
	if cpu.HasAVX512 {
		return dotAVX512(a, b)
	} else if cpu.HasAVX2 {
		return dotSIMD(a, b)
	}
	return dotScalar(a, b)
}

func DotF16(a []float32, b []uint16) float32 {
	// F16 dot currently uses scalar conversion.
	return dotScalarF16(a, b)
}

func dotScalarF16(a []float32, b []uint16) float32 {
	var sum float32
	for i, v := range a {
		sum += v * Float16ToFloat32(b[i])
	}
	return sum
}

// dotScalar computes the dot product using scalar operations.
func dotScalar(a, b []float32) float32 {
	var sum float32
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

// RMSNorm performs Root Mean Square Normalization.
func RMSNorm(dst, src, weight []float32, eps float32) {
	if cpu.HasAVX512 {
		rmsNormAVX512(dst, src, weight, eps)
		return
	} else if cpu.HasAVX2 {
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
	var sum float32
	for i := range x {
		v := fastExp(x[i] - maxv)
		x[i] = v
		sum += v
	}
	if sum == 0 {
		return
	}
	inv := 1.0 / sum
	for i := range x {
		x[i] *= inv
	}
}

// fastExp computes an approximation of exp(x) using polynomial approximation.
// Accurate for x in [-10, 10] with relative error < 0.1%.
// For neural network inference, this is sufficient.
func fastExp(x float32) float32 {
	// Clamp to safe range to avoid overflow
	if x > 88.0 {
		return 3.4028235e38 // Close to float32 max
	}
	if x < -88.0 {
		return 0.0
	}

	// Range reduction: exp(x) = 2^k * exp(r)
	// where x = k*ln(2) + r, and -ln(2)/2 <= r < ln(2)/2
	const ln2 = 0.693147180559945309417
	const ln2Inv = 1.44269504088896340736

	// k = round(x / ln(2))
	k := int32(x*ln2Inv + 0.5)
	if x < 0 {
		k = int32(x*ln2Inv - 0.5)
	}

	// r = x - k*ln(2)
	r := x - float32(k)*ln2

	// Compute exp(r) using 5th order polynomial (minimax approximation)
	// exp(r) ≈ 1 + r + r²/2 + r³/6 + r⁴/24 + r⁵/120
	r2 := r * r
	r3 := r2 * r
	r4 := r2 * r2

	poly := 1.0 + r + r2*0.5 + r3*0.16666667 + r4*0.041666667 + (r4*r)*0.008333333

	// Compute 2^k using bit manipulation
	// float32 bits: sign(1) | exponent(8) | mantissa(23)
	// For 2^k: exponent = k + 127 (bias)
	exp := uint32(k+127) << 23
	scale := math.Float32frombits(exp)

	return poly * scale
}

// fastSigmoid computes an approximation of sigmoid using fastExp.
func fastSigmoid(x float32) float32 {
	// For large |x|, sigmoid saturates
	if x > 10.0 {
		return 1.0
	}
	if x < -10.0 {
		return 0.0
	}
	return 1.0 / (1.0 + fastExp(-x))
}

// fastTanh computes an approximation of tanh(x) using fastExp.
func fastTanh(x float32) float32 {
	if x > 10.0 {
		return 1.0
	}
	if x < -10.0 {
		return -1.0
	}
	// tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
	e2x := fastExp(2.0 * x)
	return (e2x - 1.0) / (e2x + 1.0)
}

// fastSilu computes silu(x) = x * sigmoid(x) using fast approximation.
func fastSilu(x float32) float32 {
	return x * fastSigmoid(x)
}

// Sigmoid computes the logistic sigmoid activation.
func Sigmoid(x float32) float32 {
	// Use fast approximation for speed
	// If you need exact results, change to: return float32(1.0 / (1.0 + math.Exp(float64(-x))))
	return fastSigmoid(x)
}

// Silu computes the Sigmoid Linear Unit (SiLU) activation.
func Silu(x float32) float32 {
	// Use fast approximation for speed
	return fastSilu(x)
}

// Gelu computes the Gaussian Error Linear Unit activation using the tanh
// approximation matching PyTorch's gelu_pytorch_tanh:
//
//	0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
func Gelu(x float32) float32 {
	const sqrt2OverPi = 0.7978845608028654 // sqrt(2/pi)
	inner := sqrt2OverPi * (x + 0.044715*x*x*x)
	return 0.5 * x * (1 + fastTanh(inner))
}

// Softplus computes log(1+exp(x)) in a numerically stable way.
func Softplus(x float32) float32 {
	if x > 20 {
		return x
	}
	if x < -20 {
		return fastExp(x)
	}
	return float32(math.Log1p(float64(fastExp(x))))
}

// RMSNormGated applies SiLU gating to src using gate and then RMS-normalizes.
// If normBeforeGate is true, normalization is applied before gating.
func RMSNormGated(dst, src, gate, weight []float32, eps float32, normBeforeGate bool) {
	if len(src) != len(gate) || len(src) != len(weight) {
		panic("RMSNormGated input sizes do not match")
	}
	if len(dst) < len(src) {
		panic("RMSNormGated dst too small")
	}
	if normBeforeGate {
		RMSNorm(dst, src, weight, eps)
		for i := range src {
			dst[i] *= Silu(gate[i])
		}
		return
	}
	for i := range src {
		dst[i] = src[i] * Silu(gate[i])
	}
	RMSNorm(dst, dst, weight, eps)
}

// DotQ8 computes the dot product of float32 vector a and quantized int8 vector b,
// scaled by the per-position scale factor.
func DotQ8(a []float32, b []int8, scale float32) float32 {
	var sum float32
	for i, v := range a {
		sum += v * float32(b[i])
	}
	return sum * scale
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
	half := headDim / 2
	if cpu.HasAVX512 && half >= 16 {
		applyRoPEAVX512(x, nHead, headDim, pos, invFreq, attentionFactor, half)
		return
	} else if cpu.HasAVX2 && half >= 8 {
		applyRoPESIMD(x, nHead, headDim, pos, invFreq, attentionFactor, half)
		return
	}
	applyRoPEScalar(x, nHead, headDim, pos, invFreq, attentionFactor, half)
}

func applyRoPEScalar(x []float32, nHead, headDim, pos int, invFreq []float64, attentionFactor float32, half int) {
	for h := range nHead {
		base := h * headDim
		for i := range half {
			angle := float64(pos) * invFreq[i]
			c := float32(math.Cos(angle)) * attentionFactor
			s := float32(math.Sin(angle)) * attentionFactor
			i0 := base + i
			i1 := base + i + half
			x0 := x[i0]
			x1 := x[i1]
			x[i0] = x0*c - x1*s
			x[i1] = x0*s + x1*c
		}
	}
}
