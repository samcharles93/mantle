//go:build goexperiment.simd

package simd

import (
	"math"

	"simd/archsimd"
)

// addSIMD adds src to dst element-wise using AVX2 SIMD.
func addSIMD(dst, src []float32) {
	n := len(dst)
	i := 0
	for ; i+8 <= n; i += 8 {
		vd := archsimd.LoadFloat32x8Slice(dst[i:])
		vs := archsimd.LoadFloat32x8Slice(src[i:])
		vd = vd.Add(vs)
		vd.StoreSlice(dst[i:])
	}
	for ; i < n; i++ {
		dst[i] += src[i]
	}
}

// dotSIMD computes the dot product using AVX2 SIMD.
// Uses multiple accumulators to better utilize FMA units.
func dotSIMD(a, b []float32) float32 {
	n := len(a)
	if n == 0 {
		return 0
	}

	var acc0, acc1 archsimd.Float32x8

	i := 0
	for ; i+16 <= n; i += 16 {
		va0 := archsimd.LoadFloat32x8Slice(a[i:])
		vb0 := archsimd.LoadFloat32x8Slice(b[i:])
		acc0 = va0.MulAdd(vb0, acc0)

		va1 := archsimd.LoadFloat32x8Slice(a[i+8:])
		vb1 := archsimd.LoadFloat32x8Slice(b[i+8:])
		acc1 = va1.MulAdd(vb1, acc1)
	}

	for ; i+8 <= n; i += 8 {
		va := archsimd.LoadFloat32x8Slice(a[i:])
		vb := archsimd.LoadFloat32x8Slice(b[i:])
		acc0 = va.MulAdd(vb, acc0)
	}

	var tmp0, tmp1 [8]float32
	acc0.Store(&tmp0)
	acc1.Store(&tmp1)

	sum := tmp0[0] + tmp0[1] + tmp0[2] + tmp0[3] + tmp0[4] + tmp0[5] + tmp0[6] + tmp0[7] +
		tmp1[0] + tmp1[1] + tmp1[2] + tmp1[3] + tmp1[4] + tmp1[5] + tmp1[6] + tmp1[7]

	for ; i < n; i++ {
		sum += a[i] * b[i]
	}
	return sum
}

// rmsNormSIMD performs Root Mean Square Normalization using AVX2 SIMD.
func rmsNormSIMD(dst, src, weight []float32, eps float32) {
	n := len(src)
	if n == 0 {
		return
	}

	var acc archsimd.Float32x8
	i := 0
	for ; i+8 <= n; i += 8 {
		v := archsimd.LoadFloat32x8Slice(src[i:])
		acc = v.MulAdd(v, acc)
	}

	var tmp [8]float32
	acc.Store(&tmp)
	sum := tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7]

	for ; i < n; i++ {
		sum += src[i] * src[i]
	}

	mean := sum / float32(n)
	scale := float32(1.0) / float32(math.Sqrt(float64(mean+eps)))

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

// fastExpVec computes exp(x) for a Float32x8 vector using polynomial approximation.
func fastExpVec(x archsimd.Float32x8) archsimd.Float32x8 {
	ln2 := archsimd.BroadcastFloat32x8(0.693147180559945309417)
	ln2Inv := archsimd.BroadcastFloat32x8(1.44269504088896340736)
	maxVal := archsimd.BroadcastFloat32x8(88.0)
	minVal := archsimd.BroadcastFloat32x8(-88.0)
	one := archsimd.BroadcastFloat32x8(1.0)
	c2 := archsimd.BroadcastFloat32x8(0.5)
	c3 := archsimd.BroadcastFloat32x8(0.16666667)
	c4 := archsimd.BroadcastFloat32x8(0.041666667)
	c5 := archsimd.BroadcastFloat32x8(0.008333333)

	x = x.Max(minVal).Min(maxVal)

	k := x.Mul(ln2Inv).RoundToEven().ConvertToInt32()

	kf := k.ConvertToFloat32()
	r := x.Sub(kf.Mul(ln2))

	r2 := r.Mul(r)
	r3 := r2.Mul(r)
	r4 := r2.Mul(r2)
	r5 := r4.Mul(r)

	poly := one.Add(r).Add(r2.Mul(c2)).Add(r3.Mul(c3)).Add(r4.Mul(c4)).Add(r5.Mul(c5))

	bias := archsimd.BroadcastInt32x8(127)
	exp := k.Add(bias).ShiftAllLeft(23)
	scale := exp.AsFloat32x8()

	return poly.Mul(scale)
}

// fastSigmoidVec computes sigmoid for a Float32x8 vector.
func fastSigmoidVec(x archsimd.Float32x8) archsimd.Float32x8 {
	one := archsimd.BroadcastFloat32x8(1.0)
	negX := archsimd.BroadcastFloat32x8(0.0).Sub(x)
	expNegX := fastExpVec(negX)
	return one.Div(one.Add(expNegX))
}

// fastSiluVec computes silu for a Float32x8 vector.
func fastSiluVec(x archsimd.Float32x8) archsimd.Float32x8 {
	return x.Mul(fastSigmoidVec(x))
}

// fastTanhVec computes tanh for a Float32x8 vector.
// tanh(x) = 2*sigmoid(2x) - 1
func fastTanhVec(x archsimd.Float32x8) archsimd.Float32x8 {
	two := archsimd.BroadcastFloat32x8(2.0)
	one := archsimd.BroadcastFloat32x8(1.0)
	return fastSigmoidVec(x.Mul(two)).Mul(two).Sub(one)
}

// fastGeluVec computes GELU using the tanh approximation for a Float32x8 vector.
// 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
func fastGeluVec(x archsimd.Float32x8) archsimd.Float32x8 {
	sqrt2OverPi := archsimd.BroadcastFloat32x8(0.7978845608028654)
	coeff := archsimd.BroadcastFloat32x8(0.044715)
	half := archsimd.BroadcastFloat32x8(0.5)
	one := archsimd.BroadcastFloat32x8(1.0)

	x3 := x.Mul(x).Mul(x)
	inner := sqrt2OverPi.Mul(x.Add(coeff.Mul(x3)))
	return half.Mul(x).Mul(one.Add(fastTanhVec(inner)))
}

func siluAndMulSIMD(dst, x []float32) {
	d := len(x) / 2
	i := 0
	for ; i+8 <= d; i += 8 {
		vgate := archsimd.LoadFloat32x8Slice(x[i:])
		vup := archsimd.LoadFloat32x8Slice(x[d+i:])
		vsilu := fastSiluVec(vgate)
		vresult := vsilu.Mul(vup)
		var tmp [8]float32
		vresult.Store(&tmp)
		copy(dst[i:], tmp[:])
	}
	for ; i < d; i++ {
		dst[i] = Silu(x[i]) * x[d+i]
	}
}

// applyRoPESIMDWithTables applies Rotary Positional Embeddings using precomputed cosine and sine values with AVX2 SIMD.
func applyRoPESIMDWithTables(x []float32, nHead, headDim, pos int, cosTable, sinTable []float32, half int) {
	for h := range nHead {
		base := h * headDim
		tableOffset := pos * half
		lo := x[base : base+half]
		hi := x[base+half : base+headDim]

		i := 0
		for ; i+8 <= half; i += 8 {
			x0 := archsimd.LoadFloat32x8Slice(lo[i:])
			x1 := archsimd.LoadFloat32x8Slice(hi[i:])
			c := archsimd.LoadFloat32x8Slice(cosTable[tableOffset+i:])
			s := archsimd.LoadFloat32x8Slice(sinTable[tableOffset+i:])

			y0 := x0.Mul(c).Sub(x1.Mul(s))
			y1 := x0.MulAdd(s, x1.Mul(c))

			y0.StoreSlice(lo[i:])
			y1.StoreSlice(hi[i:])
		}
		for ; i < half; i++ {
			x0 := lo[i]
			x1 := hi[i]
			c := cosTable[tableOffset+i]
			s := sinTable[tableOffset+i]
			lo[i] = x0*c - x1*s
			hi[i] = x0*s + x1*c
		}
	}
}

func ApplyRoPESIMD(x []float32, nHead, headDim, pos int, invFreq []float64, attentionFactor float32) {
	if headDim%2 != 0 {
		panic("headDim must be even for RoPE")
	}
	half := len(invFreq)
	if half == 0 {
		return
	}
	if half*2 > headDim {
		panic("RoPE rotary dim exceeds headDim")
	}
	if attentionFactor == 0 {
		attentionFactor = 1
	}
	applyRoPESIMD(x, nHead, headDim, pos, invFreq, attentionFactor, half)
}

func applyRoPESIMD(x []float32, nHead, headDim, pos int, invFreq []float64, attentionFactor float32, half int) {
	const ropeTrigStack = 256
	var cosStack [ropeTrigStack]float32
	var sinStack [ropeTrigStack]float32

	var cosVals []float32
	var sinVals []float32
	if half <= ropeTrigStack {
		cosVals = cosStack[:half]
		sinVals = sinStack[:half]
	} else {
		cosVals = make([]float32, half)
		sinVals = make([]float32, half)
	}

	pos64 := float64(pos)
	if attentionFactor == 1 {
		for i := range half {
			angle := pos64 * invFreq[i]
			cosVals[i] = float32(math.Cos(angle))
			sinVals[i] = float32(math.Sin(angle))
		}
	} else {
		for i := range half {
			angle := pos64 * invFreq[i]
			cosVals[i] = float32(math.Cos(angle)) * attentionFactor
			sinVals[i] = float32(math.Sin(angle)) * attentionFactor
		}
	}

	for h := range nHead {
		base := h * headDim
		lo := x[base : base+half]
		hi := x[base+half : base+half+half]

		i := 0
		for ; i+8 <= half; i += 8 {
			x0 := archsimd.LoadFloat32x8Slice(lo[i:])
			x1 := archsimd.LoadFloat32x8Slice(hi[i:])
			c := archsimd.LoadFloat32x8Slice(cosVals[i:])
			s := archsimd.LoadFloat32x8Slice(sinVals[i:])

			y0 := x0.Mul(c).Sub(x1.Mul(s))
			y1 := x0.MulAdd(s, x1.Mul(c))

			y0.StoreSlice(lo[i:])
			y1.StoreSlice(hi[i:])
		}
		for ; i < half; i++ {
			x0 := lo[i]
			x1 := hi[i]
			c := cosVals[i]
			s := sinVals[i]
			lo[i] = x0*c - x1*s
			hi[i] = x0*s + x1*c
		}
	}
}
