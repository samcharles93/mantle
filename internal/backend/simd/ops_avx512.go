//go:build goexperiment.simd

package simd

import (
	"math"

	"simd/archsimd"
)

// addAVX512 adds src to dst element-wise using AVX-512 SIMD.
func addAVX512(dst, src []float32) {
	n := len(dst)
	i := 0
	for ; i+16 <= n; i += 16 {
		vd := archsimd.LoadFloat32x16Slice(dst[i:])
		vs := archsimd.LoadFloat32x16Slice(src[i:])
		vd = vd.Add(vs)
		vd.StoreSlice(dst[i:])
	}
	for ; i < n; i++ {
		dst[i] += src[i]
	}
}

// dotAVX512 computes the dot product using AVX-512 SIMD.
// Uses multiple accumulators to better utilize FMA units.
func dotAVX512(a, b []float32) float32 {
	n := len(a)
	if n == 0 {
		return 0
	}

	var acc0, acc1, acc2, acc3 archsimd.Float32x16

	i := 0
	for ; i+64 <= n; i += 64 {
		va0 := archsimd.LoadFloat32x16Slice(a[i:])
		vb0 := archsimd.LoadFloat32x16Slice(b[i:])
		acc0 = va0.MulAdd(vb0, acc0)

		va1 := archsimd.LoadFloat32x16Slice(a[i+16:])
		vb1 := archsimd.LoadFloat32x16Slice(b[i+16:])
		acc1 = va1.MulAdd(vb1, acc1)

		va2 := archsimd.LoadFloat32x16Slice(a[i+32:])
		vb2 := archsimd.LoadFloat32x16Slice(b[i+32:])
		acc2 = va2.MulAdd(vb2, acc2)

		va3 := archsimd.LoadFloat32x16Slice(a[i+48:])
		vb3 := archsimd.LoadFloat32x16Slice(b[i+48:])
		acc3 = va3.MulAdd(vb3, acc3)
	}

	for ; i+32 <= n; i += 32 {
		va0 := archsimd.LoadFloat32x16Slice(a[i:])
		vb0 := archsimd.LoadFloat32x16Slice(b[i:])
		acc0 = va0.MulAdd(vb0, acc0)

		va1 := archsimd.LoadFloat32x16Slice(a[i+16:])
		vb1 := archsimd.LoadFloat32x16Slice(b[i+16:])
		acc1 = va1.MulAdd(vb1, acc1)
	}

	for ; i+16 <= n; i += 16 {
		va := archsimd.LoadFloat32x16Slice(a[i:])
		vb := archsimd.LoadFloat32x16Slice(b[i:])
		acc0 = va.MulAdd(vb, acc0)
	}

	var tmp0, tmp1, tmp2, tmp3 [16]float32
	acc0.Store(&tmp0)
	acc1.Store(&tmp1)
	acc2.Store(&tmp2)
	acc3.Store(&tmp3)

	sum := tmp0[0] + tmp0[1] + tmp0[2] + tmp0[3] + tmp0[4] + tmp0[5] + tmp0[6] + tmp0[7] +
		tmp0[8] + tmp0[9] + tmp0[10] + tmp0[11] + tmp0[12] + tmp0[13] + tmp0[14] + tmp0[15] +
		tmp1[0] + tmp1[1] + tmp1[2] + tmp1[3] + tmp1[4] + tmp1[5] + tmp1[6] + tmp1[7] +
		tmp1[8] + tmp1[9] + tmp1[10] + tmp1[11] + tmp1[12] + tmp1[13] + tmp1[14] + tmp1[15] +
		tmp2[0] + tmp2[1] + tmp2[2] + tmp2[3] + tmp2[4] + tmp2[5] + tmp2[6] + tmp2[7] +
		tmp2[8] + tmp2[9] + tmp2[10] + tmp2[11] + tmp2[12] + tmp2[13] + tmp2[14] + tmp2[15] +
		tmp3[0] + tmp3[1] + tmp3[2] + tmp3[3] + tmp3[4] + tmp3[5] + tmp3[6] + tmp3[7] +
		tmp3[8] + tmp3[9] + tmp3[10] + tmp3[11] + tmp3[12] + tmp3[13] + tmp3[14] + tmp3[15]

	for ; i < n; i++ {
		sum += a[i] * b[i]
	}
	return sum
}

// rmsNormAVX512 performs Root Mean Square Normalization using AVX-512 SIMD.
func rmsNormAVX512(dst, src, weight []float32, eps float32) {
	n := len(src)
	if n == 0 {
		return
	}

	var acc archsimd.Float32x16
	i := 0
	for ; i+16 <= n; i += 16 {
		v := archsimd.LoadFloat32x16Slice(src[i:])
		acc = v.MulAdd(v, acc)
	}

	var tmp [16]float32
	acc.Store(&tmp)
	sum := tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7] +
		tmp[8] + tmp[9] + tmp[10] + tmp[11] + tmp[12] + tmp[13] + tmp[14] + tmp[15]

	for ; i < n; i++ {
		sum += src[i] * src[i]
	}

	mean := sum / float32(n)
	scale := float32(1.0) / float32(math.Sqrt(float64(mean+eps)))

	vscale := archsimd.BroadcastFloat32x16(scale)
	i = 0
	for ; i+16 <= n; i += 16 {
		vsrc := archsimd.LoadFloat32x16Slice(src[i:])
		vw := archsimd.LoadFloat32x16Slice(weight[i:])
		v := vsrc.Mul(vscale)
		v = v.Mul(vw)
		v.StoreSlice(dst[i:])
	}
	for ; i < n; i++ {
		dst[i] = src[i] * scale * weight[i]
	}
}

// siluAndMulAVX512 computes dst[i] = Silu(x[i]) * x[d+i] where d = len(x)/2 using AVX-512 SIMD.
func siluAndMulAVX512(dst, x []float32) {
	d := len(x) / 2
	i := 0
	for ; i+16 <= d; i += 16 {
		vgate := archsimd.LoadFloat32x16Slice(x[i:])
		vup := archsimd.LoadFloat32x16Slice(x[d+i:])
		vsilu := fastSiluVec16(vgate)
		vresult := vsilu.Mul(vup)
		var tmp [16]float32
		vresult.Store(&tmp)
		copy(dst[i:], tmp[:])
	}
	for ; i < d; i++ {
		dst[i] = Silu(x[i]) * x[d+i]
	}
}

// fastSiluVec16 computes silu for a Float32x16 vector.
func fastSiluVec16(x archsimd.Float32x16) archsimd.Float32x16 {
	return x.Mul(fastSigmoidVec16(x))
}

// fastSigmoidVec16 computes sigmoid for a Float32x16 vector.
func fastSigmoidVec16(x archsimd.Float32x16) archsimd.Float32x16 {
	one := archsimd.BroadcastFloat32x16(1.0)
	zero := archsimd.BroadcastFloat32x16(0.0)
	negX := zero.Sub(x)
	expNegX := fastExpVec16(negX)
	return one.Div(one.Add(expNegX))
}

// fastExpVec16 computes exp(x) for a Float32x16 vector using polynomial approximation.
func fastExpVec16(x archsimd.Float32x16) archsimd.Float32x16 {
	ln2 := archsimd.BroadcastFloat32x16(0.693147180559945309417)
	ln2Inv := archsimd.BroadcastFloat32x16(1.44269504088896340736)
	maxVal := archsimd.BroadcastFloat32x16(88.0)
	minVal := archsimd.BroadcastFloat32x16(-88.0)
	one := archsimd.BroadcastFloat32x16(1.0)
	c2 := archsimd.BroadcastFloat32x16(0.5)
	c3 := archsimd.BroadcastFloat32x16(0.16666667)
	c4 := archsimd.BroadcastFloat32x16(0.041666667)
	c5 := archsimd.BroadcastFloat32x16(0.008333333)

	x = x.Max(minVal).Min(maxVal)

	k := x.Mul(ln2Inv).RoundToEvenScaled(0).ConvertToInt32()

	kf := k.ConvertToFloat32()
	r := x.Sub(kf.Mul(ln2))

	r2 := r.Mul(r)
	r3 := r2.Mul(r)
	r4 := r2.Mul(r2)
	r5 := r4.Mul(r)

	poly := one.Add(r).Add(r2.Mul(c2)).Add(r3.Mul(c3)).Add(r4.Mul(c4)).Add(r5.Mul(c5))

	bias := archsimd.BroadcastInt32x16(127)
	exp := k.Add(bias).ShiftAllLeft(23)
	scale := exp.AsFloat32x16()

	return poly.Mul(scale)
}

// applyRoPEWithTablesAVX512 applies Rotary Positional Embeddings using precomputed cosine and sine values with AVX-512 SIMD.
func applyRoPEWithTablesAVX512(x []float32, nHead, headDim, pos int, cosTable, sinTable []float32, half int) {
	for h := range nHead {
		base := h * headDim
		tableOffset := pos * half
		lo := x[base : base+half]
		hi := x[base+half : base+headDim]

		i := 0
		for ; i+16 <= half; i += 16 {
			x0 := archsimd.LoadFloat32x16Slice(lo[i:])
			x1 := archsimd.LoadFloat32x16Slice(hi[i:])
			c := archsimd.LoadFloat32x16Slice(cosTable[tableOffset+i:])
			s := archsimd.LoadFloat32x16Slice(sinTable[tableOffset+i:])

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

// applyRoPEAVX512 applies Rotary Positional Embeddings using AVX-512 SIMD.
func applyRoPEAVX512(x []float32, nHead, headDim, pos int, invFreq []float64, attentionFactor float32, half int) {
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
			sin64, cos64 := math.Sincos(angle)
			cosVals[i] = float32(cos64)
			sinVals[i] = float32(sin64)
		}
	} else {
		for i := range half {
			angle := pos64 * invFreq[i]
			sin64, cos64 := math.Sincos(angle)
			cosVals[i] = float32(cos64) * attentionFactor
			sinVals[i] = float32(sin64) * attentionFactor
		}
	}

	for h := range nHead {
		base := h * headDim
		lo := x[base : base+half]
		hi := x[base+half : base+headDim]

		i := 0
		for ; i+16 <= half; i += 16 {
			x0 := archsimd.LoadFloat32x16Slice(lo[i:])
			x1 := archsimd.LoadFloat32x16Slice(hi[i:])
			c := archsimd.LoadFloat32x16Slice(cosVals[i:])
			s := archsimd.LoadFloat32x16Slice(sinVals[i:])

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
