package simd

import (
	"math"
	"simd/archsimd"
)

// quantActMaxAbs is the symmetric magnitude bound activation quantisation
// clamps to: every quantised activation stays in [-127, +127] and never -128.
// Both the signed int8 kernels (|w|<=128, |x|<=127 -> 2*128*127 = 32512) and
// the unsigned 4-bit int16 reduction tree (8*2*15*127 = 30480) depend on this
// invariant.
const quantActMaxAbs = 127

// Activation quantisation for the int8 kernels: each 32-element block of x
// becomes 32 int8 values and one float32 scale, maxAbs/127, with every value
// round(x*inv) (half away from zero) clamped to [-127, 127], where
// inv = 1/scale. A zero element always gives 0, so an all-zero block gets
// scale 0 and zero values. The clamp applies before the integer conversion:
// when a block is so small (all denormal) that inv overflows to +Inf, its
// nonzero elements saturate to ±127 rather than depend on how an
// out-of-range float converts. The AVX2 kernel and the scalar reference
// produce identical bytes for finite inputs; the int8 kernels' bit-exactness
// rests on that.

// Lane tables for quantizeBlocksAVX2.
var (
	// quantEvenOddIdx reorders eight int32 lanes into the four even elements
	// followed by the four odd ones.
	quantEvenOddIdx = [8]uint32{0, 2, 4, 6, 1, 3, 5, 7}
	// quantQwordOrderIdx moves 64-bit units {0, 2, 1, 3} into place, undoing
	// the per-lane split left by quantLowBytes*.
	quantQwordOrderIdx = [8]uint32{0, 1, 4, 5, 2, 3, 6, 7}
	// quantLowBytesLo gathers the low byte of each int16 of a 128-bit lane
	// into that lane's low 8 bytes and zeroes the rest; quantLowBytesHi puts
	// them in the high 8 bytes instead.
	quantLowBytesLo = [32]int8{
		0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1,
		0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1,
	}
	quantLowBytesHi = [32]int8{
		-1, -1, -1, -1, -1, -1, -1, -1, 0, 2, 4, 6, 8, 10, 12, 14,
		-1, -1, -1, -1, -1, -1, -1, -1, 0, 2, 4, 6, 8, 10, 12, 14,
	}
)

// quantizeActivationBlocks quantises the first blocks*32 elements of x, reading
// elements past len(x) as zero. It writes scales[:blocks] plus whichever
// optional outputs are non-nil: q8 in element order, q32 in element order
// packed 4 int8s per little-endian int32 (for repacked 8x4 GEMM), q16 widened
// to int16, q16Pair and q8Pair with each block as its 16 even elements then
// its 16 odd ones (the order decodePacked4RowUnsigned produces), and bias with
// 8 times each block's exact sum of quantised int8 values (for unsigned 4-bit
// sign folding).
func quantizeActivationBlocks(x []float32, blocks int, scales []float32, q8 []int8, q32 []int32, q16, q16Pair []int16, q8Pair []int8, bias []int32) {
	if blocks <= 0 {
		return
	}
	n := blocks * 32
	scales = scales[:blocks]
	if q8 != nil {
		q8 = q8[:n]
	}
	if q32 != nil {
		q32 = q32[:blocks*8]
	}
	if q16 != nil {
		q16 = q16[:n]
	}
	if q16Pair != nil {
		q16Pair = q16Pair[:n]
	}
	if q8Pair != nil {
		q8Pair = q8Pair[:n]
	}
	if bias != nil {
		bias = bias[:blocks]
	}
	full := 0
	if cpu.HasAVX2 {
		full = min(blocks, len(x)/32)
		var q8Full []int8
		if q8 != nil {
			q8Full = q8[:full*32]
		}
		var q32Full []int32
		if q32 != nil {
			q32Full = q32[:full*8]
		}
		var biasFull []int32
		if bias != nil {
			biasFull = bias[:full]
		}
		quantizeBlocksAVX2(x[:full*32], scales[:full], q8Full, q32Full, q16, q16Pair, q8Pair, biasFull)
	}
	var q [32]int8
	for b := full; b < blocks; b++ {
		lo := min(b*32, len(x))
		hi := min(b*32+32, len(x))
		scales[b] = quantizeBlockScalar(x[lo:hi], &q)
		writeQuantBlock(&q, b, q8, q32, q16, q16Pair, q8Pair, bias)
	}
}

// quantizeBlockScalar quantises one block, x holding its first len(x) <= 32
// elements and the rest read as zero, into q, and returns the block's scale.
func quantizeBlockScalar(x []float32, q *[32]int8) float32 {
	*q = [32]int8{}
	maxAbs := float32(0)
	for _, v := range x {
		if v < 0 {
			v = -v
		}
		if v > maxAbs {
			maxAbs = v
		}
	}
	if maxAbs == 0 {
		return 0
	}
	scale := maxAbs / quantActMaxAbs
	inv := float32(1.0) / scale
	for i, v := range x {
		if v == 0 {
			continue
		}
		r := math.Round(float64(v * inv))
		if r > quantActMaxAbs {
			r = quantActMaxAbs
		} else if r < -quantActMaxAbs {
			r = -quantActMaxAbs
		}
		q[i] = int8(r)
	}
	return scale
}

// writeQuantBlock stores block b's values q into each non-nil output.
func writeQuantBlock(q *[32]int8, b int, q8 []int8, q32 []int32, q16, q16Pair []int16, q8Pair []int8, bias []int32) {
	base := b * 32
	if q8 != nil {
		copy(q8[base:base+32], q[:])
	}
	if q32 != nil {
		base32 := b * 8
		for k := range 8 {
			q32[base32+k] = int32(uint32(uint8(q[4*k])) |
				uint32(uint8(q[4*k+1]))<<8 |
				uint32(uint8(q[4*k+2]))<<16 |
				uint32(uint8(q[4*k+3]))<<24)
		}
	}
	var sum int32
	for i, v := range q {
		sum += int32(v)
		if q16 != nil {
			q16[base+i] = int16(v)
		}
		pi := i/2 + (i%2)*16
		if q16Pair != nil {
			q16Pair[base+pi] = int16(v)
		}
		if q8Pair != nil {
			q8Pair[base+pi] = v
		}
	}
	if bias != nil {
		bias[b] = sum << 3
	}
}

// quantizeBlocksAVX2 quantises len(x)/32 whole blocks. Rounding half away from
// zero is built from truncation: t = trunc(y) and d = y-t are exact in
// float32, and the result is t plus y's sign when |d| >= 0.5, which equals
// math.Round(float64(y)); for y = ±Inf, d is NaN, so r stays ±Inf and the
// clamp saturates it. The scale and its reciprocal are the same two
// correctly rounded divisions the scalar path performs.
func quantizeBlocksAVX2(x []float32, scales []float32, q8 []int8, q32 []int32, q16, q16Pair []int16, q8Pair []int8, bias []int32) {
	blocks := len(x) / 32
	if blocks == 0 {
		return
	}
	evenOdd := archsimd.LoadUint32x8Array(&quantEvenOddIdx)
	qwordOrder := archsimd.LoadUint32x8Array(&quantQwordOrderIdx)
	lowLo := archsimd.LoadInt8x32Array(&quantLowBytesLo)
	lowHi := archsimd.LoadInt8x32Array(&quantLowBytesHi)
	signBit := archsimd.BroadcastUint32x8(0x80000000)
	oneBits := archsimd.BroadcastUint32x8(math.Float32bits(1))
	half := archsimd.BroadcastFloat32x8(0.5)
	one := archsimd.BroadcastFloat32x8(1)
	lim := archsimd.BroadcastFloat32x8(quantActMaxAbs)
	negLim := archsimd.BroadcastFloat32x8(-quantActMaxAbs)
	zero := archsimd.LoadFloat32x8Array(&zeroFloat32s)

	// narrow packs two int16 vectors holding elements 0-15 and 16-31 of a
	// block (values already in int8 range) into its 32 int8 values.
	narrow := func(lo, hi archsimd.Int16x16) archsimd.Int8x32 {
		a := lo.ToBits().ReshapeToUint8s().PermuteOrZeroGrouped(lowLo)
		b := hi.ToBits().ReshapeToUint8s().PermuteOrZeroGrouped(lowHi)
		return a.Or(b).ReshapeToUint32s().Permute(qwordOrder).ReshapeToUint8s().BitsToInt8()
	}

	for b := range blocks {
		off := b * 32
		xb := (*[32]float32)(x[off : off+32])
		x0 := archsimd.LoadFloat32x8Array((*[8]float32)(xb[0:8]))
		x1 := archsimd.LoadFloat32x8Array((*[8]float32)(xb[8:16]))
		x2 := archsimd.LoadFloat32x8Array((*[8]float32)(xb[16:24]))
		x3 := archsimd.LoadFloat32x8Array((*[8]float32)(xb[24:32]))

		m := x0.Abs().Max(x1.Abs()).Max(x2.Abs().Max(x3.Abs()))
		m = m.Max(m.SetLo(m.GetHi()).SetHi(m.GetLo()))
		m = m.Max(m.ToBits().PermuteScalarsGrouped(2, 3, 0, 1).BitsToFloat32())
		m = m.Max(m.ToBits().PermuteScalarsGrouped(1, 0, 3, 2).BitsToFloat32())
		scale := m.Div(lim)
		inv := one.Div(scale)
		scales[b] = math.Float32frombits(scale.ToBits().GetLo().GetElem(0))

		quant := func(v archsimd.Float32x8) archsimd.Int32x8 {
			y := v.Mul(inv)
			t := y.Trunc()
			sign := y.ToBits().And(signBit).Or(oneBits).BitsToFloat32()
			r := t.Add(sign.Masked(y.Sub(t).Abs().GreaterEqual(half)))
			// Masking zero elements also clears the NaN of 0*Inf.
			return r.Max(negLim).Min(lim).Masked(v.NotEqual(zero)).ConvertToInt32()
		}
		a0, a1, a2, a3 := quant(x0), quant(x1), quant(x2), quant(x3)

		if bias != nil {
			s := a0.Add(a1).Add(a2.Add(a3))
			s = s.Add(s.SetLo(s.GetHi()).SetHi(s.GetLo()))
			s = s.Add(s.PermuteScalarsGrouped(2, 3, 0, 1))
			s = s.Add(s.PermuteScalarsGrouped(1, 0, 3, 2))
			bias[b] = s.GetLo().GetElem(0) << 3
		}
		if q8 != nil || q32 != nil || q16 != nil {
			// VPACKSSDW packs per 128-bit lane, so pair the halves first.
			n0 := a0.SetHi(a1.GetLo()).SaturateToInt16ConcatGrouped(a1.SetLo(a0.GetHi()))
			n1 := a2.SetHi(a3.GetLo()).SaturateToInt16ConcatGrouped(a3.SetLo(a2.GetHi()))
			if q8 != nil || q32 != nil {
				packed := narrow(n0, n1)
				if q8 != nil {
					packed.StoreArray((*[32]int8)(q8[off : off+32]))
				}
				if q32 != nil {
					packed.ToBits().ReshapeToUint32s().BitsToInt32().StoreArray((*[8]int32)(q32[b*8 : b*8+8]))
				}
			}
			if q16 != nil {
				n0.StoreArray((*[16]int16)(q16[off : off+16]))
				n1.StoreArray((*[16]int16)(q16[off+16 : off+32]))
			}
		}
		if q16Pair != nil || q8Pair != nil {
			p0, p1 := a0.Permute(evenOdd), a1.Permute(evenOdd)
			p2, p3 := a2.Permute(evenOdd), a3.Permute(evenOdd)
			ev := p0.SetHi(p2.GetLo()).SaturateToInt16ConcatGrouped(p1.SetHi(p3.GetLo()))
			od := p2.SetLo(p0.GetHi()).SaturateToInt16ConcatGrouped(p3.SetLo(p1.GetHi()))
			if q16Pair != nil {
				ev.StoreArray((*[16]int16)(q16Pair[off : off+16]))
				od.StoreArray((*[16]int16)(q16Pair[off+16 : off+32]))
			}
			if q8Pair != nil {
				narrow(ev, od).StoreArray((*[32]int8)(q8Pair[off : off+32]))
			}
		}
	}
	endAVX()
}
