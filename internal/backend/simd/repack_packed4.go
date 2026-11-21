package simd

import (
	"errors"
	"math"
	"simd/archsimd"
	"sync"

	instance "github.com/samcharles93/mantle/internal/backend/core"
	"github.com/samcharles93/mantle/pkg/mcf"
)

// Compile-time guard for the 8-slice int16 accumulation in quantDotRepacked8x*:
// each VPMADDUBSW pair sum has magnitude at most 2*15*127 = 3810 (activations
// clamped strictly to [-quantActMaxAbs, +quantActMaxAbs], never -128), so
// summing all 8 slices of a 32-element block stays exact in signed int16
// (8 * 3810 = 30480 <= 32767) before widening with VPMADDWD.
const _ = uint16(math.MaxInt16 - 8*2*packed4UnsignedMax*quantActMaxAbs)

// packed4x8WantedForLoad reports whether the CPU loader should repack a loaded
// packed 4-bit weight matrix into the 8-row interleaved AVX2 layout. It
// respects quantCacheBuildEnabledForLoad so CUDA loads and --no-quant-cache
// keep the raw payload only.
func packed4x8WantedForLoad(dt mcf.TensorDType, r, c int) bool {
	return cpu.HasAVX2 && quantCacheBuildEnabledForLoad() && isPacked4DType(dt) &&
		r >= 8 && c > 0 && c%32 == 0 && c/32 <= quantMaxBlocksPerRow
}

// batchWeightsRepacked4x8 reports whether MatMatQuant can run w through the
// repacked 8-row interleaved 4-bit GEMM micro-kernels.
func batchWeightsRepacked4x8(w *Mat) bool {
	return batchWeightsPacked(w) && w.Packed4x8 != nil && w.Packed4x8.ValidFor(w)
}

// BuildPacked4x8 repacks a packed 4-bit weight matrix (q4, k4, gq*4) into
// 8-row interleaved groups of unsigned [0, 15] nibbles (w + 8). Only full
// 8-row groups (R / 8) are repacked; any trailing R%8 rows stay on the
// per-row packed path.
func BuildPacked4x8(w *Mat) (*instance.Packed4x8, error) {
	if w == nil || w.Raw == nil {
		return nil, errors.New("repack4x8: missing raw payload")
	}
	if !isPacked4DType(w.DType) {
		return nil, errors.New("repack4x8: unsupported dtype")
	}
	if w.R < 8 || w.C <= 0 || w.C%32 != 0 {
		return nil, errors.New("repack4x8: shape requires R>=8 and C%32==0")
	}
	layout, err := instance.QuantLayoutForMat(w.R, w.C, w.DType, len(w.Raw))
	if err != nil {
		return nil, err
	}
	bpr := layout.BlocksPerRow
	groups := w.R / 8
	dataLen, ok := instance.MulInt(groups, bpr)
	if !ok {
		return nil, errors.New("repack4x8: size overflow")
	}
	dataBytes, ok := instance.MulInt(dataLen, 128)
	if !ok {
		return nil, errors.New("repack4x8: byte size overflow")
	}
	src := w.Raw[layout.DataOff : layout.DataOff+layout.DataBytes]
	dst := make([]byte, dataBytes)

	repackRange := func(gs, ge int) {
		for g := gs; g < ge; g++ {
			for b := range bpr {
				outBlk := dst[(g*bpr+b)*128 : (g*bpr+b+1)*128]
				for j := range 8 {
					inBlk := src[((g*8+j)*bpr+b)*16 : ((g*8+j)*bpr+b+1)*16]
					for s := range 4 {
						b0, b1, b2, b3 := inBlk[4*s], inBlk[4*s+1], inBlk[4*s+2], inBlk[4*s+3]
						u0 := (b0 & 0x0F) ^ 8
						u1 := (b0 >> 4) ^ 8
						u2 := (b1 & 0x0F) ^ 8
						u3 := (b1 >> 4) ^ 8
						u4 := (b2 & 0x0F) ^ 8
						u5 := (b2 >> 4) ^ 8
						u6 := (b3 & 0x0F) ^ 8
						u7 := (b3 >> 4) ^ 8
						out := outBlk[s*32+j*4 : s*32+j*4+4]
						out[0] = u0 | (u4 << 4)
						out[1] = u1 | (u5 << 4)
						out[2] = u2 | (u6 << 4)
						out[3] = u3 | (u7 << 4)
					}
				}
			}
		}
	}

	workers := 1
	if groups >= 64 {
		if pool := getMatVecPool(); pool != nil && pool.size > 1 {
			workers = min(pool.size, groups/32)
		}
	}
	if workers <= 1 {
		repackRange(0, groups)
	} else {
		var wg sync.WaitGroup
		for i := range workers {
			gs := i * groups / workers
			ge := (i + 1) * groups / workers
			wg.Go(func() { repackRange(gs, ge) })
		}
		wg.Wait()
	}

	return &instance.Packed4x8{
		Data:         dst,
		Groups:       groups,
		BlocksPerRow: bpr,
	}, nil
}

// packed4GroupWeightScales fills wScales[b*8+j] with row (r+j)'s block-b
// weight scale for the eight consecutive packed 4-bit rows starting at r.
// For k-family weights with bpr%8 == 0 it uses the vectorised 8x8 superblock
// transpose (matching packed4GroupScalesK's bit semantics); q and gq weights
// fill via packed4RowScales.
func packed4GroupWeightScales(wScales []float32, layout instance.QuantLayout, raw []byte, r int, tmp []float32) {
	bpr := layout.BlocksPerRow
	if layout.Family == 'k' && bpr%8 == 0 && bpr <= packed4GroupMaxBlocks {
		packed4GroupWeightScalesK(wScales, layout, raw, r)
		return
	}
	ws := tmp[:bpr]
	for j := range 8 {
		packed4RowScales(ws, layout, raw, r+j)
		for b, s := range ws {
			wScales[b*8+j] = s
		}
	}
}

// packed4GroupWeightScalesK is packed4GroupScalesK without multiplying by an
// activation scale vector: it fills wScales[b*8+j] = wScale[r+j][b] for eight
// consecutive k-family rows. bpr must be a multiple of 8 and <= 256.
func packed4GroupWeightScalesK(wScales []float32, layout instance.QuantLayout, raw []byte, r int) {
	bpr := layout.BlocksPerRow
	spr := bpr / 8
	scalesRaw := raw[layout.ScaleOff : layout.ScaleOff+layout.ScaleCount*2]
	subRaw := raw[layout.SubScaleOff : layout.SubScaleOff+layout.SubScaleCount]
	wScales = wScales[:bpr*8]

	var supers [8 * packed4GroupMaxBlocks / 8]float32
	for j := range 8 {
		for sb := range spr {
			off := ((r+j)*layout.SuperBlocksPerRow + sb) * 2
			supers[j*spr+sb] = instance.FP16BitsToFloat32(uint16(scalesRaw[off]) | uint16(scalesRaw[off+1])<<8)
		}
	}

	low6 := archsimd.BroadcastUint32x8(0x3F)
	var zeroU archsimd.Uint32x8
	oneOver32 := archsimd.BroadcastFloat32x8(1.0 / 32)
	for sb := range spr {
		row := func(j int) archsimd.Uint32x8 {
			u6 := loadSubScales8(subRaw, (r+j)*bpr+sb*8).ExtendLo8ToUint32().And(low6)
			q := u6.BitsToInt32().ConvertToFloat32().Mul(oneOver32)
			super := archsimd.BroadcastFloat32x8(supers[j*spr+sb])
			return super.Mul(q).Masked(u6.NotEqual(zeroU)).ToBits()
		}
		v0, v1, v2, v3 := row(0), row(1), row(2), row(3)
		v4, v5, v6, v7 := row(4), row(5), row(6), row(7)
		a0, a1 := v0.InterleaveLoGrouped(v1), v0.InterleaveHiGrouped(v1)
		a2, a3 := v2.InterleaveLoGrouped(v3), v2.InterleaveHiGrouped(v3)
		a4, a5 := v4.InterleaveLoGrouped(v5), v4.InterleaveHiGrouped(v5)
		a6, a7 := v6.InterleaveLoGrouped(v7), v6.InterleaveHiGrouped(v7)
		lo64 := func(x, y archsimd.Uint32x8) archsimd.Uint32x8 {
			return x.ReshapeToUint64s().InterleaveLoGrouped(y.ReshapeToUint64s()).ReshapeToUint32s()
		}
		hi64 := func(x, y archsimd.Uint32x8) archsimd.Uint32x8 {
			return x.ReshapeToUint64s().InterleaveHiGrouped(y.ReshapeToUint64s()).ReshapeToUint32s()
		}
		b0, b1 := lo64(a0, a2), hi64(a0, a2)
		b2, b3 := lo64(a1, a3), hi64(a1, a3)
		b4, b5 := lo64(a4, a6), hi64(a4, a6)
		b6, b7 := lo64(a5, a7), hi64(a5, a7)
		out := wScales[sb*64 : (sb+1)*64]
		store := func(k int, t archsimd.Uint32x8) {
			t.BitsToFloat32().StoreArray((*[8]float32)(out[k*8 : k*8+8]))
		}
		store(0, b0.SetHi(b4.GetLo()))
		store(1, b1.SetHi(b5.GetLo()))
		store(2, b2.SetHi(b6.GetLo()))
		store(3, b3.SetHi(b7.GetLo()))
		store(4, b4.SetLo(b0.GetHi()))
		store(5, b5.SetLo(b1.GetHi()))
		store(6, b6.SetLo(b2.GetHi()))
		store(7, b7.SetLo(b3.GetHi()))
	}
	endAVX()
}

// quantDotRepacked8x4 computes 32 outputs (8 consecutive weight rows x 4
// activation rows) directly from repacked 4-bit weights wData (128 bytes per
// block) and contiguous quad-interleaved int8 activations xQuad (32 int32s per
// block: four 8-int32 chunks, each holding 4 rows for wLo in lanes 0..3 and
// 4 rows for wHi in lanes 4..7). xMeta holds 8 uint32s per block: Float32bits
// of the 4 activation scales in lanes 0..3 and uint32(bias) in lanes 4..7.
//
// Every value inside the loop stays 256-bit wide (loading 8 activation dwords
// per 256-bit load and broadcasting via VPERM2I128 + VPSHUFD instead of 128-bit
// VMOVD/VPBROADCASTD) so Go never creates 128-bit XMM temps that could spill
// with legacy-SSE MOVUPS on Golden Cove.
//
// Each 8-row weight slice is unpacked in registers without any per-call scratch
// expansion, multiplied via unsigned*signed VPMADDUBSW, and accumulated across
// all 8 slices of a 32-element block in signed int16 (zero VPHADDW). One
// VPMADDWD per activation row widens all eight weight rows to exact int32,
// subtracts the precomputed 8*sum(x) bias, and folds with the exact same
// float32 operations in block order as matVecRangeQuantCached.
//
// Invariant: activations must be clamped strictly to [-127, +127] (never -128,
// enforced by quantActMaxAbs in quant_act.go) so 8*2*15*127 = 30480 <= 32767.
//
// out receives rows 0..3 sequentially: out[m*8+j] = dot(wRow j, xRow m).
func quantDotRepacked8x4(wData []byte, wScales []float32, xQuad []int32, xMeta []uint32, out *[32]float32) {
	bpr := len(wScales) / 8
	wData = wData[:bpr*128]
	wScales = wScales[:bpr*8]
	xQuad = xQuad[:bpr*32]
	xMeta = xMeta[:bpr*8]

	mask0F := archsimd.BroadcastUint8x32(0x0F)
	ones := archsimd.BroadcastInt16x16(1)
	sum0 := archsimd.LoadFloat32x8Array(&zeroFloat32s)
	sum1 := archsimd.LoadFloat32x8Array(&zeroFloat32s)
	sum2 := archsimd.LoadFloat32x8Array(&zeroFloat32s)
	sum3 := archsimd.LoadFloat32x8Array(&zeroFloat32s)

	for b := range bpr {
		wb := (*[128]byte)(wData[b*128 : b*128+128])
		xb := (*[32]int32)(xQuad[b*32 : b*32+32])

		raw := archsimd.LoadUint8x32Array((*[32]uint8)(wb[0:32]))
		xv := archsimd.LoadInt32x8Array((*[8]int32)(xb[0:8])).ToBits()
		xLo := xv.ConcatPermute128Scalars(0, 0, xv)
		wLo := raw.And(mask0F)
		acc0 := wLo.DotProductPairsSaturated(xLo.PermuteScalarsGrouped(0, 0, 0, 0).ReshapeToUint8s().BitsToInt8())
		acc1 := wLo.DotProductPairsSaturated(xLo.PermuteScalarsGrouped(1, 1, 1, 1).ReshapeToUint8s().BitsToInt8())
		acc2 := wLo.DotProductPairsSaturated(xLo.PermuteScalarsGrouped(2, 2, 2, 2).ReshapeToUint8s().BitsToInt8())
		acc3 := wLo.DotProductPairsSaturated(xLo.PermuteScalarsGrouped(3, 3, 3, 3).ReshapeToUint8s().BitsToInt8())
		xHi := xv.ConcatPermute128Scalars(1, 1, xv)
		wHi := raw.ReshapeToUint16s().ShiftAllRight(4).ReshapeToUint8s().And(mask0F)
		acc0 = acc0.Add(wHi.DotProductPairsSaturated(xHi.PermuteScalarsGrouped(0, 0, 0, 0).ReshapeToUint8s().BitsToInt8()))
		acc1 = acc1.Add(wHi.DotProductPairsSaturated(xHi.PermuteScalarsGrouped(1, 1, 1, 1).ReshapeToUint8s().BitsToInt8()))
		acc2 = acc2.Add(wHi.DotProductPairsSaturated(xHi.PermuteScalarsGrouped(2, 2, 2, 2).ReshapeToUint8s().BitsToInt8()))
		acc3 = acc3.Add(wHi.DotProductPairsSaturated(xHi.PermuteScalarsGrouped(3, 3, 3, 3).ReshapeToUint8s().BitsToInt8()))

		for s := 1; s < 4; s++ {
			raw = archsimd.LoadUint8x32Array((*[32]uint8)(wb[s*32 : s*32+32]))
			xv = archsimd.LoadInt32x8Array((*[8]int32)(xb[s*8 : s*8+8])).ToBits()
			xLo = xv.ConcatPermute128Scalars(0, 0, xv)
			wLo = raw.And(mask0F)
			acc0 = acc0.Add(wLo.DotProductPairsSaturated(xLo.PermuteScalarsGrouped(0, 0, 0, 0).ReshapeToUint8s().BitsToInt8()))
			acc1 = acc1.Add(wLo.DotProductPairsSaturated(xLo.PermuteScalarsGrouped(1, 1, 1, 1).ReshapeToUint8s().BitsToInt8()))
			acc2 = acc2.Add(wLo.DotProductPairsSaturated(xLo.PermuteScalarsGrouped(2, 2, 2, 2).ReshapeToUint8s().BitsToInt8()))
			acc3 = acc3.Add(wLo.DotProductPairsSaturated(xLo.PermuteScalarsGrouped(3, 3, 3, 3).ReshapeToUint8s().BitsToInt8()))
			xHi = xv.ConcatPermute128Scalars(1, 1, xv)
			wHi = raw.ReshapeToUint16s().ShiftAllRight(4).ReshapeToUint8s().And(mask0F)
			acc0 = acc0.Add(wHi.DotProductPairsSaturated(xHi.PermuteScalarsGrouped(0, 0, 0, 0).ReshapeToUint8s().BitsToInt8()))
			acc1 = acc1.Add(wHi.DotProductPairsSaturated(xHi.PermuteScalarsGrouped(1, 1, 1, 1).ReshapeToUint8s().BitsToInt8()))
			acc2 = acc2.Add(wHi.DotProductPairsSaturated(xHi.PermuteScalarsGrouped(2, 2, 2, 2).ReshapeToUint8s().BitsToInt8()))
			acc3 = acc3.Add(wHi.DotProductPairsSaturated(xHi.PermuteScalarsGrouped(3, 3, 3, 3).ReshapeToUint8s().BitsToInt8()))
		}

		ws := archsimd.LoadFloat32x8Array((*[8]float32)(wScales[b*8 : b*8+8]))
		meta := archsimd.LoadUint32x8Array((*[8]uint32)(xMeta[b*8 : b*8+8]))
		sv := meta.ConcatPermute128Scalars(0, 0, meta)
		bv := meta.ConcatPermute128Scalars(1, 1, meta).BitsToInt32()

		d0 := acc0.DotProductPairs(ones).Sub(bv.PermuteScalarsGrouped(0, 0, 0, 0)).ConvertToFloat32()
		sum0 = sum0.Add(d0.Mul(ws.Mul(sv.PermuteScalarsGrouped(0, 0, 0, 0).BitsToFloat32())))
		d1 := acc1.DotProductPairs(ones).Sub(bv.PermuteScalarsGrouped(1, 1, 1, 1)).ConvertToFloat32()
		sum1 = sum1.Add(d1.Mul(ws.Mul(sv.PermuteScalarsGrouped(1, 1, 1, 1).BitsToFloat32())))
		d2 := acc2.DotProductPairs(ones).Sub(bv.PermuteScalarsGrouped(2, 2, 2, 2)).ConvertToFloat32()
		sum2 = sum2.Add(d2.Mul(ws.Mul(sv.PermuteScalarsGrouped(2, 2, 2, 2).BitsToFloat32())))
		d3 := acc3.DotProductPairs(ones).Sub(bv.PermuteScalarsGrouped(3, 3, 3, 3)).ConvertToFloat32()
		sum3 = sum3.Add(d3.Mul(ws.Mul(sv.PermuteScalarsGrouped(3, 3, 3, 3).BitsToFloat32())))
	}

	sum0.StoreArray((*[8]float32)(out[0:8]))
	sum1.StoreArray((*[8]float32)(out[8:16]))
	sum2.StoreArray((*[8]float32)(out[16:24]))
	sum3.StoreArray((*[8]float32)(out[24:32]))
	endAVX()
}

// quantDotRepacked8x2 is the 2-activation-row tail sibling of
// quantDotRepacked8x4 (16 outputs: 8 weight rows x 2 activation rows).
// xPair holds 16 int32s per block (two 8-int32 loads: chunks 0-1 in the first,
// chunks 2-3 in the second). xMeta holds 8 uint32s per block: {s0, s1, 0, 0,
// b0, b1, 0, 0}.
func quantDotRepacked8x2(wData []byte, wScales []float32, xPair []int32, xMeta []uint32, out *[16]float32) {
	bpr := len(wScales) / 8
	wData = wData[:bpr*128]
	wScales = wScales[:bpr*8]
	xPair = xPair[:bpr*16]
	xMeta = xMeta[:bpr*8]

	mask0F := archsimd.BroadcastUint8x32(0x0F)
	ones := archsimd.BroadcastInt16x16(1)
	sum0 := archsimd.LoadFloat32x8Array(&zeroFloat32s)
	sum1 := archsimd.LoadFloat32x8Array(&zeroFloat32s)

	for b := range bpr {
		wb := (*[128]byte)(wData[b*128 : b*128+128])
		xb := (*[16]int32)(xPair[b*16 : b*16+16])

		xv := archsimd.LoadInt32x8Array((*[8]int32)(xb[0:8])).ToBits()
		xLo := xv.ConcatPermute128Scalars(0, 0, xv)
		raw := archsimd.LoadUint8x32Array((*[32]uint8)(wb[0:32]))
		wLo := raw.And(mask0F)
		acc0 := wLo.DotProductPairsSaturated(xLo.PermuteScalarsGrouped(0, 0, 0, 0).ReshapeToUint8s().BitsToInt8())
		acc1 := wLo.DotProductPairsSaturated(xLo.PermuteScalarsGrouped(1, 1, 1, 1).ReshapeToUint8s().BitsToInt8())
		wHi := raw.ReshapeToUint16s().ShiftAllRight(4).ReshapeToUint8s().And(mask0F)
		acc0 = acc0.Add(wHi.DotProductPairsSaturated(xLo.PermuteScalarsGrouped(2, 2, 2, 2).ReshapeToUint8s().BitsToInt8()))
		acc1 = acc1.Add(wHi.DotProductPairsSaturated(xLo.PermuteScalarsGrouped(3, 3, 3, 3).ReshapeToUint8s().BitsToInt8()))

		xHi := xv.ConcatPermute128Scalars(1, 1, xv)
		raw = archsimd.LoadUint8x32Array((*[32]uint8)(wb[32:64]))
		wLo = raw.And(mask0F)
		acc0 = acc0.Add(wLo.DotProductPairsSaturated(xHi.PermuteScalarsGrouped(0, 0, 0, 0).ReshapeToUint8s().BitsToInt8()))
		acc1 = acc1.Add(wLo.DotProductPairsSaturated(xHi.PermuteScalarsGrouped(1, 1, 1, 1).ReshapeToUint8s().BitsToInt8()))
		wHi = raw.ReshapeToUint16s().ShiftAllRight(4).ReshapeToUint8s().And(mask0F)
		acc0 = acc0.Add(wHi.DotProductPairsSaturated(xHi.PermuteScalarsGrouped(2, 2, 2, 2).ReshapeToUint8s().BitsToInt8()))
		acc1 = acc1.Add(wHi.DotProductPairsSaturated(xHi.PermuteScalarsGrouped(3, 3, 3, 3).ReshapeToUint8s().BitsToInt8()))

		xv = archsimd.LoadInt32x8Array((*[8]int32)(xb[8:16])).ToBits()
		xLo = xv.ConcatPermute128Scalars(0, 0, xv)
		raw = archsimd.LoadUint8x32Array((*[32]uint8)(wb[64:96]))
		wLo = raw.And(mask0F)
		acc0 = acc0.Add(wLo.DotProductPairsSaturated(xLo.PermuteScalarsGrouped(0, 0, 0, 0).ReshapeToUint8s().BitsToInt8()))
		acc1 = acc1.Add(wLo.DotProductPairsSaturated(xLo.PermuteScalarsGrouped(1, 1, 1, 1).ReshapeToUint8s().BitsToInt8()))
		wHi = raw.ReshapeToUint16s().ShiftAllRight(4).ReshapeToUint8s().And(mask0F)
		acc0 = acc0.Add(wHi.DotProductPairsSaturated(xLo.PermuteScalarsGrouped(2, 2, 2, 2).ReshapeToUint8s().BitsToInt8()))
		acc1 = acc1.Add(wHi.DotProductPairsSaturated(xLo.PermuteScalarsGrouped(3, 3, 3, 3).ReshapeToUint8s().BitsToInt8()))

		xHi = xv.ConcatPermute128Scalars(1, 1, xv)
		raw = archsimd.LoadUint8x32Array((*[32]uint8)(wb[96:128]))
		wLo = raw.And(mask0F)
		acc0 = acc0.Add(wLo.DotProductPairsSaturated(xHi.PermuteScalarsGrouped(0, 0, 0, 0).ReshapeToUint8s().BitsToInt8()))
		acc1 = acc1.Add(wLo.DotProductPairsSaturated(xHi.PermuteScalarsGrouped(1, 1, 1, 1).ReshapeToUint8s().BitsToInt8()))
		wHi = raw.ReshapeToUint16s().ShiftAllRight(4).ReshapeToUint8s().And(mask0F)
		acc0 = acc0.Add(wHi.DotProductPairsSaturated(xHi.PermuteScalarsGrouped(2, 2, 2, 2).ReshapeToUint8s().BitsToInt8()))
		acc1 = acc1.Add(wHi.DotProductPairsSaturated(xHi.PermuteScalarsGrouped(3, 3, 3, 3).ReshapeToUint8s().BitsToInt8()))

		ws := archsimd.LoadFloat32x8Array((*[8]float32)(wScales[b*8 : b*8+8]))
		meta := archsimd.LoadUint32x8Array((*[8]uint32)(xMeta[b*8 : b*8+8]))
		sv := meta.ConcatPermute128Scalars(0, 0, meta)
		bv := meta.ConcatPermute128Scalars(1, 1, meta).BitsToInt32()

		d0 := acc0.DotProductPairs(ones).Sub(bv.PermuteScalarsGrouped(0, 0, 0, 0)).ConvertToFloat32()
		sum0 = sum0.Add(d0.Mul(ws.Mul(sv.PermuteScalarsGrouped(0, 0, 0, 0).BitsToFloat32())))
		d1 := acc1.DotProductPairs(ones).Sub(bv.PermuteScalarsGrouped(1, 1, 1, 1)).ConvertToFloat32()
		sum1 = sum1.Add(d1.Mul(ws.Mul(sv.PermuteScalarsGrouped(1, 1, 1, 1).BitsToFloat32())))
	}

	sum0.StoreArray((*[8]float32)(out[0:8]))
	sum1.StoreArray((*[8]float32)(out[8:16]))
	endAVX()
}

// quantDotRepacked8x1 is the 1-activation-row tail sibling of
// quantDotRepacked8x4 (8 outputs: 8 weight rows x 1 activation row).
// xMeta holds 8 uint32s per block: {s0, 0, 0, 0, b0, 0, 0, 0}.
func quantDotRepacked8x1(wData []byte, wScales []float32, xRow []int32, xMeta []uint32, out *[8]float32) {
	bpr := len(wScales) / 8
	wData = wData[:bpr*128]
	wScales = wScales[:bpr*8]
	xRow = xRow[:bpr*8]
	xMeta = xMeta[:bpr*8]

	mask0F := archsimd.BroadcastUint8x32(0x0F)
	ones := archsimd.BroadcastInt16x16(1)
	sum0 := archsimd.LoadFloat32x8Array(&zeroFloat32s)

	for b := range bpr {
		wb := (*[128]byte)(wData[b*128 : b*128+128])
		xv := archsimd.LoadInt32x8Array((*[8]int32)(xRow[b*8 : b*8+8])).ToBits()
		xLo := xv.ConcatPermute128Scalars(0, 0, xv)

		raw := archsimd.LoadUint8x32Array((*[32]uint8)(wb[0:32]))
		acc0 := raw.And(mask0F).DotProductPairsSaturated(xLo.PermuteScalarsGrouped(0, 0, 0, 0).ReshapeToUint8s().BitsToInt8())
		acc0 = acc0.Add(raw.ReshapeToUint16s().ShiftAllRight(4).ReshapeToUint8s().And(mask0F).
			DotProductPairsSaturated(xLo.PermuteScalarsGrouped(1, 1, 1, 1).ReshapeToUint8s().BitsToInt8()))

		raw = archsimd.LoadUint8x32Array((*[32]uint8)(wb[32:64]))
		acc0 = acc0.Add(raw.And(mask0F).DotProductPairsSaturated(xLo.PermuteScalarsGrouped(2, 2, 2, 2).ReshapeToUint8s().BitsToInt8()))
		acc0 = acc0.Add(raw.ReshapeToUint16s().ShiftAllRight(4).ReshapeToUint8s().And(mask0F).
			DotProductPairsSaturated(xLo.PermuteScalarsGrouped(3, 3, 3, 3).ReshapeToUint8s().BitsToInt8()))

		xHi := xv.ConcatPermute128Scalars(1, 1, xv)
		raw = archsimd.LoadUint8x32Array((*[32]uint8)(wb[64:96]))
		acc0 = acc0.Add(raw.And(mask0F).DotProductPairsSaturated(xHi.PermuteScalarsGrouped(0, 0, 0, 0).ReshapeToUint8s().BitsToInt8()))
		acc0 = acc0.Add(raw.ReshapeToUint16s().ShiftAllRight(4).ReshapeToUint8s().And(mask0F).
			DotProductPairsSaturated(xHi.PermuteScalarsGrouped(1, 1, 1, 1).ReshapeToUint8s().BitsToInt8()))

		raw = archsimd.LoadUint8x32Array((*[32]uint8)(wb[96:128]))
		acc0 = acc0.Add(raw.And(mask0F).DotProductPairsSaturated(xHi.PermuteScalarsGrouped(2, 2, 2, 2).ReshapeToUint8s().BitsToInt8()))
		acc0 = acc0.Add(raw.ReshapeToUint16s().ShiftAllRight(4).ReshapeToUint8s().And(mask0F).
			DotProductPairsSaturated(xHi.PermuteScalarsGrouped(3, 3, 3, 3).ReshapeToUint8s().BitsToInt8()))

		ws := archsimd.LoadFloat32x8Array((*[8]float32)(wScales[b*8 : b*8+8]))
		meta := archsimd.LoadUint32x8Array((*[8]uint32)(xMeta[b*8 : b*8+8]))
		sv := meta.ConcatPermute128Scalars(0, 0, meta)
		bv := meta.ConcatPermute128Scalars(1, 1, meta).BitsToInt32()

		d0 := acc0.DotProductPairs(ones).Sub(bv.PermuteScalarsGrouped(0, 0, 0, 0)).ConvertToFloat32()
		sum0 = sum0.Add(d0.Mul(ws.Mul(sv.PermuteScalarsGrouped(0, 0, 0, 0).BitsToFloat32())))
	}

	sum0.StoreArray(out)
	endAVX()
}
