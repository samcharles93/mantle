package simd

import (
	"encoding/binary"
	"simd/archsimd"

	instance "github.com/samcharles93/mantle/internal/backend/core"
	"github.com/samcharles93/mantle/pkg/mcf"
)

// Packed 4-bit kernels. q4, k4 and gq*4 store 32 signed 4-bit codes per block
// in 16 bytes (element 2i in the low nibble of byte i, element 2i+1 in the
// high nibble), so one kernel computes the int32 block dots straight from the
// payload, without expanding weights into the int8 quant cache. Only the
// per-block weight scale differs between the families.

// isPacked4DType reports whether dt uses the packed 4-bit block layout.
func isPacked4DType(dt mcf.TensorDType) bool {
	if dt == mcf.DTypeQ4 || dt == mcf.DTypeK4 {
		return true
	}
	bits, _, ok := mcf.GQParams(dt)
	return ok && bits == 4
}

// dotBlocksPacked4Int16 sets out[b] to the dot product of block b's 32 codes
// (data[b*16:(b+1)*16]) with the block's activations, given in pair order
// (16 even elements then 16 odd, see QuantVec.Q16Pair).
func dotBlocksPacked4Int16(data []byte, xPair []int16, out []int32, blocks int) {
	if !cpu.HasAVX2 {
		dotBlocksPacked4Int16Scalar(data, xPair, out, blocks)
		return
	}
	// tmp is zeroed here, before any 256-bit work: zeroing it in the tail loop
	// would be a legacy-SSE store with the upper halves dirty (see endAVX).
	var tmp [8]int32
	mask := archsimd.BroadcastUint8x16(0x0F)
	eight := archsimd.BroadcastInt8x16(8)
	blockDot := func(b int) archsimd.Int32x8 {
		v := archsimd.LoadUint8x16(data[b*16:])
		// (n ^ 8) - 8 sign-extends a 4-bit two's-complement value held in a byte.
		even := v.And(mask).BitsToInt8().Xor(eight).Sub(eight).ExtendToInt16()
		odd := v.ReshapeToUint16s().ShiftAllRight(4).ReshapeToUint8s().And(mask).BitsToInt8().Xor(eight).Sub(eight).ExtendToInt16()
		base := b * 32
		return even.DotProductPairs(archsimd.LoadInt16x16(xPair[base:])).
			Add(odd.DotProductPairs(archsimd.LoadInt16x16(xPair[base+16:])))
	}
	b := 0
	// Four blocks at a time: a horizontal-add tree reduces their eight-lane
	// partial sums to four exact block dots.
	for ; b+3 < blocks; b += 4 {
		h := blockDot(b).ConcatAddPairsGrouped(blockDot(b + 1)).
			ConcatAddPairsGrouped(blockDot(b + 2).ConcatAddPairsGrouped(blockDot(b + 3)))
		h.GetLo().Add(h.GetHi()).StoreArray((*[4]int32)(out[b : b+4]))
	}
	for ; b < blocks; b++ {
		blockDot(b).StoreArray(&tmp)
		out[b] = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7]
	}
	endAVX() // the caller folds out with scalar float maths
}

func dotBlocksPacked4Int16Scalar(data []byte, xPair []int16, out []int32, blocks int) {
	for b := range blocks {
		var sum int32
		for i := range 16 {
			v := data[b*16+i]
			even := int32(int8(v<<4) >> 4)
			odd := int32(int8(v) >> 4)
			sum += even*int32(xPair[b*32+i]) + odd*int32(xPair[b*32+16+i])
		}
		out[b] = sum
	}
}

// packed4RowScales fills ws with row r's per-block weight scales, computed
// with exactly the float operations BuildQuantCache uses, so a packed product
// matches the cached one bit for bit.
func packed4RowScales(ws []float32, layout instance.QuantLayout, raw []byte, r int) {
	bpr := layout.BlocksPerRow
	scalesRaw := raw[layout.ScaleOff : layout.ScaleOff+layout.ScaleCount*2]
	u16 := func(off int) uint16 { return uint16(scalesRaw[off]) | uint16(scalesRaw[off+1])<<8 }
	switch layout.Family {
	case 'q':
		for b := range ws {
			ws[b] = instance.FP16BitsToFloat32(u16((r*bpr + b) * 2))
		}
	case 'k':
		subRaw := raw[layout.SubScaleOff : layout.SubScaleOff+layout.SubScaleCount]
		spr := layout.SuperBlocksPerRow
		for sb := 0; sb*8 < bpr; sb++ {
			super := instance.FP16BitsToFloat32(u16((r*spr + sb) * 2))
			for b := sb * 8; b < min(sb*8+8, bpr); b++ {
				u6 := subRaw[r*bpr+b] & 0x3F
				if super == 0 || u6 == 0 {
					ws[b] = 0
					continue
				}
				ws[b] = super * (float32(u6) / 32.0)
			}
		}
	case 'g':
		gpr := layout.GroupsPerRow
		blocksPerGroup := layout.Group / 32
		for g := 0; g*blocksPerGroup < bpr; g++ {
			s := mcf.BF16ToFloat32(u16((r*gpr + g) * 2))
			for b := g * blocksPerGroup; b < min((g+1)*blocksPerGroup, bpr); b++ {
				ws[b] = s
			}
		}
	}
}

// packed4GroupRows is the number of rows matVecRangePacked4 folds together,
// one Float32x8 lane per row.
const packed4GroupRows = 8

// packed4GroupMaxBlocks bounds the row width of the grouped path, whose
// block-major scale products (8 rows x blocks float32) live on the stack.
const packed4GroupMaxBlocks = 256

// matVecRangePacked4 computes dst[r] for rows [rs, re) of a packed 4-bit
// matrix against int16-quantized activations qx. It produces the same bits as
// matVecRangeQuantCached on the expanded matrix: identical integer block
// dots, then sum += float32(dot) * (wScale * xScale) in block order.
//
// On AVX2 hosts, rows run in groups of packed4GroupRows. Each group's scale
// products are computed first, in scalar code, into block-major order; then
// matVecPacked4x8 folds the group with one SIMD lane per row, so the eight
// per-row sums advance together instead of as eight serial chains of scalar
// adds. Each lane performs exactly the scalar loop's operations in its order.
func matVecRangePacked4(dst []float32, w *Mat, rs, re int, qx *QuantVec) {
	layout, err := instance.QuantLayoutForMat(w.R, w.C, w.DType, len(w.Raw))
	if err != nil {
		panic(err)
	}
	bpr := layout.BlocksPerRow
	data := w.Raw[layout.DataOff : layout.DataOff+layout.DataBytes]
	xPair := qx.Q16Pair[:bpr*32]
	xs := qx.Scales[:bpr]

	var dotStack [512]int32
	var wsStack [512]float32
	var dots []int32
	var ws []float32
	if bpr <= len(dotStack) {
		dots, ws = dotStack[:bpr], wsStack[:bpr]
	} else {
		dots, ws = make([]int32, bpr), make([]float32, bpr)
	}
	r := rs
	if cpu.HasAVX2 && bpr <= packed4GroupMaxBlocks {
		endAVX() // the scale products below are scalar float maths
		var scStack [packed4GroupRows * packed4GroupMaxBlocks]float32
		sc := scStack[:packed4GroupRows*bpr]
		rowBytes := bpr * 16
		vecScales := layout.Family == 'k' && bpr%8 == 0
		for ; r+packed4GroupRows <= re; r += packed4GroupRows {
			if vecScales {
				packed4GroupScalesK(sc, layout, w.Raw, r, xs)
			} else {
				for j := range packed4GroupRows {
					packed4RowScales(ws, layout, w.Raw, r+j)
					for b, s := range ws {
						sc[b*packed4GroupRows+j] = s * xs[b]
					}
				}
			}
			matVecPacked4x8((*[packed4GroupRows]float32)(dst[r:r+packed4GroupRows]),
				data[r*rowBytes:(r+packed4GroupRows)*rowBytes], xPair, sc, bpr)
		}
	}
	for ; r < re; r++ {
		dotBlocksPacked4Int16(data[r*bpr*16:(r+1)*bpr*16], xPair, dots, bpr)
		packed4RowScales(ws, layout, w.Raw, r)
		var sum float32
		for b, d := range dots {
			sum += float32(d) * (ws[b] * xs[b])
		}
		dst[r] = sum
	}
}

// packed4GroupScalesK fills sc[b*8+j] = wScale[r+j][b] * xs[b] for the eight
// k-family rows starting at r, the block-major scale products matVecPacked4x8
// reads. A k superblock is exactly eight blocks, so each row's eight block
// scales for one superblock are computed in one vector,
// super * (float32(u6) / 32), and an 8x8 transpose turns the eight row
// vectors into eight block vectors. bpr must be a multiple of 8.
//
// The products carry the bits packed4RowScales and the scalar loop produce,
// with one tolerated difference: a zero superblock scale yields a signed zero
// where the scalar code writes +0. Every product only ever enters a row sum as
// float32(dot) * product, and adding a zero of either sign leaves a float32
// sum that started at +0 unchanged, so no output bit can differ. A zero u6 is
// masked to +0 as the scalar code does, which also keeps an infinite or NaN
// superblock scale from leaking into those blocks.
func packed4GroupScalesK(sc []float32, layout instance.QuantLayout, raw []byte, r int, xs []float32) {
	bpr := layout.BlocksPerRow
	spr := bpr / 8
	scalesRaw := raw[layout.ScaleOff : layout.ScaleOff+layout.ScaleCount*2]
	subRaw := raw[layout.SubScaleOff : layout.SubScaleOff+layout.SubScaleCount]
	sc = sc[:bpr*packed4GroupRows]
	xs = xs[:bpr]

	// The superblock scales are looked up first, in scalar code, so no scalar
	// float load runs between the vector instructions below (see endAVX).
	var supers [packed4GroupRows * packed4GroupMaxBlocks / 8]float32
	for j := range packed4GroupRows {
		for sb := range spr {
			off := ((r+j)*layout.SuperBlocksPerRow + sb) * 2
			supers[j*spr+sb] = instance.FP16BitsToFloat32(uint16(scalesRaw[off]) | uint16(scalesRaw[off+1])<<8)
		}
	}

	low6 := archsimd.BroadcastUint32x8(0x3F)
	var zeroU archsimd.Uint32x8
	oneOver32 := archsimd.BroadcastFloat32x8(1.0 / 32)
	for sb := range spr {
		xsv := archsimd.LoadFloat32x8Array((*[8]float32)(xs[sb*8 : sb*8+8]))
		// Row j's eight products for this superblock. The rows are named
		// values rather than an array, which Go would zero with legacy-SSE
		// stores in the middle of the vector code.
		row := func(j int) archsimd.Uint32x8 {
			u6 := loadSubScales8(subRaw, (r+j)*bpr+sb*8).ExtendLo8ToUint32().And(low6)
			q := u6.BitsToInt32().ConvertToFloat32().Mul(oneOver32)
			super := archsimd.BroadcastFloat32x8(supers[j*spr+sb])
			return super.Mul(q).Masked(u6.NotEqual(zeroU)).Mul(xsv).ToBits()
		}
		v0, v1, v2, v3 := row(0), row(1), row(2), row(3)
		v4, v5, v6, v7 := row(4), row(5), row(6), row(7)
		// 8x8 transpose: row vectors v0..v7 become block vectors.
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
		// b0..b3 hold blocks 0-3 of rows 0-3 in their low halves and blocks
		// 4-7 in their high halves; b4..b7 the same for rows 4-7.
		out := sc[sb*8*packed4GroupRows : (sb+1)*8*packed4GroupRows]
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
}

// loadSubScales8 returns a vector whose low eight bytes are sub[off:off+8].
// The bytes travel as one uint64 through a general register, which never reads
// past the slice and has no branch: a branch merging two vector loads makes Go
// copy the result with a legacy-SSE move (see endAVX).
func loadSubScales8(sub []byte, off int) archsimd.Uint8x16 {
	return archsimd.BroadcastUint64x2(binary.LittleEndian.Uint64(sub[off : off+8])).ReshapeToUint8s()
}

// matVecPacked4x8 sets out[j] to row j's product for the eight consecutive
// packed rows in data (bpr 16-byte blocks per row), given the activations in
// pair order and the block-major scale products sc[b*8+j] = wScale[j][b] *
// xScale[b]. Per block, the activations are loaded once for all eight rows,
// the eight exact int32 dots are reduced into one vector by the horizontal-add
// tree quantDot2x4 uses, and each lane folds its dot in with
// sum += float32(dot) * sc, in block order.
func matVecPacked4x8(out *[packed4GroupRows]float32, data []byte, xPair []int16, sc []float32, bpr int) {
	rowBytes := bpr * 16
	data = data[:packed4GroupRows*rowBytes]
	xPair = xPair[:bpr*32]
	sc = sc[:bpr*packed4GroupRows]
	mask := archsimd.BroadcastUint8x16(0x0F)
	eight := archsimd.BroadcastInt8x16(8)
	sum := archsimd.LoadFloat32x8Array(&zeroFloat32s)
	for b := range bpr {
		xe := archsimd.LoadInt16x16(xPair[b*32:])
		xo := archsimd.LoadInt16x16(xPair[b*32+16:])
		rowDot := func(j int) archsimd.Int32x8 {
			v := archsimd.LoadUint8x16(data[j*rowBytes+b*16:])
			// (n ^ 8) - 8 sign-extends a 4-bit two's-complement value held in a byte.
			even := v.And(mask).BitsToInt8().Xor(eight).Sub(eight).ExtendToInt16()
			odd := v.ReshapeToUint16s().ShiftAllRight(4).ReshapeToUint8s().And(mask).BitsToInt8().Xor(eight).Sub(eight).ExtendToInt16()
			return even.DotProductPairs(xe).Add(odd.DotProductPairs(xo))
		}
		t := rowDot(0).ConcatAddPairsGrouped(rowDot(1)).ConcatAddPairsGrouped(rowDot(2).ConcatAddPairsGrouped(rowDot(3)))
		u := rowDot(4).ConcatAddPairsGrouped(rowDot(5)).ConcatAddPairsGrouped(rowDot(6).ConcatAddPairsGrouped(rowDot(7)))
		// Each tree's dots are its low half plus its high half.
		dots := t.SetHi(u.GetLo()).Add(u.SetLo(t.GetHi())).ConvertToFloat32()
		sum = sum.Add(dots.Mul(archsimd.LoadFloat32x8Array((*[8]float32)(sc[b*packed4GroupRows : b*packed4GroupRows+8]))))
	}
	sum.StoreArray(out)
	endAVX()
}

// decodePacked4Row expands blocks packed 4-bit blocks into dst (blocks*32
// int8 codes), each block in pair order: its 16 even elements, then its 16
// odd ones.
func decodePacked4Row(dst []int8, data []byte, blocks int) {
	mask := archsimd.BroadcastUint8x16(0x0F)
	eight := archsimd.BroadcastInt8x16(8)
	var zero archsimd.Int8x32
	for b := range blocks {
		v := archsimd.LoadUint8x16(data[b*16:])
		even := v.And(mask).BitsToInt8().Xor(eight).Sub(eight)
		odd := v.ReshapeToUint16s().ShiftAllRight(4).ReshapeToUint8s().And(mask).BitsToInt8().Xor(eight).Sub(eight)
		zero.SetLo(even).SetHi(odd).StoreArray((*[32]int8)(dst[b*32 : b*32+32]))
	}
	endAVX() // packed4RowScales, scalar float maths, follows every call
}
