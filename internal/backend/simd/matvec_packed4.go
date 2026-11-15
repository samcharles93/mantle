package simd

import (
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
		var tmp [8]int32
		blockDot(b).StoreArray(&tmp)
		out[b] = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7]
	}
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

// matVecRangePacked4 computes dst[r] for rows [rs, re) of a packed 4-bit
// matrix against int16-quantized activations qx. It produces the same bits as
// matVecRangeQuantCached on the expanded matrix: identical integer block
// dots, then sum += float32(dot) * (wScale * xScale) in block order.
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
	for r := rs; r < re; r++ {
		dotBlocksPacked4Int16(data[r*bpr*16:(r+1)*bpr*16], xPair, dots, bpr)
		packed4RowScales(ws, layout, w.Raw, r)
		var sum float32
		for b, d := range dots {
			sum += float32(d) * (ws[b] * xs[b])
		}
		dst[r] = sum
	}
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
}
