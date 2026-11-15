package simd

import (
	"simd/archsimd"

	instance "github.com/samcharles93/mantle/internal/backend/core"
	"github.com/samcharles93/mantle/pkg/mcf"
)

// Packed 4-bit matvec. q4, k4 and gq*4 store 32 signed 4-bit codes per block
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
	for b := range blocks {
		v := archsimd.LoadUint8x16(data[b*16:])
		// (n ^ 8) - 8 sign-extends a 4-bit two's-complement value held in a byte.
		even := v.And(mask).BitsToInt8().Xor(eight).Sub(eight).ExtendToInt16()
		odd := v.ReshapeToUint16s().ShiftAllRight(4).ReshapeToUint8s().And(mask).BitsToInt8().Xor(eight).Sub(eight).ExtendToInt16()
		base := b * 32
		d := even.DotProductPairs(archsimd.LoadInt16x16(xPair[base:])).
			Add(odd.DotProductPairs(archsimd.LoadInt16x16(xPair[base+16:])))
		var tmp [8]int32
		d.StoreArray(&tmp)
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

// matVecRangePacked4 computes dst[r] for rows [rs, re) of a packed 4-bit
// matrix against int16-quantized activations qx. Each row first gets its
// combined scales (weight scale times activation scale per block), then one
// pass turns block dots into floats and accumulates them in a vector
// register, with a single horizontal sum per row.
func matVecRangePacked4(dst []float32, w *Mat, rs, re int, qx *QuantVec) {
	layout, err := instance.QuantLayoutForMat(w.R, w.C, w.DType, len(w.Raw))
	if err != nil {
		panic(err)
	}
	bpr := layout.BlocksPerRow
	data := w.Raw[layout.DataOff : layout.DataOff+layout.DataBytes]
	scalesRaw := w.Raw[layout.ScaleOff : layout.ScaleOff+layout.ScaleCount*2]
	var subRaw []byte
	if layout.Family == 'k' {
		subRaw = w.Raw[layout.SubScaleOff : layout.SubScaleOff+layout.SubScaleCount]
	}
	xPair := qx.Q16Pair[:bpr*32]
	xs := qx.Scales[:bpr]

	var stack [512]float32
	var scales []float32
	if bpr <= len(stack) {
		scales = stack[:bpr]
	} else {
		scales = make([]float32, bpr)
	}
	u16 := func(off int) uint16 { return uint16(scalesRaw[off]) | uint16(scalesRaw[off+1])<<8 }
	for r := rs; r < re; r++ {
		switch layout.Family {
		case 'q':
			for b := range scales {
				scales[b] = instance.FP16BitsToFloat32(u16((r*bpr+b)*2)) * xs[b]
			}
		case 'k':
			spr := layout.SuperBlocksPerRow
			for sb := 0; sb*8 < bpr; sb++ {
				super := instance.FP16BitsToFloat32(u16((r*spr+sb)*2)) / 32
				for b := sb * 8; b < min(sb*8+8, bpr); b++ {
					scales[b] = super * float32(subRaw[r*bpr+b]&0x3F) * xs[b]
				}
			}
		case 'g':
			gpr := layout.GroupsPerRow
			blocksPerGroup := layout.Group / 32
			for g := 0; g*blocksPerGroup < bpr; g++ {
				s := mcf.BF16ToFloat32(u16((r*gpr + g) * 2))
				for b := g * blocksPerGroup; b < min((g+1)*blocksPerGroup, bpr); b++ {
					scales[b] = s * xs[b]
				}
			}
		}
		dst[r] = dotRowPacked4(data[r*bpr*16:(r+1)*bpr*16], xPair, scales)
	}
}

// dotRowPacked4 returns sum over blocks of dot(block codes, activations) *
// scales[b], with activations in pair order.
func dotRowPacked4(data []byte, xPair []int16, scales []float32) float32 {
	if !cpu.HasAVX2 {
		var dots [1]int32
		var sum float32
		for b, s := range scales {
			dotBlocksPacked4Int16Scalar(data[b*16:], xPair[b*32:], dots[:], 1)
			sum += float32(dots[0]) * s
		}
		return sum
	}
	mask := archsimd.BroadcastUint8x16(0x0F)
	eight := archsimd.BroadcastInt8x16(8)
	acc := archsimd.BroadcastFloat32x8(0)
	for b, s := range scales {
		v := archsimd.LoadUint8x16(data[b*16:])
		even := v.And(mask).BitsToInt8().Xor(eight).Sub(eight).ExtendToInt16()
		odd := v.ReshapeToUint16s().ShiftAllRight(4).ReshapeToUint8s().And(mask).BitsToInt8().Xor(eight).Sub(eight).ExtendToInt16()
		base := b * 32
		d := even.DotProductPairs(archsimd.LoadInt16x16(xPair[base:])).
			Add(odd.DotProductPairs(archsimd.LoadInt16x16(xPair[base+16:])))
		acc = d.ConvertToFloat32().MulAdd(archsimd.BroadcastFloat32x8(s), acc)
	}
	var tmp [8]float32
	acc.StoreArray(&tmp)
	return tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7]
}
