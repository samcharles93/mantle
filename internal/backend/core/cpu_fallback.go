package core

import (
	"fmt"
	"math"

	"github.com/samcharles93/mantle/pkg/mcf"
)

// MatVecWithQuant computes dst = w * x using optional pre-quantized x.
func MatVecWithQuant(dst []float32, w *Mat, x []float32, qx *QuantVec) {
	if w == nil || w.R == 0 || w.C == 0 {
		return
	}
	if len(dst) < w.R || len(x) < w.C {
		panic("matvec shape mismatch")
	}

	if w.Raw != nil && mcf.DTypeRequiresAligned64(w.DType) {
		if w.Quant == nil || !w.Quant.ValidFor(w) {
			qc, err := BuildQuantCache(w)
			if err != nil {
				panic(fmt.Errorf("quant matvec cache build failed: %w", err))
			}
			w.Quant = qc
		}
		matVecQuantCached(dst, w, x, qx)
		return
	}

	row := make([]float32, w.C)
	for r := range w.R {
		if w.Raw == nil || w.DType == mcf.DTypeF32 {
			base := r * w.Stride
			dst[r] = dotFloat32(w.Data[base:base+w.C], x[:w.C])
			continue
		}
		w.RowTo(row, r)
		dst[r] = dotFloat32(row, x[:w.C])
	}
}

func matVecQuantCached(dst []float32, w *Mat, x []float32, qx *QuantVec) {
	qc := w.Quant
	if qc == nil {
		panic("quant cache is nil")
	}
	blocksPerRow := qc.BlocksPerRow
	useInt8 := qx != nil && len(qx.Q) >= blocksPerRow*32 && len(qx.Q16) >= blocksPerRow*32 && len(qx.Scales) >= blocksPerRow

	for r := range w.R {
		blockBase := r * blocksPerRow
		rowOff := blockBase * 32
		qRow := qc.Q[rowOff : rowOff+blocksPerRow*32]
		scales := qc.Scales[blockBase : blockBase+blocksPerRow]

		var sum float32
		for b := range blocksPerRow {
			col := b * 32
			n := w.C - col
			if n <= 0 {
				break
			}
			if n > 32 {
				n = 32
			}
			scale := scales[b]
			if scale == 0 {
				continue
			}
			off := b * 32
			qBlock := qRow[off : off+32]
			if useInt8 {
				xScale := qx.Scales[b]
				if xScale == 0 {
					continue
				}
				xBlock := qx.Q16[off : off+32]
				dot := dotInt8Int16(qBlock, xBlock, 32)
				sum += float32(dot) * (scale * xScale)
			} else {
				sum += scale * dotInt8Float32(qBlock, x[col:], n)
			}
		}
		dst[r] = sum
	}
}

func dotInt8Int16(a []int8, b []int16, n int) int32 {
	var sum int32
	for i := range n {
		sum += int32(a[i]) * int32(b[i])
	}
	return sum
}

func dotInt8Float32(a []int8, b []float32, n int) float32 {
	var sum float32
	for i := range n {
		sum += float32(a[i]) * b[i]
	}
	return sum
}

func dotFloat32(a []float32, b []float32) float32 {
	var sum float32
	for i := range len(a) {
		sum += a[i] * b[i]
	}
	return sum
}

// Softmax applies numerically-stable softmax in place.
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
	inv := float32(1.0) / sum
	for i := range x {
		x[i] *= inv
	}
}

func fastExp(x float32) float32 {
	if x > 88.0 {
		return 3.4028235e38
	}
	if x < -88.0 {
		return 0.0
	}

	const ln2 = 0.693147180559945309417
	const ln2Inv = 1.44269504088896340736

	k := int32(x*ln2Inv + 0.5)
	if x < 0 {
		k = int32(x*ln2Inv - 0.5)
	}

	r := x - float32(k)*ln2
	r2 := r * r
	r3 := r2 * r
	r4 := r2 * r2

	poly := 1.0 + r + r2*0.5 + r3*0.16666667 + r4*0.041666667 + (r4*r)*0.008333333
	exp := uint32(k+127) << 23
	scale := math.Float32frombits(exp)

	return poly * scale
}
