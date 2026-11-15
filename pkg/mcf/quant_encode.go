package mcf

import (
	"errors"
	"fmt"
	"math"
	"runtime"
	"sync"
)

// Symmetric scalar encoders for the q* and k* families.
//
// Each 32-element block gets the scale that minimises its reconstruction
// error weighted by x^2, found by scanning a grid of candidate scales and
// refining the best one with least squares, rather than the absmax scale. The
// x^2 weight protects large weights; on LFM3-1.7B k4 it measured closest to
// bf16 by next-token KL of the weightings tried, with the grid kept near
// absmax (from 0.85) so large weights are rarely clipped.
// Codes span the full two's-complement range of the bit width ([-8, 7] at 4
// bits); 8-bit codes stay in [-127, 127]. Every candidate is scored against
// the scale exactly as it will be stored, so the payload holds what was scored.

// QuantCodeRange returns the signed code range writers use for a bit width.
func QuantCodeRange(bits int) (lo, hi int) {
	if bits >= 8 {
		return -127, 127
	}
	n := 1 << (bits - 1)
	return -n, n - 1
}

// scaleGrid holds the candidate span for the scale search, as fractions of the
// largest code magnitude that the group's largest |x| maps onto.
type scaleGrid struct {
	from, to float64
	steps    int
}

func gridForBits(bits int) scaleGrid {
	if bits >= 8 {
		return scaleGrid{from: 0.9, to: 1.0, steps: 10}
	}
	return scaleGrid{from: 0.85, to: 1.2, steps: 24}
}

// bestScale returns the scale s minimising sum(x[i]^2 * (x[i] - s*q[i])^2), where
// q[i] = clamp(round(x[i]/s), lo, hi). With signed set, s may be negative and
// its sign is chosen so the largest-magnitude element maps towards lo, the
// code with the larger magnitude. Without it, s is never negative. It returns
// 0 for an all-zero group.
func bestScale(x []float32, lo, hi int, signed bool, grid scaleGrid) float32 {
	var amax, xm float32
	for _, v := range x {
		if a := float32(math.Abs(float64(v))); a > amax {
			amax, xm = a, v
		}
	}
	if amax == 0 {
		return 0
	}
	sign := 1.0
	if signed && xm > 0 && -lo > hi {
		sign = -1
	}
	nmax := float64(-lo)
	bestScore := -1.0
	bestS := 0.0
	try := func(t float64) {
		score, s := fitScore(x, sign*t/float64(amax), lo, hi)
		if score > bestScore {
			bestScore, bestS = score, s
		}
	}
	// The absmax scale (largest |x| on the largest positive code) and the
	// full-range scale (largest |x| on lo) are always candidates, so the
	// result is never worse than plain absmax rounding.
	try(float64(hi))
	try(nmax)
	for i := 0; i <= grid.steps; i++ {
		try(nmax * (grid.from + (grid.to-grid.from)*float64(i)/float64(grid.steps)))
	}
	for range 2 {
		if bestS == 0 {
			break
		}
		score, s := fitScore(x, 1/bestS, lo, hi)
		if score <= bestScore {
			break
		}
		bestScore, bestS = score, s
	}
	return float32(bestS)
}

// fitScore quantizes x with the inverse scale iscale and returns the
// x^2-weighted least-squares scale for the resulting codes together with its
// score (sum(w*x*q))^2 / sum(w*q*q), w = x^2. Maximising the score minimises
// the weighted squared error.
func fitScore(x []float32, iscale float64, lo, hi int) (float64, float64) {
	var sxq, sqq float64
	for _, v := range x {
		q := float64(clampCode(math.Round(float64(v)*iscale), lo, hi))
		w := float64(v) * float64(v)
		sxq += w * float64(v) * q
		sqq += w * q * q
	}
	if sqq == 0 {
		return -1, 0
	}
	return sxq * sxq / sqq, sxq / sqq
}

func clampCode(r float64, lo, hi int) int {
	if r < float64(lo) {
		return lo
	}
	if r > float64(hi) {
		return hi
	}
	return int(r)
}

// quantizeWithScale fills q with the nearest codes for x under scale s and
// returns the x^2-weighted squared reconstruction error. A zero scale yields
// zero codes.
func quantizeWithScale(x []float32, q []int8, s float32, lo, hi int) float64 {
	clear(q)
	if s == 0 {
		var e float64
		for _, v := range x {
			v2 := float64(v) * float64(v)
			e += v2 * v2
		}
		return e
	}
	inv := 1 / float64(s)
	var e float64
	for i, v := range x {
		c := clampCode(math.Round(float64(v)*inv), lo, hi)
		q[i] = int8(c)
		d := float64(v) - float64(s)*float64(c)
		e += float64(v) * float64(v) * d * d
	}
	return e
}

// packCodes writes 32 signed codes as bits-wide two's-complement fields, LSB
// first, into dst, which must hold 32*bits/8 bytes.
func packCodes(dst []byte, q *[32]int8, bits int) {
	clear(dst)
	switch bits {
	case 8:
		for i, v := range q {
			dst[i] = byte(v)
		}
	case 4:
		for i := 0; i < 32; i += 2 {
			dst[i/2] = byte(q[i])&0x0F | byte(q[i+1])<<4
		}
	default:
		mask := uint32(1)<<uint(bits) - 1
		bitPos := 0
		for _, v := range q {
			u := uint32(uint8(v)) & mask
			for b := range bits {
				if u>>uint(b)&1 == 1 {
					dst[bitPos>>3] |= 1 << uint(bitPos&7)
				}
				bitPos++
			}
		}
	}
}

// parallelRows runs fn over [0, rows) split into contiguous ranges, one per
// worker. Each row's output depends only on that row, so the result is the
// same for any split.
func parallelRows(rows int, fn func(r0, r1 int)) {
	workers := min(runtime.GOMAXPROCS(0), rows)
	if workers <= 1 {
		fn(0, rows)
		return
	}
	chunk := (rows + workers - 1) / workers
	var wg sync.WaitGroup
	for r0 := 0; r0 < rows; r0 += chunk {
		wg.Go(func() { fn(r0, min(r0+chunk, rows)) })
	}
	wg.Wait()
}

func checkEncodeShape(src []float32, rows, cols int) error {
	if rows <= 0 || cols <= 0 {
		return errors.New("mcf: invalid quant tensor shape")
	}
	if rows > math.MaxInt/cols {
		return errors.New("mcf: quant tensor too large")
	}
	if len(src) != rows*cols {
		return fmt.Errorf("mcf: quant tensor has %d values, want %d", len(src), rows*cols)
	}
	for _, v := range src {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return errors.New("mcf: quant tensor contains NaN or Inf")
		}
	}
	return nil
}

// QDTypeForBits maps a q* bit width to its dtype.
func QDTypeForBits(bits int) (TensorDType, error) {
	switch bits {
	case 8:
		return DTypeQ8, nil
	case 4:
		return DTypeQ4, nil
	}
	return DTypeUnknown, fmt.Errorf("mcf: unsupported q* bit width %d", bits)
}

// KDTypeForBits maps a k* bit width to its dtype.
func KDTypeForBits(bits int) (TensorDType, error) {
	switch bits {
	case 6:
		return DTypeK6, nil
	case 4:
		return DTypeK4, nil
	case 3:
		return DTypeK3, nil
	case 2:
		return DTypeK2, nil
	}
	return DTypeUnknown, fmt.Errorf("mcf: unsupported k* bit width %d", bits)
}

// EncodeQTensor encodes a row-major rank-2 tensor as a q4 or q8 payload: one
// f16 scale per 32-element block, then the packed codes. Block scales may be
// negative, which lets the largest-magnitude element of every block use the
// most negative code.
func EncodeQTensor(src []float32, rows, cols, bits int) ([]byte, error) {
	dt, err := QDTypeForBits(bits)
	if err != nil {
		return nil, err
	}
	if err := checkEncodeShape(src, rows, cols); err != nil {
		return nil, err
	}
	size, err := QuantPayloadSize([]uint64{uint64(rows), uint64(cols)}, dt)
	if err != nil {
		return nil, err
	}
	blocksPerRow := (cols + 31) / 32
	blockBytes := 32 * bits / 8
	dataOff, _ := align64(uint64(rows*blocksPerRow) * 2)
	payload := make([]byte, size)
	scales := payload[:dataOff]
	data := payload[dataOff:]
	lo, hi := QuantCodeRange(bits)
	grid := gridForBits(bits)

	parallelRows(rows, func(r0, r1 int) {
		var q [32]int8
		for r := r0; r < r1; r++ {
			row := src[r*cols : (r+1)*cols]
			for b := range blocksPerRow {
				x := row[b*32 : min((b+1)*32, cols)]
				s := bestScale(x, lo, hi, true, grid)
				h := float32ToFP16Bits(s)
				quantizeWithScale(x, q[:len(x)], fp16ToFloat32(h), lo, hi)
				clear(q[len(x):])
				blockIdx := r*blocksPerRow + b
				putF16LE(scales, uint64(blockIdx)*2, h)
				packCodes(data[blockIdx*blockBytes:(blockIdx+1)*blockBytes], &q, bits)
			}
		}
	})
	return payload, nil
}

// EncodeKTensor encodes a row-major rank-2 tensor as a k2, k3, k4 or k6
// payload. Each 256-element super-block stores an f16 scale S and each
// 32-element block a 6-bit sub-scale u, giving block scale S*u/32. S is set so
// the largest block scale lands on u = 63, and each block then takes the u
// near its own best scale with the lowest error.
func EncodeKTensor(src []float32, rows, cols, bits int) ([]byte, error) {
	dt, err := KDTypeForBits(bits)
	if err != nil {
		return nil, err
	}
	if err := checkEncodeShape(src, rows, cols); err != nil {
		return nil, err
	}
	size, err := QuantPayloadSize([]uint64{uint64(rows), uint64(cols)}, dt)
	if err != nil {
		return nil, err
	}
	blocksPerRow := (cols + 31) / 32
	superPerRow := (blocksPerRow + 7) / 8
	totalBlocks := rows * blocksPerRow
	blockBytes := 32 * bits / 8
	subOff, _ := align64(uint64(rows*superPerRow) * 2)
	dataOff, _ := align64(subOff + uint64(totalBlocks))
	payload := make([]byte, size)
	supers := payload[:subOff]
	subs := payload[subOff : subOff+uint64(totalBlocks)]
	data := payload[dataOff:]
	lo, hi := QuantCodeRange(bits)
	grid := gridForBits(bits)

	parallelRows(rows, func(r0, r1 int) {
		var q, bestQ [32]int8
		var blockS [8]float32
		for r := r0; r < r1; r++ {
			row := src[r*cols : (r+1)*cols]
			for sb := range superPerRow {
				b0 := sb * 8
				b1 := min(b0+8, blocksPerRow)
				var smax float32
				for b := b0; b < b1; b++ {
					x := row[b*32 : min((b+1)*32, cols)]
					blockS[b-b0] = bestScale(x, lo, hi, false, grid)
					smax = max(smax, blockS[b-b0])
				}
				h := float32ToFP16Bits(smax * 32 / 63)
				super := fp16ToFloat32(h)
				putF16LE(supers, uint64(r*superPerRow+sb)*2, h)
				for b := b0; b < b1; b++ {
					x := row[b*32 : min((b+1)*32, cols)]
					u := fitSubScale(x, &q, &bestQ, blockS[b-b0], super, lo, hi)
					blockIdx := r*blocksPerRow + b
					subs[blockIdx] = u
					packCodes(data[blockIdx*blockBytes:(blockIdx+1)*blockBytes], &bestQ, bits)
				}
			}
		}
	})
	return payload, nil
}

// fitSubScale picks the 6-bit sub-scale u near s/super*32 whose block scale
// super*u/32 gives the lowest error for x, leaving the codes in bestQ.
func fitSubScale(x []float32, q, bestQ *[32]int8, s, super float32, lo, hi int) uint8 {
	clear(bestQ[:])
	if super == 0 || s == 0 {
		return 0
	}
	base := int(math.Floor(float64(s / super * 32)))
	bestErr := math.Inf(1)
	bestU := 0
	for u := max(base-1, 0); u <= min(base+2, 63); u++ {
		e := quantizeWithScale(x, q[:len(x)], super*float32(u)/32, lo, hi)
		if e < bestErr {
			bestErr, bestU = e, u
			copy(bestQ[:], q[:len(x)])
		}
	}
	return uint8(bestU)
}
