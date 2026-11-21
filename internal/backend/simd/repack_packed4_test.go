package simd

import (
	"fmt"
	"math"
	"testing"

	instance "github.com/samcharles93/mantle/internal/backend/core"
	"github.com/samcharles93/mantle/pkg/mcf"
)

// TestMatMatQuantRepacked4x8MatchesMatVec verifies that BuildPacked4x8 plus the
// repacked 8x4/8x2/8x1 kernels (and R%8 tail fallback) match per-row MatVec bit
// for bit across every packed 4-bit dtype and every batch tail shape.
func TestMatMatQuantRepacked4x8MatchesMatVec(t *testing.T) {
	if !cpu.HasAVX2 {
		t.Skip("repacked 4x8 kernels need AVX2")
	}
	for _, dt := range packed4DTypes(t) {
		for _, rows := range []int{64, 67} {
			for _, n := range []int{1, 2, 3, 4, 6, 7, 33} {
				const cols = 768
				t.Run(fmt.Sprintf("%#x_r%d_n%d", dt, rows, n), func(t *testing.T) {
					w := packed4Mat(t, dt, rows, cols)
					p4, err := BuildPacked4x8(&w)
					if err != nil {
						t.Fatalf("BuildPacked4x8: %v", err)
					}
					w.Packed4x8 = p4
					if !batchWeightsRepacked4x8(&w) {
						t.Fatalf("batchWeightsRepacked4x8 returned false")
					}
					x := quantTestActivations(n, cols)
					got := make([]float32, n*rows)
					MatMatQuant(got, &w, x, n)

					want := make([]float32, rows)
					for r := range n {
						MatVec(want, &w, x[r*cols:(r+1)*cols])
						for i, v := range want {
							if g := got[r*rows+i]; math.Float32bits(g) != math.Float32bits(v) {
								t.Fatalf("row %d col %d: repacked MatMatQuant = %v (%#x), MatVec = %v (%#x)",
									r, i, g, math.Float32bits(g), v, math.Float32bits(v))
							}
						}
					}
				})
			}
		}
	}
}

// TestSharedQuantRowsMixedRepackedMatchesMatVec covers mixed-recipe groups
// where a repacked k4 weight and a cached k6 weight share one quantRows.
func TestSharedQuantRowsMixedRepackedMatchesMatVec(t *testing.T) {
	if !cpu.HasAVX2 {
		t.Skip("repacked 4x8 kernels need AVX2")
	}
	for _, rows := range []int{64, 67} {
		const cols = 512
		wRepacked := packed4Mat(t, mcf.DTypeK4, rows, cols)
		p4, err := BuildPacked4x8(&wRepacked)
		if err != nil {
			t.Fatalf("BuildPacked4x8: %v", err)
		}
		wRepacked.Packed4x8 = p4
		wCached := cachedQuantMat(t, mcf.DTypeK6, rows, cols)

		for _, n := range []int{1, 2, 3, 4, 6, 7, 17} {
			t.Run(fmt.Sprintf("r%d_n%d", rows, n), func(t *testing.T) {
				x := quantTestActivations(n, cols)
				qr := prepareBatchQuantRows(x, n, cols, &wRepacked, wCached)
				defer putQuantRows(qr)
				if n >= 2 && (qr == nil || len(qr.q8) == 0 || len(qr.q32) == 0) {
					t.Fatalf("mixed repacked+cached weights must populate both q8 and q32")
				}
				if n >= 2 && rows%8 == 0 && len(qr.q8Pair) != 0 {
					t.Fatalf("multiple-of-8 repacked weight should not build q8Pair")
				}
				gotRep := make([]float32, n*rows)
				gotCached := make([]float32, n*rows)
				projectRowsWithQuant(gotRep, &wRepacked, x, n, qr)
				projectRowsWithQuant(gotCached, wCached, x, n, qr)

				want := make([]float32, rows)
				for r := range n {
					MatVec(want, &wRepacked, x[r*cols:(r+1)*cols])
					for i, v := range want {
						if g := gotRep[r*rows+i]; g != v {
							t.Fatalf("repacked row %d col %d: got %v want %v", r, i, g, v)
						}
					}
					MatVec(want, wCached, x[r*cols:(r+1)*cols])
					for i, v := range want {
						if g := gotCached[r*rows+i]; g != v {
							t.Fatalf("cached row %d col %d: got %v want %v", r, i, g, v)
						}
					}
				}
			})
		}
	}
}

// TestQuantDotRepacked8xWorstCaseInt16Bound drives every lane of
// quantDotRepacked8x4, quantDotRepacked8x2 and quantDotRepacked8x1 to the
// exact ±30,480 signed int16 peak (all weights 15, all activations ±127) and
// checks against an exact int64 reference.
func TestQuantDotRepacked8xWorstCaseInt16Bound(t *testing.T) {
	if !cpu.HasAVX2 {
		t.Skip("repacked 4x8 kernels need AVX2")
	}
	const blocks = 4
	const cols = blocks * 32

	// 8 rows: even rows all 15 (signed +7 -> hits ±30,480 in int16), odd rows all 0 (signed -8).
	wData := make([]byte, blocks*128)
	var rowU [8]uint8
	for j := range 8 {
		if j%2 == 0 {
			rowU[j] = packed4UnsignedMax
		} else {
			rowU[j] = 0
		}
	}
	for b := range blocks {
		outBlk := wData[b*128 : (b+1)*128]
		for s := range 4 {
			for j := range 8 {
				packed := rowU[j] | (rowU[j] << 4)
				for k := range 4 {
					outBlk[s*32+j*4+k] = packed
				}
			}
		}
	}

	wScales := make([]float32, blocks*8)
	for i := range wScales {
		wScales[i] = 1
	}

	pack4 := func(v int8) int32 {
		u := uint32(uint8(v))
		return int32(u | u<<8 | u<<16 | u<<24)
	}
	actVals := [4]int8{quantActMaxAbs, -quantActMaxAbs, -quantActMaxAbs, quantActMaxAbs}

	xQuad := make([]int32, blocks*32)
	xPair := make([]int32, blocks*16)
	xRow := make([]int32, blocks*8)
	xMeta4 := make([]uint32, blocks*8)
	xMeta2 := make([]uint32, blocks*8)
	xMeta1 := make([]uint32, blocks*8)

	oneBits := math.Float32bits(1)
	for b := range blocks {
		for k := range 8 {
			for m := range 4 {
				xQuad[b*32+k*4+m] = pack4(actVals[m])
			}
			xPair[b*16+k*2] = pack4(actVals[0])
			xPair[b*16+k*2+1] = pack4(actVals[1])
			xRow[b*8+k] = pack4(actVals[0])
		}
		for m := range 4 {
			xMeta4[b*8+m] = oneBits
			xMeta4[b*8+4+m] = uint32(int32(int64(actVals[m]) * 32 * 8))
		}
		xMeta2[b*8], xMeta2[b*8+1] = oneBits, oneBits
		xMeta2[b*8+4], xMeta2[b*8+5] = xMeta4[b*8+4], xMeta4[b*8+5]
		xMeta1[b*8] = oneBits
		xMeta1[b*8+4] = xMeta4[b*8+4]
	}

	want := func(j, m int) float32 {
		return float32((int64(rowU[j]) - 8) * int64(actVals[m]) * cols)
	}

	var out4 [32]float32
	quantDotRepacked8x4(wData, wScales, xQuad, xMeta4, &out4)
	for m := range 4 {
		for j := range 8 {
			if got, exp := out4[m*8+j], want(j, m); got != exp {
				t.Fatalf("quantDotRepacked8x4[row=%d, w=%d] = %v, want %v", m, j, got, exp)
			}
		}
	}

	var out2 [16]float32
	quantDotRepacked8x2(wData, wScales, xPair, xMeta2, &out2)
	for m := range 2 {
		for j := range 8 {
			if got, exp := out2[m*8+j], want(j, m); got != exp {
				t.Fatalf("quantDotRepacked8x2[row=%d, w=%d] = %v, want %v", m, j, got, exp)
			}
		}
	}

	var out1 [8]float32
	quantDotRepacked8x1(wData, wScales, xRow, xMeta1, &out1)
	for j := range 8 {
		if got, exp := out1[j], want(j, 0); got != exp {
			t.Fatalf("quantDotRepacked8x1[w=%d] = %v, want %v", j, got, exp)
		}
	}
}

// TestRepack4x8ZeroSuperblockAndZeroBlock exercises zero superblock scales and
// zero sub-scales in packed4GroupWeightScalesK to confirm parity with MatVec.
func TestRepack4x8ZeroSuperblockAndZeroBlock(t *testing.T) {
	if !cpu.HasAVX2 {
		t.Skip("repacked 4x8 kernels need AVX2")
	}
	const rows, cols = 16, 512
	w := packed4Mat(t, mcf.DTypeK4, rows, cols)
	layout, err := instance.QuantLayoutForMat(rows, cols, mcf.DTypeK4, len(w.Raw))
	if err != nil {
		t.Fatal(err)
	}
	// Zero one superblock scale and one sub-scale in the first 8-row group.
	w.Raw[layout.ScaleOff] = 0
	w.Raw[layout.ScaleOff+1] = 0
	w.Raw[layout.SubScaleOff+3*layout.BlocksPerRow+5] = 0

	p4, err := BuildPacked4x8(&w)
	if err != nil {
		t.Fatal(err)
	}
	w.Packed4x8 = p4

	x := quantTestActivations(6, cols)
	got := make([]float32, 6*rows)
	MatMatQuant(got, &w, x, 6)
	want := make([]float32, rows)
	for r := range 6 {
		MatVec(want, &w, x[r*cols:(r+1)*cols])
		for i, v := range want {
			if g := got[r*rows+i]; math.Float32bits(g) != math.Float32bits(v) {
				t.Fatalf("row %d col %d: got %v (%#x), want %v (%#x)",
					r, i, g, math.Float32bits(g), v, math.Float32bits(v))
			}
		}
	}
}
