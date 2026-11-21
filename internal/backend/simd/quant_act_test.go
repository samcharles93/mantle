package simd

import (
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"testing"
)

// quantActRef is the scalar reference for quantizeActivationBlocks: every
// block through quantizeBlockScalar and writeQuantBlock.
func quantActRef(x []float32, blocks int) (scales []float32, q8 []int8, q32 []int32, q16, q16Pair []int16, q8Pair []int8, bias []int32) {
	n := blocks * 32
	scales, q8, q32 = make([]float32, blocks), make([]int8, n), make([]int32, blocks*8)
	q16, q16Pair, q8Pair = make([]int16, n), make([]int16, n), make([]int8, n)
	bias = make([]int32, blocks)
	var q [32]int8
	for b := range blocks {
		lo, hi := min(b*32, len(x)), min(b*32+32, len(x))
		scales[b] = quantizeBlockScalar(x[lo:hi], &q)
		writeQuantBlock(&q, b, q8, q32, q16, q16Pair, q8Pair, bias)
	}
	return scales, q8, q32, q16, q16Pair, q8Pair, bias
}

// quantActInput builds activations that exercise the rounding edges: exact
// half-way products, values at the clamp, a zero block, a block holding only
// negative zero, denormals and a wide spread of magnitudes.
func quantActInput(n int, seed uint64) []float32 {
	rng := rand.New(rand.NewPCG(seed, 1))
	x := make([]float32, n)
	for i := range x {
		x[i] = float32(rng.NormFloat64()) * float32(math.Pow(10, float64(rng.IntN(7)-3)))
	}
	if n >= 64 {
		// Block 0: max 127, so inv is 1 and k+0.5 is an exact tie.
		for i := range 32 {
			x[i] = float32(i%16) + 0.5
			if i%2 == 1 {
				x[i] = -x[i]
			}
		}
		x[0] = 127
		// Block 1: all zero apart from negative zeros.
		for i := 32; i < 64; i++ {
			x[i] = 0
		}
		x[33] = float32(math.Copysign(0, -1))
	}
	if n >= 96 {
		for i := 64; i < 96; i++ {
			x[i] = math.Float32frombits(uint32(i)) // denormals
		}
	}
	return x
}

func TestQuantizeActivationBlocksMatchesScalar(t *testing.T) {
	for _, n := range []int{0, 5, 32, 33, 96, 330, 2048, 7680, 8224} {
		for seed := range uint64(4) {
			t.Run(fmt.Sprintf("n=%d/seed=%d", n, seed), func(t *testing.T) {
				x := quantActInput(n, seed)
				blocks := (n + 31) / 32
				ws, w8, w32, w16, w16p, w8p, wbias := quantActRef(x, blocks)
				m := blocks * 32
				gs, g8, g32 := make([]float32, blocks), make([]int8, m), make([]int32, blocks*8)
				g16, g16p, g8p := make([]int16, m), make([]int16, m), make([]int8, m)
				gbias := make([]int32, blocks)
				quantizeActivationBlocks(x, blocks, gs, g8, g32, g16, g16p, g8p, gbias)
				for b := range blocks {
					if math.Float32bits(gs[b]) != math.Float32bits(ws[b]) {
						t.Fatalf("scale[%d] = %g, want %g", b, gs[b], ws[b])
					}
				}
				if !slices.Equal(gbias, wbias) {
					t.Fatalf("bias differs from scalar reference")
				}
				if !slices.Equal(g32, w32) {
					t.Fatalf("q32 differs from scalar reference")
				}
				for _, c := range []struct {
					name      string
					got, want any
				}{{"q8", g8, w8}, {"q16", g16, w16}, {"q16Pair", g16p, w16p}, {"q8Pair", g8p, w8p}} {
					if !equalAny(c.got, c.want) {
						t.Fatalf("%s differs from the scalar reference: first mismatch at %d", c.name, firstDiff(c.got, c.want))
					}
				}

				// Optional layouts are independent: q8 alone, q8Pair+bias alone, or q32+bias alone.
				o8 := make([]int8, m)
				quantizeActivationBlocks(x, blocks, make([]float32, blocks), o8, nil, nil, nil, nil, nil)
				if !slices.Equal(o8, w8) {
					t.Fatalf("q8 with no optional layouts differs at %d", firstDiff(o8, w8))
				}
				o8p := make([]int8, m)
				obias := make([]int32, blocks)
				quantizeActivationBlocks(x, blocks, make([]float32, blocks), nil, nil, nil, nil, o8p, obias)
				if !slices.Equal(o8p, w8p) || !slices.Equal(obias, wbias) {
					t.Fatalf("q8Pair+bias without q8 differs")
				}
				o32 := make([]int32, blocks*8)
				obias32 := make([]int32, blocks)
				quantizeActivationBlocks(x, blocks, make([]float32, blocks), nil, o32, nil, nil, nil, obias32)
				if !slices.Equal(o32, w32) || !slices.Equal(obias32, wbias) {
					t.Fatalf("q32+bias without q8 differs")
				}
			})
		}
	}
}

// TestQuantizeActivationBlocksPadsShortInput covers blocks reaching past
// len(x), as when a weight's padded width exceeds the activation length.
func TestQuantizeActivationBlocksPadsShortInput(t *testing.T) {
	x := quantActInput(70, 9)
	const blocks = 4
	ws, w8, w32, _, w16p, _, wbias := quantActRef(x, blocks)
	gs, g8, g32, g16p, gbias := make([]float32, blocks), make([]int8, blocks*32), make([]int32, blocks*8), make([]int16, blocks*32), make([]int32, blocks)
	quantizeActivationBlocks(x, blocks, gs, g8, g32, nil, g16p, nil, gbias)
	if !slices.Equal(gs, ws) || !slices.Equal(g8, w8) || !slices.Equal(g32, w32) || !slices.Equal(g16p, w16p) || !slices.Equal(gbias, wbias) {
		t.Fatalf("padded quantisation differs: scales %v want %v", gs, ws)
	}
	if gs[3] != 0 || gbias[3] != 0 || slices.ContainsFunc(g8[96:], func(v int8) bool { return v != 0 }) {
		t.Fatal("block past the input must be zero")
	}
}

func equalAny(a, b any) bool {
	switch a := a.(type) {
	case []int8:
		return slices.Equal(a, b.([]int8))
	case []int16:
		return slices.Equal(a, b.([]int16))
	}
	panic("unsupported type")
}

func firstDiff(a, b any) int {
	switch a := a.(type) {
	case []int8:
		bb := b.([]int8)
		for i := range a {
			if a[i] != bb[i] {
				return i
			}
		}
	case []int16:
		bb := b.([]int16)
		for i := range a {
			if a[i] != bb[i] {
				return i
			}
		}
	}
	return -1
}
