package simd

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	instance "github.com/samcharles93/mantle/internal/backend/core"
	"github.com/samcharles93/mantle/pkg/mcf"
)

func packed4Mat(tb testing.TB, dt mcf.TensorDType, rows, cols int) Mat {
	tb.Helper()
	rng := rand.New(rand.NewPCG(51, uint64(dt)))
	src := make([]float32, rows*cols)
	for i := range src {
		src[i] = float32(rng.NormFloat64() * 0.02)
	}
	var payload []byte
	var err error
	switch dt {
	case mcf.DTypeQ4:
		payload, err = mcf.EncodeQTensor(src, rows, cols, 4)
	case mcf.DTypeK4:
		payload, err = mcf.EncodeKTensor(src, rows, cols, 4)
	default:
		_, group, _ := mcf.GQParams(dt)
		_, payload, err = mcf.EncodeGQTensor(src, rows, cols, 4, group)
	}
	if err != nil {
		tb.Fatal(err)
	}
	return Mat{R: rows, C: cols, Stride: cols, DType: dt, Raw: payload}
}

func packed4DTypes(t testing.TB) []mcf.TensorDType {
	out := []mcf.TensorDType{mcf.DTypeQ4, mcf.DTypeK4}
	for _, g := range []int{32, 64, 128} {
		dt, err := mcf.GQDType(4, g)
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, dt)
	}
	return out
}

func TestDotBlocksPacked4MatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewPCG(52, 53))
	const blocks = 37
	data := make([]byte, blocks*16)
	for i := range data {
		data[i] = byte(rng.IntN(256))
	}
	x := make([]int16, blocks*32)
	for i := range x {
		x[i] = int16(rng.IntN(65535) - 32767)
	}
	got := make([]int32, blocks)
	want := make([]int32, blocks)
	dotBlocksPacked4Int16(data, x, got, blocks)
	dotBlocksPacked4Int16Scalar(data, x, want, blocks)
	for b := range blocks {
		if got[b] != want[b] {
			t.Fatalf("block %d: SIMD %d, scalar %d", b, got[b], want[b])
		}
	}
	// The scalar path itself must agree with an element-order decode.
	for b := range blocks {
		var ref int32
		for i := range 32 {
			v := data[b*16+i/2]
			var code int32
			if i%2 == 0 {
				code = int32(int8(v<<4) >> 4)
			} else {
				code = int32(int8(v) >> 4)
			}
			pos := b*32 + i/2
			if i%2 == 1 {
				pos += 16
			}
			ref += code * int32(x[pos])
		}
		if ref != want[b] {
			t.Fatalf("block %d: scalar %d, element-order reference %d", b, want[b], ref)
		}
	}
}

// TestMatVecPacked4MatchesCached checks the packed kernel against the int8
// quant-cache kernel on the same weights and activations: the results must be
// bit-identical, so switching between the two never changes output.
func TestMatVecPacked4MatchesCached(t *testing.T) {
	const rows, cols = 48, 330 // cols not a multiple of 32
	x := make([]float32, cols)
	for i := range x {
		x[i] = float32((i%17)-8) * 0.07
	}
	for _, dt := range packed4DTypes(t) {
		w := packed4Mat(t, dt, rows, cols)
		qx := PrepareQuantVec(x)
		packed := make([]float32, rows)
		MatVecWithQuant(packed, &w, x, qx)

		qc, err := instance.BuildQuantCache(&w)
		if err != nil {
			t.Fatal(err)
		}
		wc := w
		wc.Quant = qc
		cached := make([]float32, rows)
		MatVecWithQuant(cached, &wc, x, qx)
		ReleaseQuantVec(qx)

		for r := range rows {
			if packed[r] != cached[r] {
				t.Fatalf("%#x row %d: packed %v, cached %v", dt, r, packed[r], cached[r])
			}
		}
	}
}

// TestMatVecRangePacked4GroupsMatchCached runs matVecRangePacked4 over whole
// row ranges, so the eight-row grouped path and the per-row tail both run (a
// pooled MatVec gives each worker too few rows to reach a group in small
// tests), and requires every output to carry the int8-cache kernel's bits.
func TestMatVecRangePacked4GroupsMatchCached(t *testing.T) {
	cases := []struct{ rows, cols, rs, re int }{
		{67, 330, 0, 67},  // eight groups and a 3-row tail; a partial last block
		{40, 2048, 5, 37}, // groups starting off a multiple of eight
		{16, 7680, 0, 16}, // 240 blocks: the widest LFM3 row
		{12, 8224, 0, 12}, // 257 blocks: past the grouped path's limit
	}
	for _, dt := range packed4DTypes(t) {
		for _, tc := range cases {
			t.Run(fmt.Sprintf("%#x_%dx%d_%d-%d", dt, tc.rows, tc.cols, tc.rs, tc.re), func(t *testing.T) {
				w := packed4Mat(t, dt, tc.rows, tc.cols)
				x := make([]float32, tc.cols)
				rng := rand.New(rand.NewPCG(54, uint64(tc.cols)))
				for i := range x {
					x[i] = float32(rng.NormFloat64())
				}
				clear(x[32:64]) // a zero activation block
				qx := PrepareQuantVec(x)
				defer ReleaseQuantVec(qx)

				got := make([]float32, tc.rows)
				matVecRangePacked4(got, &w, tc.rs, tc.re, qx)

				qc, err := instance.BuildQuantCache(&w)
				if err != nil {
					t.Fatal(err)
				}
				wc := w
				wc.Quant = qc
				want := make([]float32, tc.rows)
				matVecRangeQuantCached(want, &wc, x, tc.rs, tc.re, qx)

				for r := tc.rs; r < tc.re; r++ {
					if math.Float32bits(got[r]) != math.Float32bits(want[r]) {
						t.Fatalf("row %d: packed %v, cached %v", r, got[r], want[r])
					}
				}
			})
		}
	}
}

// TestMatVecRangePacked4ZeroScalesMatchCached gives the grouped k path zero
// superblock scales (an all-zero row) and zero sub-scales (zero blocks), where
// the vector scale path may produce -0 instead of +0, and requires the outputs
// to keep the int8-cache kernel's bits.
func TestMatVecRangePacked4ZeroScalesMatchCached(t *testing.T) {
	const rows, cols = 24, 2048
	rng := rand.New(rand.NewPCG(55, 56))
	src := make([]float32, rows*cols)
	for i := range src {
		src[i] = float32(rng.NormFloat64() * 0.02)
	}
	clear(src[3*cols : 4*cols])            // row 3: every scale zero
	clear(src[5*cols+256 : 5*cols+512])    // row 5: one whole superblock zero
	clear(src[9*cols+64 : 9*cols+96])      // row 9: a single zero block
	for i := 11 * cols; i < 12*cols; i++ { // row 11: negative-only weights
		src[i] = -float32(math.Abs(float64(src[i])))
	}
	payload, err := mcf.EncodeKTensor(src, rows, cols, 4)
	if err != nil {
		t.Fatal(err)
	}
	w := Mat{R: rows, C: cols, Stride: cols, DType: mcf.DTypeK4, Raw: payload}
	x := make([]float32, cols)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	qx := PrepareQuantVec(x)
	defer ReleaseQuantVec(qx)
	got := make([]float32, rows)
	matVecRangePacked4(got, &w, 0, rows, qx)

	qc, err := instance.BuildQuantCache(&w)
	if err != nil {
		t.Fatal(err)
	}
	wc := w
	wc.Quant = qc
	want := make([]float32, rows)
	matVecRangeQuantCached(want, &wc, x, 0, rows, qx)
	for r := range rows {
		if math.Float32bits(got[r]) != math.Float32bits(want[r]) {
			t.Fatalf("row %d: packed %v (%#x), cached %v (%#x)", r, got[r], math.Float32bits(got[r]), want[r], math.Float32bits(want[r]))
		}
	}
	if got[3] != 0 {
		t.Fatalf("all-zero row gave %v", got[3])
	}
}

func benchMatVec4(b *testing.B, dt mcf.TensorDType, mode string) {
	const rows, cols = 2048, 2048
	w := packed4Mat(b, dt, rows, cols)
	switch mode {
	case "cached":
		qc, err := instance.BuildQuantCache(&w)
		if err != nil {
			b.Fatal(err)
		}
		w.Quant = qc
	case "packed":
	}
	x := make([]float32, cols)
	for i := range x {
		x[i] = float32((i%13)-6) * 0.11
	}
	dst := make([]float32, rows)
	qx := PrepareQuantVec(x)
	defer ReleaseQuantVec(qx)
	b.SetBytes(int64(len(w.Raw)))
	for b.Loop() {
		MatVecWithQuant(dst, &w, x, qx)
	}
}

func BenchmarkMatVecK4Packed(b *testing.B)  { benchMatVec4(b, mcf.DTypeK4, "packed") }
func BenchmarkMatVecK4Cached2(b *testing.B) { benchMatVec4(b, mcf.DTypeK4, "cached") }
func BenchmarkMatVecGQ4Packed(b *testing.B) {
	dt, _ := mcf.GQDType(4, 128)
	benchMatVec4(b, dt, "packed")
}
