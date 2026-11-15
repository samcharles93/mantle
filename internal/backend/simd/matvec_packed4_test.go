package simd

import (
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
