package core

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/samcharles93/mantle/pkg/mcf"
)

func roundTripWeights(rows, cols int) []float32 {
	rng := rand.New(rand.NewPCG(11, 12))
	out := make([]float32, rows*cols)
	for i := range out {
		v := rng.NormFloat64() * 0.02
		if rng.IntN(200) == 0 {
			v *= 6
		}
		out[i] = float32(v)
	}
	return out
}

// TestEncodedPayloadsDecodeInRuntime encodes with the pkg/mcf writers and
// decodes with the runtime readers. RowTo and the int8 quant cache must agree
// exactly, and the reconstruction error must stay under a per-dtype bound.
func TestEncodedPayloadsDecodeInRuntime(t *testing.T) {
	const rows, cols = 24, 300 // cols not a multiple of 32
	src := roundTripWeights(rows, cols)
	var srcSq float64
	for _, v := range src {
		srcSq += float64(v) * float64(v)
	}
	cases := []struct {
		dt     mcf.TensorDType
		bits   int
		encode func() ([]byte, error)
		maxRel float64
	}{
		{mcf.DTypeQ8, 8, func() ([]byte, error) { return mcf.EncodeQTensor(src, rows, cols, 8) }, 0.007},
		{mcf.DTypeQ4, 4, func() ([]byte, error) { return mcf.EncodeQTensor(src, rows, cols, 4) }, 0.12},
		{mcf.DTypeK6, 6, func() ([]byte, error) { return mcf.EncodeKTensor(src, rows, cols, 6) }, 0.03},
		{mcf.DTypeK4, 4, func() ([]byte, error) { return mcf.EncodeKTensor(src, rows, cols, 4) }, 0.13},
		{mcf.DTypeK3, 3, func() ([]byte, error) { return mcf.EncodeKTensor(src, rows, cols, 3) }, 0.27},
		{mcf.DTypeK2, 2, func() ([]byte, error) { return mcf.EncodeKTensor(src, rows, cols, 2) }, 0.49},
	}
	for _, tc := range cases {
		raw, err := tc.encode()
		if err != nil {
			t.Fatalf("%v: encode: %v", tc.dt, err)
		}
		m := Mat{R: rows, C: cols, Stride: cols, DType: tc.dt, Raw: raw}
		qc, err := BuildQuantCache(&m)
		if err != nil {
			t.Fatalf("%v: BuildQuantCache: %v", tc.dt, err)
		}
		row := make([]float32, cols)
		var errSq float64
		for r := range rows {
			m.RowTo(row, r)
			for c, v := range row {
				b := r*qc.BlocksPerRow + c/32
				cached := float32(qc.Q[b*32+c%32]) * qc.Scales[b]
				if cached != v {
					t.Fatalf("%v: row %d col %d: RowTo %v, quant cache %v", tc.dt, r, c, v, cached)
				}
				d := float64(v) - float64(src[r*cols+c])
				errSq += d * d
			}
		}
		rel := math.Sqrt(errSq / srcSq)
		base := absmaxBlockError(src, rows, cols, max(1, (1<<(tc.bits-1))-1))
		t.Logf("%v: relative RMSE %.4f, absmax rounding %.4f", tc.dt, rel, math.Sqrt(base/srcSq))
		if rel > tc.maxRel {
			t.Errorf("%v: relative RMSE %.4f exceeds %.4f", tc.dt, rel, tc.maxRel)
		}
		if tc.bits < 8 && errSq >= base {
			t.Errorf("%v: squared error %.6g not below per-block absmax rounding %.6g", tc.dt, errSq, base)
		}
	}
}

// absmaxBlockError is the squared error of rounding each 32-element block on
// its absmax scale with codes in [-hi, hi], the scheme the encoders replaced.
func absmaxBlockError(src []float32, rows, cols, hi int) float64 {
	var e float64
	for r := range rows {
		for c0 := 0; c0 < cols; c0 += 32 {
			x := src[r*cols+c0 : r*cols+min(c0+32, cols)]
			var amax float64
			for _, v := range x {
				amax = max(amax, math.Abs(float64(v)))
			}
			if amax == 0 {
				continue
			}
			s := amax / float64(hi)
			for _, v := range x {
				q := math.Max(-float64(hi), math.Min(float64(hi), math.Round(float64(v)/s)))
				d := float64(v) - s*q
				e += d * d
			}
		}
	}
	return e
}
