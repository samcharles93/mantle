package simd

import (
	"testing"

	"github.com/samcharles93/mantle/pkg/mcf"
)

// TestMatVecMatchesCachedPath covers the two decode paths a quantized weight can
// take: the pre-expanded int8 cache built at load, and the on-the-fly block decode
// used when that cache is skipped (--no-quant-cache). The two must produce the same
// matrix-vector product, otherwise dropping the cache changes model output as well
// as memory use.
func TestMatVecMatchesCachedPath(t *testing.T) {
	const (
		rows = 8
		cols = 256
	)

	cases := []struct {
		name    string
		dtype   mcf.TensorDType
		payload func(scale float32, qvals []int8) []byte
	}{
		{
			name:  "k4",
			dtype: mcf.DTypeK4,
			payload: func(scale float32, qvals []int8) []byte {
				return buildK4Payload(rows, cols, scale, qvals)
			},
		},
		{
			name:  "q8",
			dtype: mcf.DTypeQ8,
			payload: func(scale float32, qvals []int8) []byte {
				return buildQPayload(rows, cols, 8, scale, qvals)
			},
		},
	}

	x := make([]float32, cols)
	for i := range x {
		x[i] = float32((i%17)-8) * 0.125
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			const scale = 0.031
			qvals := makeQVals(rows * cols)
			payload := tc.payload(scale, qvals)
			verifyPayloadSize(t, payload, rows, cols, tc.dtype)

			uncached := Mat{R: rows, C: cols, Stride: cols, DType: tc.dtype, Raw: payload}
			cached := Mat{R: rows, C: cols, Stride: cols, DType: tc.dtype, Raw: payload}
			cache, err := BuildQuantCache(&cached)
			if err != nil {
				t.Fatalf("build quant cache: %v", err)
			}
			cached.Quant = cache

			want := make([]float32, rows)
			got := make([]float32, rows)
			MatVec(want, &cached, x)
			MatVec(got, &uncached, x)

			for i := range want {
				if diff := got[i] - want[i]; diff < -1e-4 || diff > 1e-4 {
					t.Fatalf("row %d: uncached %v, cached %v", i, got[i], want[i])
				}
			}
		})
	}
}
