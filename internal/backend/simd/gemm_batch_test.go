package simd

import (
	"math"
	"testing"

	"github.com/samcharles93/mantle/pkg/mcf"
)

// naiveWT is the reference for C = A*Wᵀ. It decodes W through Mat.RowTo, the
// canonical decoder, so it stays independent of the GEMM's own tile packing but
// agrees on the decoded weight values.
func naiveWT(C, A, W *Mat) {
	tmp := make([]float32, W.C)
	for i := 0; i < A.R; i++ {
		aRow := A.Row(i)
		for j := 0; j < W.R; j++ {
			W.RowTo(tmp, j)
			var sum float32
			for k := 0; k < A.C; k++ {
				sum += aRow[k] * tmp[k]
			}
			C.Row(i)[j] = sum
		}
	}
}

func fillDeterministic(dst []float32, seed uint32) {
	x := seed
	for i := range dst {
		x = x*1664525 + 1013904223
		dst[i] = float32(int32(x>>8))/float32(1<<23) - 1
	}
}

func bf16Raw(vals []float32) []byte {
	raw := make([]byte, len(vals)*2)
	for i, v := range vals {
		u := uint16(math.Float32bits(v) >> 16)
		raw[i*2] = byte(u)
		raw[i*2+1] = byte(u >> 8)
	}
	return raw
}

// f16Raw truncates toward zero. The reference decodes the same bytes, so the
// exact rounding mode does not matter for the comparison.
func f16Raw(vals []float32) []byte {
	raw := make([]byte, len(vals)*2)
	for i, v := range vals {
		b := math.Float32bits(v)
		sign := uint16((b >> 16) & 0x8000)
		exp := int32((b>>23)&0xff) - 127 + 15
		mant := b & 0x7fffff
		var u uint16
		switch {
		case exp <= 0:
			u = sign
		case exp >= 31:
			u = sign | 0x7c00
		default:
			u = sign | uint16(exp)<<10 | uint16(mant>>13)
		}
		raw[i*2] = byte(u)
		raw[i*2+1] = byte(u >> 8)
	}
	return raw
}

func TestGemmParWTMatchesNaive(t *testing.T) {
	t.Parallel()

	type shape struct{ m, k, n int }
	shapes := []shape{
		{50, 70, 45},
		{8, 2048, 256},
		{8, 2048, 2048},
	}

	for _, sh := range shapes {
		aData := make([]float32, sh.m*sh.k)
		fillDeterministic(aData, 12345)
		wData := make([]float32, sh.n*sh.k)
		fillDeterministic(wData, 98765)

		a := NewMatFromData(sh.m, sh.k, aData)

		f32W := NewMatFromData(sh.n, sh.k, append([]float32(nil), wData...))
		bf16Mat, err := NewMatFromRaw(sh.n, sh.k, mcf.DTypeBF16, bf16Raw(wData))
		if err != nil {
			t.Fatalf("bf16 mat: %v", err)
		}
		f16Mat, err := NewMatFromRaw(sh.n, sh.k, mcf.DTypeF16, f16Raw(wData))
		if err != nil {
			t.Fatalf("f16 mat: %v", err)
		}

		for _, tc := range []struct {
			name string
			w    *Mat
			tol  float64
		}{
			{"f32", &f32W, 1e-3},
			{"bf16", &bf16Mat, 1e-2},
			{"f16", &f16Mat, 1e-2},
		} {
			want := NewMat(sh.m, sh.n)
			naiveWT(&want, &a, tc.w)

			// workers=1 takes the scalar, no-pack fallback; workers=4 exercises
			// the pooled packed SIMD path. Both must agree with the reference.
			for _, workers := range []int{1, 4} {
				got := NewMat(sh.m, sh.n)
				cfg := SelectGemmConfig(sh.m, sh.k, sh.n)
				GemmParWT(cfg, &got, &a, tc.w, 1, 0, workers)

				if diff := maxAbsDiff(want.Data, got.Data); diff > tc.tol {
					t.Fatalf("shape %dx%dx%d %s workers=%d: max abs diff %g > %g", sh.m, sh.k, sh.n, tc.name, workers, diff, tc.tol)
				}
			}
		}
	}
}

func TestGemmParWTMatchesNaiveAlphaBeta(t *testing.T) {
	t.Parallel()

	const m, k, n = 17, 23, 11
	aData := make([]float32, m*k)
	fillDeterministic(aData, 7)
	wData := make([]float32, n*k)
	fillDeterministic(wData, 11)
	a := NewMatFromData(m, k, aData)
	w := NewMatFromData(n, k, append([]float32(nil), wData...))

	want := NewMat(m, n)
	naiveWT(&want, &a, &w)
	for i := range want.Data {
		want.Data[i] = want.Data[i]*0.5 + 0.25
	}

	got := NewMat(m, n)
	for i := range got.Data {
		got.Data[i] = 0.5 // beta != 0 pre-load
	}
	cfg := SelectGemmConfig(m, k, n)
	GemmParWT(cfg, &got, &a, &w, 0.5, 0.5, 1)

	if diff := maxAbsDiff(want.Data, got.Data); diff > 1e-4 {
		t.Fatalf("alpha/beta: max abs diff %g", diff)
	}
}

func TestGemmParWTNoAllocs(t *testing.T) {
	a := NewMat(16, 16)
	w := NewMat(16, 16)
	c := NewMat(16, 16)

	cfg := DefaultGemmConfig()
	allocs := testing.AllocsPerRun(100, func() {
		GemmParWT(cfg, &c, &a, &w, 1, 0, 2)
	})
	if allocs != 0 {
		t.Fatalf("unexpected allocs: %v", allocs)
	}
}

func BenchmarkGemmParWT(b *testing.B) {
	cases := []struct {
		name string
		m    int
		n    int
	}{
		{"m64_n2048", 64, 2048},
		{"m64_n6144", 64, 6144},
		{"m256_n2048", 256, 2048},
		{"m512_n6144", 512, 6144},
	}
	for _, tc := range cases {
		const k = 2048
		aData := make([]float32, tc.m*k)
		fillDeterministic(aData, 3)
		wData := make([]float32, tc.n*k)
		fillDeterministic(wData, 5)
		a := NewMatFromData(tc.m, k, aData)
		w, err := NewMatFromRaw(tc.n, k, mcf.DTypeBF16, bf16Raw(wData))
		if err != nil {
			b.Fatalf("bf16 mat: %v", err)
		}
		c := NewMat(tc.m, tc.n)
		cfg := SelectGemmConfig(tc.m, k, tc.n)

		b.Run(tc.name, func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				GemmParWT(cfg, &c, &a, &w, 1, 0, 0)
			}
		})
	}
}
