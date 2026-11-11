package simd

import (
	"math"
	"testing"

	instance "github.com/samcharles93/mantle/internal/backend/core"
)

// TestF16x8ToF32Exhaustive checks every half bit pattern against the scalar
// conversion, including subnormals, infinities and NaN payloads.
func TestF16x8ToF32Exhaustive(t *testing.T) {
	if !cpu.HasAVX2 {
		t.Skip("needs AVX2")
	}
	k := newF16Consts()
	var in [8]uint16
	var out [8]float32
	for base := 0; base < 1<<16; base += 8 {
		for i := range in {
			in[i] = uint16(base + i)
		}
		f16x8ToF32(&in, &k).StoreArray(&out)
		for i, h := range in {
			want := instance.Float16ToFloat32(h)
			if math.Float32bits(out[i]) != math.Float32bits(want) {
				t.Fatalf("half %#04x: got %#08x, want %#08x", h, math.Float32bits(out[i]), math.Float32bits(want))
			}
		}
	}
}

func TestDotF16SIMDMatchesScalar(t *testing.T) {
	if !cpu.HasAVX2 || !cpu.HasFMA {
		t.Skip("needs AVX2 and FMA")
	}
	for _, n := range []int{1, 7, 8, 16, 64, 129} {
		a := make([]float32, n)
		b := make([]uint16, n)
		for i := range a {
			a[i] = float32((i*17)%23-11) * 0.07
			b[i] = instance.Float32ToFloat16(float32((i*29)%31-15) * 0.05)
		}
		got, want := dotF16SIMD(a, b), dotScalarF16(a, b)
		if d := math.Abs(float64(got - want)); d > 1e-5*math.Max(1, math.Abs(float64(want))) {
			t.Fatalf("n=%d: SIMD %v, scalar %v", n, got, want)
		}
	}
}

// TestRunAttnHeadsVectorVMatchesScalar requires the position-outer V
// accumulation to equal the d-outer scalar loop exactly, for f16 and f32
// caches, including a ring-wrapped window.
func TestRunAttnHeadsVectorVMatchesScalar(t *testing.T) {
	if !cpu.HasAVX2 {
		t.Skip("needs AVX2")
	}
	const nHead, kvHeads, headDim, cacheLen = 4, 2, 16, 12
	kvStride := kvHeads * headDim
	q := make([]float32, nHead*headDim)
	for i := range q {
		q[i] = float32((i*13)%17-8) * 0.1
	}
	k32 := make([]float32, cacheLen*kvStride)
	v32 := make([]float32, cacheLen*kvStride)
	k16 := make([]uint16, len(k32))
	v16 := make([]uint16, len(v32))
	for i := range k32 {
		k32[i] = float32((i*7)%19-9) * 0.05
		v32[i] = float32((i*11)%23-11) * 0.04
		k16[i] = instance.Float32ToFloat16(k32[i])
		v16[i] = instance.Float32ToFloat16(v32[i])
	}
	scalarV := func(ctx *AttnContext, scores []float32, h int) []float32 {
		out := make([]float32, headDim)
		kvHead := h * kvHeads / nHead
		for d := range headDim {
			var sum float32
			for t := ctx.Start; t <= ctx.Pos; t++ {
				voff := (t%cacheLen)*kvStride + kvHead*headDim + d
				if ctx.CacheV16 != nil {
					sum += scores[t-ctx.Start] * instance.Float16ToFloat32(ctx.CacheV16[voff])
				} else {
					sum += scores[t-ctx.Start] * ctx.CacheV[voff]
				}
			}
			out[d] = sum
		}
		return out
	}
	for _, f16 := range []bool{false, true} {
		for _, pos := range []int{5, 17} { // 17 wraps the 12-slot ring
			ctx := &AttnContext{
				Q: q, AttnOut: make([]float32, nHead*headDim), Pos: pos, Start: max(0, pos-cacheLen+1),
				HeadDim: headDim, KvStride: kvStride, NHead: nHead, KvHeads: kvHeads, Scale: 0.25, CacheLen: cacheLen,
			}
			if f16 {
				ctx.CacheK16, ctx.CacheV16 = k16, v16
			} else {
				ctx.CacheK, ctx.CacheV = k32, v32
			}
			scores := make([]float32, cacheLen)
			for h := range nHead {
				RunAttnHeads(ctx, scores, h, h+1)
				want := scalarV(ctx, scores, h)
				got := ctx.AttnOut[h*headDim : (h+1)*headDim]
				for d := range want {
					if math.Float32bits(got[d]) != math.Float32bits(want[d]) {
						t.Fatalf("f16=%v pos=%d head %d dim %d: got %v, want %v", f16, pos, h, d, got[d], want[d])
					}
				}
			}
		}
	}
}
