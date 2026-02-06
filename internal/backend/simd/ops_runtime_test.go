package simd

import "testing"

func TestDefaultOpsStoreKVF32(t *testing.T) {
	ops := DefaultOps{}
	const stride = 4
	kDst := make([]float32, stride*3)
	vDst := make([]float32, stride*3)
	k := []float32{1, 2, 3, 4}
	v := []float32{5, 6, 7, 8}

	ops.StoreKV(0, 1, stride, kDst, vDst, nil, nil, k, v)

	for i, want := range k {
		if got := kDst[stride+i]; got != want {
			t.Fatalf("k[%d]: got %v want %v", i, got, want)
		}
	}
	for i, want := range v {
		if got := vDst[stride+i]; got != want {
			t.Fatalf("v[%d]: got %v want %v", i, got, want)
		}
	}
}

func TestDefaultOpsStoreKVF16(t *testing.T) {
	ops := DefaultOps{}
	const stride = 2
	kDst := make([]uint16, stride*2)
	vDst := make([]uint16, stride*2)
	k := []float32{1.5, -2.0}
	v := []float32{3.25, 4.5}

	ops.StoreKV(0, 1, stride, nil, nil, kDst, vDst, k, v)

	for i, want := range k {
		got := Float16ToFloat32(kDst[stride+i])
		if got != want {
			t.Fatalf("k[%d]: got %v want %v", i, got, want)
		}
	}
	for i, want := range v {
		got := Float16ToFloat32(vDst[stride+i])
		if got != want {
			t.Fatalf("v[%d]: got %v want %v", i, got, want)
		}
	}
}

func TestDefaultOpsApplyRoPEMatchesFunction(t *testing.T) {
	ops := DefaultOps{}
	x0 := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	x1 := append([]float32(nil), x0...)
	invFreq := []float64{1.0, 0.5}

	ops.ApplyRoPE(x0, 2, 4, 3, invFreq, 1.0)
	ApplyRoPE(x1, 2, 4, 3, invFreq, 1.0)

	for i := range x0 {
		if x0[i] != x1[i] {
			t.Fatalf("x[%d]: got %v want %v", i, x0[i], x1[i])
		}
	}
}
