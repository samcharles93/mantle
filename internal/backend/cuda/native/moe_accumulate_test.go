//go:build cuda

package native

import (
	"math/rand"
	"testing"
	"unsafe"
)

func TestMoEAccumulateF32Parity(t *testing.T) {
	count, err := DeviceCount()
	if err != nil {
		t.Fatalf("DeviceCount: %v", err)
	}
	if count < 1 {
		t.Skip("no cuda device available")
	}
	stream, err := NewStream()
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	defer stream.Destroy()

	cases := []struct {
		n int
		w float32
	}{
		{1, 0.5},
		{31, 1.0},
		{256, 0.25},
		{1024, -0.75},
		{4097, 0.123},
		{8192, 1.5},
	}

	rng := rand.New(rand.NewSource(0xACC0))
	for _, c := range cases {
		accum := make([]float32, c.n)
		src := make([]float32, c.n)
		for i := range c.n {
			accum[i] = (rng.Float32()*2 - 1) * 4
			src[i] = (rng.Float32()*2 - 1) * 4
		}
		want := make([]float32, c.n)
		for i := range c.n {
			want[i] = accum[i] + c.w*src[i]
		}

		accDev, err := AllocDevice(int64(c.n) * 4)
		if err != nil {
			t.Fatalf("alloc accum: %v", err)
		}
		srcDev, err := AllocDevice(int64(c.n) * 4)
		if err != nil {
			t.Fatalf("alloc src: %v", err)
		}
		if err := MemcpyH2D(accDev, unsafe.Pointer(&accum[0]), int64(c.n)*4); err != nil {
			t.Fatalf("h2d accum: %v", err)
		}
		if err := MemcpyH2D(srcDev, unsafe.Pointer(&src[0]), int64(c.n)*4); err != nil {
			t.Fatalf("h2d src: %v", err)
		}
		if err := MoEAccumulateF32(accDev, srcDev, c.w, c.n, stream); err != nil {
			t.Fatalf("MoEAccumulateF32: %v", err)
		}
		if err := stream.Synchronize(); err != nil {
			t.Fatalf("sync: %v", err)
		}
		got := make([]float32, c.n)
		if err := MemcpyD2H(unsafe.Pointer(&got[0]), accDev, int64(c.n)*4); err != nil {
			t.Fatalf("d2h: %v", err)
		}
		_ = accDev.Free()
		_ = srcDev.Free()

		for i := range c.n {
			if !approxEqual(got[i], want[i], 1e-6) {
				t.Fatalf("n=%d i=%d: got=%g want=%g", c.n, i, got[i], want[i])
			}
		}
	}
}

func TestMoEAccumulateF32Zero(t *testing.T) {
	var empty DeviceBuffer
	if err := MoEAccumulateF32(empty, empty, 1.0, 0, Stream{}); err != nil {
		t.Fatalf("n=0 must be no-op, got: %v", err)
	}
}

func TestMoEAccumulateF32NilBuffer(t *testing.T) {
	var empty DeviceBuffer
	if err := MoEAccumulateF32(empty, empty, 1.0, 8, Stream{}); err == nil {
		t.Fatal("expected error for nil buffers when n>0")
	}
}
