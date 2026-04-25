//go:build cuda

package native

import (
	"math"
	"math/rand"
	"testing"
	"unsafe"
)

func l2normRef(x []float32, eps float32) []float32 {
	var sum float64
	for _, v := range x {
		sum += float64(v) * float64(v)
	}
	scale := float32(1.0 / math.Sqrt(sum+float64(eps)))
	out := make([]float32, len(x))
	for i, v := range x {
		out[i] = v * scale
	}
	return out
}

func TestDeltaNetL2NormF32Parity(t *testing.T) {
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
	defer func() {
		if err := stream.Destroy(); err != nil {
			t.Fatalf("stream destroy: %v", err)
		}
	}()

	cases := []struct {
		headDim int
		nHeads  int
	}{
		{32, 1},
		{64, 4},
		{128, 8},
		{128, 32},
		{256, 16},
	}
	rng := rand.New(rand.NewSource(0xDE17A))
	const eps = float32(1e-6)

	for _, c := range cases {
		total := c.headDim * c.nHeads
		input := make([]float32, total)
		for i := range total {
			input[i] = (rng.Float32()*2 - 1) * 4
		}
		want := make([]float32, total)
		for h := range c.nHeads {
			off := h * c.headDim
			ref := l2normRef(input[off:off+c.headDim], eps)
			copy(want[off:off+c.headDim], ref)
		}

		bytes := int64(total * 4)
		buf, err := AllocDevice(bytes)
		if err != nil {
			t.Fatalf("AllocDevice(hd=%d nh=%d): %v", c.headDim, c.nHeads, err)
		}
		if err := MemcpyH2D(buf, unsafe.Pointer(&input[0]), bytes); err != nil {
			t.Fatalf("MemcpyH2D: %v", err)
		}
		if err := DeltaNetL2NormF32(buf, eps, c.headDim, c.nHeads, stream); err != nil {
			t.Fatalf("DeltaNetL2NormF32: %v", err)
		}
		if err := stream.Synchronize(); err != nil {
			t.Fatalf("stream sync: %v", err)
		}
		got := make([]float32, total)
		if err := MemcpyD2H(unsafe.Pointer(&got[0]), buf, bytes); err != nil {
			t.Fatalf("MemcpyD2H: %v", err)
		}
		if err := buf.Free(); err != nil {
			t.Fatalf("buf free: %v", err)
		}

		for i := range total {
			if !approxEqual(got[i], want[i], 1e-5) {
				t.Fatalf("hd=%d nh=%d idx=%d: got=%g want=%g", c.headDim, c.nHeads, i, got[i], want[i])
			}
		}
	}
}

func TestDeltaNetL2NormF32NilBuffer(t *testing.T) {
	var empty DeviceBuffer
	if err := DeltaNetL2NormF32(empty, 1e-6, 32, 1, Stream{}); err == nil {
		t.Fatal("expected error for nil buffer")
	}
}

func TestDeltaNetL2NormF32InvalidDims(t *testing.T) {
	count, err := DeviceCount()
	if err != nil {
		t.Fatalf("DeviceCount: %v", err)
	}
	if count < 1 {
		t.Skip("no cuda device available")
	}
	buf, err := AllocDevice(4)
	if err != nil {
		t.Fatalf("AllocDevice: %v", err)
	}
	defer buf.Free()
	if err := DeltaNetL2NormF32(buf, 1e-6, 0, 1, Stream{}); err == nil {
		t.Fatal("expected error for head_dim=0")
	}
	if err := DeltaNetL2NormF32(buf, 1e-6, 32, 0, Stream{}); err == nil {
		t.Fatal("expected error for n_heads=0")
	}
}
