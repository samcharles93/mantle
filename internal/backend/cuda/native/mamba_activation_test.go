//go:build cuda

package native

import (
	"math"
	"math/rand"
	"testing"
	"unsafe"
)

func siluRef(x float32) float32 {
	return x / (1.0 + float32(math.Exp(float64(-x))))
}

func TestMambaSiluF32InPlaceFixed(t *testing.T) {
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

	input := []float32{-4.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0}
	want := make([]float32, len(input))
	for i, v := range input {
		want[i] = siluRef(v)
	}

	n := len(input)
	bytes := int64(n * 4)
	buf, err := AllocDevice(bytes)
	if err != nil {
		t.Fatalf("AllocDevice: %v", err)
	}
	defer func() {
		if err := buf.Free(); err != nil {
			t.Fatalf("buf free: %v", err)
		}
	}()

	if err := MemcpyH2D(buf, unsafe.Pointer(&input[0]), bytes); err != nil {
		t.Fatalf("MemcpyH2D: %v", err)
	}

	if err := SiluF32InPlace(buf, n, stream); err != nil {
		t.Fatalf("SiluF32InPlace: %v", err)
	}
	if err := stream.Synchronize(); err != nil {
		t.Fatalf("stream sync: %v", err)
	}

	got := make([]float32, n)
	if err := MemcpyD2H(unsafe.Pointer(&got[0]), buf, bytes); err != nil {
		t.Fatalf("MemcpyD2H: %v", err)
	}

	for i := range n {
		if !approxEqual(got[i], want[i], 1e-6) {
			t.Fatalf("index %d: got=%g want=%g (in=%g)", i, got[i], want[i], input[i])
		}
	}
}

func TestMambaSiluF32InPlaceParity(t *testing.T) {
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

	sizes := []int{1, 31, 256, 1024, 4096, 65537}
	rng := rand.New(rand.NewSource(0xC0FFEE))

	for _, n := range sizes {
		input := make([]float32, n)
		for i := range n {
			input[i] = (rng.Float32()*2 - 1) * 8
		}
		want := make([]float32, n)
		for i, v := range input {
			want[i] = siluRef(v)
		}

		bytes := int64(n * 4)
		buf, err := AllocDevice(bytes)
		if err != nil {
			t.Fatalf("AllocDevice(n=%d): %v", n, err)
		}
		if err := MemcpyH2D(buf, unsafe.Pointer(&input[0]), bytes); err != nil {
			t.Fatalf("MemcpyH2D(n=%d): %v", n, err)
		}
		if err := SiluF32InPlace(buf, n, stream); err != nil {
			t.Fatalf("SiluF32InPlace(n=%d): %v", n, err)
		}
		if err := stream.Synchronize(); err != nil {
			t.Fatalf("stream sync(n=%d): %v", n, err)
		}

		got := make([]float32, n)
		if err := MemcpyD2H(unsafe.Pointer(&got[0]), buf, bytes); err != nil {
			t.Fatalf("MemcpyD2H(n=%d): %v", n, err)
		}
		if err := buf.Free(); err != nil {
			t.Fatalf("buf free(n=%d): %v", n, err)
		}

		for i := range n {
			if !approxEqual(got[i], want[i], 5e-6) {
				t.Fatalf("n=%d index %d: got=%g want=%g (in=%g)", n, i, got[i], want[i], input[i])
			}
		}
	}
}

func TestMambaSiluF32InPlaceNilBuffer(t *testing.T) {
	var empty DeviceBuffer
	if err := SiluF32InPlace(empty, 16, Stream{}); err == nil {
		t.Fatal("expected error for nil buffer, got nil")
	}
}

func TestMambaSiluF32InPlaceInvalidN(t *testing.T) {
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
	if err := SiluF32InPlace(buf, 0, Stream{}); err == nil {
		t.Fatal("expected error for n=0, got nil")
	}
	if err := SiluF32InPlace(buf, -1, Stream{}); err == nil {
		t.Fatal("expected error for n=-1, got nil")
	}
}
