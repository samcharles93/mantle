//go:build cuda

package native

import (
	"math"
	"math/rand"
	"testing"
	"unsafe"
)

func TestMambaDtSoftplusClampF32(t *testing.T) {
	if n, err := DeviceCount(); err != nil || n == 0 {
		t.Skip("no CUDA device available")
	}

	t.Run("Parity", func(t *testing.T) {
		const n = 2048
		rng := rand.New(rand.NewSource(7))
		dt := make([]float32, n)
		bias := make([]float32, n)
		for i := range dt {
			dt[i] = rng.Float32()*40 - 20
			bias[i] = rng.Float32()*2 - 1
		}

		tMin := float32(0.001)
		tMax := float32(100.0)
		tFloor := float32(0.01)

		ref := make([]float32, n)
		for i := range dt {
			v := float64(dt[i] + bias[i])
			var sp float64
			switch {
			case v > 20:
				sp = v
			case v < -20:
				sp = math.Exp(v)
			default:
				sp = math.Log1p(math.Exp(v))
			}
			f := float32(sp)
			if f < tMin {
				f = tMin
			}
			if f > tMax {
				f = tMax
			}
			if tFloor > 0 && f < tFloor {
				f = tFloor
			}
			ref[i] = f
		}

		stream, err := NewStream()
		if err != nil {
			t.Fatalf("NewStream: %v", err)
		}
		defer stream.Destroy()

		dtBuf := mustAllocAndCopy(t, dt)
		defer dtBuf.Free()
		biasBuf := mustAllocAndCopy(t, bias)
		defer biasBuf.Free()

		if err := MambaDtSoftplusClampF32(dtBuf, biasBuf, n, tMin, tMax, tFloor, stream); err != nil {
			t.Fatalf("MambaDtSoftplusClampF32: %v", err)
		}
		if err := stream.Synchronize(); err != nil {
			t.Fatalf("Synchronize: %v", err)
		}

		got := make([]float32, n)
		if err := MemcpyD2H(unsafe.Pointer(&got[0]), dtBuf, int64(n*4)); err != nil {
			t.Fatalf("MemcpyD2H: %v", err)
		}

		const tol = float32(5e-6)
		for i := range ref {
			if !approxEqual(got[i], ref[i], tol) {
				t.Fatalf("dt[%d]: got %.8f want %.8f (diff %.2e)", i, got[i], ref[i], got[i]-ref[i])
			}
		}
	})

	t.Run("NilBuffer", func(t *testing.T) {
		stream, _ := NewStream()
		defer stream.Destroy()
		var nilBuf DeviceBuffer
		if err := MambaDtSoftplusClampF32(nilBuf, nilBuf, 1, 0, 1, 0, stream); err == nil {
			t.Fatal("expected error for nil buffers")
		}
	})

	t.Run("InvalidN", func(t *testing.T) {
		stream, _ := NewStream()
		defer stream.Destroy()
		buf, _ := AllocDevice(16)
		defer buf.Free()
		if err := MambaDtSoftplusClampF32(buf, buf, 0, 0, 1, 0, stream); err == nil {
			t.Fatal("expected error for n=0")
		}
	})
}

func TestScaleF32InPlace(t *testing.T) {
	if n, err := DeviceCount(); err != nil || n == 0 {
		t.Skip("no CUDA device available")
	}

	t.Run("Parity", func(t *testing.T) {
		const n = 4096
		rng := rand.New(rand.NewSource(9))
		x := make([]float32, n)
		for i := range x {
			x[i] = rng.Float32()*4 - 2
		}
		scale := float32(0.123)
		ref := make([]float32, n)
		for i := range x {
			ref[i] = x[i] * scale
		}

		stream, err := NewStream()
		if err != nil {
			t.Fatalf("NewStream: %v", err)
		}
		defer stream.Destroy()

		xBuf := mustAllocAndCopy(t, x)
		defer xBuf.Free()

		if err := ScaleF32InPlace(xBuf, n, scale, stream); err != nil {
			t.Fatalf("ScaleF32InPlace: %v", err)
		}
		if err := stream.Synchronize(); err != nil {
			t.Fatalf("Synchronize: %v", err)
		}

		got := make([]float32, n)
		if err := MemcpyD2H(unsafe.Pointer(&got[0]), xBuf, int64(n*4)); err != nil {
			t.Fatalf("MemcpyD2H: %v", err)
		}
		const tol = float32(1e-6)
		for i := range ref {
			if !approxEqual(got[i], ref[i], tol) {
				t.Fatalf("x[%d]: got %.8f want %.8f", i, got[i], ref[i])
			}
		}
	})

	t.Run("IdentityShortCircuit", func(t *testing.T) {
		stream, _ := NewStream()
		defer stream.Destroy()
		x := []float32{1, 2, 3}
		buf := mustAllocAndCopy(t, x)
		defer buf.Free()
		if err := ScaleF32InPlace(buf, len(x), 1.0, stream); err != nil {
			t.Fatalf("ScaleF32InPlace(1.0): %v", err)
		}
		if err := stream.Synchronize(); err != nil {
			t.Fatalf("Synchronize: %v", err)
		}
		got := make([]float32, len(x))
		if err := MemcpyD2H(unsafe.Pointer(&got[0]), buf, int64(len(x)*4)); err != nil {
			t.Fatalf("MemcpyD2H: %v", err)
		}
		for i := range x {
			if got[i] != x[i] {
				t.Fatalf("scale=1 should be no-op, x[%d]=%v want %v", i, got[i], x[i])
			}
		}
	})

	t.Run("NilBuffer", func(t *testing.T) {
		stream, _ := NewStream()
		defer stream.Destroy()
		var nilBuf DeviceBuffer
		if err := ScaleF32InPlace(nilBuf, 1, 2.0, stream); err == nil {
			t.Fatal("expected error for nil buffer")
		}
	})
}
