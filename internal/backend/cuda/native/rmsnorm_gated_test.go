//go:build cuda

package native

import (
	"math"
	"math/rand"
	"testing"
	"unsafe"
)

func refRMSNormGated(y, z, weight []float32, eps float32, normBeforeGate bool) []float32 {
	n := len(y)
	out := make([]float32, n)
	silu := func(v float32) float32 {
		return v / (1.0 + float32(math.Exp(-float64(v))))
	}
	if normBeforeGate {
		var ss float64
		for i := range y {
			ss += float64(y[i]) * float64(y[i])
		}
		rms := float32(math.Sqrt(ss/float64(n) + float64(eps)))
		inv := 1.0 / rms
		for i := range y {
			out[i] = y[i] * inv * weight[i] * silu(z[i])
		}
		return out
	}
	t := make([]float32, n)
	var ss float64
	for i := range y {
		t[i] = y[i] * silu(z[i])
		ss += float64(t[i]) * float64(t[i])
	}
	rms := float32(math.Sqrt(ss/float64(n) + float64(eps)))
	inv := 1.0 / rms
	for i := range t {
		out[i] = t[i] * inv * weight[i]
	}
	return out
}

func runRMSNormGatedCase(t *testing.T, n int, normBeforeGate bool, tol float32) {
	t.Helper()
	rng := rand.New(rand.NewSource(int64(n) + boolSeed(normBeforeGate)))
	y := make([]float32, n)
	z := make([]float32, n)
	w := make([]float32, n)
	for i := range y {
		y[i] = rng.Float32()*2 - 1
		z[i] = rng.Float32()*2 - 1
		w[i] = rng.Float32()*0.5 + 0.75
	}
	eps := float32(1e-5)
	ref := refRMSNormGated(y, z, w, eps, normBeforeGate)

	stream, err := NewStream()
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	defer stream.Destroy()

	yBuf := mustAllocAndCopy(t, y)
	defer yBuf.Free()
	zBuf := mustAllocAndCopy(t, z)
	defer zBuf.Free()
	wBuf := mustAllocAndCopy(t, w)
	defer wBuf.Free()
	outBuf, err := AllocDevice(int64(n * 4))
	if err != nil {
		t.Fatalf("AllocDevice: %v", err)
	}
	defer outBuf.Free()

	if err := RMSNormGatedF32(outBuf, yBuf, zBuf, wBuf, n, eps, normBeforeGate, stream); err != nil {
		t.Fatalf("RMSNormGatedF32: %v", err)
	}
	if err := stream.Synchronize(); err != nil {
		t.Fatalf("Synchronize: %v", err)
	}

	got := make([]float32, n)
	if err := MemcpyD2H(unsafe.Pointer(&got[0]), outBuf, int64(n*4)); err != nil {
		t.Fatalf("MemcpyD2H: %v", err)
	}
	for i := range ref {
		if !approxEqual(got[i], ref[i], tol) {
			t.Fatalf("n=%d normBeforeGate=%v out[%d]: got %.8f want %.8f (diff %.2e)",
				n, normBeforeGate, i, got[i], ref[i], got[i]-ref[i])
		}
	}
}

func boolSeed(b bool) int64 {
	if b {
		return 101
	}
	return 202
}

func TestRMSNormGatedF32(t *testing.T) {
	if n, err := DeviceCount(); err != nil || n == 0 {
		t.Skip("no CUDA device available")
	}

	sizes := []int{128, 1024, 4096}
	const tol = float32(5e-6)
	for _, n := range sizes {
		for _, gateFirst := range []bool{true, false} {
			name := "NormBeforeGate"
			if !gateFirst {
				name = "GateBeforeNorm"
			}
			t.Run(name+"/n"+itoa(n), func(t *testing.T) {
				runRMSNormGatedCase(t, n, gateFirst, tol)
			})
		}
	}

	t.Run("NilBuffer", func(t *testing.T) {
		stream, _ := NewStream()
		defer stream.Destroy()
		var nilBuf DeviceBuffer
		if err := RMSNormGatedF32(nilBuf, nilBuf, nilBuf, nilBuf, 4, 1e-5, true, stream); err == nil {
			t.Fatal("expected error for nil buffers")
		}
	})

	t.Run("InvalidN", func(t *testing.T) {
		stream, _ := NewStream()
		defer stream.Destroy()
		buf, _ := AllocDevice(16)
		defer buf.Free()
		if err := RMSNormGatedF32(buf, buf, buf, buf, 0, 1e-5, true, stream); err == nil {
			t.Fatal("expected error for n=0")
		}
	})
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	neg := n < 0
	if neg {
		n = -n
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		buf[i] = '-'
	}
	return string(buf[i:])
}
