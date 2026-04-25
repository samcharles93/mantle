//go:build cuda

package native

import (
	"math"
	"math/rand"
	"testing"
	"unsafe"
)

func safeSoftplus(x float32) float32 {
	if x > 20 {
		return x
	}
	if x < -20 {
		return float32(math.Exp(float64(x)))
	}
	return float32(math.Log1p(math.Exp(float64(x))))
}

func safeSigmoid(x float32) float32 {
	return float32(1.0 / (1.0 + math.Exp(-float64(x))))
}

func deltaNetRecurrentRef(
	state, q, kBuf, v []float32,
	aLog, deltaA, deltaB, dtBias []float32,
	scale float32,
	headKeyDim, headValueDim, nValueHeads, nKeyHeads, groupSize int,
) []float32 {
	out := make([]float32, nValueHeads*headValueDim)
	for hv := range nValueHeads {
		hk := hv
		if groupSize > 1 {
			hk = hv / groupSize
		}
		qH := q[hk*headKeyDim : (hk+1)*headKeyDim]
		kH := kBuf[hk*headKeyDim : (hk+1)*headKeyDim]
		vH := v[hv*headValueDim : (hv+1)*headValueDim]
		stH := state[hv*headKeyDim*headValueDim : (hv+1)*headKeyDim*headValueDim]
		oH := out[hv*headValueDim : (hv+1)*headValueDim]

		da := deltaA[hv] + dtBias[hv]
		sp := safeSoftplus(da)
		dec := float32(math.Exp(-float64(float32(math.Exp(float64(aLog[hv]))) * sp)))
		bet := safeSigmoid(deltaB[hv])

		for i := range stH {
			stH[i] *= dec
		}
		delta := make([]float32, headValueDim)
		for vi := range headValueDim {
			var kvMem float32
			for kk := range headKeyDim {
				kvMem += stH[kk*headValueDim+vi] * kH[kk]
			}
			delta[vi] = (vH[vi] - kvMem) * bet
		}
		for kk := range headKeyDim {
			base := kk * headValueDim
			kVal := kH[kk]
			for vi := range headValueDim {
				stH[base+vi] += kVal * delta[vi]
			}
		}
		for vi := range headValueDim {
			var sum float32
			for kk := range headKeyDim {
				sum += stH[kk*headValueDim+vi] * qH[kk]
			}
			oH[vi] = sum * scale
		}
	}
	return out
}

func TestDeltaNetRecurrentF32Parity(t *testing.T) {
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
		hk, hv, nv, nk int
	}{
		{32, 32, 4, 4},
		{64, 64, 8, 4},
		{128, 128, 16, 8},
	}
	const steps = 16
	const scale = float32(0.0883883)
	rng := rand.New(rand.NewSource(0xD17A7))

	for _, c := range cases {
		group := c.nv / c.nk
		stateLen := c.nv * c.hk * c.hv
		qLen := c.nk * c.hk
		vLen := c.nv * c.hv
		outLen := vLen

		stateRef := make([]float32, stateLen)
		for i := range stateLen {
			stateRef[i] = (rng.Float32()*2 - 1) * 0.1
		}
		stateInit := make([]float32, stateLen)
		copy(stateInit, stateRef)

		aLog := make([]float32, c.nv)
		deltaA := make([]float32, c.nv)
		deltaB := make([]float32, c.nv)
		dtBias := make([]float32, c.nv)
		for i := range c.nv {
			aLog[i] = rng.Float32()*1.5 - 0.5
			deltaA[i] = rng.Float32()*2 - 1
			deltaB[i] = rng.Float32()*2 - 1
			dtBias[i] = rng.Float32()*0.4 - 0.2
		}

		stateBuf, err := AllocDevice(int64(stateLen * 4))
		if err != nil {
			t.Fatalf("alloc state: %v", err)
		}
		qBuf, err := AllocDevice(int64(qLen * 4))
		if err != nil {
			t.Fatalf("alloc q: %v", err)
		}
		kBuf, err := AllocDevice(int64(qLen * 4))
		if err != nil {
			t.Fatalf("alloc k: %v", err)
		}
		vBuf, err := AllocDevice(int64(vLen * 4))
		if err != nil {
			t.Fatalf("alloc v: %v", err)
		}
		aLogBuf, err := AllocDevice(int64(c.nv * 4))
		if err != nil {
			t.Fatalf("alloc aLog: %v", err)
		}
		deltaABuf, err := AllocDevice(int64(c.nv * 4))
		if err != nil {
			t.Fatalf("alloc deltaA: %v", err)
		}
		deltaBBuf, err := AllocDevice(int64(c.nv * 4))
		if err != nil {
			t.Fatalf("alloc deltaB: %v", err)
		}
		dtBiasBuf, err := AllocDevice(int64(c.nv * 4))
		if err != nil {
			t.Fatalf("alloc dtBias: %v", err)
		}
		outBuf, err := AllocDevice(int64(outLen * 4))
		if err != nil {
			t.Fatalf("alloc out: %v", err)
		}

		if err := MemcpyH2D(stateBuf, unsafe.Pointer(&stateInit[0]), int64(stateLen*4)); err != nil {
			t.Fatalf("h2d state: %v", err)
		}
		if err := MemcpyH2D(aLogBuf, unsafe.Pointer(&aLog[0]), int64(c.nv*4)); err != nil {
			t.Fatalf("h2d aLog: %v", err)
		}
		if err := MemcpyH2D(deltaABuf, unsafe.Pointer(&deltaA[0]), int64(c.nv*4)); err != nil {
			t.Fatalf("h2d deltaA: %v", err)
		}
		if err := MemcpyH2D(deltaBBuf, unsafe.Pointer(&deltaB[0]), int64(c.nv*4)); err != nil {
			t.Fatalf("h2d deltaB: %v", err)
		}
		if err := MemcpyH2D(dtBiasBuf, unsafe.Pointer(&dtBias[0]), int64(c.nv*4)); err != nil {
			t.Fatalf("h2d dtBias: %v", err)
		}

		for step := range steps {
			q := make([]float32, qLen)
			k := make([]float32, qLen)
			v := make([]float32, vLen)
			for i := range qLen {
				q[i] = (rng.Float32()*2 - 1) * 0.5
				k[i] = (rng.Float32()*2 - 1) * 0.5
			}
			for i := range vLen {
				v[i] = (rng.Float32()*2 - 1) * 0.5
			}

			wantOut := deltaNetRecurrentRef(stateRef, q, k, v, aLog, deltaA, deltaB, dtBias, scale, c.hk, c.hv, c.nv, c.nk, group)

			if err := MemcpyH2D(qBuf, unsafe.Pointer(&q[0]), int64(qLen*4)); err != nil {
				t.Fatalf("h2d q: %v", err)
			}
			if err := MemcpyH2D(kBuf, unsafe.Pointer(&k[0]), int64(qLen*4)); err != nil {
				t.Fatalf("h2d k: %v", err)
			}
			if err := MemcpyH2D(vBuf, unsafe.Pointer(&v[0]), int64(vLen*4)); err != nil {
				t.Fatalf("h2d v: %v", err)
			}
			if err := DeltaNetRecurrentF32(stateBuf, qBuf, kBuf, vBuf, aLogBuf, deltaABuf, deltaBBuf, dtBiasBuf, outBuf, scale, c.hk, c.hv, c.nv, c.nk, group, stream); err != nil {
				t.Fatalf("DeltaNetRecurrentF32 step=%d: %v", step, err)
			}
			if err := stream.Synchronize(); err != nil {
				t.Fatalf("sync: %v", err)
			}

			gotOut := make([]float32, outLen)
			if err := MemcpyD2H(unsafe.Pointer(&gotOut[0]), outBuf, int64(outLen*4)); err != nil {
				t.Fatalf("d2h out: %v", err)
			}
			gotState := make([]float32, stateLen)
			if err := MemcpyD2H(unsafe.Pointer(&gotState[0]), stateBuf, int64(stateLen*4)); err != nil {
				t.Fatalf("d2h state: %v", err)
			}

			for i := range outLen {
				if !approxEqual(gotOut[i], wantOut[i], 5e-5) {
					t.Fatalf("case hk=%d nv=%d step=%d out[%d]: got=%g want=%g", c.hk, c.nv, step, i, gotOut[i], wantOut[i])
				}
			}
			for i := range stateLen {
				if !approxEqual(gotState[i], stateRef[i], 5e-5) {
					t.Fatalf("case hk=%d nv=%d step=%d state[%d]: got=%g want=%g", c.hk, c.nv, step, i, gotState[i], stateRef[i])
				}
			}
		}

		_ = stateBuf.Free()
		_ = qBuf.Free()
		_ = kBuf.Free()
		_ = vBuf.Free()
		_ = aLogBuf.Free()
		_ = deltaABuf.Free()
		_ = deltaBBuf.Free()
		_ = dtBiasBuf.Free()
		_ = outBuf.Free()
	}
}

func TestDeltaNetRecurrentF32NilBuffer(t *testing.T) {
	var empty DeviceBuffer
	if err := DeltaNetRecurrentF32(empty, empty, empty, empty, empty, empty, empty, empty, empty, 0.1, 32, 32, 4, 4, 1, Stream{}); err == nil {
		t.Fatal("expected error for nil buffer")
	}
}

func TestDeltaNetRecurrentF32InvalidDims(t *testing.T) {
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
	if err := DeltaNetRecurrentF32(buf, buf, buf, buf, buf, buf, buf, buf, buf, 0.1, 0, 32, 4, 4, 1, Stream{}); err == nil {
		t.Fatal("expected error for headKeyDim=0")
	}
	if err := DeltaNetRecurrentF32(buf, buf, buf, buf, buf, buf, buf, buf, buf, 0.1, 32, 32, 0, 4, 1, Stream{}); err == nil {
		t.Fatal("expected error for nValueHeads=0")
	}
}
