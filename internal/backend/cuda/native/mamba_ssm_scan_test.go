//go:build cuda

package native

import (
	"math"
	"math/rand"
	"testing"
	"unsafe"
)

func mambaScanRef(out, state, x, dt, b, c, aLog, dVec []float32, headCount, headDim, dState, groupSize int) {
	for h := 0; h < headCount; h++ {
		group := h / groupSize
		a := -float32(math.Exp(float64(aLog[h])))
		dtH := dt[h]
		dA := float32(math.Exp(float64(a * dtH)))
		bGroup := b[group*dState : (group+1)*dState]
		cGroup := c[group*dState : (group+1)*dState]
		for p := 0; p < headDim; p++ {
			xhp := x[h*headDim+p]
			stateBase := (h*headDim + p) * dState
			var sum float32
			for n := 0; n < dState; n++ {
				idx := stateBase + n
				state[idx] = state[idx]*dA + dtH*bGroup[n]*xhp
				sum += cGroup[n] * state[idx]
			}
			out[h*headDim+p] = sum + dVec[h]*xhp
		}
	}
}

func TestMambaSSMScan(t *testing.T) {
	if n, err := DeviceCount(); err != nil || n == 0 {
		t.Skip("no CUDA device available")
	}

	t.Run("Parity", func(t *testing.T) {
		headCount := 8
		headDim := 4
		dState := 16
		groupSize := 4
		groups := headCount / groupSize

		rng := rand.New(rand.NewSource(42))
		x := make([]float32, headCount*headDim)
		dt := make([]float32, headCount)
		b := make([]float32, groups*dState)
		c := make([]float32, groups*dState)
		aLog := make([]float32, headCount)
		dVec := make([]float32, headCount)
		state := make([]float32, headCount*headDim*dState)

		for i := range x {
			x[i] = rng.Float32()*2 - 1
		}
		for i := range dt {
			dt[i] = rng.Float32()*0.5 + 0.01
		}
		for i := range b {
			b[i] = rng.Float32()*2 - 1
		}
		for i := range c {
			c[i] = rng.Float32()*2 - 1
		}
		for i := range aLog {
			aLog[i] = rng.Float32()*0.5 - 0.25
		}
		for i := range dVec {
			dVec[i] = rng.Float32()*2 - 1
		}
		for i := range state {
			state[i] = rng.Float32()*0.2 - 0.1
		}

		refState := append([]float32(nil), state...)
		refOut := make([]float32, headCount*headDim)
		mambaScanRef(refOut, refState, x, dt, b, c, aLog, dVec, headCount, headDim, dState, groupSize)

		stream, err := NewStream()
		if err != nil {
			t.Fatalf("NewStream: %v", err)
		}
		defer stream.Destroy()

		xBuf := mustAllocAndCopy(t, x)
		defer xBuf.Free()
		dtBuf := mustAllocAndCopy(t, dt)
		defer dtBuf.Free()
		bBuf := mustAllocAndCopy(t, b)
		defer bBuf.Free()
		cBuf := mustAllocAndCopy(t, c)
		defer cBuf.Free()
		aLogBuf := mustAllocAndCopy(t, aLog)
		defer aLogBuf.Free()
		dVecBuf := mustAllocAndCopy(t, dVec)
		defer dVecBuf.Free()
		stateBuf := mustAllocAndCopy(t, state)
		defer stateBuf.Free()
		outBuf, err := AllocDevice(int64(headCount * headDim * 4))
		if err != nil {
			t.Fatalf("AllocDevice out: %v", err)
		}
		defer outBuf.Free()

		if err := MambaSSMScan(xBuf, dtBuf, bBuf, cBuf, aLogBuf, dVecBuf, stateBuf, outBuf, headCount, headDim, dState, groupSize, stream); err != nil {
			t.Fatalf("MambaSSMScan: %v", err)
		}
		if err := stream.Synchronize(); err != nil {
			t.Fatalf("Synchronize: %v", err)
		}

		gotOut := make([]float32, headCount*headDim)
		if err := MemcpyD2H(unsafe.Pointer(&gotOut[0]), outBuf, int64(len(gotOut)*4)); err != nil {
			t.Fatalf("MemcpyD2H out: %v", err)
		}
		gotState := make([]float32, len(state))
		if err := MemcpyD2H(unsafe.Pointer(&gotState[0]), stateBuf, int64(len(gotState)*4)); err != nil {
			t.Fatalf("MemcpyD2H state: %v", err)
		}

		const tol = float32(5e-5)
		for i := range refOut {
			if !approxEqual(gotOut[i], refOut[i], tol) {
				t.Fatalf("out[%d]: got %.8f want %.8f (diff %.2e)", i, gotOut[i], refOut[i], gotOut[i]-refOut[i])
			}
		}
		for i := range refState {
			if !approxEqual(gotState[i], refState[i], tol) {
				t.Fatalf("state[%d]: got %.8f want %.8f (diff %.2e)", i, gotState[i], refState[i], gotState[i]-refState[i])
			}
		}
	})

	t.Run("NilBuffer", func(t *testing.T) {
		stream, _ := NewStream()
		defer stream.Destroy()
		var nilBuf DeviceBuffer
		if err := MambaSSMScan(nilBuf, nilBuf, nilBuf, nilBuf, nilBuf, nilBuf, nilBuf, nilBuf, 1, 1, 1, 1, stream); err == nil {
			t.Fatal("expected error for nil buffers")
		}
	})

	t.Run("InvalidDims", func(t *testing.T) {
		stream, _ := NewStream()
		defer stream.Destroy()
		buf, _ := AllocDevice(64)
		defer buf.Free()
		if err := MambaSSMScan(buf, buf, buf, buf, buf, buf, buf, buf, 0, 1, 1, 1, stream); err == nil {
			t.Fatal("expected error for headCount=0")
		}
		if err := MambaSSMScan(buf, buf, buf, buf, buf, buf, buf, buf, 6, 1, 1, 4, stream); err == nil {
			t.Fatal("expected error for non-divisible groupSize")
		}
		if err := MambaSSMScan(buf, buf, buf, buf, buf, buf, buf, buf, 4, 1, 2048, 1, stream); err == nil {
			t.Fatal("expected error for dState > 1024")
		}
	})
}

func mustAllocAndCopy(t *testing.T, data []float32) DeviceBuffer {
	t.Helper()
	buf, err := AllocDevice(int64(len(data) * 4))
	if err != nil {
		t.Fatalf("AllocDevice: %v", err)
	}
	if err := MemcpyH2D(buf, unsafe.Pointer(&data[0]), int64(len(data)*4)); err != nil {
		buf.Free()
		t.Fatalf("MemcpyH2D: %v", err)
	}
	return buf
}
