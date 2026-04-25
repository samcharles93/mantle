//go:build cuda

package native

import (
	"math/rand"
	"testing"
	"unsafe"
)

func mambaDepthwiseConvRef(in, convW, bias, state, out []float32, channels, klen int, hasBias bool) {
	for c := range channels {
		kw := convW[c*klen : (c+1)*klen]
		xc := in[c]
		var sum float32
		if hasBias {
			sum = bias[c]
		}
		for k := 0; k < klen-1; k++ {
			sum += kw[k] * state[k*channels+c]
		}
		sum += kw[klen-1] * xc
		out[c] = sum
	}
	if klen <= 1 {
		return
	}
	for k := 0; k < klen-2; k++ {
		for c := range channels {
			state[k*channels+c] = state[(k+1)*channels+c]
		}
	}
	for c := range channels {
		state[(klen-2)*channels+c] = in[c]
	}
}

func TestMambaDepthwiseConvFixed(t *testing.T) {
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

	// Channels=2, klen=3. conv_w row-major [channels*klen]; state row-major [(klen-1)*channels].
	// out[0] = 1*1 + 2*3 + 3*5 = 22; out[1] = 4*2 + 5*4 + 6*6 = 64.
	// Post-state: shift left + append in at row klen-2.
	const (
		channels = 2
		klen     = 3
	)
	in := []float32{5, 6}
	convW := []float32{1, 2, 3, 4, 5, 6}
	state := []float32{1, 2, 3, 4}
	wantOut := []float32{22, 64}
	wantState := []float32{3, 4, 5, 6}

	out := make([]float32, channels)

	fsize := int64(unsafe.Sizeof(float32(0)))

	inDev, err := AllocDevice(int64(len(in)) * fsize)
	if err != nil {
		t.Fatalf("AllocDevice in: %v", err)
	}
	defer inDev.Free()
	wDev, err := AllocDevice(int64(len(convW)) * fsize)
	if err != nil {
		t.Fatalf("AllocDevice convW: %v", err)
	}
	defer wDev.Free()
	stDev, err := AllocDevice(int64(len(state)) * fsize)
	if err != nil {
		t.Fatalf("AllocDevice state: %v", err)
	}
	defer stDev.Free()
	outDev, err := AllocDevice(int64(len(out)) * fsize)
	if err != nil {
		t.Fatalf("AllocDevice out: %v", err)
	}
	defer outDev.Free()

	if err := MemcpyH2D(inDev, unsafe.Pointer(&in[0]), int64(len(in))*fsize); err != nil {
		t.Fatalf("MemcpyH2D in: %v", err)
	}
	if err := MemcpyH2D(wDev, unsafe.Pointer(&convW[0]), int64(len(convW))*fsize); err != nil {
		t.Fatalf("MemcpyH2D convW: %v", err)
	}
	if err := MemcpyH2D(stDev, unsafe.Pointer(&state[0]), int64(len(state))*fsize); err != nil {
		t.Fatalf("MemcpyH2D state: %v", err)
	}

	// nil bias buffer signals has_bias=0 to the kernel.
	var nilBias DeviceBuffer
	if err := MambaDepthwiseConv(inDev, wDev, nilBias, stDev, outDev, channels, klen, stream); err != nil {
		t.Fatalf("MambaDepthwiseConv: %v", err)
	}
	if err := stream.Synchronize(); err != nil {
		t.Fatalf("stream synchronize: %v", err)
	}

	gotState := make([]float32, len(state))
	if err := MemcpyD2H(unsafe.Pointer(&out[0]), outDev, int64(len(out))*fsize); err != nil {
		t.Fatalf("MemcpyD2H out: %v", err)
	}
	if err := MemcpyD2H(unsafe.Pointer(&gotState[0]), stDev, int64(len(gotState))*fsize); err != nil {
		t.Fatalf("MemcpyD2H state: %v", err)
	}

	for i := range wantOut {
		if !approxEqual(wantOut[i], out[i], 1e-6) {
			t.Fatalf("out[%d]: got %v want %v", i, out[i], wantOut[i])
		}
	}
	for i := range wantState {
		if !approxEqual(wantState[i], gotState[i], 1e-6) {
			t.Fatalf("state[%d]: got %v want %v", i, gotState[i], wantState[i])
		}
	}
}

func TestMambaDepthwiseConvRandomized(t *testing.T) {
	count, err := DeviceCount()
	if err != nil {
		t.Fatalf("DeviceCount: %v", err)
	}
	if count < 1 {
		t.Skip("no cuda device available")
	}

	cases := []struct {
		name     string
		channels int
		klen     int
		hasBias  bool
	}{
		{"klen1_nobias", 17, 1, false},
		{"klen1_bias", 33, 1, true},
		{"klen2_nobias", 65, 2, false},
		{"klen2_bias", 48, 2, true},
		{"klen4_nobias", 96, 4, false},
		{"klen4_bias", 129, 4, true}, // spans >1 block (256 threads)
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			stream, err := NewStream()
			if err != nil {
				t.Fatalf("NewStream: %v", err)
			}
			defer func() {
				if err := stream.Destroy(); err != nil {
					t.Fatalf("stream destroy: %v", err)
				}
			}()

			rng := rand.New(rand.NewSource(int64(tc.channels*131 + tc.klen*17)))

			in := make([]float32, tc.channels)
			convW := make([]float32, tc.channels*tc.klen)
			var state []float32
			if tc.klen > 1 {
				state = make([]float32, (tc.klen-1)*tc.channels)
			}
			var bias []float32
			if tc.hasBias {
				bias = make([]float32, tc.channels)
			}
			for i := range in {
				in[i] = rng.Float32()*2 - 1
			}
			for i := range convW {
				convW[i] = rng.Float32()*2 - 1
			}
			for i := range state {
				state[i] = rng.Float32()*2 - 1
			}
			for i := range bias {
				bias[i] = rng.Float32()*2 - 1
			}

			// Reference on host (clone buffers so device state doesn't affect us).
			refState := append([]float32(nil), state...)
			refOut := make([]float32, tc.channels)
			mambaDepthwiseConvRef(in, convW, bias, refState, refOut, tc.channels, tc.klen, tc.hasBias)

			fsize := int64(unsafe.Sizeof(float32(0)))

			inDev, err := AllocDevice(int64(len(in)) * fsize)
			if err != nil {
				t.Fatalf("AllocDevice in: %v", err)
			}
			defer inDev.Free()
			wDev, err := AllocDevice(int64(len(convW)) * fsize)
			if err != nil {
				t.Fatalf("AllocDevice convW: %v", err)
			}
			defer wDev.Free()

			var stDev DeviceBuffer
			if len(state) > 0 {
				stDev, err = AllocDevice(int64(len(state)) * fsize)
				if err != nil {
					t.Fatalf("AllocDevice state: %v", err)
				}
				defer stDev.Free()
			} else {
				stDev, err = AllocDevice(fsize)
				if err != nil {
					t.Fatalf("AllocDevice state placeholder: %v", err)
				}
				defer stDev.Free()
			}

			var biasDev DeviceBuffer
			if tc.hasBias {
				biasDev, err = AllocDevice(int64(len(bias)) * fsize)
				if err != nil {
					t.Fatalf("AllocDevice bias: %v", err)
				}
				defer biasDev.Free()
			}

			outDev, err := AllocDevice(int64(tc.channels) * fsize)
			if err != nil {
				t.Fatalf("AllocDevice out: %v", err)
			}
			defer outDev.Free()

			if err := MemcpyH2D(inDev, unsafe.Pointer(&in[0]), int64(len(in))*fsize); err != nil {
				t.Fatalf("MemcpyH2D in: %v", err)
			}
			if err := MemcpyH2D(wDev, unsafe.Pointer(&convW[0]), int64(len(convW))*fsize); err != nil {
				t.Fatalf("MemcpyH2D convW: %v", err)
			}
			if len(state) > 0 {
				if err := MemcpyH2D(stDev, unsafe.Pointer(&state[0]), int64(len(state))*fsize); err != nil {
					t.Fatalf("MemcpyH2D state: %v", err)
				}
			}
			if tc.hasBias {
				if err := MemcpyH2D(biasDev, unsafe.Pointer(&bias[0]), int64(len(bias))*fsize); err != nil {
					t.Fatalf("MemcpyH2D bias: %v", err)
				}
			}

			if err := MambaDepthwiseConv(inDev, wDev, biasDev, stDev, outDev, tc.channels, tc.klen, stream); err != nil {
				t.Fatalf("MambaDepthwiseConv: %v", err)
			}
			if err := stream.Synchronize(); err != nil {
				t.Fatalf("stream synchronize: %v", err)
			}

			gotOut := make([]float32, tc.channels)
			if err := MemcpyD2H(unsafe.Pointer(&gotOut[0]), outDev, int64(len(gotOut))*fsize); err != nil {
				t.Fatalf("MemcpyD2H out: %v", err)
			}
			for i := range refOut {
				if !approxEqual(refOut[i], gotOut[i], 1e-6) {
					t.Fatalf("out mismatch at c=%d: got %v want %v", i, gotOut[i], refOut[i])
				}
			}

			if len(state) > 0 {
				gotState := make([]float32, len(state))
				if err := MemcpyD2H(unsafe.Pointer(&gotState[0]), stDev, int64(len(gotState))*fsize); err != nil {
					t.Fatalf("MemcpyD2H state: %v", err)
				}
				for i := range refState {
					if !approxEqual(refState[i], gotState[i], 1e-6) {
						t.Fatalf("state mismatch at %d (klen=%d, channels=%d): got %v want %v",
							i, tc.klen, tc.channels, gotState[i], refState[i])
					}
				}
			}
		})
	}
}
