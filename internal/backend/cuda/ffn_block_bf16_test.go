//go:build cuda

package cuda

import (
	"encoding/binary"
	"math"
	"testing"
	"unsafe"

	"github.com/samcharles93/mantle/internal/backend/core"
	"github.com/samcharles93/mantle/internal/backend/cuda/native"
	"github.com/samcharles93/mantle/pkg/mcf"
)

// TestFFNBlockBF16DeviceInput runs the dense FFN fast path on BF16 weights with
// its input left on device by DeviceRMSNorm, as Gemma-3 decode does. It pins
// two regressions: the host copy of the normed input is stale (NaN here), so
// FFNBlock must read the device copy; and the result must be published as the
// pending device result for out, not copied to out behind lastResult's back.
func TestFFNBlockBF16DeviceInput(t *testing.T) {
	count, err := native.DeviceCount()
	if err != nil || count < 1 {
		t.Skip("no CUDA device available")
	}
	stream, err := native.NewStream()
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	defer func() { _ = stream.Destroy() }()
	blas, err := native.NewBlasHandle(stream)
	if err != nil {
		t.Fatalf("NewBlasHandle: %v", err)
	}
	defer func() { _ = blas.Destroy() }()
	ops := NewOps(stream, blas)
	defer func() { _ = ops.Close() }()

	const (
		hidden = 64
		interm = 128
		eps    = 1e-6
	)
	bf16Mat := func(r, c int, f func(i int) float32) (*core.Mat, []float32) {
		raw := make([]byte, r*c*2)
		vals := make([]float32, r*c)
		for i := range vals {
			b := bf16FromF32(f(i))
			binary.LittleEndian.PutUint16(raw[2*i:], b)
			vals[i] = math.Float32frombits(uint32(b) << 16)
		}
		m, err := core.NewMatFromRaw(r, c, mcf.DTypeBF16, raw)
		if err != nil {
			t.Fatalf("NewMatFromRaw: %v", err)
		}
		return &m, vals
	}
	wave := func(scale float64, phase float64) func(int) float32 {
		return func(i int) float32 { return float32(scale * math.Sin(float64(i)*0.37+phase)) }
	}
	up, upV := bf16Mat(interm, hidden, wave(0.2, 0.1))
	gate, gateV := bf16Mat(interm, hidden, wave(0.25, 1.3))
	down, downV := bf16Mat(hidden, interm, wave(0.15, 2.2))
	layer := &core.Layer{FfnUp: up, FfnGate: gate, FfnDown: down, FFNActivation: "gelu_pytorch_tanh"}

	x := make([]float32, hidden)
	normW := make([]float32, hidden)
	for i := range x {
		x[i] = float32(math.Cos(float64(i) * 0.71))
		normW[i] = 1 + 0.01*float32(i%7)
	}

	ops.BeginToken(x)
	normed := make([]float32, hidden)
	for i := range normed {
		normed[i] = float32(math.NaN())
	}
	if !ops.DeviceRMSNorm(normed, x, normW, eps) {
		t.Fatal("DeviceRMSNorm declined a device-resident input")
	}
	out := make([]float32, hidden)
	if !ops.FFNBlock(layer, normed, out) {
		t.Fatalf("FFNBlock declined BF16 weights: %v", ops.ConsumeFastPathError())
	}
	if !ops.lastResultValid || ops.lastResultHostPtr != uintptr(unsafe.Pointer(&out[0])) {
		t.Fatal("FFNBlock did not publish its result as the pending device result for out")
	}
	if err := ops.FlushBlockResult(); err != nil {
		t.Fatalf("FlushBlockResult: %v", err)
	}

	// Host reference: RMSNorm, BF16 input, tanh GELU, BF16 activation.
	var ss float64
	for _, v := range x {
		ss += float64(v) * float64(v)
	}
	inv := 1 / math.Sqrt(ss/hidden+eps)
	bf := func(v float64) float64 { return float64(math.Float32frombits(uint32(bf16FromF32(float32(v))) << 16)) }
	xb := make([]float64, hidden)
	for i, v := range x {
		xb[i] = bf(float64(float32(float64(v) * inv * float64(normW[i]))))
	}
	act := make([]float64, interm)
	for r := range interm {
		var u, g float64
		for c := range hidden {
			u += float64(upV[r*hidden+c]) * xb[c]
			g += float64(gateV[r*hidden+c]) * xb[c]
		}
		gelu := 0.5 * g * (1 + math.Tanh(0.7978845608028654*(g+0.044715*g*g*g)))
		act[r] = bf(gelu * u)
	}
	for r := range hidden {
		var want float64
		for c := range interm {
			want += float64(downV[r*interm+c]) * act[c]
		}
		if got := float64(out[r]); math.IsNaN(got) || math.Abs(got-want) > 2e-3*(1+math.Abs(want)) {
			t.Fatalf("out[%d] = %g, want %g", r, got, want)
		}
	}
	ops.EndToken(x)
}
