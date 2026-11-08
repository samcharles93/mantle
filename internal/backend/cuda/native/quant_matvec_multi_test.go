//go:build cuda

package native

import (
	"math"
	"math/rand"
	"testing"
	"unsafe"
)

// quantAct constants must match the .cu kernel contract.
const (
	quantActNone = 0
	quantActSilu = 1
	quantActGelu = 2
)

type qmvFixture struct {
	q       []int8
	scales  []float32
	x       []float32
	bpr     int
	cols    int
	qBuf    DeviceBuffer
	sBuf    DeviceBuffer
	xBuf    DeviceBuffer
	stream  Stream
	rngRows int
}

func makeQMVFixture(t testing.TB, rows, cols int) *qmvFixture {
	t.Helper()
	bpr := (cols + 31) / 32
	f := &qmvFixture{
		q:       make([]int8, rows*bpr*32),
		scales:  make([]float32, rows*bpr),
		x:       make([]float32, cols),
		bpr:     bpr,
		cols:    cols,
		rngRows: rows,
	}
	rng := rand.New(rand.NewSource(42))
	for i := range f.q {
		f.q[i] = int8(rng.Intn(255) - 127)
	}
	for i := range f.scales {
		f.scales[i] = float32(rng.Float64()) * 0.01
	}
	// sprinkle zero scales to exercise the s == 0 skip path
	for i := 0; i < len(f.scales); i += 17 {
		f.scales[i] = 0
	}
	for i := range f.x {
		f.x[i] = float32(rng.Float64()*2 - 1)
	}
	var err error
	if f.qBuf, err = AllocDevice(int64(len(f.q))); err != nil {
		t.Fatalf("AllocDevice q: %v", err)
	}
	t.Cleanup(func() { _ = f.qBuf.Free() })
	if f.sBuf, err = AllocDevice(int64(len(f.scales)) * 4); err != nil {
		t.Fatalf("AllocDevice scales: %v", err)
	}
	t.Cleanup(func() { _ = f.sBuf.Free() })
	if f.xBuf, err = AllocDevice(int64(len(f.x)) * 4); err != nil {
		t.Fatalf("AllocDevice x: %v", err)
	}
	t.Cleanup(func() { _ = f.xBuf.Free() })
	if err = MemcpyH2D(f.qBuf, unsafe.Pointer(&f.q[0]), int64(len(f.q))); err != nil {
		t.Fatalf("H2D q: %v", err)
	}
	if err = MemcpyH2D(f.sBuf, unsafe.Pointer(&f.scales[0]), int64(len(f.scales))*4); err != nil {
		t.Fatalf("H2D scales: %v", err)
	}
	if err = MemcpyH2D(f.xBuf, unsafe.Pointer(&f.x[0]), int64(len(f.x))*4); err != nil {
		t.Fatalf("H2D x: %v", err)
	}
	if f.stream, err = NewStream(); err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	t.Cleanup(func() { f.stream.Destroy() })
	return f
}

func (f *qmvFixture) allocY(t testing.TB, rows int) (DeviceBuffer, []float32) {
	t.Helper()
	buf, err := AllocDevice(int64(rows) * 4)
	if err != nil {
		t.Fatalf("AllocDevice y: %v", err)
	}
	t.Cleanup(func() { _ = buf.Free() })
	host := make([]float32, rows)
	return buf, host
}

func (f *qmvFixture) readBack(t testing.TB, buf DeviceBuffer, host []float32) {
	t.Helper()
	if err := MemcpyD2H(unsafe.Pointer(&host[0]), buf, int64(len(host))*4); err != nil {
		t.Fatalf("D2H y: %v", err)
	}
	if err := f.stream.Synchronize(); err != nil {
		t.Fatalf("sync: %v", err)
	}
}

func requireBitExact(t testing.TB, got, want []float32, label string) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length mismatch got %d want %d", label, len(got), len(want))
	}
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("%s: not bit-exact at %d: got %v (%08x) want %v (%08x)", label, i, got[i], math.Float32bits(got[i]), want[i], math.Float32bits(want[i]))
		}
	}
}

// TestQuantMatVecInt8MultiSingleSegBitExact requires the multi-segment kernel
// with one segment to be bit-identical to the legacy xshared kernel.
func TestQuantMatVecInt8MultiSingleSegBitExact(t *testing.T) {
	count, err := DeviceCount()
	if err != nil || count < 1 {
		t.Skip("no cuda device available")
	}
	for _, shape := range []struct{ rows, cols int }{
		{rows: 2048, cols: 2048},
		{rows: 17, cols: 100}, // tail block + ragged rows
		{rows: 512, cols: 7680},
	} {
		f := makeQMVFixture(t, shape.rows, shape.cols)
		yLegacyBuf, yLegacy := f.allocY(t, shape.rows)
		yMultiBuf, yMulti := f.allocY(t, shape.rows)

		if err := QuantMatVecInt8BlocksF32Legacy(f.qBuf, f.sBuf, f.xBuf, yLegacyBuf, shape.rows, f.bpr, f.cols, f.stream); err != nil {
			t.Fatalf("legacy matvec rows=%d cols=%d: %v", shape.rows, shape.cols, err)
		}
		segs := []QuantMatVecSegment{{Q: f.qBuf, Scales: f.sBuf, Y: yMultiBuf, Rows: shape.rows}}
		if err := QuantMatVecInt8BlocksF32Multi(segs, f.xBuf, DeviceBuffer{}, quantActNone, f.cols, f.bpr, f.stream); err != nil {
			t.Fatalf("multi matvec rows=%d cols=%d: %v", shape.rows, shape.cols, err)
		}
		f.readBack(t, yLegacyBuf, yLegacy)
		f.readBack(t, yMultiBuf, yMulti)
		requireBitExact(t, yMulti, yLegacy, "multi-1seg vs legacy")
	}
}

// TestQuantMatVecInt8MultiSegmentsBitExact requires a 4-segment launch
// (QKV+gate shape) to be bit-identical to four separate legacy launches.
func TestQuantMatVecInt8MultiSegmentsBitExact(t *testing.T) {
	count, err := DeviceCount()
	if err != nil || count < 1 {
		t.Skip("no cuda device available")
	}
	const cols = 2048
	shapes := []int{2048, 16, 512, 512} // Wq, gate, Wk, Wv
	fixtures := make([]*qmvFixture, len(shapes))
	legacyY := make([]DeviceBuffer, len(shapes))
	legacyHost := make([][]float32, len(shapes))
	multiY := make([]DeviceBuffer, len(shapes))
	multiHost := make([][]float32, len(shapes))
	segs := make([]QuantMatVecSegment, len(shapes))
	for i, rows := range shapes {
		fixtures[i] = makeQMVFixture(t, rows, cols)
		legacyY[i], legacyHost[i] = fixtures[i].allocY(t, rows)
		multiY[i], multiHost[i] = fixtures[i].allocY(t, rows)
		segs[i] = QuantMatVecSegment{Q: fixtures[i].qBuf, Scales: fixtures[i].sBuf, Y: multiY[i], Rows: rows}
	}
	x := fixtures[0].xBuf
	bpr := fixtures[0].bpr
	for i, rows := range shapes {
		if err := QuantMatVecInt8BlocksF32Legacy(fixtures[i].qBuf, fixtures[i].sBuf, x, legacyY[i], rows, bpr, cols, fixtures[i].stream); err != nil {
			t.Fatalf("legacy seg %d: %v", i, err)
		}
	}
	if err := QuantMatVecInt8BlocksF32Multi(segs, x, DeviceBuffer{}, quantActNone, cols, bpr, fixtures[0].stream); err != nil {
		t.Fatalf("multi 4seg: %v", err)
	}
	for i := range shapes {
		fixtures[i].readBack(t, legacyY[i], legacyHost[i])
		fixtures[i].readBack(t, multiY[i], multiHost[i])
		requireBitExact(t, multiHost[i], legacyHost[i], "multi-4seg seg")
	}
}

// TestQuantMatVecInt8MultiActBitExact requires the activation-fused x-load
// (silu/gelu applied inline) to be bit-identical to silu_mul/gelu_mul followed
// by a legacy matvec over the activated buffer.
func TestQuantMatVecInt8MultiActBitExact(t *testing.T) {
	count, err := DeviceCount()
	if err != nil || count < 1 {
		t.Skip("no cuda device available")
	}
	for _, tc := range []struct {
		name string
		act  int
	}{
		{"silu", quantActSilu},
		{"gelu", quantActGelu},
	} {
		const rows, cols = 256, 512
		f := makeQMVFixture(t, rows, cols)
		gate := make([]float32, cols)
		up := make([]float32, cols)
		rng := rand.New(rand.NewSource(7))
		for i := range gate {
			gate[i] = float32(rng.Float64()*4 - 2)
			up[i] = float32(rng.Float64()*2 - 1)
		}
		gateBuf, err := AllocDevice(int64(cols) * 4)
		if err != nil {
			t.Fatalf("AllocDevice gate: %v", err)
		}
		defer gateBuf.Free()
		upBuf, err := AllocDevice(int64(cols) * 4)
		if err != nil {
			t.Fatalf("AllocDevice up: %v", err)
		}
		defer upBuf.Free()
		actBuf, err := AllocDevice(int64(cols) * 4)
		if err != nil {
			t.Fatalf("AllocDevice act: %v", err)
		}
		defer actBuf.Free()
		if err := MemcpyH2D(gateBuf, unsafe.Pointer(&gate[0]), int64(cols)*4); err != nil {
			t.Fatalf("H2D gate: %v", err)
		}
		if err := MemcpyH2D(upBuf, unsafe.Pointer(&up[0]), int64(cols)*4); err != nil {
			t.Fatalf("H2D up: %v", err)
		}

		// reference: act kernel then legacy matvec over actBuf
		if tc.act == quantActSilu {
			if err := SiluMulF32(gateBuf, upBuf, actBuf, cols, f.stream); err != nil {
				t.Fatalf("SiluMulF32: %v", err)
			}
		} else {
			if err := GeluMulF32(gateBuf, upBuf, actBuf, cols, f.stream); err != nil {
				t.Fatalf("GeluMulF32: %v", err)
			}
		}
		yRefBuf, yRef := f.allocY(t, rows)
		if err := QuantMatVecInt8BlocksF32Legacy(f.qBuf, f.sBuf, actBuf, yRefBuf, rows, f.bpr, f.cols, f.stream); err != nil {
			t.Fatalf("legacy matvec over act: %v", err)
		}
		// fused: multi kernel with act applied to (up, gate)
		yFusedBuf, yFused := f.allocY(t, rows)
		segs := []QuantMatVecSegment{{Q: f.qBuf, Scales: f.sBuf, Y: yFusedBuf, Rows: rows}}
		if err := QuantMatVecInt8BlocksF32Multi(segs, upBuf, gateBuf, tc.act, f.cols, f.bpr, f.stream); err != nil {
			t.Fatalf("multi act=%s: %v", tc.name, err)
		}
		f.readBack(t, yRefBuf, yRef)
		f.readBack(t, yFusedBuf, yFused)
		requireBitExact(t, yFused, yRef, "multi-act "+tc.name)
	}
}

// BenchmarkQuantMatVecInt8BlocksF32Legacy measures the legacy xshared decode
// matvec kernel.
func BenchmarkQuantMatVecInt8BlocksF32Legacy(b *testing.B) {
	benchmarkQuantMatVec(b, true, 1)
}

// BenchmarkQuantMatVecInt8Multi measures the fused multi-segment kernel with
// one segment (same shape as the legacy benchmark).
func BenchmarkQuantMatVecInt8Multi(b *testing.B) {
	benchmarkQuantMatVec(b, false, 1)
}

// BenchmarkQuantMatVecInt8MultiQKV measures a 4-segment QKV+gate launch.
func BenchmarkQuantMatVecInt8MultiQKV(b *testing.B) {
	benchmarkQuantMatVec(b, false, 4)
}

func benchmarkQuantMatVec(b *testing.B, legacy bool, nSegs int) {
	if _, err := DeviceCount(); err != nil || mustDeviceCount(b) < 1 {
		b.Skip("no cuda device available")
	}
	const cols = 2048
	segRows := []int{2048, 16, 512, 512}
	f := makeQMVFixture(b, segRows[0], cols)
	segs := make([]QuantMatVecSegment, nSegs)
	ys := make([]DeviceBuffer, nSegs)
	for i := 0; i < nSegs; i++ {
		rows := segRows[i]
		var qBuf, sBuf DeviceBuffer
		var err error
		if i == 0 {
			qBuf, sBuf = f.qBuf, f.sBuf
		} else {
			sf := makeQMVFixture(b, rows, cols)
			qBuf, sBuf = sf.qBuf, sf.sBuf
		}
		yBuf, err := AllocDevice(int64(rows) * 4)
		if err != nil {
			b.Fatal(err)
		}
		b.Cleanup(func() { _ = yBuf.Free() })
		ys[i] = yBuf
		segs[i] = QuantMatVecSegment{Q: qBuf, Scales: sBuf, Y: yBuf, Rows: rows}
	}
	launch := func() error {
		if legacy {
			return QuantMatVecInt8BlocksF32Legacy(f.qBuf, f.sBuf, f.xBuf, ys[0], segRows[0], f.bpr, cols, f.stream)
		}
		return QuantMatVecInt8BlocksF32Multi(segs, f.xBuf, DeviceBuffer{}, quantActNone, cols, f.bpr, f.stream)
	}
	if err := launch(); err != nil {
		b.Fatal(err)
	}
	if err := f.stream.Synchronize(); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := launch(); err != nil {
			b.Fatal(err)
		}
	}
	if err := f.stream.Synchronize(); err != nil {
		b.Fatal(err)
	}
}

func mustDeviceCount(b *testing.B) int {
	n, err := DeviceCount()
	if err != nil {
		b.Fatalf("DeviceCount: %v", err)
	}
	return n
}
