//go:build cuda

package native

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"
	"unsafe"
)

// k4TestMatrix is a k4 weight in the three device buffers QuantMatVecK4F32
// reads, plus its dequantised values for a float64 reference.
type k4TestMatrix struct {
	rows, cols, bpr      int
	q, super, sub        []byte
	deq                  []float64 // [rows, bpr*32]
	qDev, supDev, subDev DeviceBuffer
}

// newK4TestMatrix builds a random k4 matrix. Every 17th block gets a zero
// 6-bit sub-scale and every 5th superblock of row 1 a zero super-scale, the
// two ways a block contributes nothing.
func newK4TestMatrix(t testing.TB, rows, cols int, seed uint64) *k4TestMatrix {
	t.Helper()
	rng := rand.New(rand.NewPCG(seed, 2))
	bpr := (cols + 31) / 32
	spr := (bpr + 7) / 8
	m := &k4TestMatrix{rows: rows, cols: cols, bpr: bpr}
	m.q = make([]byte, rows*bpr*16)
	m.sub = make([]byte, rows*bpr)
	supers := make([]uint16, rows*spr)
	m.deq = make([]float64, rows*bpr*32)
	for i := range supers {
		supers[i] = f32ToF16BitsRef(float32(0.002 + 0.01*rng.Float64()))
		if i/spr == 1 && i%5 == 0 {
			supers[i] = 0
		}
	}
	var block [32]int8
	for r := range rows {
		for b := range bpr {
			blk := r*bpr + b
			for i := range block {
				block[i] = int8(rng.IntN(16) - 8)
			}
			packQ4Block(m.q[blk*16:(blk+1)*16], block[:])
			u6 := byte(rng.IntN(64)) | byte(rng.IntN(4))<<6 // high bits must be ignored
			if blk%17 == 0 {
				u6 &^= 0x3F
			}
			m.sub[blk] = u6
			s := float64(fp16ToF32(supers[r*spr+b/8])) * float64(u6&0x3F) / 32
			for i, v := range block {
				m.deq[blk*32+i] = s * float64(v)
			}
		}
	}
	m.super = make([]byte, len(supers)*2)
	for i, v := range supers {
		m.super[2*i], m.super[2*i+1] = byte(v), byte(v>>8)
	}
	upload := func(b []byte) DeviceBuffer {
		d, err := AllocDevice(int64(len(b)))
		if err != nil {
			t.Fatalf("AllocDevice: %v", err)
		}
		if err := MemcpyH2D(d, unsafe.Pointer(&b[0]), int64(len(b))); err != nil {
			t.Fatalf("MemcpyH2D: %v", err)
		}
		return d
	}
	m.qDev, m.supDev, m.subDev = upload(m.q), upload(m.super), upload(m.sub)
	return m
}

func (m *k4TestMatrix) free() {
	_ = m.qDev.Free()
	_ = m.supDev.Free()
	_ = m.subDev.Free()
}

// TestQuantMatVecK4F32Shapes checks the k4 matvec against a float64 reference
// at LFM3-1.7B projection shapes and at widths with a partial last block and
// fewer rows than one warp's row group.
func TestQuantMatVecK4F32Shapes(t *testing.T) {
	if count, err := DeviceCount(); err != nil || count < 1 {
		t.Skip("no cuda device available")
	}
	stream, err := NewStream()
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	defer func() { _ = stream.Destroy() }()

	for _, sh := range []struct{ rows, cols int }{
		{512, 2048}, {2048, 7680}, {7680, 2048}, {3, 2048}, {37, 330}, {9, 8224}, {5, 20},
	} {
		t.Run(fmt.Sprintf("%dx%d", sh.rows, sh.cols), func(t *testing.T) {
			m := newK4TestMatrix(t, sh.rows, sh.cols, uint64(sh.rows*31+sh.cols))
			defer m.free()
			rng := rand.New(rand.NewPCG(9, uint64(sh.cols)))
			x := make([]float32, sh.cols)
			for i := range x {
				x[i] = float32(rng.NormFloat64())
			}
			got := runQuantMatVecK4(t, m, x, stream)
			for r := range sh.rows {
				var want, mag float64
				for c := range sh.cols {
					p := m.deq[r*m.bpr*32+c] * float64(x[c])
					want += p
					mag += math.Abs(p)
				}
				if d := math.Abs(float64(got[r]) - want); d > 1e-5*mag+1e-6 {
					t.Fatalf("row %d = %g, want %g (|err| %g, sum|p| %g)", r, got[r], want, d, mag)
				}
			}
		})
	}
}

func runQuantMatVecK4(t testing.TB, m *k4TestMatrix, x []float32, stream Stream) []float32 {
	t.Helper()
	xDev, err := AllocDevice(int64(len(x) * 4))
	if err != nil {
		t.Fatalf("AllocDevice x: %v", err)
	}
	defer func() { _ = xDev.Free() }()
	yDev, err := AllocDevice(int64(m.rows * 4))
	if err != nil {
		t.Fatalf("AllocDevice y: %v", err)
	}
	defer func() { _ = yDev.Free() }()
	if err := MemcpyH2D(xDev, unsafe.Pointer(&x[0]), int64(len(x)*4)); err != nil {
		t.Fatalf("MemcpyH2D x: %v", err)
	}
	if err := QuantMatVecK4F32(m.qDev, m.supDev, m.subDev, xDev, yDev, m.rows, m.bpr, m.cols, stream); err != nil {
		t.Fatalf("QuantMatVecK4F32: %v", err)
	}
	got := make([]float32, m.rows)
	if err := MemcpyD2H(unsafe.Pointer(&got[0]), yDev, int64(len(got)*4)); err != nil {
		t.Fatalf("MemcpyD2H y: %v", err)
	}
	return got
}

// BenchmarkQuantMatVecK4F32 measures the k4 decode matvec at the LFM3-1.7B
// projection shapes; wGB/s is the packed weight payload streamed per second,
// to compare against the card's VRAM read rate.
func BenchmarkQuantMatVecK4F32(b *testing.B) {
	if count, err := DeviceCount(); err != nil || count < 1 {
		b.Skip("no cuda device available")
	}
	stream, err := NewStream()
	if err != nil {
		b.Fatal(err)
	}
	defer func() { _ = stream.Destroy() }()
	for _, sh := range []struct {
		name       string
		rows, cols int
	}{
		{"attn_kv_512x2048", 512, 2048},
		{"attn_qo_2048x2048", 2048, 2048},
		{"conv_in_6144x2048", 6144, 2048},
		{"ffn_up_7680x2048", 7680, 2048},
		{"ffn_down_2048x7680", 2048, 7680},
	} {
		b.Run(sh.name, func(b *testing.B) {
			m := newK4TestMatrix(b, sh.rows, sh.cols, 1)
			defer m.free()
			xDev, err := AllocDevice(int64(sh.cols * 4))
			if err != nil {
				b.Fatal(err)
			}
			defer func() { _ = xDev.Free() }()
			yDev, err := AllocDevice(int64(sh.rows * 4))
			if err != nil {
				b.Fatal(err)
			}
			defer func() { _ = yDev.Free() }()
			x := make([]float32, sh.cols)
			for i := range x {
				x[i] = float32(i%13) * 0.1
			}
			if err := MemcpyH2D(xDev, unsafe.Pointer(&x[0]), int64(len(x)*4)); err != nil {
				b.Fatal(err)
			}
			const launches = 50
			for b.Loop() {
				for range launches {
					if err := QuantMatVecK4F32(m.qDev, m.supDev, m.subDev, xDev, yDev, sh.rows, m.bpr, sh.cols, stream); err != nil {
						b.Fatal(err)
					}
				}
				if err := stream.Synchronize(); err != nil {
					b.Fatal(err)
				}
			}
			bytes := float64(len(m.q) + len(m.sub) + len(m.super))
			sec := b.Elapsed().Seconds() / float64(b.N*launches)
			b.ReportMetric(sec*1e6, "us/launch")
			b.ReportMetric(bytes/sec/1e9, "wGB/s")
		})
	}
}
