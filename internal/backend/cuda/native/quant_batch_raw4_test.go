//go:build cuda

package native

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"
	"unsafe"
)

type raw4BatchFixture struct {
	rows, cols, n, bpr int
	stream             Stream

	// Cached-int8 oracle buffers
	cachedQ      DeviceBuffer
	cachedScales DeviceBuffer

	// Raw K4 buffers
	k4Q     DeviceBuffer
	k4Super DeviceBuffer
	k4Sub   DeviceBuffer

	// Raw Q4 buffers
	q4Q      DeviceBuffer
	q4Scales DeviceBuffer

	// Activations & quant scratch
	x, xq, xsc, xq2, xsc2 DeviceBuffer
}

func newRaw4BatchFixture(tb testing.TB, rows, cols, n int, seed uint64) *raw4BatchFixture {
	tb.Helper()
	stream, err := NewStream()
	if err != nil {
		tb.Fatal(err)
	}
	bpr := (cols + 31) / 32
	spr := (bpr + 7) / 8
	rng := rand.New(rand.NewPCG(seed, 7))

	rawQ := make([]byte, rows*bpr*16)
	k4Sub := make([]byte, rows*bpr)
	k4Supers := make([]uint16, rows*spr)
	q4ScalesF16 := make([]uint16, rows*bpr)

	cachedQ := make([]int8, rows*bpr*32)
	cachedK4Scales := make([]float32, rows*bpr)
	cachedQ4Scales := make([]float32, rows*bpr)

	for i := range k4Supers {
		k4Supers[i] = f32ToF16BitsRef(float32(0.002 + 0.01*rng.Float64()))
		if i/spr == 1 && i%5 == 0 {
			k4Supers[i] = 0
		}
	}

	var blk [32]int8
	for r := range rows {
		for b := range bpr {
			idx := r*bpr + b
			for i := range blk {
				if idx == 0 {
					// Cover all 16 nibble values [-8..+7] twice in block 0.
					blk[i] = int8((i % 16) - 8)
				} else {
					blk[i] = int8(rng.IntN(16) - 8)
				}
			}
			packQ4Block(rawQ[idx*16:(idx+1)*16], blk[:])

			u6 := byte(rng.IntN(64)) | byte(rng.IntN(4))<<6 // top 2 bits must be masked off
			if idx%17 == 0 {
				u6 &^= 0x3F
			}
			k4Sub[idx] = u6

			sup := fp16ToF32(k4Supers[r*spr+b/8])
			if sup != 0 && (u6&0x3F) != 0 {
				cachedK4Scales[idx] = sup * (float32(u6&0x3F) / 32.0)
			}

			s16 := f32ToF16BitsRef(float32(0.002 + 0.02*rng.Float64()))
			if idx%19 == 0 {
				s16 = 0
			}
			q4ScalesF16[idx] = s16
			cachedQ4Scales[idx] = fp16ToF32(s16)

			copy(cachedQ[idx*32:(idx+1)*32], blk[:])
		}
	}

	// Build two separate cached-int8 Q buffers where zero-scale blocks are
	// zeroed (matching core.BuildQuantCache) so we test bit-exact parity.
	cachedK4Q := make([]int8, len(cachedQ))
	copy(cachedK4Q, cachedQ)
	for idx, sc := range cachedK4Scales {
		if sc == 0 {
			clear(cachedK4Q[idx*32 : (idx+1)*32])
		}
	}

	k4SuperBytes := make([]byte, len(k4Supers)*2)
	for i, v := range k4Supers {
		k4SuperBytes[2*i], k4SuperBytes[2*i+1] = byte(v), byte(v>>8)
	}
	q4ScaleBytes := make([]byte, len(q4ScalesF16)*2)
	for i, v := range q4ScalesF16 {
		q4ScaleBytes[2*i], q4ScaleBytes[2*i+1] = byte(v), byte(v>>8)
	}

	xh := make([]float32, n*cols)
	for i := range xh {
		xh[i] = float32(rng.NormFloat64() * 0.5)
	}

	f := &raw4BatchFixture{rows: rows, cols: cols, n: n, bpr: bpr, stream: stream}
	tb.Cleanup(func() {
		_ = f.cachedQ.Free()
		_ = f.cachedScales.Free()
		_ = f.k4Q.Free()
		_ = f.k4Super.Free()
		_ = f.k4Sub.Free()
		_ = f.q4Q.Free()
		_ = f.q4Scales.Free()
		_ = f.x.Free()
		_ = f.xq.Free()
		_ = f.xsc.Free()
		_ = f.xq2.Free()
		_ = f.xsc2.Free()
		_ = f.stream.Destroy()
	})

	uploadBytes := func(src []byte) DeviceBuffer {
		d, err := AllocDevice(int64(len(src)))
		if err != nil {
			tb.Fatal(err)
		}
		if err := MemcpyH2D(d, unsafe.Pointer(&src[0]), int64(len(src))); err != nil {
			tb.Fatal(err)
		}
		return d
	}
	uploadF32 := func(src []float32) DeviceBuffer {
		d, err := AllocDevice(int64(len(src)) * 4)
		if err != nil {
			tb.Fatal(err)
		}
		if err := MemcpyH2D(d, unsafe.Pointer(&src[0]), int64(len(src))*4); err != nil {
			tb.Fatal(err)
		}
		return d
	}
	uploadI8 := func(src []int8) DeviceBuffer {
		d, err := AllocDevice(int64(len(src)))
		if err != nil {
			tb.Fatal(err)
		}
		if err := MemcpyH2D(d, unsafe.Pointer(&src[0]), int64(len(src))); err != nil {
			tb.Fatal(err)
		}
		return d
	}

	f.cachedQ = uploadI8(cachedK4Q)
	f.cachedScales = uploadF32(cachedK4Scales)
	f.k4Q = uploadBytes(rawQ)
	f.k4Super = uploadBytes(k4SuperBytes)
	f.k4Sub = uploadBytes(k4Sub)

	f.q4Q = uploadBytes(rawQ)
	f.q4Scales = uploadBytes(q4ScaleBytes)

	f.x = uploadF32(xh)
	var allocErr error
	if f.xq, allocErr = AllocDevice(int64(n * bpr * 32)); allocErr != nil {
		tb.Fatal(allocErr)
	}
	if f.xsc, allocErr = AllocDevice(int64(n*bpr) * 4); allocErr != nil {
		tb.Fatal(allocErr)
	}
	if f.xq2, allocErr = AllocDevice(int64(n * bpr * 32)); allocErr != nil {
		tb.Fatal(allocErr)
	}
	if f.xsc2, allocErr = AllocDevice(int64(n*bpr) * 4); allocErr != nil {
		tb.Fatal(allocErr)
	}

	// Keep Q4 cached buffers handy by swapping when testing Q4.
	_ = cachedQ4Scales
	return f
}

func (f *raw4BatchFixture) readOutput(tb testing.TB, launch func(y DeviceBuffer) error) []float32 {
	tb.Helper()
	y, err := AllocDevice(int64(f.n*f.rows) * 4)
	if err != nil {
		tb.Fatal(err)
	}
	defer func() { _ = y.Free() }()
	if err := launch(y); err != nil {
		tb.Fatal(err)
	}
	out := make([]float32, f.n*f.rows)
	if err := MemcpyD2H(unsafe.Pointer(&out[0]), y, int64(len(out))*4); err != nil {
		tb.Fatal(err)
	}
	return out
}

func assertBitExactF32(t *testing.T, name string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: len %d != %d", name, len(got), len(want))
	}
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("%s[%d]: got %v (0x%08x), want %v (0x%08x)",
				name, i, got[i], math.Float32bits(got[i]), want[i], math.Float32bits(want[i]))
		}
	}
}

func TestQuantBatchK4MatchesCachedBitExact(t *testing.T) {
	shapes := []struct {
		rows, cols, n int
		mma           bool
	}{
		{64, 128, 64, true},
		{70, 96, 17, true},
		{130, 512, 37, true},
		{35, 100, 19, false}, // non-multiple-of-32 cols tail
	}
	for idx, s := range shapes {
		t.Run(fmt.Sprintf("%dx%dx%d", s.rows, s.cols, s.n), func(t *testing.T) {
			f := newRaw4BatchFixture(t, s.rows, s.cols, s.n, uint64(100+idx))

			wantF32 := f.readOutput(t, func(y DeviceBuffer) error {
				return QuantMatVecCachedBatch(f.cachedQ, f.cachedScales, f.x, y, f.rows, f.n, f.cols, f.bpr, f.stream)
			})
			gotF32 := f.readOutput(t, func(y DeviceBuffer) error {
				return QuantMatVecK4Batch(f.k4Q, f.k4Super, f.k4Sub, f.x, y, f.rows, f.n, f.cols, f.bpr, f.stream)
			})
			assertBitExactF32(t, "k4_f32", gotF32, wantF32)

			wantDp4a := f.readOutput(t, func(y DeviceBuffer) error {
				return QuantMatVecCachedBatchDp4a(f.cachedQ, f.cachedScales, f.x, f.xq, f.xsc, y, f.rows, f.n, f.cols, f.bpr, f.stream)
			})
			gotDp4a := f.readOutput(t, func(y DeviceBuffer) error {
				return QuantMatVecK4BatchDp4a(f.k4Q, f.k4Super, f.k4Sub, f.x, f.xq, f.xsc, y, f.rows, f.n, f.cols, f.bpr, f.stream)
			})
			assertBitExactF32(t, "k4_dp4a", gotDp4a, wantDp4a)

			if s.mma && QuantBatchMMASupported() {
				wantMMA := f.readOutput(t, func(y DeviceBuffer) error {
					return QuantMatVecCachedBatchMMA(f.cachedQ, f.cachedScales, f.x, f.xq, f.xsc, y, f.rows, f.n, f.cols, f.bpr, f.stream)
				})
				gotMMA := f.readOutput(t, func(y DeviceBuffer) error {
					return QuantMatVecK4BatchMMA(f.k4Q, f.k4Super, f.k4Sub, f.x, f.xq, f.xsc, y, f.rows, f.n, f.cols, f.bpr, f.stream)
				})
				assertBitExactF32(t, "k4_mma", gotMMA, wantMMA)

				wantHilo := f.readOutput(t, func(y DeviceBuffer) error {
					return QuantMatVecCachedBatchMMAHilo(f.cachedQ, f.cachedScales, f.x, f.xq, f.xsc, f.xq2, f.xsc2, y, f.rows, f.n, f.cols, f.bpr, f.stream)
				})
				gotHilo := f.readOutput(t, func(y DeviceBuffer) error {
					return QuantMatVecK4BatchMMAHilo(f.k4Q, f.k4Super, f.k4Sub, f.x, f.xq, f.xsc, f.xq2, f.xsc2, y, f.rows, f.n, f.cols, f.bpr, f.stream)
				})
				assertBitExactF32(t, "k4_mma_hilo", gotHilo, wantHilo)
			}
		})
	}
}

func TestQuantBatchQ4MatchesCachedBitExact(t *testing.T) {
	shapes := []struct {
		rows, cols, n int
		mma           bool
	}{
		{64, 128, 64, true},
		{70, 96, 17, true},
		{130, 512, 37, true},
		{35, 100, 19, false},
	}
	for idx, s := range shapes {
		t.Run(fmt.Sprintf("%dx%dx%d", s.rows, s.cols, s.n), func(t *testing.T) {
			stream, err := NewStream()
			if err != nil {
				t.Fatal(err)
			}
			defer func() { _ = stream.Destroy() }()

			bpr := (s.cols + 31) / 32
			rng := rand.New(rand.NewPCG(uint64(200+idx), 11))
			rawQ := make([]byte, s.rows*bpr*16)
			scalesF16 := make([]byte, s.rows*bpr*2)
			cachedQ := make([]int8, s.rows*bpr*32)
			cachedScales := make([]float32, s.rows*bpr)

			var blk [32]int8
			for r := range s.rows {
				for b := range bpr {
					bi := r*bpr + b
					for i := range blk {
						if bi == 0 {
							blk[i] = int8((i % 16) - 8)
						} else {
							blk[i] = int8(rng.IntN(16) - 8)
						}
					}
					packQ4Block(rawQ[bi*16:(bi+1)*16], blk[:])
					s16 := f32ToF16BitsRef(float32(0.002 + 0.02*rng.Float64()))
					if bi%19 == 0 {
						s16 = 0
					}
					scalesF16[2*bi], scalesF16[2*bi+1] = byte(s16), byte(s16>>8)
					sc := fp16ToF32(s16)
					cachedScales[bi] = sc
					if sc != 0 {
						copy(cachedQ[bi*32:(bi+1)*32], blk[:])
					}
				}
			}

			xh := make([]float32, s.n*s.cols)
			for i := range xh {
				xh[i] = float32(rng.NormFloat64() * 0.5)
			}

			allocUpload := func(ptr unsafe.Pointer, bytes int64) DeviceBuffer {
				d, err := AllocDevice(bytes)
				if err != nil {
					t.Fatal(err)
				}
				t.Cleanup(func() { _ = d.Free() })
				if err := MemcpyH2D(d, ptr, bytes); err != nil {
					t.Fatal(err)
				}
				return d
			}
			allocZero := func(bytes int64) DeviceBuffer {
				d, err := AllocDevice(bytes)
				if err != nil {
					t.Fatal(err)
				}
				t.Cleanup(func() { _ = d.Free() })
				return d
			}

			f := &raw4BatchFixture{
				rows: s.rows, cols: s.cols, n: s.n, bpr: bpr, stream: stream,
				cachedQ:      allocUpload(unsafe.Pointer(&cachedQ[0]), int64(len(cachedQ))),
				cachedScales: allocUpload(unsafe.Pointer(&cachedScales[0]), int64(len(cachedScales))*4),
				q4Q:          allocUpload(unsafe.Pointer(&rawQ[0]), int64(len(rawQ))),
				q4Scales:     allocUpload(unsafe.Pointer(&scalesF16[0]), int64(len(scalesF16))),
				x:            allocUpload(unsafe.Pointer(&xh[0]), int64(len(xh))*4),
				xq:           allocZero(int64(s.n * bpr * 32)),
				xsc:          allocZero(int64(s.n*bpr) * 4),
				xq2:          allocZero(int64(s.n * bpr * 32)),
				xsc2:         allocZero(int64(s.n*bpr) * 4),
			}

			wantF32 := f.readOutput(t, func(y DeviceBuffer) error {
				return QuantMatVecCachedBatch(f.cachedQ, f.cachedScales, f.x, y, f.rows, f.n, f.cols, f.bpr, f.stream)
			})
			gotF32 := f.readOutput(t, func(y DeviceBuffer) error {
				return QuantMatVecQ4Batch(f.q4Q, f.q4Scales, f.x, y, f.rows, f.n, f.cols, f.bpr, f.stream)
			})
			assertBitExactF32(t, "q4_f32", gotF32, wantF32)

			wantDp4a := f.readOutput(t, func(y DeviceBuffer) error {
				return QuantMatVecCachedBatchDp4a(f.cachedQ, f.cachedScales, f.x, f.xq, f.xsc, y, f.rows, f.n, f.cols, f.bpr, f.stream)
			})
			gotDp4a := f.readOutput(t, func(y DeviceBuffer) error {
				return QuantMatVecQ4BatchDp4a(f.q4Q, f.q4Scales, f.x, f.xq, f.xsc, y, f.rows, f.n, f.cols, f.bpr, f.stream)
			})
			assertBitExactF32(t, "q4_dp4a", gotDp4a, wantDp4a)

			if s.mma && QuantBatchMMASupported() {
				wantMMA := f.readOutput(t, func(y DeviceBuffer) error {
					return QuantMatVecCachedBatchMMA(f.cachedQ, f.cachedScales, f.x, f.xq, f.xsc, y, f.rows, f.n, f.cols, f.bpr, f.stream)
				})
				gotMMA := f.readOutput(t, func(y DeviceBuffer) error {
					return QuantMatVecQ4BatchMMA(f.q4Q, f.q4Scales, f.x, f.xq, f.xsc, y, f.rows, f.n, f.cols, f.bpr, f.stream)
				})
				assertBitExactF32(t, "q4_mma", gotMMA, wantMMA)

				wantHilo := f.readOutput(t, func(y DeviceBuffer) error {
					return QuantMatVecCachedBatchMMAHilo(f.cachedQ, f.cachedScales, f.x, f.xq, f.xsc, f.xq2, f.xsc2, y, f.rows, f.n, f.cols, f.bpr, f.stream)
				})
				gotHilo := f.readOutput(t, func(y DeviceBuffer) error {
					return QuantMatVecQ4BatchMMAHilo(f.q4Q, f.q4Scales, f.x, f.xq, f.xsc, f.xq2, f.xsc2, y, f.rows, f.n, f.cols, f.bpr, f.stream)
				})
				assertBitExactF32(t, "q4_mma_hilo", gotHilo, wantHilo)
			}
		})
	}
}

func BenchmarkQuantMatVecK4BatchMMA(b *testing.B) {
	benchmarkRaw4BatchMMA(b, true, false)
}

func BenchmarkQuantMatVecK4BatchMMAHilo(b *testing.B) {
	benchmarkRaw4BatchMMA(b, true, true)
}

func BenchmarkQuantMatVecQ4BatchMMAHilo(b *testing.B) {
	benchmarkRaw4BatchMMA(b, false, true)
}

func benchmarkRaw4BatchMMA(b *testing.B, isK4, hilo bool) {
	if !QuantBatchMMASupported() {
		b.Skip("tensor-core quant GEMM needs compute capability 8.0+")
	}
	const rows, cols, n = 2048, 2048, 610
	f := newRaw4BatchFixture(b, rows, cols, n, 42)
	y, err := AllocDevice(int64(n*rows) * 4)
	if err != nil {
		b.Fatal(err)
	}
	defer func() { _ = y.Free() }()
	run := func() error {
		switch {
		case isK4 && hilo:
			return QuantMatVecK4BatchMMAHilo(f.k4Q, f.k4Super, f.k4Sub, f.x, f.xq, f.xsc, f.xq2, f.xsc2, y, rows, n, cols, f.bpr, f.stream)
		case isK4:
			return QuantMatVecK4BatchMMA(f.k4Q, f.k4Super, f.k4Sub, f.x, f.xq, f.xsc, y, rows, n, cols, f.bpr, f.stream)
		default:
			return QuantMatVecQ4BatchMMAHilo(f.q4Q, f.q4Scales, f.x, f.xq, f.xsc, f.xq2, f.xsc2, y, rows, n, cols, f.bpr, f.stream)
		}
	}
	if err := run(); err != nil {
		b.Fatal(err)
	}
	if err := f.stream.Synchronize(); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for b.Loop() {
		if err := run(); err != nil {
			b.Fatal(err)
		}
		if err := f.stream.Synchronize(); err != nil {
			b.Fatal(err)
		}
	}
	macs := float64(rows) * cols * n
	b.ReportMetric(macs/(float64(b.Elapsed().Nanoseconds())/float64(b.N))/1e3, "TMAC/s")
}
