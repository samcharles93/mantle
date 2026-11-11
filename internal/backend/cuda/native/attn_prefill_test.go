//go:build cuda

package native

import (
	"fmt"
	"math"
	"testing"
	"unsafe"
)

func uploadF32(tb testing.TB, host []float32) DeviceBuffer {
	tb.Helper()
	buf, err := AllocDevice(int64(len(host)) * 4)
	if err != nil {
		tb.Fatal(err)
	}
	tb.Cleanup(func() { buf.Free() })
	if err := MemcpyH2D(buf, unsafe.Pointer(&host[0]), int64(len(host))*4); err != nil {
		tb.Fatal(err)
	}
	return buf
}

func allocZeroed(tb testing.TB, bytes int) DeviceBuffer {
	tb.Helper()
	host := make([]byte, bytes)
	buf, err := AllocDevice(int64(bytes))
	if err != nil {
		tb.Fatal(err)
	}
	tb.Cleanup(func() { buf.Free() })
	if err := MemcpyH2D(buf, unsafe.Pointer(&host[0]), int64(bytes)); err != nil {
		tb.Fatal(err)
	}
	return buf
}

func downloadF32(tb testing.TB, buf DeviceBuffer, n int) []float32 {
	tb.Helper()
	out := make([]float32, n)
	if err := MemcpyD2H(unsafe.Pointer(&out[0]), buf, int64(n)*4); err != nil {
		tb.Fatal(err)
	}
	return out
}

func downloadU16(tb testing.TB, buf DeviceBuffer, n int) []uint16 {
	tb.Helper()
	out := make([]uint16, n)
	if err := MemcpyD2H(unsafe.Pointer(&out[0]), buf, int64(n)*2); err != nil {
		tb.Fatal(err)
	}
	return out
}

func patternF32(n, salt int, amp float32) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32((i*31+salt*17)%97-48) / 48 * amp
	}
	return out
}

func sub(buf DeviceBuffer, byteOff int) DeviceBuffer {
	return DeviceBuffer{ptr: unsafe.Add(buf.ptr, byteOff)}
}

// TestRoPERowsAndStoreRowsMatchPerRow requires the batched RoPE and KV store
// kernels to reproduce the per-row kernels bit for bit.
func TestRoPERowsAndStoreRowsMatchPerRow(t *testing.T) {
	const nHeads, headDim, half, rows, startPos, kvStride, moduloLen = 4, 64, 32, 9, 5, 4 * 64, 6
	stream, err := NewStream()
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Destroy()
	invFreq := make([]float32, half)
	for i := range invFreq {
		invFreq[i] = float32(math.Pow(10000, -2*float64(i)/headDim))
	}
	freq := uploadF32(t, invFreq)
	x := patternF32(rows*nHeads*headDim, 1, 1.5)
	batched, perRow := uploadF32(t, x), uploadF32(t, x)
	if err := ApplyRoPERowsF32(batched, freq, startPos, 0.9, headDim, half, nHeads, rows, stream); err != nil {
		t.Fatal(err)
	}
	rowBytes := nHeads * headDim * 4
	for r := range rows {
		if err := ApplyRoPEInplaceF32(sub(perRow, r*rowBytes), freq, startPos+r, 0.9, headDim, half, nHeads, stream); err != nil {
			t.Fatal(err)
		}
	}
	got, want := downloadF32(t, batched, len(x)), downloadF32(t, perRow, len(x))
	for i := range got {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("rope element %d: batched %v, per-row %v", i, got[i], want[i])
		}
	}

	k, v := uploadF32(t, patternF32(rows*kvStride, 2, 3)), uploadF32(t, patternF32(rows*kvStride, 3, 3))
	cacheBytes := moduloLen * kvStride * 2
	bk, bv := allocZeroed(t, cacheBytes), allocZeroed(t, cacheBytes)
	pk, pv := allocZeroed(t, cacheBytes), allocZeroed(t, cacheBytes)
	row0 := rows - moduloLen
	if err := StoreKVF16Rows(bk, bv, k, v, startPos, row0, rows, kvStride, moduloLen, stream); err != nil {
		t.Fatal(err)
	}
	for r := range rows {
		slot := (startPos + r) % moduloLen
		if err := StoreKVF16Row(pk, sub(k, r*kvStride*4), slot, kvStride, stream); err != nil {
			t.Fatal(err)
		}
		if err := StoreKVF16Row(pv, sub(v, r*kvStride*4), slot, kvStride, stream); err != nil {
			t.Fatal(err)
		}
	}
	for name, pair := range map[string][2]DeviceBuffer{"K": {bk, pk}, "V": {bv, pv}} {
		g, w := downloadU16(t, pair[0], moduloLen*kvStride), downloadU16(t, pair[1], moduloLen*kvStride)
		for i := range g {
			if g[i] != w[i] {
				t.Fatalf("%s cache element %d: batched %#04x, per-row %#04x", name, i, g[i], w[i])
			}
		}
	}
}

// TestAttentionPrefillMatchesPerRow compares the prefill kernel with the
// sequential oracle: store each row into the cache, then run the single-query
// kernel at that position. Summation order differs, so values agree to f32
// rounding rather than bit for bit.
func TestAttentionPrefillMatchesPerRow(t *testing.T) {
	cases := []struct {
		headDim, nHead, kvHeads, n, startPos, cacheLen, window int
		softcap                                                float32
	}{
		{128, 16, 4, 70, 0, 128, 0, 0},  // LFM3 full attention, fresh prompt
		{128, 16, 4, 45, 30, 128, 0, 0}, // second chunk over cached positions
		{64, 8, 8, 100, 20, 40, 40, 0},  // sliding window on a ring shorter than the chunk
		{64, 8, 2, 33, 7, 64, 16, 30},   // window inside a larger cache, softcap
		{256, 4, 1, 12, 3, 32, 0, 0},
	}
	for _, c := range cases {
		t.Run(fmt.Sprintf("hd%d_h%d_kv%d_n%d_start%d_cache%d_w%d", c.headDim, c.nHead, c.kvHeads, c.n, c.startPos, c.cacheLen, c.window), func(t *testing.T) {
			if !AttentionPrefillF16Supported(c.headDim, c.nHead, c.kvHeads) {
				t.Skip("layout not supported on this device")
			}
			stream, err := NewStream()
			if err != nil {
				t.Fatal(err)
			}
			defer stream.Destroy()
			kvStride := c.kvHeads * c.headDim
			qDim := c.nHead * c.headDim
			scale := float32(1 / math.Sqrt(float64(c.headDim)))

			// Positions before startPos: written by the per-row store, which
			// both paths then read from the cache.
			total := c.startPos + c.n
			allK := patternF32(total*kvStride, 4, 2)
			allV := patternF32(total*kvStride, 5, 2)
			cacheBytes := c.cacheLen * kvStride * 2
			ck, cv := allocZeroed(t, cacheBytes), allocZeroed(t, cacheBytes)
			kAll, vAll := uploadF32(t, allK), uploadF32(t, allV)
			for p := range c.startPos {
				slot := p % c.cacheLen
				if err := StoreKVF16Row(ck, sub(kAll, p*kvStride*4), slot, kvStride, stream); err != nil {
					t.Fatal(err)
				}
				if err := StoreKVF16Row(cv, sub(vAll, p*kvStride*4), slot, kvStride, stream); err != nil {
					t.Fatal(err)
				}
			}
			oracleK, oracleV := allocZeroed(t, cacheBytes), allocZeroed(t, cacheBytes)
			if err := MemcpyD2DAsync(oracleK, ck, int64(cacheBytes), stream); err != nil {
				t.Fatal(err)
			}
			if err := MemcpyD2DAsync(oracleV, cv, int64(cacheBytes), stream); err != nil {
				t.Fatal(err)
			}

			q := uploadF32(t, patternF32(c.n*qDim, 6, 1))
			kNew := sub(kAll, c.startPos*kvStride*4)
			vNew := sub(vAll, c.startPos*kvStride*4)
			out := allocZeroed(t, c.n*qDim*4)
			if err := AttentionPrefillF16(q, kNew, vNew, ck, cv, out, c.n, c.startPos, kvStride, c.headDim, c.nHead, c.kvHeads, c.cacheLen, c.window, scale, c.softcap, stream); err != nil {
				t.Fatal(err)
			}
			got := downloadF32(t, out, c.n*qDim)

			want := make([]float32, c.n*qDim)
			rowOut := allocZeroed(t, qDim*4)
			for r := range c.n {
				pos := c.startPos + r
				slot := pos % c.cacheLen
				if err := StoreKVF16Row(oracleK, sub(kNew, r*kvStride*4), slot, kvStride, stream); err != nil {
					t.Fatal(err)
				}
				if err := StoreKVF16Row(oracleV, sub(vNew, r*kvStride*4), slot, kvStride, stream); err != nil {
					t.Fatal(err)
				}
				start := 0
				if c.window > 0 {
					start = max(pos-c.window+1, 0)
				}
				if err := AttentionInnerF16CacheF32(sub(q, r*qDim*4), oracleK, oracleV, rowOut, pos, start, kvStride, c.headDim, c.nHead, c.kvHeads, c.cacheLen, scale, c.softcap, stream); err != nil {
					t.Fatal(err)
				}
				copy(want[r*qDim:], downloadF32(t, rowOut, qDim))
			}
			worst := 0.0
			for i := range got {
				d := math.Abs(float64(got[i] - want[i]))
				worst = max(worst, d)
				if d > 2e-5+2e-5*math.Abs(float64(want[i])) {
					t.Fatalf("row %d dim %d: prefill %v, oracle %v", i/qDim, i%qDim, got[i], want[i])
				}
			}
			t.Logf("worst abs diff %g", worst)
		})
	}
}

// BenchmarkAttentionPrefill times one LFM3 full-attention layer over an
// 812-token prompt.
func BenchmarkAttentionPrefill(b *testing.B) {
	const headDim, nHead, kvHeads, n, cacheLen = 128, 16, 4, 812, 1024
	if !AttentionPrefillF16Supported(headDim, nHead, kvHeads) {
		b.Skip("not supported")
	}
	stream, err := NewStream()
	if err != nil {
		b.Fatal(err)
	}
	defer stream.Destroy()
	kvStride := kvHeads * headDim
	q := uploadF32(b, patternF32(n*nHead*headDim, 1, 1))
	k := uploadF32(b, patternF32(n*kvStride, 2, 1))
	v := uploadF32(b, patternF32(n*kvStride, 3, 1))
	ck, cv := allocZeroed(b, cacheLen*kvStride*2), allocZeroed(b, cacheLen*kvStride*2)
	out := allocZeroed(b, n*nHead*headDim*4)
	for b.Loop() {
		if err := AttentionPrefillF16(q, k, v, ck, cv, out, n, 0, kvStride, headDim, nHead, kvHeads, cacheLen, 0, 0.088, 0, stream); err != nil {
			b.Fatal(err)
		}
		if err := stream.Synchronize(); err != nil {
			b.Fatal(err)
		}
	}
}
