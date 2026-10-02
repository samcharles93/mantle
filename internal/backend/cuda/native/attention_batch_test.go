//go:build cuda

package native

import (
	"math"
	"testing"
	"unsafe"
)

// Shared geometry for the batched-attention equivalence tests. headDim=32 and
// kvStride=64 are divisible by 32 so the mixed kernel's per-32-element Q8 scale
// blocks are exercised, and kvHeads=2 < nHead=4 exercises GQA head mapping.
const (
	batchTestHeadDim  = 32
	batchTestNHead    = 4
	batchTestKVHeads  = 2
	batchTestKVStride = batchTestKVHeads * batchTestHeadDim
	batchTestCacheLen = 4
	batchTestScale    = float32(0.125)
)

// batchTestQValue is a deterministic F32 query value with enough sign and
// magnitude variety to move the softmax weights around.
func batchTestQValue(seed int) float32 {
	return float32(math.Sin(float64(seed)*0.7+0.3) * 0.8)
}

// exactF16Bits returns the F16 bit pattern for v. Callers pass small multiples
// of 1/8 (see batchTestF16Value) so the value is exactly representable and no
// rounding is involved.
func exactF16Bits(v float32) uint16 {
	b := math.Float32bits(v)
	sign := uint16(b>>31) << 15
	if v == 0 {
		return sign
	}
	exp := int32((b>>23)&0xff) - 127
	frac := b & 0x7fffff
	return sign | uint16(exp+15)<<10 | uint16(frac>>13)
}

// batchTestF16Value produces exact F16 cache values spanning positive and
// negative multiples of 0.125.
func batchTestF16Value(seed int) uint16 {
	m := (seed%33 - 16) // -16..16
	return exactF16Bits(float32(m) * 0.125)
}

func fillBatchF16Cache(cache []uint16, kvStride, cacheLen, salt int) {
	for p := range cacheLen {
		for d := range kvStride {
			cache[p*kvStride+d] = batchTestF16Value(p*kvStride + d*3 + salt)
		}
	}
}

func fillBatchQ8Cache(q []int8, scales []float32, kvStride, cacheLen, scaleStride, salt int) {
	for p := range cacheLen {
		for d := range kvStride {
			q[p*kvStride+d] = int8((p*97+d*13+salt)%255 - 127)
		}
		for b := range scaleStride {
			// Index p*scaleStride+b is exactly the per-(cachePos, kvHead)
			// scale slot the kernel reads (headDim/32 == 1).
			scales[p*scaleStride+b] = 0.01 + 0.002*float32(p) + 0.0007*float32(salt%7) + 0.0002*float32(b)
		}
	}
}

func allocPinnedU16(t *testing.T, n int) (HostBuffer, []uint16) {
	t.Helper()
	buf, err := AllocHostPinned(int64(n) * int64(unsafe.Sizeof(uint16(0))))
	if err != nil {
		t.Fatalf("AllocHostPinned u16: %v", err)
	}
	return buf, unsafe.Slice((*uint16)(buf.Ptr()), n)
}

func allocPinnedI8(t *testing.T, n int) (HostBuffer, []int8) {
	t.Helper()
	buf, err := AllocHostPinned(int64(n))
	if err != nil {
		t.Fatalf("AllocHostPinned i8: %v", err)
	}
	return buf, unsafe.Slice((*int8)(buf.Ptr()), n)
}

// assertBitwiseEqual requires the batched output to match the per-row
// single-query reference bit for bit. The batch kernel runs the identical
// per-block recurrence as the single-query kernel, so any difference is a bug
// rather than floating-point reordering.
func assertBitwiseEqual(t *testing.T, label string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length mismatch %d vs %d", label, len(got), len(want))
	}
	for i := range got {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("%s: element %d differs: batch=%v single=%v (bits %08x vs %08x)",
				label, i, got[i], want[i], math.Float32bits(got[i]), math.Float32bits(want[i]))
		}
	}
}

func TestAttentionInnerF16CacheF32BatchMatchesSingleQuery(t *testing.T) {
	count, err := DeviceCount()
	if err != nil || count < 1 {
		t.Skip("no cuda device available")
	}
	stream, err := NewStream()
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	defer func() { _ = stream.Destroy() }()

	cases := []struct {
		name     string
		nTokens  int
		startPos int
		softcap  float32
	}{
		{"nTokens1_beforeRing", 1, 2, 0},
		{"nTokens3_gqa_ringBoundary", 3, 3, 0},
		{"nTokens_cacheLenPlus1_wrap", batchTestCacheLen + 1, 3, 0},
		{"nTokens3_softcap", 3, 3, 8.0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			runF16BatchCase(t, stream, tc.nTokens, tc.startPos, tc.softcap)
		})
	}
}

func runF16BatchCase(t *testing.T, stream Stream, nTokens, startPos int, softcap float32) {
	t.Helper()

	qCount := nTokens * batchTestNHead * batchTestHeadDim
	rowBytes := int64(batchTestNHead * batchTestHeadDim * 4)
	outBytes := int64(qCount) * 4
	cacheElems := batchTestKVStride * batchTestCacheLen
	cacheBytes := int64(cacheElems) * 2

	qHost, qSlice := allocPinnedF32(t, qCount)
	t.Cleanup(func() { _ = qHost.Free() })
	for i := range qSlice {
		qSlice[i] = batchTestQValue(i)
	}

	kHost, kSlice := allocPinnedU16(t, cacheElems)
	t.Cleanup(func() { _ = kHost.Free() })
	vHost, vSlice := allocPinnedU16(t, cacheElems)
	t.Cleanup(func() { _ = vHost.Free() })
	fillBatchF16Cache(kSlice, batchTestKVStride, batchTestCacheLen, 0)
	fillBatchF16Cache(vSlice, batchTestKVStride, batchTestCacheLen, 101)

	batchHost, batchSlice := allocPinnedF32(t, qCount)
	t.Cleanup(func() { _ = batchHost.Free() })
	refHost, refSlice := allocPinnedF32(t, qCount)
	t.Cleanup(func() { _ = refHost.Free() })

	qDev, err := AllocDevice(outBytes)
	if err != nil {
		t.Fatalf("AllocDevice q: %v", err)
	}
	t.Cleanup(func() { _ = qDev.Free() })
	kDev, err := AllocDevice(cacheBytes)
	if err != nil {
		t.Fatalf("AllocDevice k: %v", err)
	}
	t.Cleanup(func() { _ = kDev.Free() })
	vDev, err := AllocDevice(cacheBytes)
	if err != nil {
		t.Fatalf("AllocDevice v: %v", err)
	}
	t.Cleanup(func() { _ = vDev.Free() })
	outDev, err := AllocDevice(outBytes)
	if err != nil {
		t.Fatalf("AllocDevice out: %v", err)
	}
	t.Cleanup(func() { _ = outDev.Free() })
	refDev, err := AllocDevice(outBytes)
	if err != nil {
		t.Fatalf("AllocDevice ref: %v", err)
	}
	t.Cleanup(func() { _ = refDev.Free() })

	if err := MemcpyH2DAsync(qDev, qHost.Ptr(), outBytes, stream); err != nil {
		t.Fatalf("H2D q: %v", err)
	}
	if err := MemcpyH2DAsync(kDev, kHost.Ptr(), cacheBytes, stream); err != nil {
		t.Fatalf("H2D k: %v", err)
	}
	if err := MemcpyH2DAsync(vDev, vHost.Ptr(), cacheBytes, stream); err != nil {
		t.Fatalf("H2D v: %v", err)
	}

	if err := AttentionInnerF16CacheF32Batch(qDev, kDev, vDev, outDev, nTokens, startPos, batchTestKVStride, batchTestHeadDim, batchTestNHead, batchTestKVHeads, batchTestCacheLen, batchTestScale, softcap, stream); err != nil {
		t.Fatalf("AttentionInnerF16CacheF32Batch: %v", err)
	}

	for i := range nTokens {
		qRow := DeviceBufferFromRaw(unsafe.Add(qDev.Ptr(), int(rowBytes)*i))
		outRow := DeviceBufferFromRaw(unsafe.Add(refDev.Ptr(), int(rowBytes)*i))
		if err := AttentionInnerF16CacheF32(qRow, kDev, vDev, outRow, startPos+i, 0, batchTestKVStride, batchTestHeadDim, batchTestNHead, batchTestKVHeads, batchTestCacheLen, batchTestScale, softcap, stream); err != nil {
			t.Fatalf("single-query row %d: %v", i, err)
		}
	}

	if err := MemcpyD2HAsync(batchHost.Ptr(), outDev, outBytes, stream); err != nil {
		t.Fatalf("D2H batch: %v", err)
	}
	if err := MemcpyD2HAsync(refHost.Ptr(), refDev, outBytes, stream); err != nil {
		t.Fatalf("D2H ref: %v", err)
	}
	if err := stream.Synchronize(); err != nil {
		t.Fatalf("sync: %v", err)
	}

	assertBitwiseEqual(t, "f16 batch", batchSlice, refSlice)
}

func TestAttentionInnerMixedCacheF32BatchMatchesSingleQuery(t *testing.T) {
	count, err := DeviceCount()
	if err != nil || count < 1 {
		t.Skip("no cuda device available")
	}
	stream, err := NewStream()
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	defer func() { _ = stream.Destroy() }()

	cases := []struct {
		name           string
		nTokens        int
		startPos       int
		useQ8K, useQ8V bool
		softcap        float32
	}{
		{"f16_cache", 3, 3, false, false, 0},
		{"q8k_only", 3, 3, true, false, 0},
		{"q8v_only", 3, 3, false, true, 0},
		{"q8_kv_softcap", 3, 3, true, true, 8.0},
		{"q8_kv_wrap", batchTestCacheLen + 1, 3, true, true, 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			runMixedBatchCase(t, stream, tc.nTokens, tc.startPos, tc.useQ8K, tc.useQ8V, tc.softcap)
		})
	}
}

func runMixedBatchCase(t *testing.T, stream Stream, nTokens, startPos int, useQ8K, useQ8V bool, softcap float32) {
	t.Helper()

	const scaleStride = batchTestKVStride / 32 // 32-element Q8 scale blocks per cache position

	qCount := nTokens * batchTestNHead * batchTestHeadDim
	rowBytes := int64(batchTestNHead * batchTestHeadDim * 4)
	outBytes := int64(qCount) * 4
	cacheElems := batchTestKVStride * batchTestCacheLen
	cacheBytes := int64(cacheElems) * 2
	scaleCount := batchTestCacheLen * scaleStride
	scaleBytes := int64(scaleCount) * 4

	qHost, qSlice := allocPinnedF32(t, qCount)
	t.Cleanup(func() { _ = qHost.Free() })
	for i := range qSlice {
		qSlice[i] = batchTestQValue(i + 17)
	}

	kF16Host, kF16Slice := allocPinnedU16(t, cacheElems)
	t.Cleanup(func() { _ = kF16Host.Free() })
	vF16Host, vF16Slice := allocPinnedU16(t, cacheElems)
	t.Cleanup(func() { _ = vF16Host.Free() })
	fillBatchF16Cache(kF16Slice, batchTestKVStride, batchTestCacheLen, 0)
	fillBatchF16Cache(vF16Slice, batchTestKVStride, batchTestCacheLen, 101)

	kQ8Host, kQ8Slice := allocPinnedI8(t, cacheElems)
	t.Cleanup(func() { _ = kQ8Host.Free() })
	vQ8Host, vQ8Slice := allocPinnedI8(t, cacheElems)
	t.Cleanup(func() { _ = vQ8Host.Free() })
	kScaleHost, kScaleSlice := allocPinnedF32(t, scaleCount)
	t.Cleanup(func() { _ = kScaleHost.Free() })
	vScaleHost, vScaleSlice := allocPinnedF32(t, scaleCount)
	t.Cleanup(func() { _ = vScaleHost.Free() })
	fillBatchQ8Cache(kQ8Slice, kScaleSlice, batchTestKVStride, batchTestCacheLen, scaleStride, 7)
	fillBatchQ8Cache(vQ8Slice, vScaleSlice, batchTestKVStride, batchTestCacheLen, scaleStride, 53)

	batchHost, batchSlice := allocPinnedF32(t, qCount)
	t.Cleanup(func() { _ = batchHost.Free() })
	refHost, refSlice := allocPinnedF32(t, qCount)
	t.Cleanup(func() { _ = refHost.Free() })

	qDev, err := AllocDevice(outBytes)
	if err != nil {
		t.Fatalf("AllocDevice q: %v", err)
	}
	t.Cleanup(func() { _ = qDev.Free() })
	alloc := func(name string, bytes int64) DeviceBuffer {
		t.Helper()
		buf, err := AllocDevice(bytes)
		if err != nil {
			t.Fatalf("AllocDevice %s: %v", name, err)
		}
		return buf
	}
	kF16Dev := alloc("kF16", cacheBytes)
	t.Cleanup(func() { _ = kF16Dev.Free() })
	vF16Dev := alloc("vF16", cacheBytes)
	t.Cleanup(func() { _ = vF16Dev.Free() })
	kQ8Dev := alloc("kQ8", int64(cacheElems))
	t.Cleanup(func() { _ = kQ8Dev.Free() })
	vQ8Dev := alloc("vQ8", int64(cacheElems))
	t.Cleanup(func() { _ = vQ8Dev.Free() })
	kScaleDev := alloc("kScale", scaleBytes)
	t.Cleanup(func() { _ = kScaleDev.Free() })
	vScaleDev := alloc("vScale", scaleBytes)
	t.Cleanup(func() { _ = vScaleDev.Free() })
	outDev := alloc("out", outBytes)
	t.Cleanup(func() { _ = outDev.Free() })
	refDev := alloc("ref", outBytes)
	t.Cleanup(func() { _ = refDev.Free() })

	if err := MemcpyH2DAsync(qDev, qHost.Ptr(), outBytes, stream); err != nil {
		t.Fatalf("H2D q: %v", err)
	}
	if err := MemcpyH2DAsync(kF16Dev, kF16Host.Ptr(), cacheBytes, stream); err != nil {
		t.Fatalf("H2D kF16: %v", err)
	}
	if err := MemcpyH2DAsync(vF16Dev, vF16Host.Ptr(), cacheBytes, stream); err != nil {
		t.Fatalf("H2D vF16: %v", err)
	}
	if err := MemcpyH2DAsync(kQ8Dev, kQ8Host.Ptr(), int64(cacheElems), stream); err != nil {
		t.Fatalf("H2D kQ8: %v", err)
	}
	if err := MemcpyH2DAsync(vQ8Dev, vQ8Host.Ptr(), int64(cacheElems), stream); err != nil {
		t.Fatalf("H2D vQ8: %v", err)
	}
	if err := MemcpyH2DAsync(kScaleDev, kScaleHost.Ptr(), scaleBytes, stream); err != nil {
		t.Fatalf("H2D kScale: %v", err)
	}
	if err := MemcpyH2DAsync(vScaleDev, vScaleHost.Ptr(), scaleBytes, stream); err != nil {
		t.Fatalf("H2D vScale: %v", err)
	}

	if err := AttentionInnerMixedCacheF32Batch(
		qDev, kF16Dev, vF16Dev, kQ8Dev, vQ8Dev, kScaleDev, vScaleDev, outDev,
		useQ8K, useQ8V, nTokens, startPos, batchTestKVStride, batchTestHeadDim,
		batchTestNHead, batchTestKVHeads, batchTestCacheLen, batchTestScale, softcap, stream,
	); err != nil {
		t.Fatalf("AttentionInnerMixedCacheF32Batch: %v", err)
	}

	for i := range nTokens {
		qRow := DeviceBufferFromRaw(unsafe.Add(qDev.Ptr(), int(rowBytes)*i))
		outRow := DeviceBufferFromRaw(unsafe.Add(refDev.Ptr(), int(rowBytes)*i))
		if err := AttentionInnerMixedCacheF32(
			qRow, kF16Dev, vF16Dev, kQ8Dev, vQ8Dev, kScaleDev, vScaleDev, outRow,
			useQ8K, useQ8V, startPos+i, 0, batchTestKVStride, batchTestHeadDim,
			batchTestNHead, batchTestKVHeads, batchTestCacheLen, batchTestScale, softcap, stream,
		); err != nil {
			t.Fatalf("single-query mixed row %d: %v", i, err)
		}
	}

	if err := MemcpyD2HAsync(batchHost.Ptr(), outDev, outBytes, stream); err != nil {
		t.Fatalf("D2H batch: %v", err)
	}
	if err := MemcpyD2HAsync(refHost.Ptr(), refDev, outBytes, stream); err != nil {
		t.Fatalf("D2H ref: %v", err)
	}
	if err := stream.Synchronize(); err != nil {
		t.Fatalf("sync: %v", err)
	}

	assertBitwiseEqual(t, "mixed batch", batchSlice, refSlice)
}

func TestAttentionInnerBatchValidation(t *testing.T) {
	count, err := DeviceCount()
	if err != nil || count < 1 {
		t.Skip("no cuda device available")
	}
	stream, err := NewStream()
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	defer func() { _ = stream.Destroy() }()

	dev := func(t *testing.T, bytes int64) DeviceBuffer {
		t.Helper()
		buf, err := AllocDevice(bytes)
		if err != nil {
			t.Fatalf("AllocDevice: %v", err)
		}
		t.Cleanup(func() { _ = buf.Free() })
		return buf
	}
	q := dev(t, 512)
	k := dev(t, 512)
	v := dev(t, 512)
	out := dev(t, 512)

	valid := func() error {
		return AttentionInnerF16CacheF32Batch(q, k, v, out, 1, 0, batchTestKVStride, batchTestHeadDim, batchTestNHead, batchTestKVHeads, batchTestCacheLen, batchTestScale, 0, stream)
	}
	if err := valid(); err != nil {
		t.Fatalf("valid call rejected: %v", err)
	}
	if err := AttentionInnerF16CacheF32Batch(DeviceBuffer{}, k, v, out, 1, 0, batchTestKVStride, batchTestHeadDim, batchTestNHead, batchTestKVHeads, batchTestCacheLen, batchTestScale, 0, stream); err == nil {
		t.Fatalf("nil q buffer accepted")
	}
	if err := AttentionInnerF16CacheF32Batch(q, k, v, out, 0, 0, batchTestKVStride, batchTestHeadDim, batchTestNHead, batchTestKVHeads, batchTestCacheLen, batchTestScale, 0, stream); err == nil {
		t.Fatalf("nTokens=0 accepted")
	}
	if err := AttentionInnerF16CacheF32Batch(q, k, v, out, 1, -1, batchTestKVStride, batchTestHeadDim, batchTestNHead, batchTestKVHeads, batchTestCacheLen, batchTestScale, 0, stream); err == nil {
		t.Fatalf("negative startPos accepted")
	}
	if err := AttentionInnerMixedCacheF32Batch(q, k, v, DeviceBuffer{}, DeviceBuffer{}, DeviceBuffer{}, DeviceBuffer{}, out, false, false, 1, 0, batchTestKVStride, batchTestHeadDim, batchTestNHead, batchTestKVHeads, batchTestCacheLen, batchTestScale, 0, stream); err != nil {
		t.Fatalf("valid mixed call rejected: %v", err)
	}
	if err := AttentionInnerMixedCacheF32Batch(DeviceBuffer{}, k, v, DeviceBuffer{}, DeviceBuffer{}, DeviceBuffer{}, DeviceBuffer{}, out, false, false, 1, 0, batchTestKVStride, batchTestHeadDim, batchTestNHead, batchTestKVHeads, batchTestCacheLen, batchTestScale, 0, stream); err == nil {
		t.Fatalf("nil mixed q buffer accepted")
	}
	if err := AttentionInnerMixedCacheF32Batch(q, k, v, DeviceBuffer{}, DeviceBuffer{}, DeviceBuffer{}, DeviceBuffer{}, out, false, false, 1, -3, batchTestKVStride, batchTestHeadDim, batchTestNHead, batchTestKVHeads, batchTestCacheLen, batchTestScale, 0, stream); err == nil {
		t.Fatalf("negative mixed startPos accepted")
	}
}
