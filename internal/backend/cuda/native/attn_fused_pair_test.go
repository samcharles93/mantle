//go:build cuda

package native

import (
	"math"
	"math/rand"
	"testing"
	"unsafe"
)

// TestApplyRoPEPairF32BitExact requires the fused Q+K RoPE launch to be
// bit-identical to two separate legacy ApplyRoPEInplaceF32 launches.
func TestApplyRoPEPairF32BitExact(t *testing.T) {
	count, err := DeviceCount()
	if err != nil || count < 1 {
		t.Skip("no cuda device available")
	}
	stream, err := NewStream()
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	defer stream.Destroy()

	const (
		nQ      = 16
		nK      = 4
		headDim = 128
		half    = 32
		pos     = 137
	)
	attnScale := float32(1.0 / math.Sqrt(128.0))

	rng := rand.New(rand.NewSource(99))
	qHost := make([]float32, nQ*headDim)
	kHost := make([]float32, nK*headDim)
	for i := range qHost {
		qHost[i] = float32(rng.Float64()*2 - 1)
	}
	for i := range kHost {
		kHost[i] = float32(rng.Float64()*2 - 1)
	}
	ifHost := make([]float32, half)
	for i := range ifHost {
		ifHost[i] = float32(1.0 / math.Pow(10000, float64(2*i)/float64(headDim)))
	}

	alloc := func(h []float32) (DeviceBuffer, HostBuffer) {
		db, err := AllocDevice(int64(len(h)) * 4)
		if err != nil {
			t.Fatalf("AllocDevice: %v", err)
		}
		hb, slice := allocPinnedF32(t, len(h))
		copy(slice, h)
		if err := MemcpyH2DAsync(db, hb.Ptr(), int64(len(h))*4, stream); err != nil {
			t.Fatalf("H2D: %v", err)
		}
		return db, hb
	}
	qDev, qHB := alloc(qHost)
	defer qDev.Free()
	defer qHB.Free()
	kDev, kHB := alloc(kHost)
	defer kDev.Free()
	defer kHB.Free()
	ifDev, ifHB := alloc(ifHost)
	defer ifDev.Free()
	defer ifHB.Free()

	// reference: two legacy launches on separate buffers
	qRef, qRefHB := alloc(qHost)
	defer qRef.Free()
	defer qRefHB.Free()
	kRef, kRefHB := alloc(kHost)
	defer kRef.Free()
	defer kRefHB.Free()
	if err := ApplyRoPEInplaceF32(qRef, ifDev, pos, attnScale, headDim, half, nQ, stream); err != nil {
		t.Fatalf("legacy rope Q: %v", err)
	}
	if err := ApplyRoPEInplaceF32(kRef, ifDev, pos, attnScale, headDim, half, nK, stream); err != nil {
		t.Fatalf("legacy rope K: %v", err)
	}
	// fused pair launch
	if err := ApplyRoPEPairF32(qDev, kDev, ifDev, pos, attnScale, headDim, half, nQ, nK, stream); err != nil {
		t.Fatalf("pair rope: %v", err)
	}
	if err := stream.Synchronize(); err != nil {
		t.Fatalf("sync: %v", err)
	}
	if err := MemcpyD2HAsync(qHB.Ptr(), qDev, int64(len(qHost))*4, stream); err != nil {
		t.Fatalf("D2H q: %v", err)
	}
	if err := MemcpyD2HAsync(kHB.Ptr(), kDev, int64(len(kHost))*4, stream); err != nil {
		t.Fatalf("D2H k: %v", err)
	}
	if err := MemcpyD2HAsync(qRefHB.Ptr(), qRef, int64(len(qHost))*4, stream); err != nil {
		t.Fatalf("D2H qRef: %v", err)
	}
	if err := MemcpyD2HAsync(kRefHB.Ptr(), kRef, int64(len(kHost))*4, stream); err != nil {
		t.Fatalf("D2H kRef: %v", err)
	}
	if err := stream.Synchronize(); err != nil {
		t.Fatalf("sync2: %v", err)
	}
	qs, _ := readF32(qHB, len(qHost))
	ks, _ := readF32(kHB, len(kHost))
	qrs, _ := readF32(qRefHB, len(qHost))
	krs, _ := readF32(kRefHB, len(kHost))
	requireBitExact(t, qs, qrs, "rope pair Q")
	requireBitExact(t, ks, krs, "rope pair K")
}

// TestStoreKVPairsBitExact requires the fused K+V store launches (f16 and q8)
// to be bit-identical to two separate legacy stores.
func TestStoreKVPairsBitExact(t *testing.T) {
	count, err := DeviceCount()
	if err != nil || count < 1 {
		t.Skip("no cuda device available")
	}
	stream, err := NewStream()
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	defer stream.Destroy()

	const (
		cacheLen = 8
		kvStride = 512
		cachePos = 5
		bpr      = kvStride / 32
	)
	rng := rand.New(rand.NewSource(7))
	kSrc := make([]float32, kvStride)
	vSrc := make([]float32, kvStride)
	for i := range kSrc {
		kSrc[i] = float32(rng.Float64()*2 - 1)
		vSrc[i] = float32(rng.Float64()*2 - 1)
	}

	// ---- F16 path ----
	mkBuf := func(n int, fill byte) DeviceBuffer {
		b, err := AllocDevice(int64(n))
		if err != nil {
			t.Fatalf("AllocDevice: %v", err)
		}
		t.Cleanup(func() { _ = b.Free() })
		host := make([]byte, n)
		for i := range host {
			host[i] = fill
		}
		if err := MemcpyH2D(b, unsafe.Pointer(&host[0]), int64(n)); err != nil {
			t.Fatalf("H2D fill: %v", err)
		}
		return b
	}
	srcBuf := func(h []float32) DeviceBuffer {
		b := mkBuf(len(h)*4, 0)
		if err := MemcpyH2D(b, unsafe.Pointer(&h[0]), int64(len(h))*4); err != nil {
			t.Fatalf("H2D: %v", err)
		}
		return b
	}
	kSrcBuf := srcBuf(kSrc)
	vSrcBuf := srcBuf(vSrc)

	// reference
	kRef := mkBuf(cacheLen*kvStride*2, 0xAB)
	vRef := mkBuf(cacheLen*kvStride*2, 0xAB)
	kPair := mkBuf(cacheLen*kvStride*2, 0xAB)
	vPair := mkBuf(cacheLen*kvStride*2, 0xAB)
	if err := StoreKVF16Row(kRef, kSrcBuf, cachePos, kvStride, stream); err != nil {
		t.Fatalf("legacy store K: %v", err)
	}
	if err := StoreKVF16Row(vRef, vSrcBuf, cachePos, kvStride, stream); err != nil {
		t.Fatalf("legacy store V: %v", err)
	}
	if err := StoreKVPairsF16(kPair, vPair, kSrcBuf, vSrcBuf, cachePos, kvStride, stream); err != nil {
		t.Fatalf("pair store f16: %v", err)
	}
	if err := stream.Synchronize(); err != nil {
		t.Fatalf("sync: %v", err)
	}
	requireDeviceBytesEqual(t, kPair, kRef, "store pair f16 K", stream)
	requireDeviceBytesEqual(t, vPair, vRef, "store pair f16 V", stream)

	// ---- Q8 path ----
	kqRef := mkBuf(cacheLen*kvStride, 0xCD)
	vqRef := mkBuf(cacheLen*kvStride, 0xCD)
	ksRef := mkBuf(cacheLen*bpr*4, 0)
	vsRef := mkBuf(cacheLen*bpr*4, 0)
	kqPair := mkBuf(cacheLen*kvStride, 0xCD)
	vqPair := mkBuf(cacheLen*kvStride, 0xCD)
	ksPair := mkBuf(cacheLen*bpr*4, 0)
	vsPair := mkBuf(cacheLen*bpr*4, 0)
	if err := StoreKVQ8RowBroadcast(kqRef, ksRef, kSrcBuf, cachePos, kvStride, bpr, stream); err != nil {
		t.Fatalf("legacy store K q8: %v", err)
	}
	if err := StoreKVQ8RowBroadcast(vqRef, vsRef, vSrcBuf, cachePos, kvStride, bpr, stream); err != nil {
		t.Fatalf("legacy store V q8: %v", err)
	}
	if err := StoreKVPairsQ8RowBroadcast(kqPair, ksPair, vqPair, vsPair, kSrcBuf, vSrcBuf, cachePos, kvStride, bpr, stream); err != nil {
		t.Fatalf("pair store q8: %v", err)
	}
	if err := stream.Synchronize(); err != nil {
		t.Fatalf("sync: %v", err)
	}
	requireDeviceBytesEqual(t, kqPair, kqRef, "store pair q8 K", stream)
	requireDeviceBytesEqual(t, vqPair, vqRef, "store pair q8 V", stream)
	requireDeviceBytesEqual(t, ksPair, ksRef, "store pair q8 K scales", stream)
	requireDeviceBytesEqual(t, vsPair, vsRef, "store pair q8 V scales", stream)
}

// TestRMSNormBatchedPairF32BitExact requires the fused Q+K batched-norm launch
// to be bit-identical to two separate legacy batched-norm launches.
func TestRMSNormBatchedPairF32BitExact(t *testing.T) {
	count, err := DeviceCount()
	if err != nil || count < 1 {
		t.Skip("no cuda device available")
	}
	stream, err := NewStream()
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	defer stream.Destroy()

	const (
		nQ      = 16
		nK      = 4
		headDim = 128
		eps     = float32(1e-5)
	)
	rng := rand.New(rand.NewSource(5))
	qHost := make([]float32, nQ*headDim)
	kHost := make([]float32, nK*headDim)
	wQ := make([]float32, headDim)
	wK := make([]float32, headDim)
	for i := range qHost {
		qHost[i] = float32(rng.Float64()*2 - 1)
	}
	for i := range kHost {
		kHost[i] = float32(rng.Float64()*2 - 1)
	}
	for i := range wQ {
		wQ[i] = float32(rng.Float64()) + 0.5
		wK[i] = float32(rng.Float64()) + 0.5
	}

	up := func(h []float32) (DeviceBuffer, HostBuffer) {
		db, err := AllocDevice(int64(len(h)) * 4)
		if err != nil {
			t.Fatalf("AllocDevice: %v", err)
		}
		hb, slice := allocPinnedF32(t, len(h))
		copy(slice, h)
		if err := MemcpyH2DAsync(db, hb.Ptr(), int64(len(h))*4, stream); err != nil {
			t.Fatalf("H2D: %v", err)
		}
		return db, hb
	}
	qDev, qHB := up(qHost)
	defer qDev.Free()
	defer qHB.Free()
	kDev, kHB := up(kHost)
	defer kDev.Free()
	defer kHB.Free()
	wQDev, wQHB := up(wQ)
	defer wQDev.Free()
	defer wQHB.Free()
	wKDev, wKHB := up(wK)
	defer wKDev.Free()
	defer wKHB.Free()
	qRef, qRefHB := up(qHost)
	defer qRef.Free()
	defer qRefHB.Free()
	kRef, kRefHB := up(kHost)
	defer kRef.Free()
	defer kRefHB.Free()
	qOut, qOutHB := up(make([]float32, len(qHost)))
	defer qOut.Free()
	defer qOutHB.Free()
	kOut, kOutHB := up(make([]float32, len(kHost)))
	defer kOut.Free()
	defer kOutHB.Free()
	qOutRef, _ := up(make([]float32, len(qHost)))
	defer qOutRef.Free()
	kOutRef, _ := up(make([]float32, len(kHost)))
	defer kOutRef.Free()

	RecordRMSNorm()
	if err := RMSNormBatchedF32(qOutRef, qRef, wQDev, eps, headDim, nQ, stream); err != nil {
		t.Fatalf("legacy norm Q: %v", err)
	}
	RecordRMSNorm()
	if err := RMSNormBatchedF32(kOutRef, kRef, wKDev, eps, headDim, nK, stream); err != nil {
		t.Fatalf("legacy norm K: %v", err)
	}
	if err := RMSNormBatchedPairF32(qOut, qDev, wQDev, kOut, kDev, wKDev, eps, headDim, nQ, nK, stream); err != nil {
		t.Fatalf("pair norm: %v", err)
	}
	if err := stream.Synchronize(); err != nil {
		t.Fatalf("sync: %v", err)
	}
	d2h := func(db DeviceBuffer, hb HostBuffer, n int) []float32 {
		if err := MemcpyD2HAsync(hb.Ptr(), db, int64(n)*4, stream); err != nil {
			t.Fatalf("D2H: %v", err)
		}
		if err := stream.Synchronize(); err != nil {
			t.Fatalf("sync: %v", err)
		}
		out := make([]float32, n)
		copy(out, unsafe.Slice((*float32)(hb.Ptr()), n))
		return out
	}
	requireBitExact(t, d2h(qOut, qOutHB, len(qHost)), d2h(qOutRef, qRefHB, len(qHost)), "norm pair Q")
	requireBitExact(t, d2h(kOut, kOutHB, len(kHost)), d2h(kOutRef, kRefHB, len(kHost)), "norm pair K")
}

func readF32(hb HostBuffer, n int) ([]float32, error) {
	out := make([]float32, n)
	copy(out, unsafe.Slice((*float32)(hb.Ptr()), n))
	return out, nil
}

func requireDeviceBytesEqual(t *testing.T, a, b DeviceBuffer, label string, stream Stream) {
	t.Helper()
	n := a.Nbytes()
	if n != b.Nbytes() {
		t.Fatalf("%s: size mismatch %d vs %d", label, n, b.Nbytes())
	}
	ha, _ := allocPinnedF32(t, int(n)/4+1)
	defer ha.Free()
	hb, _ := allocPinnedF32(t, int(n)/4+1)
	defer hb.Free()
	if err := MemcpyD2HAsync(ha.Ptr(), a, n, stream); err != nil {
		t.Fatalf("%s: D2H a: %v", label, err)
	}
	if err := MemcpyD2HAsync(hb.Ptr(), b, n, stream); err != nil {
		t.Fatalf("%s: D2H b: %v", label, err)
	}
	if err := stream.Synchronize(); err != nil {
		t.Fatalf("%s: sync: %v", label, err)
	}
	sa := unsafe.Slice((*byte)(ha.Ptr()), n)
	sb := unsafe.Slice((*byte)(hb.Ptr()), n)
	for i := 0; i < int(n); i++ {
		if sa[i] != sb[i] {
			t.Fatalf("%s: byte mismatch at %d: %02x vs %02x", label, i, sa[i], sb[i])
		}
	}
}

// TestRMSNormResidualF32BitExact requires the fused norm+residual kernel to be
// bit-identical to RMSNormF32 into a temp followed by AddVectorsF32.
func TestRMSNormResidualF32BitExact(t *testing.T) {
	count, err := DeviceCount()
	if err != nil || count < 1 {
		t.Skip("no cuda device available")
	}
	stream, err := NewStream()
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	defer stream.Destroy()

	for _, n := range []int{2048, 100, 37} {
		rng := rand.New(rand.NewSource(int64(n)))
		xHost := make([]float32, n)
		outHost := make([]float32, n)
		wHost := make([]float32, n)
		for i := range xHost {
			xHost[i] = float32(rng.Float64()*2 - 1)
			outHost[i] = float32(rng.Float64()*2 - 1)
			wHost[i] = float32(rng.Float64()) + 0.5
		}
		const eps = float32(1e-5)

		up := func(h []float32) (DeviceBuffer, HostBuffer) {
			db, err := AllocDevice(int64(len(h)) * 4)
			if err != nil {
				t.Fatalf("AllocDevice: %v", err)
			}
			hb, slice := allocPinnedF32(t, len(h))
			copy(slice, h)
			if err := MemcpyH2DAsync(db, hb.Ptr(), int64(len(h))*4, stream); err != nil {
				t.Fatalf("H2D: %v", err)
			}
			return db, hb
		}
		xDev, xHB := up(xHost)
		defer xDev.Free()
		defer xHB.Free()
		wDev, wHB := up(wHost)
		defer wDev.Free()
		defer wHB.Free()

		// reference: norm into tmp, then dst += tmp
		tmpDev, _ := up(make([]float32, n))
		defer tmpDev.Free()
		refDev, refHB := up(outHost)
		defer refDev.Free()
		defer refHB.Free()
		if err := RMSNormF32(tmpDev, xDev, wDev, eps, n, stream); err != nil {
			t.Fatalf("RMSNormF32: %v", err)
		}
		if err := AddVectorsF32(refDev, tmpDev, n, stream); err != nil {
			t.Fatalf("AddVectorsF32: %v", err)
		}
		// fused
		fusedDev, fusedHB := up(outHost)
		defer fusedDev.Free()
		defer fusedHB.Free()
		if err := RMSNormResidualF32(fusedDev, xDev, wDev, eps, n, stream); err != nil {
			t.Fatalf("RMSNormResidualF32: %v", err)
		}
		if err := stream.Synchronize(); err != nil {
			t.Fatalf("sync: %v", err)
		}
		d2h := func(db DeviceBuffer, hb HostBuffer) []float32 {
			if err := MemcpyD2HAsync(hb.Ptr(), db, int64(n)*4, stream); err != nil {
				t.Fatalf("D2H: %v", err)
			}
			if err := stream.Synchronize(); err != nil {
				t.Fatalf("sync: %v", err)
			}
			out := make([]float32, n)
			copy(out, unsafe.Slice((*float32)(hb.Ptr()), n))
			return out
		}
		requireBitExact(t, d2h(fusedDev, fusedHB), d2h(refDev, refHB), "rmsnorm+residual")
	}
}
