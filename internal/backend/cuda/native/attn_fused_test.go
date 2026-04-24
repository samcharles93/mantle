//go:build cuda

package native

import (
	"math"
	"testing"
	"unsafe"
)

// applyRoPEScalarRef mirrors internal/backend/simd/ops.go:applyRoPEScalar.
func applyRoPEScalarRef(x []float32, nHead, headDim, pos int, invFreq []float32, attnScale float32, half int) {
	if attnScale == 0 {
		attnScale = 1
	}
	for h := 0; h < nHead; h++ {
		base := h * headDim
		for i := 0; i < half; i++ {
			angle := float64(pos) * float64(invFreq[i])
			c := float32(math.Cos(angle)) * attnScale
			s := float32(math.Sin(angle)) * attnScale
			i0 := base + i
			i1 := base + i + half
			x0 := x[i0]
			x1 := x[i1]
			x[i0] = x0*c - x1*s
			x[i1] = x0*s + x1*c
		}
	}
}

func TestApplyRoPEInplaceF32(t *testing.T) {
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
	defer stream.Destroy()

	const (
		nHead   = 4
		headDim = 16
		half    = headDim / 2
		pos     = 7
	)
	attnScale := float32(1.25)

	xHost, xSlice := allocPinnedF32(t, nHead*headDim)
	defer xHost.Free()
	ifHost, ifSlice := allocPinnedF32(t, half)
	defer ifHost.Free()
	ref := make([]float32, nHead*headDim)

	for i := range xSlice {
		xSlice[i] = float32(i)*0.1 - 1.5
		ref[i] = xSlice[i]
	}
	for i := range ifSlice {
		ifSlice[i] = float32(1.0 / math.Pow(10000, float64(2*i)/float64(headDim)))
	}
	applyRoPEScalarRef(ref, nHead, headDim, pos, ifSlice, attnScale, half)

	xDev, err := AllocDevice(int64(len(xSlice)) * 4)
	if err != nil {
		t.Fatalf("AllocDevice x: %v", err)
	}
	defer xDev.Free()
	ifDev, err := AllocDevice(int64(len(ifSlice)) * 4)
	if err != nil {
		t.Fatalf("AllocDevice invFreq: %v", err)
	}
	defer ifDev.Free()

	if err := MemcpyH2DAsync(xDev, xHost.Ptr(), int64(len(xSlice))*4, stream); err != nil {
		t.Fatalf("H2D x: %v", err)
	}
	if err := MemcpyH2DAsync(ifDev, ifHost.Ptr(), int64(len(ifSlice))*4, stream); err != nil {
		t.Fatalf("H2D invFreq: %v", err)
	}
	if err := ApplyRoPEInplaceF32(xDev, ifDev, pos, attnScale, headDim, half, nHead, stream); err != nil {
		t.Fatalf("ApplyRoPEInplaceF32: %v", err)
	}
	if err := MemcpyD2HAsync(xHost.Ptr(), xDev, int64(len(xSlice))*4, stream); err != nil {
		t.Fatalf("D2H x: %v", err)
	}
	if err := stream.Synchronize(); err != nil {
		t.Fatalf("sync: %v", err)
	}

	for i := range ref {
		if !approxEqual(ref[i], xSlice[i], 1e-4) {
			t.Fatalf("rope mismatch at %d: got %v want %v", i, xSlice[i], ref[i])
		}
	}
}

func TestApplyRoPEInplaceF32ZeroAttnScaleTreatedAsOne(t *testing.T) {
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
		nHead   = 2
		headDim = 8
		half    = 4
		pos     = 3
	)
	xHost, xSlice := allocPinnedF32(t, nHead*headDim)
	defer xHost.Free()
	ifHost, ifSlice := allocPinnedF32(t, half)
	defer ifHost.Free()
	for i := range xSlice {
		xSlice[i] = float32(i) - 3
	}
	for i := range ifSlice {
		ifSlice[i] = float32(1.0 / math.Pow(10000, float64(2*i)/float64(headDim)))
	}
	ref := make([]float32, len(xSlice))
	copy(ref, xSlice)
	applyRoPEScalarRef(ref, nHead, headDim, pos, ifSlice, 1.0, half)

	xDev, err := AllocDevice(int64(len(xSlice)) * 4)
	if err != nil {
		t.Fatalf("AllocDevice: %v", err)
	}
	defer xDev.Free()
	ifDev, err := AllocDevice(int64(len(ifSlice)) * 4)
	if err != nil {
		t.Fatalf("AllocDevice: %v", err)
	}
	defer ifDev.Free()
	if err := MemcpyH2DAsync(xDev, xHost.Ptr(), int64(len(xSlice))*4, stream); err != nil {
		t.Fatalf("H2D: %v", err)
	}
	if err := MemcpyH2DAsync(ifDev, ifHost.Ptr(), int64(len(ifSlice))*4, stream); err != nil {
		t.Fatalf("H2D: %v", err)
	}
	if err := ApplyRoPEInplaceF32(xDev, ifDev, pos, 0.0, headDim, half, nHead, stream); err != nil {
		t.Fatalf("rope: %v", err)
	}
	if err := MemcpyD2HAsync(xHost.Ptr(), xDev, int64(len(xSlice))*4, stream); err != nil {
		t.Fatalf("D2H: %v", err)
	}
	if err := stream.Synchronize(); err != nil {
		t.Fatalf("sync: %v", err)
	}
	for i := range ref {
		if !approxEqual(ref[i], xSlice[i], 1e-4) {
			t.Fatalf("rope mismatch at %d: got %v want %v", i, xSlice[i], ref[i])
		}
	}
}

func TestStoreKVF16Row(t *testing.T) {
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
		kvStride = 64
		cacheLen = 4
		cachePos = 2
	)
	cacheBytes := int64(kvStride * cacheLen * 2)

	srcHost, srcSlice := allocPinnedF32(t, kvStride)
	defer srcHost.Free()
	for i := range srcSlice {
		srcSlice[i] = float32(i)*0.25 - 2.0
	}

	// Pre-fill the whole cache with a sentinel so we can detect writes outside
	// the target row.
	hostCacheBuf, err := AllocHostPinned(cacheBytes)
	if err != nil {
		t.Fatalf("AllocHostPinned: %v", err)
	}
	defer hostCacheBuf.Free()
	hostCache := unsafe.Slice((*uint16)(hostCacheBuf.Ptr()), kvStride*cacheLen)
	for i := range hostCache {
		hostCache[i] = 0xBEEF
	}

	srcDev, err := AllocDevice(int64(kvStride) * 4)
	if err != nil {
		t.Fatalf("AllocDevice src: %v", err)
	}
	defer srcDev.Free()
	cacheDev, err := AllocDevice(cacheBytes)
	if err != nil {
		t.Fatalf("AllocDevice cache: %v", err)
	}
	defer cacheDev.Free()

	if err := MemcpyH2DAsync(cacheDev, hostCacheBuf.Ptr(), cacheBytes, stream); err != nil {
		t.Fatalf("H2D cache: %v", err)
	}
	if err := MemcpyH2DAsync(srcDev, srcHost.Ptr(), int64(kvStride)*4, stream); err != nil {
		t.Fatalf("H2D src: %v", err)
	}
	if err := StoreKVF16Row(cacheDev, srcDev, cachePos, kvStride, stream); err != nil {
		t.Fatalf("StoreKVF16Row: %v", err)
	}
	if err := MemcpyD2HAsync(hostCacheBuf.Ptr(), cacheDev, cacheBytes, stream); err != nil {
		t.Fatalf("D2H cache: %v", err)
	}
	if err := stream.Synchronize(); err != nil {
		t.Fatalf("sync: %v", err)
	}

	for row := 0; row < cacheLen; row++ {
		for c := 0; c < kvStride; c++ {
			got := hostCache[row*kvStride+c]
			if row != cachePos {
				if got != 0xBEEF {
					t.Fatalf("row %d col %d: sentinel overwritten (got 0x%04x)", row, c, got)
				}
				continue
			}
			want := fp16ToF32(f32ToF16BitsRef(srcSlice[c]))
			gotF := fp16ToF32(got)
			if !approxEqual(gotF, want, 5e-3) {
				t.Fatalf("row %d col %d: got %v want %v", row, c, gotF, want)
			}
		}
	}
}

// f32ToF16BitsRef mirrors instance.Float32ToFloat16 for test-only use.
// CUDA's __float2half uses round-to-nearest-even; the reference path on
// the host uses IEEE round-to-nearest-even as well, so the bit patterns
// should match for well-represented values. We compare the resulting F32
// with a loose epsilon (5e-3) to tolerate any tie-breaking edge case.
func f32ToF16BitsRef(v float32) uint16 {
	bits := math.Float32bits(v)
	sign := uint16((bits >> 31) & 0x1)
	expF32 := int32((bits>>23)&0xff) - 127
	fracF32 := bits & 0x7fffff

	if expF32 == 128 { // Inf/NaN
		return (sign << 15) | 0x7c00 | uint16(fracF32>>13)
	}
	if expF32 > 15 { // overflow
		return (sign << 15) | 0x7c00
	}
	if expF32 < -14 { // subnormal or underflow
		if expF32 < -24 {
			return sign << 15
		}
		frac := (fracF32 | 0x800000) >> uint32(-expF32-1)
		// round to nearest, ties to even
		out := uint16(frac >> 13)
		rem := frac & 0x1fff
		if rem > 0x1000 || (rem == 0x1000 && (out&1) == 1) {
			out++
		}
		return (sign << 15) | out
	}
	expF16 := uint16(expF32+15) << 10
	fracF16 := uint16(fracF32 >> 13)
	rem := fracF32 & 0x1fff
	result := (sign << 15) | expF16 | fracF16
	if rem > 0x1000 || (rem == 0x1000 && (fracF16&1) == 1) {
		result++
	}
	return result
}

func TestStoreKVQ8RowBroadcast(t *testing.T) {
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
		kvStride      = 64 // 2 blocks of 32
		blocksPerRow  = 2
		cacheLen      = 3
		cachePos      = 1
		scaleStripLen = cacheLen * blocksPerRow
	)

	srcHost, srcSlice := allocPinnedF32(t, kvStride)
	defer srcHost.Free()
	var absmax float32
	for i := range srcSlice {
		v := float32(i) - 31.5
		srcSlice[i] = v
		if v < 0 {
			v = -v
		}
		if v > absmax {
			absmax = v
		}
	}
	wantScale := absmax / 127.0

	qHostBuf, err := AllocHostPinned(int64(kvStride * cacheLen))
	if err != nil {
		t.Fatalf("pin q: %v", err)
	}
	defer qHostBuf.Free()
	qHost := unsafe.Slice((*int8)(qHostBuf.Ptr()), kvStride*cacheLen)
	for i := range qHost {
		qHost[i] = 99
	}
	sHostBuf, err := AllocHostPinned(int64(scaleStripLen * 4))
	if err != nil {
		t.Fatalf("pin s: %v", err)
	}
	defer sHostBuf.Free()
	sHost := unsafe.Slice((*float32)(sHostBuf.Ptr()), scaleStripLen)
	for i := range sHost {
		sHost[i] = -7.0
	}

	srcDev, err := AllocDevice(int64(kvStride) * 4)
	if err != nil {
		t.Fatalf("AllocDevice src: %v", err)
	}
	defer srcDev.Free()
	qDev, err := AllocDevice(int64(kvStride * cacheLen))
	if err != nil {
		t.Fatalf("AllocDevice q: %v", err)
	}
	defer qDev.Free()
	sDev, err := AllocDevice(int64(scaleStripLen) * 4)
	if err != nil {
		t.Fatalf("AllocDevice scales: %v", err)
	}
	defer sDev.Free()

	if err := MemcpyH2DAsync(qDev, qHostBuf.Ptr(), int64(kvStride*cacheLen), stream); err != nil {
		t.Fatalf("H2D q: %v", err)
	}
	if err := MemcpyH2DAsync(sDev, sHostBuf.Ptr(), int64(scaleStripLen)*4, stream); err != nil {
		t.Fatalf("H2D s: %v", err)
	}
	if err := MemcpyH2DAsync(srcDev, srcHost.Ptr(), int64(kvStride)*4, stream); err != nil {
		t.Fatalf("H2D src: %v", err)
	}
	if err := StoreKVQ8RowBroadcast(qDev, sDev, srcDev, cachePos, kvStride, blocksPerRow, stream); err != nil {
		t.Fatalf("StoreKVQ8RowBroadcast: %v", err)
	}
	if err := MemcpyD2HAsync(qHostBuf.Ptr(), qDev, int64(kvStride*cacheLen), stream); err != nil {
		t.Fatalf("D2H q: %v", err)
	}
	if err := MemcpyD2HAsync(sHostBuf.Ptr(), sDev, int64(scaleStripLen)*4, stream); err != nil {
		t.Fatalf("D2H s: %v", err)
	}
	if err := stream.Synchronize(); err != nil {
		t.Fatalf("sync: %v", err)
	}

	// Sentinel rows untouched.
	for r := 0; r < cacheLen; r++ {
		if r == cachePos {
			continue
		}
		for c := 0; c < kvStride; c++ {
			if qHost[r*kvStride+c] != 99 {
				t.Fatalf("q sentinel overwritten at row=%d col=%d", r, c)
			}
		}
		for b := 0; b < blocksPerRow; b++ {
			if sHost[r*blocksPerRow+b] != -7.0 {
				t.Fatalf("scale sentinel overwritten at row=%d block=%d", r, b)
			}
		}
	}

	// Target row scales: all blocks must equal wantScale.
	for b := 0; b < blocksPerRow; b++ {
		got := sHost[cachePos*blocksPerRow+b]
		if !approxEqual(got, wantScale, 1e-6) {
			t.Fatalf("scale[%d]=%v want %v", b, got, wantScale)
		}
	}

	// Dequantize row and compare with source within abs tolerance of one
	// quant step (scale).
	tol := wantScale * 1.1
	for c := 0; c < kvStride; c++ {
		got := float32(qHost[cachePos*kvStride+c]) * wantScale
		if !approxEqual(got, srcSlice[c], tol) {
			t.Fatalf("col %d: got %v want %v (tol=%v)", c, got, srcSlice[c], tol)
		}
	}
}

func TestStoreKVQ8RowBroadcastZeroVector(t *testing.T) {
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
		kvStride     = 32
		blocksPerRow = 1
		cachePos     = 0
	)

	srcHost, srcSlice := allocPinnedF32(t, kvStride)
	defer srcHost.Free()
	for i := range srcSlice {
		srcSlice[i] = 0
	}
	qHostBuf, _ := AllocHostPinned(int64(kvStride))
	defer qHostBuf.Free()
	qHost := unsafe.Slice((*int8)(qHostBuf.Ptr()), kvStride)
	for i := range qHost {
		qHost[i] = 33
	}
	sHostBuf, _ := AllocHostPinned(4)
	defer sHostBuf.Free()
	sHost := unsafe.Slice((*float32)(sHostBuf.Ptr()), 1)
	sHost[0] = -42

	srcDev, _ := AllocDevice(int64(kvStride) * 4)
	defer srcDev.Free()
	qDev, _ := AllocDevice(int64(kvStride))
	defer qDev.Free()
	sDev, _ := AllocDevice(4)
	defer sDev.Free()

	if err := MemcpyH2DAsync(qDev, qHostBuf.Ptr(), int64(kvStride), stream); err != nil {
		t.Fatalf("H2D q: %v", err)
	}
	if err := MemcpyH2DAsync(sDev, sHostBuf.Ptr(), 4, stream); err != nil {
		t.Fatalf("H2D s: %v", err)
	}
	if err := MemcpyH2DAsync(srcDev, srcHost.Ptr(), int64(kvStride)*4, stream); err != nil {
		t.Fatalf("H2D src: %v", err)
	}
	if err := StoreKVQ8RowBroadcast(qDev, sDev, srcDev, cachePos, kvStride, blocksPerRow, stream); err != nil {
		t.Fatalf("StoreKVQ8RowBroadcast: %v", err)
	}
	if err := MemcpyD2HAsync(qHostBuf.Ptr(), qDev, int64(kvStride), stream); err != nil {
		t.Fatalf("D2H q: %v", err)
	}
	if err := MemcpyD2HAsync(sHostBuf.Ptr(), sDev, 4, stream); err != nil {
		t.Fatalf("D2H s: %v", err)
	}
	if err := stream.Synchronize(); err != nil {
		t.Fatalf("sync: %v", err)
	}
	if sHost[0] != 0 {
		t.Fatalf("scale for zero vector: got %v want 0", sHost[0])
	}
	for i, q := range qHost {
		if q != 0 {
			t.Fatalf("q[%d]=%d want 0 (zero vector)", i, q)
		}
	}
}
