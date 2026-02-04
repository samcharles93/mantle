//go:build cuda

package native

import (
	"runtime"
	"testing"
	"unsafe"
)

func TestPinnedAllocAndMemcpyRoundTrip(t *testing.T) {
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

	const n = 256
	hostIn, err := AllocHostPinned(n * int64(unsafe.Sizeof(float32(0))))
	if err != nil {
		t.Fatalf("AllocHostPinned input: %v", err)
	}
	defer func() {
		if err := hostIn.Free(); err != nil {
			t.Fatalf("host input free: %v", err)
		}
	}()

	hostOut, err := AllocHostPinned(n * int64(unsafe.Sizeof(float32(0))))
	if err != nil {
		t.Fatalf("AllocHostPinned output: %v", err)
	}
	defer func() {
		if err := hostOut.Free(); err != nil {
			t.Fatalf("host output free: %v", err)
		}
	}()

	dev, err := AllocDevice(n * int64(unsafe.Sizeof(float32(0))))
	if err != nil {
		t.Fatalf("AllocDevice: %v", err)
	}
	defer func() {
		if err := dev.Free(); err != nil {
			t.Fatalf("device free: %v", err)
		}
	}()

	inSlice := unsafe.Slice((*float32)(hostIn.Ptr()), n)
	outSlice := unsafe.Slice((*float32)(hostOut.Ptr()), n)
	for i := range inSlice {
		inSlice[i] = float32(i) * 1.25
		outSlice[i] = 0
	}

	if err := MemcpyH2DAsync(dev, hostIn.Ptr(), n*int64(unsafe.Sizeof(float32(0))), stream); err != nil {
		t.Fatalf("MemcpyH2DAsync: %v", err)
	}
	if err := MemcpyD2HAsync(hostOut.Ptr(), dev, n*int64(unsafe.Sizeof(float32(0))), stream); err != nil {
		t.Fatalf("MemcpyD2HAsync: %v", err)
	}
	if err := stream.Synchronize(); err != nil {
		t.Fatalf("stream synchronize: %v", err)
	}

	for i := range inSlice {
		if inSlice[i] != outSlice[i] {
			t.Fatalf("mismatch at %d: got %v want %v", i, outSlice[i], inSlice[i])
		}
	}
	runtime.KeepAlive(inSlice)
	runtime.KeepAlive(outSlice)
}

func TestCublasGemmExF32(t *testing.T) {
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

	blas, err := NewBlasHandle(stream)
	if err != nil {
		t.Fatalf("NewBlasHandle: %v", err)
	}
	defer func() {
		if err := blas.Destroy(); err != nil {
			t.Fatalf("blas destroy: %v", err)
		}
	}()

	const (
		m = 2
		n = 3
		k = 4
	)

	aHost, aSlice := allocPinnedF32(t, m*k)
	defer aHost.Free()
	bHost, bSlice := allocPinnedF32(t, k*n)
	defer bHost.Free()
	cHost, cSlice := allocPinnedF32(t, m*n)
	defer cHost.Free()
	ref := make([]float32, m*n)

	fillColMajor(aSlice, m, k, 0.5)
	fillColMajor(bSlice, k, n, -0.25)
	for i := range cSlice {
		cSlice[i] = 0
	}

	refGemmColMajor(ref, aSlice, bSlice, m, n, k)

	aDev, err := AllocDevice(int64(len(aSlice)) * int64(unsafe.Sizeof(float32(0))))
	if err != nil {
		t.Fatalf("AllocDevice A: %v", err)
	}
	defer aDev.Free()
	bDev, err := AllocDevice(int64(len(bSlice)) * int64(unsafe.Sizeof(float32(0))))
	if err != nil {
		t.Fatalf("AllocDevice B: %v", err)
	}
	defer bDev.Free()
	cDev, err := AllocDevice(int64(len(cSlice)) * int64(unsafe.Sizeof(float32(0))))
	if err != nil {
		t.Fatalf("AllocDevice C: %v", err)
	}
	defer cDev.Free()

	if err := MemcpyH2DAsync(aDev, aHost.Ptr(), int64(len(aSlice))*int64(unsafe.Sizeof(float32(0))), stream); err != nil {
		t.Fatalf("MemcpyH2DAsync A: %v", err)
	}
	if err := MemcpyH2DAsync(bDev, bHost.Ptr(), int64(len(bSlice))*int64(unsafe.Sizeof(float32(0))), stream); err != nil {
		t.Fatalf("MemcpyH2DAsync B: %v", err)
	}

	if err := GemmEx(blas, BlasOpN, BlasOpN, m, n, k, 1.0, aDev, BlasF32, m, bDev, BlasF32, k, 0.0, cDev, BlasF32, m, BlasComputeF32, BlasGemmDefault); err != nil {
		t.Fatalf("GemmEx: %v", err)
	}
	if err := MemcpyD2HAsync(cHost.Ptr(), cDev, int64(len(cSlice))*int64(unsafe.Sizeof(float32(0))), stream); err != nil {
		t.Fatalf("MemcpyD2HAsync C: %v", err)
	}
	if err := stream.Synchronize(); err != nil {
		t.Fatalf("stream synchronize: %v", err)
	}

	for i := range ref {
		if !approxEqual(ref[i], cSlice[i], 1e-4) {
			t.Fatalf("gemm mismatch at %d: got %v want %v", i, cSlice[i], ref[i])
		}
	}
}

func TestCublasGemvF32(t *testing.T) {
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

	blas, err := NewBlasHandle(stream)
	if err != nil {
		t.Fatalf("NewBlasHandle: %v", err)
	}
	defer func() {
		if err := blas.Destroy(); err != nil {
			t.Fatalf("blas destroy: %v", err)
		}
	}()

	const (
		m = 4
		n = 3
	)

	aHost, aSlice := allocPinnedF32(t, m*n)
	defer aHost.Free()
	xHost, xSlice := allocPinnedF32(t, n)
	defer xHost.Free()
	yHost, ySlice := allocPinnedF32(t, m)
	defer yHost.Free()
	ref := make([]float32, m)

	fillColMajor(aSlice, m, n, 1.0)
	for i := range xSlice {
		xSlice[i] = float32(i) * -0.5
	}
	for i := range ySlice {
		ySlice[i] = 0
	}

	refGemvColMajor(ref, aSlice, xSlice, m, n)

	aDev, err := AllocDevice(int64(len(aSlice)) * int64(unsafe.Sizeof(float32(0))))
	if err != nil {
		t.Fatalf("AllocDevice A: %v", err)
	}
	defer aDev.Free()
	xDev, err := AllocDevice(int64(len(xSlice)) * int64(unsafe.Sizeof(float32(0))))
	if err != nil {
		t.Fatalf("AllocDevice X: %v", err)
	}
	defer xDev.Free()
	yDev, err := AllocDevice(int64(len(ySlice)) * int64(unsafe.Sizeof(float32(0))))
	if err != nil {
		t.Fatalf("AllocDevice Y: %v", err)
	}
	defer yDev.Free()

	if err := MemcpyH2DAsync(aDev, aHost.Ptr(), int64(len(aSlice))*int64(unsafe.Sizeof(float32(0))), stream); err != nil {
		t.Fatalf("MemcpyH2DAsync A: %v", err)
	}
	if err := MemcpyH2DAsync(xDev, xHost.Ptr(), int64(len(xSlice))*int64(unsafe.Sizeof(float32(0))), stream); err != nil {
		t.Fatalf("MemcpyH2DAsync X: %v", err)
	}

	if err := GemvF32(blas, BlasOpN, m, n, 1.0, aDev, m, xDev, 1, 0.0, yDev, 1); err != nil {
		t.Fatalf("GemvF32: %v", err)
	}
	if err := MemcpyD2HAsync(yHost.Ptr(), yDev, int64(len(ySlice))*int64(unsafe.Sizeof(float32(0))), stream); err != nil {
		t.Fatalf("MemcpyD2HAsync Y: %v", err)
	}
	if err := stream.Synchronize(); err != nil {
		t.Fatalf("stream synchronize: %v", err)
	}

	for i := range ref {
		if !approxEqual(ref[i], ySlice[i], 1e-4) {
			t.Fatalf("gemv mismatch at %d: got %v want %v", i, ySlice[i], ref[i])
		}
	}
}

func allocPinnedF32(t *testing.T, n int) (HostBuffer, []float32) {
	t.Helper()
	buf, err := AllocHostPinned(int64(n) * int64(unsafe.Sizeof(float32(0))))
	if err != nil {
		t.Fatalf("AllocHostPinned: %v", err)
	}
	slice := unsafe.Slice((*float32)(buf.Ptr()), n)
	return buf, slice
}

func fillColMajor(dst []float32, rows, cols int, scale float32) {
	idx := 0
	for c := 0; c < cols; c++ {
		for r := 0; r < rows; r++ {
			dst[idx] = (float32(r+1) + float32(c+1)) * scale
			idx++
		}
	}
}

func refGemmColMajor(dst, a, b []float32, m, n, k int) {
	for col := 0; col < n; col++ {
		for row := 0; row < m; row++ {
			var sum float32
			for i := 0; i < k; i++ {
				av := a[row+i*m]
				bv := b[i+col*k]
				sum += av * bv
			}
			dst[row+col*m] = sum
		}
	}
}

func refGemvColMajor(dst, a, x []float32, m, n int) {
	for row := 0; row < m; row++ {
		var sum float32
		for col := 0; col < n; col++ {
			sum += a[row+col*m] * x[col]
		}
		dst[row] = sum
	}
}

func approxEqual(a, b, eps float32) bool {
	diff := a - b
	if diff < 0 {
		diff = -diff
	}
	return diff <= eps
}
