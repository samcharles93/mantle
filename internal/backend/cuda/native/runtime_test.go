//go:build cuda

package native

import (
	"math"
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

func TestSoftmaxRowsF32(t *testing.T) {
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

	const (
		rows = 2
		cols = 8
		n    = rows * cols
	)

	hostIn, hostSlice := allocPinnedF32(t, n)
	defer hostIn.Free()
	ref := make([]float32, n)

	for i := range hostSlice {
		hostSlice[i] = float32((i%cols)-3) * 0.5
		ref[i] = hostSlice[i]
	}
	for r := 0; r < rows; r++ {
		softmaxRef(ref[r*cols : (r+1)*cols])
	}

	dev, err := AllocDevice(int64(n) * int64(unsafe.Sizeof(float32(0))))
	if err != nil {
		t.Fatalf("AllocDevice: %v", err)
	}
	defer dev.Free()

	if err := MemcpyH2DAsync(dev, hostIn.Ptr(), int64(n)*int64(unsafe.Sizeof(float32(0))), stream); err != nil {
		t.Fatalf("MemcpyH2DAsync: %v", err)
	}
	if err := SoftmaxRowsF32(dev, rows, cols, stream); err != nil {
		t.Fatalf("SoftmaxRowsF32: %v", err)
	}
	if err := MemcpyD2HAsync(hostIn.Ptr(), dev, int64(n)*int64(unsafe.Sizeof(float32(0))), stream); err != nil {
		t.Fatalf("MemcpyD2HAsync: %v", err)
	}
	if err := stream.Synchronize(); err != nil {
		t.Fatalf("stream synchronize: %v", err)
	}

	for i := range ref {
		if !approxEqual(ref[i], hostSlice[i], 1e-4) {
			t.Fatalf("softmax mismatch at %d: got %v want %v", i, hostSlice[i], ref[i])
		}
	}
}

func TestQuantMatVecQ4F32(t *testing.T) {
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

	const (
		rows         = 3
		cols         = 40
		blocksPerRow = 2
	)
	totalBlocks := rows * blocksPerRow
	qData := make([]byte, totalBlocks*16)
	qDecoded := make([]int8, totalBlocks*32)
	for b := 0; b < totalBlocks; b++ {
		var block [32]int8
		for i := range 32 {
			v := int8((i+b)%15 - 7)
			block[i] = v
			qDecoded[b*32+i] = v
		}
		packQ4Block(qData[b*16:(b+1)*16], block[:])
	}
	scaleRaw := make([]uint16, totalBlocks)
	for i := range scaleRaw {
		scaleRaw[i] = 0x3c00 // fp16(1.0)
	}
	xHost := make([]float32, cols)
	for i := range xHost {
		xHost[i] = float32((i%11)-5) * 0.25
	}
	want := refQuantMatVecQ4(qDecoded, xHost, rows, blocksPerRow, cols)
	got := make([]float32, rows)

	qDev, err := AllocDevice(int64(len(qData)))
	if err != nil {
		t.Fatalf("AllocDevice q: %v", err)
	}
	defer qDev.Free()
	scalesDev, err := AllocDevice(int64(len(scaleRaw) * 2))
	if err != nil {
		t.Fatalf("AllocDevice scales: %v", err)
	}
	defer scalesDev.Free()
	xDev, err := AllocDevice(int64(len(xHost) * 4))
	if err != nil {
		t.Fatalf("AllocDevice x: %v", err)
	}
	defer xDev.Free()
	yDev, err := AllocDevice(int64(len(got) * 4))
	if err != nil {
		t.Fatalf("AllocDevice y: %v", err)
	}
	defer yDev.Free()

	if err := MemcpyH2D(qDev, unsafe.Pointer(&qData[0]), int64(len(qData))); err != nil {
		t.Fatalf("MemcpyH2D q: %v", err)
	}
	if err := MemcpyH2D(scalesDev, unsafe.Pointer(&scaleRaw[0]), int64(len(scaleRaw)*2)); err != nil {
		t.Fatalf("MemcpyH2D scales: %v", err)
	}
	if err := MemcpyH2D(xDev, unsafe.Pointer(&xHost[0]), int64(len(xHost)*4)); err != nil {
		t.Fatalf("MemcpyH2D x: %v", err)
	}
	if err := QuantMatVecQ4F32(qDev, scalesDev, xDev, yDev, rows, blocksPerRow, cols, stream); err != nil {
		t.Fatalf("QuantMatVecQ4F32: %v", err)
	}
	if err := MemcpyD2H(unsafe.Pointer(&got[0]), yDev, int64(len(got)*4)); err != nil {
		t.Fatalf("MemcpyD2H y: %v", err)
	}

	for i := range want {
		if !approxEqual(want[i], got[i], 1e-4) {
			t.Fatalf("q4 matvec mismatch at %d: got %v want %v", i, got[i], want[i])
		}
	}
}

func TestQuantMatVecK4F32(t *testing.T) {
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

	const (
		rows         = 2
		cols         = 96
		blocksPerRow = 3
	)
	superBlocksPerRow := (blocksPerRow + 7) / 8
	totalBlocks := rows * blocksPerRow
	totalSuper := rows * superBlocksPerRow
	qData := make([]byte, totalBlocks*16)
	qDecoded := make([]int8, totalBlocks*32)
	for b := 0; b < totalBlocks; b++ {
		var block [32]int8
		for i := range 32 {
			v := int8((i+b)%13 - 6)
			block[i] = v
			qDecoded[b*32+i] = v
		}
		packQ4Block(qData[b*16:(b+1)*16], block[:])
	}
	superRaw := make([]uint16, totalSuper)
	for i := range superRaw {
		superRaw[i] = 0x3c00 // fp16(1.0)
	}
	subScales := make([]byte, totalBlocks)
	for i := range subScales {
		subScales[i] = 32 // scale factor == 1.0
	}
	xHost := make([]float32, cols)
	for i := range xHost {
		xHost[i] = float32((i%17)-8) * 0.125
	}
	want := refQuantMatVecQ4(qDecoded, xHost, rows, blocksPerRow, cols)
	got := make([]float32, rows)

	qDev, err := AllocDevice(int64(len(qData)))
	if err != nil {
		t.Fatalf("AllocDevice q: %v", err)
	}
	defer qDev.Free()
	superDev, err := AllocDevice(int64(len(superRaw) * 2))
	if err != nil {
		t.Fatalf("AllocDevice super scales: %v", err)
	}
	defer superDev.Free()
	subDev, err := AllocDevice(int64(len(subScales)))
	if err != nil {
		t.Fatalf("AllocDevice sub scales: %v", err)
	}
	defer subDev.Free()
	xDev, err := AllocDevice(int64(len(xHost) * 4))
	if err != nil {
		t.Fatalf("AllocDevice x: %v", err)
	}
	defer xDev.Free()
	yDev, err := AllocDevice(int64(len(got) * 4))
	if err != nil {
		t.Fatalf("AllocDevice y: %v", err)
	}
	defer yDev.Free()

	if err := MemcpyH2D(qDev, unsafe.Pointer(&qData[0]), int64(len(qData))); err != nil {
		t.Fatalf("MemcpyH2D q: %v", err)
	}
	if err := MemcpyH2D(superDev, unsafe.Pointer(&superRaw[0]), int64(len(superRaw)*2)); err != nil {
		t.Fatalf("MemcpyH2D super scales: %v", err)
	}
	if err := MemcpyH2D(subDev, unsafe.Pointer(&subScales[0]), int64(len(subScales))); err != nil {
		t.Fatalf("MemcpyH2D sub scales: %v", err)
	}
	if err := MemcpyH2D(xDev, unsafe.Pointer(&xHost[0]), int64(len(xHost)*4)); err != nil {
		t.Fatalf("MemcpyH2D x: %v", err)
	}
	if err := QuantMatVecK4F32(qDev, superDev, subDev, xDev, yDev, rows, blocksPerRow, cols, stream); err != nil {
		t.Fatalf("QuantMatVecK4F32: %v", err)
	}
	if err := MemcpyD2H(unsafe.Pointer(&got[0]), yDev, int64(len(got)*4)); err != nil {
		t.Fatalf("MemcpyD2H y: %v", err)
	}

	for i := range want {
		if !approxEqual(want[i], got[i], 1e-4) {
			t.Fatalf("k4 matvec mismatch at %d: got %v want %v", i, got[i], want[i])
		}
	}
}

func TestDequantizeQ4ToF16(t *testing.T) {
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
		rows         = 2
		cols         = 40
		blocksPerRow = 2
	)
	totalBlocks := rows * blocksPerRow
	qData := make([]byte, totalBlocks*16)
	scalesRaw := make([]uint16, totalBlocks)
	for i := range scalesRaw {
		scalesRaw[i] = 0x3c00 // 1.0
	}
	ref := make([]float32, rows*cols)
	for b := 0; b < totalBlocks; b++ {
		var block [32]int8
		for i := range 32 {
			v := int8((i+b)%15 - 7)
			block[i] = v
		}
		packQ4Block(qData[b*16:(b+1)*16], block[:])
		row := b / blocksPerRow
		blockInRow := b % blocksPerRow
		colBase := blockInRow * 32
		for i := range 32 {
			col := colBase + i
			if col < cols {
				ref[row*cols+col] = float32(block[i])
			}
		}
	}
	outRaw := make([]uint16, rows*cols)

	qDev, err := AllocDevice(int64(len(qData)))
	if err != nil {
		t.Fatalf("AllocDevice q: %v", err)
	}
	defer qDev.Free()
	sDev, err := AllocDevice(int64(len(scalesRaw) * 2))
	if err != nil {
		t.Fatalf("AllocDevice scales: %v", err)
	}
	defer sDev.Free()
	outDev, err := AllocDevice(int64(len(outRaw) * 2))
	if err != nil {
		t.Fatalf("AllocDevice out: %v", err)
	}
	defer outDev.Free()

	if err := MemcpyH2D(qDev, unsafe.Pointer(&qData[0]), int64(len(qData))); err != nil {
		t.Fatalf("MemcpyH2D q: %v", err)
	}
	if err := MemcpyH2D(sDev, unsafe.Pointer(&scalesRaw[0]), int64(len(scalesRaw)*2)); err != nil {
		t.Fatalf("MemcpyH2D scales: %v", err)
	}
	if err := DequantizeQ4ToF16(qDev, sDev, outDev, rows, blocksPerRow, cols, stream); err != nil {
		t.Fatalf("DequantizeQ4ToF16: %v", err)
	}
	if err := MemcpyD2H(unsafe.Pointer(&outRaw[0]), outDev, int64(len(outRaw)*2)); err != nil {
		t.Fatalf("MemcpyD2H out: %v", err)
	}

	for i := range ref {
		got := fp16ToF32(outRaw[i])
		if !approxEqual(got, ref[i], 1e-3) {
			t.Fatalf("dequant q4 mismatch at %d: got %v want %v", i, got, ref[i])
		}
	}
}

func TestDequantizeK4ToF16(t *testing.T) {
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
		rows         = 2
		cols         = 64
		blocksPerRow = 2
	)
	totalBlocks := rows * blocksPerRow
	superBlocksPerRow := (blocksPerRow + 7) / 8
	totalSuper := rows * superBlocksPerRow
	qData := make([]byte, totalBlocks*16)
	superRaw := make([]uint16, totalSuper)
	for i := range superRaw {
		superRaw[i] = 0x3c00 // 1.0
	}
	subRaw := make([]byte, totalBlocks)
	for i := range subRaw {
		subRaw[i] = 32 // factor 1.0
	}
	ref := make([]float32, rows*cols)
	for b := 0; b < totalBlocks; b++ {
		var block [32]int8
		for i := range 32 {
			v := int8((i+b)%13 - 6)
			block[i] = v
		}
		packQ4Block(qData[b*16:(b+1)*16], block[:])
		row := b / blocksPerRow
		blockInRow := b % blocksPerRow
		colBase := blockInRow * 32
		for i := range 32 {
			col := colBase + i
			if col < cols {
				ref[row*cols+col] = float32(block[i])
			}
		}
	}
	outRaw := make([]uint16, rows*cols)

	qDev, err := AllocDevice(int64(len(qData)))
	if err != nil {
		t.Fatalf("AllocDevice q: %v", err)
	}
	defer qDev.Free()
	superDev, err := AllocDevice(int64(len(superRaw) * 2))
	if err != nil {
		t.Fatalf("AllocDevice super: %v", err)
	}
	defer superDev.Free()
	subDev, err := AllocDevice(int64(len(subRaw)))
	if err != nil {
		t.Fatalf("AllocDevice sub: %v", err)
	}
	defer subDev.Free()
	outDev, err := AllocDevice(int64(len(outRaw) * 2))
	if err != nil {
		t.Fatalf("AllocDevice out: %v", err)
	}
	defer outDev.Free()

	if err := MemcpyH2D(qDev, unsafe.Pointer(&qData[0]), int64(len(qData))); err != nil {
		t.Fatalf("MemcpyH2D q: %v", err)
	}
	if err := MemcpyH2D(superDev, unsafe.Pointer(&superRaw[0]), int64(len(superRaw)*2)); err != nil {
		t.Fatalf("MemcpyH2D super: %v", err)
	}
	if err := MemcpyH2D(subDev, unsafe.Pointer(&subRaw[0]), int64(len(subRaw))); err != nil {
		t.Fatalf("MemcpyH2D sub: %v", err)
	}
	if err := DequantizeK4ToF16(qDev, superDev, subDev, outDev, rows, blocksPerRow, cols, stream); err != nil {
		t.Fatalf("DequantizeK4ToF16: %v", err)
	}
	if err := MemcpyD2H(unsafe.Pointer(&outRaw[0]), outDev, int64(len(outRaw)*2)); err != nil {
		t.Fatalf("MemcpyD2H out: %v", err)
	}

	for i := range ref {
		got := fp16ToF32(outRaw[i])
		if !approxEqual(got, ref[i], 1e-3) {
			t.Fatalf("dequant k4 mismatch at %d: got %v want %v", i, got, ref[i])
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

func packQ4Block(dst []byte, values []int8) {
	for i := range 16 {
		lo := encodeQ4Nibble(values[2*i])
		hi := encodeQ4Nibble(values[2*i+1])
		dst[i] = lo | (hi << 4)
	}
}

func encodeQ4Nibble(v int8) byte {
	if v >= 0 {
		return byte(v)
	}
	return byte(int(v) + 16)
}

func refQuantMatVecQ4(qDecoded []int8, x []float32, rows, blocksPerRow, cols int) []float32 {
	out := make([]float32, rows)
	for r := range rows {
		var sum float32
		rowBase := r * blocksPerRow
		for b := range blocksPerRow {
			colBase := b * 32
			n := cols - colBase
			if n <= 0 {
				break
			}
			if n > 32 {
				n = 32
			}
			qb := qDecoded[(rowBase+b)*32 : (rowBase+b+1)*32]
			for i := range n {
				sum += float32(qb[i]) * x[colBase+i]
			}
		}
		out[r] = sum
	}
	return out
}

func fp16ToF32(h uint16) float32 {
	sign := uint32(h>>15) & 0x1
	exp := int32((h >> 10) & 0x1f)
	frac := uint32(h & 0x03ff)
	switch exp {
	case 0:
		if frac == 0 {
			return math.Float32frombits(sign << 31)
		}
		for (frac & 0x0400) == 0 {
			frac <<= 1
			exp -= 1
		}
		exp += 1
		frac &= 0x03ff
	case 0x1f:
		return math.Float32frombits((sign << 31) | 0x7f800000 | (frac << 13))
	}
	exp = exp + (127 - 15)
	return math.Float32frombits((sign << 31) | (uint32(exp) << 23) | (frac << 13))
}

func softmaxRef(x []float32) {
	if len(x) == 0 {
		return
	}
	maxv := x[0]
	for i := 1; i < len(x); i++ {
		if x[i] > maxv {
			maxv = x[i]
		}
	}
	var sum float64
	for i := range x {
		v := math.Exp(float64(x[i] - maxv))
		x[i] = float32(v)
		sum += v
	}
	if sum == 0 {
		return
	}
	inv := float32(1.0 / sum)
	for i := range x {
		x[i] *= inv
	}
}
