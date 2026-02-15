package simd

import (
	"testing"

	"github.com/samcharles93/mantle/pkg/mcf"
)

func TestMatVecQ4MatchesDequant(t *testing.T) {
	const (
		rows  = 3
		cols  = 64
		scale = 0.05
	)
	qvals := makeQVals(rows * cols)
	payload := buildQPayload(rows, cols, 4, scale, qvals)
	verifyPayloadSize(t, payload, rows, cols, mcf.DTypeQ4)

	x := make([]float32, cols)
	for i := range x {
		x[i] = float32((i%11)-5) * 0.2
	}
	scaleUsed := Float16ToFloat32(Float32ToFloat16(scale))
	want := matVecExpected(rows, cols, scaleUsed, qvals, x)

	w := Mat{R: rows, C: cols, Stride: cols, DType: mcf.DTypeQ4, Raw: payload}
	got := make([]float32, rows)
	MatVec(got, &w, x)

	assertCloseSlice(t, got, want, 1e-4)
}

func TestMatVecQ4CachedMatchesDequant(t *testing.T) {
	const (
		rows  = 3
		cols  = 64
		scale = 0.05
	)
	qvals := makeQVals(rows * cols)
	payload := buildQPayload(rows, cols, 4, scale, qvals)
	verifyPayloadSize(t, payload, rows, cols, mcf.DTypeQ4)

	x := make([]float32, cols)
	for i := range x {
		x[i] = float32((i%11)-5) * 0.2
	}
	scaleUsed := Float16ToFloat32(Float32ToFloat16(scale))
	want := matVecExpected(rows, cols, scaleUsed, qvals, x)

	w := Mat{R: rows, C: cols, Stride: cols, DType: mcf.DTypeQ4, Raw: payload}
	cache, err := BuildQuantCache(&w)
	if err != nil {
		t.Fatalf("BuildQuantCache: %v", err)
	}
	w.Quant = cache

	got := make([]float32, rows)
	MatVec(got, &w, x)

	assertCloseSlice(t, got, want, 1e-4)
}

func TestMatVecQ8MatchesDequant(t *testing.T) {
	const (
		rows  = 4
		cols  = 64
		scale = 0.015
	)
	qvals := makeQVals(rows * cols)
	payload := buildQPayload(rows, cols, 8, scale, qvals)
	verifyPayloadSize(t, payload, rows, cols, mcf.DTypeQ8)

	x := make([]float32, cols)
	for i := range x {
		x[i] = float32((i%9)-4) * 0.15
	}
	scaleUsed := Float16ToFloat32(Float32ToFloat16(scale))
	want := matVecExpected(rows, cols, scaleUsed, qvals, x)

	w := Mat{R: rows, C: cols, Stride: cols, DType: mcf.DTypeQ8, Raw: payload}
	got := make([]float32, rows)
	MatVec(got, &w, x)

	assertCloseSlice(t, got, want, 1e-4)
}

func TestMatVecK4MatchesDequant(t *testing.T) {
	const (
		rows  = 2
		cols  = 64
		scale = 0.075
	)
	qvals := makeQVals(rows * cols)
	payload := buildKPayload(rows, cols, 4, scale, qvals)
	verifyPayloadSize(t, payload, rows, cols, mcf.DTypeK4)

	x := make([]float32, cols)
	for i := range x {
		x[i] = float32((i%7)-3) * 0.3
	}
	scaleUsed := Float16ToFloat32(Float32ToFloat16(scale))
	want := matVecExpected(rows, cols, scaleUsed, qvals, x)

	w := Mat{R: rows, C: cols, Stride: cols, DType: mcf.DTypeK4, Raw: payload}
	got := make([]float32, rows)
	MatVec(got, &w, x)

	assertCloseSlice(t, got, want, 1e-4)
}

func TestMatVecK4CachedMatchesDequant(t *testing.T) {
	const (
		rows  = 2
		cols  = 64
		scale = 0.075
	)
	qvals := makeQVals(rows * cols)
	payload := buildKPayload(rows, cols, 4, scale, qvals)
	verifyPayloadSize(t, payload, rows, cols, mcf.DTypeK4)

	x := make([]float32, cols)
	for i := range x {
		x[i] = float32((i%7)-3) * 0.3
	}
	scaleUsed := Float16ToFloat32(Float32ToFloat16(scale))
	want := matVecExpected(rows, cols, scaleUsed, qvals, x)

	w := Mat{R: rows, C: cols, Stride: cols, DType: mcf.DTypeK4, Raw: payload}
	cache, err := BuildQuantCache(&w)
	if err != nil {
		t.Fatalf("BuildQuantCache: %v", err)
	}
	w.Quant = cache

	got := make([]float32, rows)
	MatVec(got, &w, x)

	assertCloseSlice(t, got, want, 1e-4)
}

func TestMatVecK3MatchesDequant(t *testing.T) {
	const (
		rows  = 2
		cols  = 64
		scale = 0.05
	)
	qvals := makeKVals(3, rows*cols)
	payload := buildKPayload(rows, cols, 3, scale, qvals)
	verifyPayloadSize(t, payload, rows, cols, mcf.DTypeK3)

	x := make([]float32, cols)
	for i := range x {
		x[i] = float32((i%9)-4) * 0.2
	}
	scaleUsed := Float16ToFloat32(Float32ToFloat16(scale))
	want := matVecExpected(rows, cols, scaleUsed, qvals, x)

	w := Mat{R: rows, C: cols, Stride: cols, DType: mcf.DTypeK3, Raw: payload}
	got := make([]float32, rows)
	MatVec(got, &w, x)

	assertCloseSlice(t, got, want, 1e-4)
}

func TestMatVecK6MatchesDequant(t *testing.T) {
	const (
		rows  = 2
		cols  = 64
		scale = 0.03
	)
	qvals := makeKVals(6, rows*cols)
	payload := buildKPayload(rows, cols, 6, scale, qvals)
	verifyPayloadSize(t, payload, rows, cols, mcf.DTypeK6)

	x := make([]float32, cols)
	for i := range x {
		x[i] = float32((i%11)-5) * 0.12
	}
	scaleUsed := Float16ToFloat32(Float32ToFloat16(scale))
	want := matVecExpected(rows, cols, scaleUsed, qvals, x)

	w := Mat{R: rows, C: cols, Stride: cols, DType: mcf.DTypeK6, Raw: payload}
	got := make([]float32, rows)
	MatVec(got, &w, x)

	assertCloseSlice(t, got, want, 1e-4)
}

func TestRowToQuant(t *testing.T) {
	const (
		rows  = 2
		cols  = 64
		scale = 0.04
	)
	qvals := makeQVals(rows * cols)
	payload := buildQPayload(rows, cols, 4, scale, qvals)
	verifyPayloadSize(t, payload, rows, cols, mcf.DTypeQ4)

	w := Mat{R: rows, C: cols, Stride: cols, DType: mcf.DTypeQ4, Raw: payload}
	row := make([]float32, cols)
	w.RowTo(row, 1)

	scaleUsed := Float16ToFloat32(Float32ToFloat16(scale))
	for i := range cols {
		want := float32(qvals[cols+i]) * scaleUsed
		if diff := row[i] - want; diff < -1e-6 || diff > 1e-6 {
			t.Fatalf("row[%d]=%v want %v", i, row[i], want)
		}
	}
}

func BenchmarkMatVecQ4(b *testing.B) {
	const (
		rows  = 512
		cols  = 512
		scale = 0.05
	)
	qvals := makeQVals(rows * cols)
	payload := buildQPayload(rows, cols, 4, scale, qvals)
	w := Mat{R: rows, C: cols, Stride: cols, DType: mcf.DTypeQ4, Raw: payload}
	x := make([]float32, cols)
	for i := range x {
		x[i] = float32((i%13)-6) * 0.11
	}
	dst := make([]float32, rows)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MatVec(dst, &w, x)
	}
}

func BenchmarkMatVecK4(b *testing.B) {
	const (
		rows  = 512
		cols  = 512
		scale = 0.05
	)
	qvals := makeQVals(rows * cols)
	payload := buildK4Payload(rows, cols, scale, qvals)
	w := Mat{R: rows, C: cols, Stride: cols, DType: mcf.DTypeK4, Raw: payload}
	x := make([]float32, cols)
	for i := range x {
		x[i] = float32((i%13)-6) * 0.11
	}
	dst := make([]float32, rows)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MatVec(dst, &w, x)
	}
}

func BenchmarkMatVecK4Triple(b *testing.B) {
	const (
		rows  = 512
		cols  = 512
		scale = 0.05
	)
	qvals := makeQVals(rows * cols)
	payload := buildK4Payload(rows, cols, scale, qvals)
	w := Mat{R: rows, C: cols, Stride: cols, DType: mcf.DTypeK4, Raw: payload}
	x := make([]float32, cols)
	for i := range x {
		x[i] = float32((i%13)-6) * 0.11
	}
	dst0 := make([]float32, rows)
	dst1 := make([]float32, rows)
	dst2 := make([]float32, rows)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MatVec(dst0, &w, x)
		MatVec(dst1, &w, x)
		MatVec(dst2, &w, x)
	}
}

func BenchmarkMatVecK4TripleSharedQx(b *testing.B) {
	const (
		rows  = 512
		cols  = 512
		scale = 0.05
	)
	qvals := makeQVals(rows * cols)
	payload := buildK4Payload(rows, cols, scale, qvals)
	w := Mat{R: rows, C: cols, Stride: cols, DType: mcf.DTypeK4, Raw: payload}
	x := make([]float32, cols)
	for i := range x {
		x[i] = float32((i%13)-6) * 0.11
	}
	dst0 := make([]float32, rows)
	dst1 := make([]float32, rows)
	dst2 := make([]float32, rows)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		qx := PrepareQuantVec(x)
		MatVecWithQuant(dst0, &w, x, qx)
		MatVecWithQuant(dst1, &w, x, qx)
		MatVecWithQuant(dst2, &w, x, qx)
		ReleaseQuantVec(qx)
	}
}

func BenchmarkMatVecK3(b *testing.B) {
	const (
		rows  = 512
		cols  = 512
		scale = 0.05
	)
	qvals := makeKVals(3, rows*cols)
	payload := buildKPayload(rows, cols, 3, scale, qvals)
	w := Mat{R: rows, C: cols, Stride: cols, DType: mcf.DTypeK3, Raw: payload}
	x := make([]float32, cols)
	for i := range x {
		x[i] = float32((i%13)-6) * 0.11
	}
	dst := make([]float32, rows)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MatVec(dst, &w, x)
	}
}

func BenchmarkMatVecK6(b *testing.B) {
	const (
		rows  = 512
		cols  = 512
		scale = 0.05
	)
	qvals := makeKVals(6, rows*cols)
	payload := buildKPayload(rows, cols, 6, scale, qvals)
	w := Mat{R: rows, C: cols, Stride: cols, DType: mcf.DTypeK6, Raw: payload}
	x := make([]float32, cols)
	for i := range x {
		x[i] = float32((i%13)-6) * 0.11
	}
	dst := make([]float32, rows)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MatVec(dst, &w, x)
	}
}

func matVecExpected(rows, cols int, scale float32, qvals []int8, x []float32) []float32 {
	if cpu.HasAVX2 {
		return matVecExpectedInt8(rows, cols, scale, qvals, x)
	}
	return matVecExpectedFloat(rows, cols, scale, qvals, x)
}

func matVecExpectedFloat(rows, cols int, scale float32, qvals []int8, x []float32) []float32 {
	out := make([]float32, rows)
	for r := range rows {
		var sum float32
		base := r * cols
		for c := range cols {
			sum += float32(qvals[base+c]) * scale * x[c]
		}
		out[r] = sum
	}
	return out
}

func matVecExpectedInt8(rows, cols int, scale float32, qvals []int8, x []float32) []float32 {
	blocksPerRow := (cols + 31) / 32
	qx8 := make([]int8, blocksPerRow*32)
	qx16 := make([]int16, blocksPerRow*32)
	xScales := make([]float32, blocksPerRow)
	blockSums := make([]int32, blocksPerRow)
	quantizeVecBlocksInto(x, blocksPerRow, qx8, qx16, xScales, blockSums)

	out := make([]float32, rows)
	for r := range rows {
		var sum float32
		rowBase := r * cols
		for b := range blocksPerRow {
			xScale := xScales[b]
			if xScale == 0 {
				continue
			}
			wBlock := qvals[rowBase+b*32 : rowBase+b*32+32]
			if cpu.HasAVXVNNI {
				// VPDPBUSD path with offset correction
				var dot int32
				for i := range 32 {
					u8w := uint8(wBlock[i]) ^ 0x80
					dot += int32(u8w) * int32(qx8[b*32+i])
				}
				corrected := dot - 128*blockSums[b]
				sum += float32(corrected) * (scale * xScale)
			} else {
				xBlock := qx16[b*32 : b*32+32]
				dot := dotInt8Int16Scalar(wBlock, xBlock, 32)
				sum += float32(dot) * (scale * xScale)
			}
		}
		out[r] = sum
	}
	return out
}

func assertCloseSlice(t *testing.T, got, want []float32, tol float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length mismatch: %d != %d", len(got), len(want))
	}
	for i := range got {
		diff := got[i] - want[i]
		if diff < -tol || diff > tol {
			t.Fatalf("idx %d: got %v want %v", i, got[i], want[i])
		}
	}
}

func makeQVals(n int) []int8 {
	out := make([]int8, n)
	for i := range out {
		out[i] = int8((i % 15) - 7)
	}
	return out
}

func makeKVals(bits, n int) []int8 {
	if bits <= 1 {
		return make([]int8, n)
	}
	min := -1 << (bits - 1)
	max := (1 << (bits - 1)) - 1
	rangeSize := max - min + 1
	out := make([]int8, n)
	for i := range out {
		out[i] = int8(min + (i % rangeSize))
	}
	return out
}

func buildQPayload(rows, cols, bits int, scale float32, qvals []int8) []byte {
	blocksPerRow := (cols + 31) / 32
	totalBlocks := rows * blocksPerRow
	scales := make([]byte, totalBlocks*2)
	f16 := Float32ToFloat16(scale)
	for i := range totalBlocks {
		scales[i*2] = byte(f16)
		scales[i*2+1] = byte(f16 >> 8)
	}
	buf := append([]byte{}, scales...)
	buf = align64(buf)

	blockBytes := (32 * bits) / 8
	data := make([]byte, totalBlocks*blockBytes)
	if bits == 8 {
		for i, v := range qvals {
			data[i] = byte(v)
		}
	} else {
		for i := 0; i < rows*cols; i += 2 {
			u0 := uint8(qvals[i]) & 0x0F
			u1 := uint8(qvals[i+1]) & 0x0F
			data[i/2] = u0 | (u1 << 4)
		}
	}
	buf = append(buf, data...)
	return buf
}

func buildK4Payload(rows, cols int, scale float32, qvals []int8) []byte {
	return buildKPayload(rows, cols, 4, scale, qvals)
}

func buildKPayload(rows, cols, bits int, scale float32, qvals []int8) []byte {
	blocksPerRow := (cols + 31) / 32
	superBlocksPerRow := (blocksPerRow + 7) / 8
	totalBlocks := rows * blocksPerRow
	totalSuper := rows * superBlocksPerRow

	superScales := make([]byte, totalSuper*2)
	f16 := Float32ToFloat16(scale)
	for i := range totalSuper {
		superScales[i*2] = byte(f16)
		superScales[i*2+1] = byte(f16 >> 8)
	}
	buf := append([]byte{}, superScales...)
	buf = align64(buf)

	subScales := make([]byte, totalBlocks)
	for i := range subScales {
		subScales[i] = 32
	}
	buf = append(buf, subScales...)
	buf = align64(buf)

	blockBytes := (32 * bits) / 8
	data := make([]byte, totalBlocks*blockBytes)
	padded := make([]int8, totalBlocks*32)
	copy(padded, qvals)
	for block := range totalBlocks {
		start := block * 32
		end := start + 32
		segment := padded[start:end]
		startOff := block * blockBytes
		packQuantBlock(data[startOff:startOff+blockBytes], bits, segment)
	}
	buf = append(buf, data...)
	return buf
}

func packQuantBlock(dst []byte, bits int, values []int8) {
	mask := uint64((1 << bits) - 1)
	var bitBuf uint64
	var bitCount uint
	dstIdx := 0
	for i := range 32 {
		val := uint64(uint8(values[i])) & mask
		bitBuf |= val << bitCount
		bitCount += uint(bits)
		for bitCount >= 8 && dstIdx < len(dst) {
			dst[dstIdx] = byte(bitBuf)
			dstIdx++
			bitBuf >>= 8
			bitCount -= 8
		}
	}
	for bitCount >= 8 && dstIdx < len(dst) {
		dst[dstIdx] = byte(bitBuf)
		dstIdx++
		bitBuf >>= 8
		bitCount -= 8
	}
	if bitCount > 0 && dstIdx < len(dst) {
		dst[dstIdx] = byte(bitBuf)
		dstIdx++
	}
	for dstIdx < len(dst) {
		dst[dstIdx] = 0
		dstIdx++
	}
}

func align64(buf []byte) []byte {
	rem := len(buf) % 64
	if rem == 0 {
		return buf
	}
	return append(buf, make([]byte, 64-rem)...)
}

func verifyPayloadSize(t *testing.T, payload []byte, rows, cols int, dt mcf.TensorDType) {
	t.Helper()
	want, err := mcf.QuantPayloadSize([]uint64{uint64(rows), uint64(cols)}, dt)
	if err != nil {
		t.Fatalf("QuantPayloadSize: %v", err)
	}
	if uint64(len(payload)) != want {
		t.Fatalf("payload size mismatch: got %d want %d", len(payload), want)
	}
}

// TestDotUint8Int8MatchesScalar verifies VPDPBUSD SIMD matches scalar reference
func TestDotUint8Int8MatchesScalar(t *testing.T) {
	const n = 128

	weights := make([]uint8, n)
	activations := make([]int8, n)

	// Fill with deterministic test data
	for i := 0; i < n; i++ {
		weights[i] = uint8((i % 200) + 1)  // [1, 200]
		activations[i] = int8((i % 15) - 7) // [-7, 7]
	}

	want := dotUint8Int8Scalar(weights, activations, n)
	got := dotUint8Int8(weights, activations, n)

	if got != want {
		t.Fatalf("SIMD mismatch: got %d want %d", got, want)
	}
}

// TestConvertInt8ToUint8 verifies int8→uint8 conversion correctness
func TestConvertInt8ToUint8(t *testing.T) {
	src := []int8{-127, -1, 0, 1, 127}
	want := []uint8{1, 127, 128, 129, 255}

	qx := getQuantVec(1)
	defer putQuantVec(qx)

	got := convertInt8ToUint8(qx, src, len(src))

	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("idx %d: got %d want %d", i, got[i], want[i])
		}
	}
}

// TestOffsetCorrection verifies that the VPDPBUSD offset correction formula
// produces the same result as direct int8×int8 dot product.
func TestOffsetCorrection(t *testing.T) {
	const n = 32
	// Test with various weight/activation patterns
	testCases := []struct {
		name string
		w    [n]int8
		a    [n]int8
	}{
		{"positive", func() [n]int8 {
			var v [n]int8
			for i := range v {
				v[i] = int8(i%7 + 1)
			}
			return v
		}(), func() [n]int8 {
			var v [n]int8
			for i := range v {
				v[i] = int8(i%5 + 1)
			}
			return v
		}()},
		{"mixed", func() [n]int8 {
			var v [n]int8
			for i := range v {
				v[i] = int8((i % 15) - 7)
			}
			return v
		}(), func() [n]int8 {
			var v [n]int8
			for i := range v {
				v[i] = int8((i % 11) - 5)
			}
			return v
		}()},
		{"extremes", func() [n]int8 {
			var v [n]int8
			for i := range v {
				if i%2 == 0 {
					v[i] = 127
				} else {
					v[i] = -127
				}
			}
			return v
		}(), func() [n]int8 {
			var v [n]int8
			for i := range v {
				if i%3 == 0 {
					v[i] = 127
				} else {
					v[i] = -127
				}
			}
			return v
		}()},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Reference: direct int8×int8 dot product
			var wantDot int32
			for i := range n {
				wantDot += int32(tc.w[i]) * int32(tc.a[i])
			}

			// VPDPBUSD with offset correction
			var u8w [n]uint8
			for i := range n {
				u8w[i] = uint8(tc.w[i]) ^ 0x80
			}
			vpdpbusdResult := dotUint8Int8Scalar(u8w[:], tc.a[:], n)

			var aSum int32
			for i := range n {
				aSum += int32(tc.a[i])
			}
			gotDot := vpdpbusdResult - 128*aSum

			if gotDot != wantDot {
				t.Fatalf("offset correction mismatch: got %d want %d (vpdpbusd=%d, aSum=%d)", gotDot, wantDot, vpdpbusdResult, aSum)
			}
		})
	}
}

// TestMatVecQ4CachedVPDPBUSD verifies that the cached path works correctly
// with VPDPBUSD offset correction on AVX-VNNI hardware.
func TestMatVecQ4CachedVPDPBUSD(t *testing.T) {
	const (
		rows  = 4
		cols  = 128
		scale = 0.05
	)

	qvals := makeQVals(rows * cols)
	payload := buildQPayload(rows, cols, 4, scale, qvals)

	x := make([]float32, cols)
	for i := range x {
		x[i] = float32((i%11)-5) * 0.2
	}

	scaleUsed := Float16ToFloat32(Float32ToFloat16(scale))
	want := matVecExpected(rows, cols, scaleUsed, qvals, x)

	// Build cache and run with cached quantized data
	w := Mat{R: rows, C: cols, Stride: cols, DType: mcf.DTypeQ4, Raw: payload}
	cache, err := BuildQuantCache(&w)
	if err != nil {
		t.Fatalf("BuildQuantCache: %v", err)
	}
	w.Quant = cache

	got := make([]float32, rows)
	MatVec(got, &w, x)

	// Verify results match expected values within tolerance
	assertCloseSlice(t, got, want, 1e-4)
}

// TestMatVecK4CachedVPDPBUSD verifies K4 cached path works correctly
// with VPDPBUSD offset correction on AVX-VNNI hardware.
func TestMatVecK4CachedVPDPBUSD(t *testing.T) {
	const (
		rows  = 2
		cols  = 128
		scale = 0.075
	)

	qvals := makeQVals(rows * cols)
	payload := buildK4Payload(rows, cols, scale, qvals)

	x := make([]float32, cols)
	for i := range x {
		x[i] = float32((i%7)-3) * 0.3
	}

	scaleUsed := Float16ToFloat32(Float32ToFloat16(scale))
	want := matVecExpected(rows, cols, scaleUsed, qvals, x)

	// Build cache and run
	w := Mat{R: rows, C: cols, Stride: cols, DType: mcf.DTypeK4, Raw: payload}
	cache, err := BuildQuantCache(&w)
	if err != nil {
		t.Fatalf("BuildQuantCache: %v", err)
	}
	w.Quant = cache

	got := make([]float32, rows)
	MatVec(got, &w, x)

	assertCloseSlice(t, got, want, 1e-4)
}

// BenchmarkDotUint8Int8SIMD benchmarks the VPDPBUSD dot product
func BenchmarkDotUint8Int8SIMD(b *testing.B) {
	const n = 512

	weights := make([]uint8, n)
	activations := make([]int8, n)

	for i := 0; i < n; i++ {
		weights[i] = uint8(i % 255)
		activations[i] = int8((i % 15) - 7)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = dotUint8Int8SIMD(weights, activations, n)
	}
}

// BenchmarkDotInt8Int16SIMD benchmarks the existing VPMADDWD path
func BenchmarkDotInt8Int16SIMD(b *testing.B) {
	const n = 512

	weights := make([]int8, n)
	weightsI16 := make([]int16, n)

	for i := 0; i < n; i++ {
		weights[i] = int8((i % 15) - 7)
		weightsI16[i] = int16(weights[i])
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = dotInt8Int16SIMD(weights, weightsI16, n)
	}
}

// BenchmarkConvertInt8ToUint8 benchmarks the int8→uint8 conversion
func BenchmarkConvertInt8ToUint8(b *testing.B) {
	const n = 512

	weights := make([]int8, n)
	for i := 0; i < n; i++ {
		weights[i] = int8((i % 15) - 7)
	}

	qx := getQuantVec(1)
	defer putQuantVec(qx)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = convertInt8ToUint8(qx, weights, n)
	}
}
