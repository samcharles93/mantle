package simd

import (
	"math"
	"simd/archsimd"
	"testing"
)

// Benchmark comparison between different SIMD implementations
func BenchmarkDotProductScalar(b *testing.B) {
	size := 1024
	a := make([]float32, size)
	bv := make([]float32, size)

	for i := range size {
		a[i] = float32(i)
		bv[i] = float32(i * 2)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = dotScalar(a, bv)
	}
}

func BenchmarkDotProductAVX2(b *testing.B) {
	if !cpu.HasAVX2 {
		b.Skip("AVX2 not available")
	}

	size := 1024
	a := make([]float32, size)
	bv := make([]float32, size)

	for i := range size {
		a[i] = float32(i)
		bv[i] = float32(i * 2)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = dotSIMD(a, bv)
	}
}

func BenchmarkDotProductAVX512(b *testing.B) {
	if !archsimd.X86.AVX512() {
		b.Skip("AVX-512 not available")
	}

	size := 1024
	a := make([]float32, size)
	bv := make([]float32, size)

	for i := range size {
		a[i] = float32(i)
		bv[i] = float32(i * 2)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = dotAVX512(a, bv)
	}
}

// Benchmark comparison for Add operations
func BenchmarkAddScalar(b *testing.B) {
	size := 1024
	dst := make([]float32, size)
	src := make([]float32, size)

	for i := range size {
		dst[i] = float32(i)
		src[i] = float32(i * 2)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		addScalar(dst, src)
	}
}

func BenchmarkAddAVX2(b *testing.B) {
	if !cpu.HasAVX2 {
		b.Skip("AVX2 not available")
	}

	size := 1024
	dst := make([]float32, size)
	src := make([]float32, size)

	for i := range size {
		dst[i] = float32(i)
		src[i] = float32(i * 2)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		addSIMD(dst, src)
	}
}

func BenchmarkAddAVX512(b *testing.B) {
	if !archsimd.X86.AVX512() {
		b.Skip("AVX-512 not available")
	}

	size := 1024
	dst := make([]float32, size)
	src := make([]float32, size)

	for i := range size {
		dst[i] = float32(i)
		src[i] = float32(i * 2)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		addAVX512(dst, src)
	}
}

// Benchmark comparison for RMSNorm operations
func BenchmarkRMSNormScalar(b *testing.B) {
	size := 512
	dst := make([]float32, size)
	src := make([]float32, size)
	weight := make([]float32, size)
	eps := float32(1e-5)

	for i := range size {
		src[i] = float32(i) * 0.1
		weight[i] = float32(i%10) * 0.1
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rmsNormScalar(dst, src, weight, eps)
	}
}

func BenchmarkRMSNormAVX2(b *testing.B) {
	if !cpu.HasAVX2 {
		b.Skip("AVX2 not available")
	}

	size := 512
	dst := make([]float32, size)
	src := make([]float32, size)
	weight := make([]float32, size)
	eps := float32(1e-5)

	for i := range size {
		src[i] = float32(i) * 0.1
		weight[i] = float32(i%10) * 0.1
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rmsNormSIMD(dst, src, weight, eps)
	}
}

func BenchmarkRMSNormAVX512(b *testing.B) {
	if !archsimd.X86.AVX512() {
		b.Skip("AVX-512 not available")
	}

	size := 512
	dst := make([]float32, size)
	src := make([]float32, size)
	weight := make([]float32, size)
	eps := float32(1e-5)

	for i := range size {
		src[i] = float32(i) * 0.1
		weight[i] = float32(i%10) * 0.1
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rmsNormAVX512(dst, src, weight, eps)
	}
}

// Benchmark RoPE operations
func BenchmarkApplyRoPEScalar(b *testing.B) {
	nHead := 8
	headDim := 64
	size := nHead * headDim
	x := make([]float32, size)
	invFreq := make([]float64, headDim/2)
	attentionFactor := float32(1.0)
	pos := 10

	for i := range x {
		x[i] = float32(i%20) * 0.1
	}
	for i := range invFreq {
		invFreq[i] = float64(i+1) * 0.01
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		applyRoPEScalar(x, nHead, headDim, pos, invFreq, attentionFactor, headDim/2)
	}
}

func BenchmarkApplyRoPEAVX2(b *testing.B) {
	if !cpu.HasAVX2 {
		b.Skip("AVX2 not available")
	}

	nHead := 8
	headDim := 64
	size := nHead * headDim
	x := make([]float32, size)
	invFreq := make([]float64, headDim/2)
	attentionFactor := float32(1.0)
	pos := 10

	for i := range x {
		x[i] = float32(i%20) * 0.1
	}
	for i := range invFreq {
		invFreq[i] = float64(i+1) * 0.01
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		applyRoPESIMD(x, nHead, headDim, pos, invFreq, attentionFactor, headDim/2)
	}
}

func BenchmarkApplyRoPEAVX512(b *testing.B) {
	if !archsimd.X86.AVX512() {
		b.Skip("AVX-512 not available")
	}

	nHead := 8
	headDim := 64
	size := nHead * headDim
	x := make([]float32, size)
	invFreq := make([]float64, headDim/2)
	attentionFactor := float32(1.0)
	pos := 10

	for i := range x {
		x[i] = float32(i%20) * 0.1
	}
	for i := range invFreq {
		invFreq[i] = float64(i+1) * 0.01
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		applyRoPEAVX512(x, nHead, headDim, pos, invFreq, attentionFactor, headDim/2)
	}
}

// Benchmark RoPE with precomputed tables
func BenchmarkApplyRoPEWithTablesScalar(b *testing.B) {
	nHead := 8
	headDim := 64
	size := nHead * headDim
	x := make([]float32, size)
	cosTable := make([]float32, 100*headDim/2) // Precomputed for 100 positions
	sinTable := make([]float32, 100*headDim/2)
	pos := 10

	for i := range x {
		x[i] = float32(i%20) * 0.1
	}

	// Precompute tables
	for p := range 100 {
		for i := 0; i < headDim/2; i++ {
			angle := float64(p) * float64(i+1) * 0.01
			cosTable[p*(headDim/2)+i] = float32(math.Cos(angle))
			sinTable[p*(headDim/2)+i] = float32(math.Sin(angle))
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		applyRoPEScalarWithTables(x, nHead, headDim, pos, cosTable, sinTable, headDim/2)
	}
}

func BenchmarkApplyRoPEWithTablesAVX2(b *testing.B) {
	if !cpu.HasAVX2 {
		b.Skip("AVX2 not available")
	}

	nHead := 8
	headDim := 64
	size := nHead * headDim
	x := make([]float32, size)
	cosTable := make([]float32, 100*headDim/2) // Precomputed for 100 positions
	sinTable := make([]float32, 100*headDim/2)
	pos := 10

	for i := range x {
		x[i] = float32(i%20) * 0.1
	}

	// Precompute tables
	for p := range 100 {
		for i := 0; i < headDim/2; i++ {
			angle := float64(p) * float64(i+1) * 0.01
			cosTable[p*(headDim/2)+i] = float32(math.Cos(angle))
			sinTable[p*(headDim/2)+i] = float32(math.Sin(angle))
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		applyRoPESIMDWithTables(x, nHead, headDim, pos, cosTable, sinTable, headDim/2)
	}
}

func BenchmarkApplyRoPEWithTablesAVX512(b *testing.B) {
	if !archsimd.X86.AVX512() {
		b.Skip("AVX-512 not available")
	}

	nHead := 8
	headDim := 64
	size := nHead * headDim
	x := make([]float32, size)
	cosTable := make([]float32, 100*headDim/2) // Precomputed for 100 positions
	sinTable := make([]float32, 100*headDim/2)
	pos := 10

	for i := range x {
		x[i] = float32(i%20) * 0.1
	}

	// Precompute tables
	for p := range 100 {
		for i := 0; i < headDim/2; i++ {
			angle := float64(p) * float64(i+1) * 0.01
			cosTable[p*(headDim/2)+i] = float32(math.Cos(angle))
			sinTable[p*(headDim/2)+i] = float32(math.Sin(angle))
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		applyRoPEWithTablesAVX512(x, nHead, headDim, pos, cosTable, sinTable, headDim/2)
	}
}

// Benchmark SiluAndMul operations
func BenchmarkSiluAndMulScalar(b *testing.B) {
	size := 512
	dst := make([]float32, size)
	x := make([]float32, size*2)

	for i := range x {
		x[i] = float32(i%20) * 0.1
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		siluAndMulScalar(dst, x)
	}
}

func BenchmarkSiluAndMulAVX2(b *testing.B) {
	if !cpu.HasAVX2 {
		b.Skip("AVX2 not available")
	}

	size := 512
	dst := make([]float32, size)
	x := make([]float32, size*2)

	for i := range x {
		x[i] = float32(i%20) * 0.1
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		siluAndMulSIMD(dst, x)
	}
}

func BenchmarkSiluAndMulAVX512(b *testing.B) {
	if !archsimd.X86.AVX512() {
		b.Skip("AVX-512 not available")
	}

	size := 512
	dst := make([]float32, size)
	x := make([]float32, size*2)

	for i := range x {
		x[i] = float32(i%20) * 0.1
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		siluAndMulAVX512(dst, x)
	}
}
