package simd

import (
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
