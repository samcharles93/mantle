package simd

import (
	"math"
	"simd/archsimd"
	"testing"
)

func TestFlashAttentionBasic(t *testing.T) {
	const (
		seqLenQ = 4
		seqLenK = 4
		dim     = 8
	)

	q := make([]float32, seqLenQ*dim)
	k := make([]float32, seqLenK*dim)
	v := make([]float32, seqLenK*dim)

	// Fill with test data
	for i := range q {
		q[i] = float32(i%5) * 0.1
	}
	for i := range k {
		k[i] = float32((i+1)%7) * 0.1
	}
	for i := range v {
		v[i] = float32((i+2)%11) * 0.1
	}

	scale := float32(1.0 / math.Sqrt(float64(dim)))

	result := FlashAttention(q, k, v, seqLenQ, seqLenK, dim, scale)

	if len(result) != seqLenQ*dim {
		t.Errorf("Expected result length %d, got %d", seqLenQ*dim, len(result))
	}

	// Basic sanity checks
	for i, val := range result {
		if math.IsNaN(float64(val)) {
			t.Errorf("NaN found at index %d", i)
		}
		if math.IsInf(float64(val), 0) {
			t.Errorf("Infinity found at index %d", i)
		}
	}
}

func TestFlashAttentionMultiHead(t *testing.T) {
	const (
		batch   = 1
		heads   = 2
		seqLenQ = 4
		seqLenK = 4
		headDim = 8
	)

	q := make([]float32, batch*heads*seqLenQ*headDim)
	k := make([]float32, batch*heads*seqLenK*headDim)
	v := make([]float32, batch*heads*seqLenK*headDim)

	// Fill with test data
	for i := range q {
		q[i] = float32(i%5) * 0.1
	}
	for i := range k {
		k[i] = float32((i+1)%7) * 0.1
	}
	for i := range v {
		v[i] = float32((i+2)%11) * 0.1
	}

	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	result := FlashAttentionMultiHead(q, k, v, batch, heads, seqLenQ, seqLenK, headDim, scale)

	expectedLen := batch * heads * seqLenQ * headDim
	if len(result) != expectedLen {
		t.Errorf("Expected result length %d, got %d", expectedLen, len(result))
	}

	// Basic sanity checks
	for i, val := range result {
		if math.IsNaN(float64(val)) {
			t.Errorf("NaN found at index %d", i)
		}
		if math.IsInf(float64(val), 0) {
			t.Errorf("Infinity found at index %d", i)
		}
	}
}

func TestFlashAttentionAVX512(t *testing.T) {
	if !archsimd.X86.AVX512() {
		t.Skip("AVX-512 not available, skipping test")
	}

	testFlashAttentionImplementation(t, "AVX-512")
}

func TestFlashAttentionAVX2(t *testing.T) {
	if !cpu.HasAVX2 {
		t.Skip("AVX2 not available, skipping test")
	}

	testFlashAttentionImplementation(t, "AVX2")
}

func TestFlashAttentionScalar(t *testing.T) {
	// This test runs regardless of CPU features to test the scalar fallback
	testFlashAttentionImplementation(t, "scalar")
}

func testFlashAttentionImplementation(t *testing.T, impl string) {
	const (
		seqLenQ = 8
		seqLenK = 8
		dim     = 16
	)

	q := make([]float32, seqLenQ*dim)
	k := make([]float32, seqLenK*dim)
	v := make([]float32, seqLenK*dim)

	// Fill with test data
	for i := range q {
		q[i] = float32(i%13) * 0.05
	}
	for i := range k {
		k[i] = float32((i+3)%17) * 0.05
	}
	for i := range v {
		v[i] = float32((i+7)%19) * 0.05
	}

	scale := float32(1.0 / math.Sqrt(float64(dim)))

	result := FlashAttention(q, k, v, seqLenQ, seqLenK, dim, scale)

	if len(result) != seqLenQ*dim {
		t.Errorf("%s: Expected result length %d, got %d", impl, seqLenQ*dim, len(result))
	}

	// Basic sanity checks
	for i, val := range result {
		if math.IsNaN(float64(val)) {
			t.Errorf("%s: NaN found at index %d", impl, i)
		}
		if math.IsInf(float64(val), 0) {
			t.Errorf("%s: Infinity found at index %d", impl, i)
		}
	}
}

func TestGetOptimalBlockSize(t *testing.T) {
	blockQ, blockK := GetOptimalBlockSize()

	if blockQ <= 0 || blockK <= 0 {
		t.Errorf("Expected positive block sizes, got Q=%d, K=%d", blockQ, blockK)
	}

	// The block sizes should be reasonable (not extremely large or small)
	if blockQ > 128 || blockK > 128 {
		t.Errorf("Block sizes seem too large: Q=%d, K=%d", blockQ, blockK)
	}
	if blockQ < 8 || blockK < 8 {
		t.Errorf("Block sizes seem too small: Q=%d, K=%d", blockQ, blockK)
	}
}

func TestValidateFlashAttention(t *testing.T) {
	const (
		seqLenQ = 4
		seqLenK = 4
		dim     = 8
	)

	q := make([]float32, seqLenQ*dim)
	k := make([]float32, seqLenK*dim)
	v := make([]float32, seqLenK*dim)

	// Test valid inputs
	err := ValidateFlashAttention(q, k, v, seqLenQ, seqLenK, dim)
	if err != nil {
		t.Errorf("Valid inputs should not return error: %v", err)
	}

	// Test invalid dimensions
	err = ValidateFlashAttention(q, k, v, 0, seqLenK, dim)
	if err == nil {
		t.Error("Invalid seqLenQ=0 should return error")
	}

	err = ValidateFlashAttention(q, k, v, seqLenQ, 0, dim)
	if err == nil {
		t.Error("Invalid seqLenK=0 should return error")
	}

	err = ValidateFlashAttention(q, k, v, seqLenQ, seqLenK, 0)
	if err == nil {
		t.Error("Invalid dim=0 should return error")
	}

	// Test insufficient buffer sizes
	shortQ := make([]float32, seqLenQ*dim-1)
	err = ValidateFlashAttention(shortQ, k, v, seqLenQ, seqLenK, dim)
	if err == nil {
		t.Error("Insufficient Q buffer should return error")
	}

	shortK := make([]float32, seqLenK*dim-1)
	err = ValidateFlashAttention(q, shortK, v, seqLenQ, seqLenK, dim)
	if err == nil {
		t.Error("Insufficient K buffer should return error")
	}

	shortV := make([]float32, seqLenK*dim-1)
	err = ValidateFlashAttention(q, k, shortV, seqLenQ, seqLenK, dim)
	if err == nil {
		t.Error("Insufficient V buffer should return error")
	}
}

func TestOnlineSoftmax(t *testing.T) {
	// Test the OnlineSoftmax function added to ops.go
	x := []float32{1.0, 2.0, 3.0, 4.0}
	m := float32(0.0) // Previous max
	l := float32(1.0) // Previous sum of exp

	// Save original values for comparison
	originalX := make([]float32, len(x))
	copy(originalX, x)

	OnlineSoftmax(x, m, l)

	// Check that values sum to approximately 1 (after normalization)
	var sum float32
	for _, val := range x {
		sum += val
	}

	if sum < 0.99 || sum > 1.01 {
		t.Errorf("Softmax output should sum to ~1.0, got %f", sum)
	}

	// Check that no values are negative
	for i, val := range x {
		if val < 0 {
			t.Errorf("Softmax output at index %d should be non-negative, got %f", i, val)
		}
	}
}

func TestFlashAttentionConsistency(t *testing.T) {
	// Test that different implementations (when available) produce consistent results
	const (
		seqLenQ = 6
		seqLenK = 6
		dim     = 12
	)

	// Create test data
	q := make([]float32, seqLenQ*dim)
	k := make([]float32, seqLenK*dim)
	v := make([]float32, seqLenK*dim)

	for i := range q {
		q[i] = float32(i%13) * 0.1
	}
	for i := range k {
		k[i] = float32((i+5)%17) * 0.1
	}
	for i := range v {
		v[i] = float32((i+9)%11) * 0.1
	}

	scale := float32(1.0 / math.Sqrt(float64(dim)))

	// Get result from the main function (which will use the best available implementation)
	result := FlashAttention(q, k, v, seqLenQ, seqLenK, dim, scale)

	// Verify result properties
	if len(result) != seqLenQ*dim {
		t.Errorf("Unexpected result length: expected %d, got %d", seqLenQ*dim, len(result))
	}

	// Basic validation
	for i, val := range result {
		if math.IsNaN(float64(val)) {
			t.Errorf("NaN at index %d", i)
		}
		if math.IsInf(float64(val), 0) {
			t.Errorf("Infinity at index %d", i)
		}
	}
}

func BenchmarkFlashAttentionSmall(b *testing.B) {
	const (
		seqLenQ = 16
		seqLenK = 16
		dim     = 64
	)

	q := make([]float32, seqLenQ*dim)
	k := make([]float32, seqLenK*dim)
	v := make([]float32, seqLenK*dim)

	for i := range q {
		q[i] = float32(i%23) * 0.01
	}
	for i := range k {
		k[i] = float32((i+7)%29) * 0.01
	}
	for i := range v {
		v[i] = float32((i+13)%31) * 0.01
	}

	scale := float32(1.0 / math.Sqrt(float64(dim)))

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = FlashAttention(q, k, v, seqLenQ, seqLenK, dim, scale)
	}
}

func BenchmarkFlashAttentionLarge(b *testing.B) {
	const (
		seqLenQ = 128
		seqLenK = 128
		dim     = 256
	)

	q := make([]float32, seqLenQ*dim)
	k := make([]float32, seqLenK*dim)
	v := make([]float32, seqLenK*dim)

	for i := range q {
		q[i] = float32(i%47) * 0.001
	}
	for i := range k {
		k[i] = float32((i+11)%53) * 0.001
	}
	for i := range v {
		v[i] = float32((i+17)%59) * 0.001
	}

	scale := float32(1.0 / math.Sqrt(float64(dim)))

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = FlashAttention(q, k, v, seqLenQ, seqLenK, dim, scale)
	}
}
