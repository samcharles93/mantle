package simd

import (
	"math"
	"testing"
)

// TestNewMatDimensions verifies that NewMat creates a matrix with the correct
// dimensions and stride, and that the backing slice has the expected length.
func TestNewMatDimensions(t *testing.T) {
	r, c := 5, 7
	m := NewMat(r, c)
	if m.R != r || m.C != c {
		t.Fatalf("expected dimensions %dx%d, got %dx%d", r, c, m.R, m.C)
	}
	if m.Stride != c {
		t.Fatalf("expected stride %d, got %d", c, m.Stride)
	}
	if len(m.Data) != r*c {
		t.Fatalf("expected backing slice length %d, got %d", r*c, len(m.Data))
	}
}

// TestRowSlicing ensures that Row returns a slice of the correct length and
// that modifications to the returned slice affect the underlying matrix.
func TestRowSlicing(t *testing.T) {
	m := NewMat(3, 4)
	// Set a value via Row and ensure it propagates to Data.
	row1 := m.Row(1)
	if len(row1) != 4 {
		t.Fatalf("expected row length 4, got %d", len(row1))
	}
	row1[2] = 42
	// Index in underlying Data is 1*stride + 2
	idx := 1*m.Stride + 2
	if m.Data[idx] != 42 {
		t.Fatalf("expected Data[%d] to be 42, got %f", idx, m.Data[idx])
	}
}

// TestFillRandDeterminism checks that FillRand produces deterministic results
// for the same seed and that the values lie within the expected range.
func TestFillRandDeterminism(t *testing.T) {
	m1 := NewMat(2, 3)
	m2 := NewMat(2, 3)
	FillRand(&m1, 1234)
	FillRand(&m2, 1234)
	for i := range m1.Data {
		v := m1.Data[i]
		// Values should be approximately in (-0.01, 0.01).
		if math.Abs(float64(v)) > 0.02 {
			t.Fatalf("value out of range: %f", v)
		}
		if v != m2.Data[i] {
			t.Fatalf("determinism failed: index %d got %f vs %f", i, v, m2.Data[i])
		}
	}
}
