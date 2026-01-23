package tensor

import (
    "math/rand"
)

// Mat represents a dense row‑major matrix of float32 values.
//
// R and C represent the number of rows and columns respectively.  Stride is the
// number of elements between the starts of two consecutive rows (for row‑major
// matrices this is equal to C).  Data holds the flattened matrix values.
//
// Mat does not perform any memory safety beyond the checks performed by Go's
// slice types; out‑of‑range indices will panic.
type Mat struct {
    R, C   int
    Stride int
    Data   []float32
}

// NewMat allocates a new matrix with the given number of rows and columns.
// The underlying slice is zero initialised.  The stride is set to the
// number of columns.
func NewMat(r, c int) Mat {
    if r < 0 || c < 0 {
        panic("negative dimension for matrix")
    }
    return Mat{
        R:      r,
        C:      c,
        Stride: c,
        Data:   make([]float32, r*c),
    }
}

// Row returns a view of the i‑th row of the matrix as a slice.  The slice
// has length equal to the number of columns.  Modifications to the returned
// slice update the underlying matrix values.
func (m *Mat) Row(i int) []float32 {
    if i < 0 || i >= m.R {
        panic("row index out of range")
    }
    start := i * m.Stride
    return m.Data[start : start+m.C]
}

// FillRand fills the matrix with reproducible pseudo‑random values.  A small
// range around zero is used to avoid overflow in accumulations.  The seed
// controls the random sequence; multiple calls with the same seed produce
// identical matrices.
func FillRand(m *Mat, seed int64) {
    rng := rand.New(rand.NewSource(seed))
    for i := range m.Data {
        m.Data[i] = (rng.Float32() - 0.5) * 0.02 // roughly in (-0.01,0.01)
    }
}