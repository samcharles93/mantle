package tensor

import (
	"math/rand"

	"github.com/samcharles93/mantle/pkg/mcf"
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

	// DType describes the underlying element encoding. For f32 weights we keep
	// Data populated for fast access. For f16/bf16 weights we keep Raw and
	// decode inline in MatVec/RowTo to reduce memory bandwidth pressure.
	DType mcf.TensorDType
	Data  []float32
	Raw   []byte
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
		DType:  mcf.DTypeF32,
		Data:   make([]float32, r*c),
	}
}

// NewMatFromData creates a matrix from existing data.
// It checks that the data length matches r*c.
func NewMatFromData(r, c int, data []float32) Mat {
	if r*c != len(data) {
		panic("data length mismatch")
	}
	return Mat{
		R:      r,
		C:      c,
		Stride: c,
		DType:  mcf.DTypeF32,
		Data:   data,
	}
}

// NewMatFromRaw creates a matrix backed by raw bytes in the provided dtype.
// The raw slice must contain exactly r*c elements in row-major layout.
func NewMatFromRaw(r, c int, dtype mcf.TensorDType, raw []byte) (Mat, error) {
	if r < 0 || c < 0 {
		return Mat{}, errNegativeDim
	}
	elemSize, ok := dtypeElemSize(dtype)
	if !ok || elemSize == 0 {
		return Mat{}, errUnsupportedDType
	}
	want := r * c
	if r != 0 && want/r != c {
		return Mat{}, errMatTooLarge
	}
	wantBytes := want * elemSize
	if want != 0 && wantBytes/want != elemSize {
		return Mat{}, errMatTooLarge
	}
	if len(raw) != wantBytes {
		return Mat{}, errRawSizeMismatch
	}
	return Mat{
		R:      r,
		C:      c,
		Stride: c,
		DType:  dtype,
		Raw:    raw,
	}, nil
}

// Row returns a view of the i‑th row of the matrix as a slice.  The slice
// has length equal to the number of columns.  Modifications to the returned
// slice update the underlying matrix values.
func (m *Mat) Row(i int) []float32 {
	if i < 0 || i >= m.R {
		panic("row index out of range")
	}
	if m.Raw == nil || m.DType == mcf.DTypeF32 {
		start := i * m.Stride
		return m.Data[start : start+m.C]
	}
	row := make([]float32, m.C)
	m.RowTo(row, i)
	return row
}

// RowTo decodes the i-th row into dst. dst must have length >= C.
func (m *Mat) RowTo(dst []float32, i int) {
	if i < 0 || i >= m.R {
		panic("row index out of range")
	}
	if len(dst) < m.C {
		panic("row buffer too small")
	}
	start := i * m.Stride
	if m.Raw == nil || m.DType == mcf.DTypeF32 {
		copy(dst[:m.C], m.Data[start:start+m.C])
		return
	}

	elemSize, ok := dtypeElemSize(m.DType)
	if !ok || elemSize == 0 {
		panic("unsupported dtype for row decode")
	}
	rowBytes := m.Stride * elemSize
	off := i * rowBytes
	switch m.DType {
	case mcf.DTypeBF16:
		for j := 0; j < m.C; j++ {
			u := u16le(m.Raw, off+j*2)
			dst[j] = bf16ToF32(u)
		}
	case mcf.DTypeF16:
		for j := 0; j < m.C; j++ {
			u := u16le(m.Raw, off+j*2)
			dst[j] = fp16ToF32(u)
		}
	default:
		panic("unsupported dtype for row decode")
	}
}

// FillRand fills the matrix with reproducible pseudo‑random values.  A small
// range around zero is used to avoid overflow in accumulations.  The seed
// controls the random sequence; multiple calls with the same seed produce
// identical matrices.
func FillRand(m *Mat, seed int64) {
	rng := rand.New(rand.NewSource(seed))
	if m.Raw != nil && m.DType != mcf.DTypeF32 {
		panic("FillRand only supports f32 mats")
	}
	for i := range m.Data {
		m.Data[i] = (rng.Float32() - 0.5) * 0.02 // roughly in (-0.01,0.01)
	}
}

var (
	errNegativeDim      = fmtError("negative dimension for matrix")
	errUnsupportedDType = fmtError("unsupported dtype for raw matrix")
	errMatTooLarge      = fmtError("matrix too large")
	errRawSizeMismatch  = fmtError("raw data length mismatch")
)

type fmtError string

func (e fmtError) Error() string { return string(e) }
