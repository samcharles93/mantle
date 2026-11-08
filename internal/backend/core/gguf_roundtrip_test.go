package core

import (
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"testing"

	"github.com/samcharles93/mantle/pkg/mcf"
)

// mustWrite writes v to buf; bytes.Buffer writes never fail, but errcheck
// requires the error to be checked.
func mustWrite(buf *bytes.Buffer, v any) {
	if err := binary.Write(buf, binary.LittleEndian, v); err != nil {
		panic(err)
	}
}

// ggufTestTensor builds a synthetic GGUF v3 container holding one Q8_0 tensor
// with the given per-block f16 scales and int8 data.
func ggufTestTensor(t *testing.T, rows, cols int, scales []uint16, data []int8) []byte {
	t.Helper()
	blocks := rows * (cols / 32)
	if len(scales) != blocks || len(data) != blocks*32 {
		t.Fatalf("bad test fixture: scales=%d data=%d blocks=%d", len(scales), len(data), blocks)
	}
	payload := make([]byte, blocks*34)
	for b := 0; b < blocks; b++ {
		binary.LittleEndian.PutUint16(payload[b*34:], scales[b])
		for i := 0; i < 32; i++ {
			payload[b*34+2+i] = byte(data[b*32+i])
		}
	}
	var buf bytes.Buffer
	buf.WriteString("GGUF")
	mustWrite(&buf, uint32(3))
	mustWrite(&buf, uint64(1)) // tensors
	mustWrite(&buf, uint64(0)) // kv
	name := "w"
	mustWrite(&buf, uint64(len(name)))
	buf.WriteString(name)
	mustWrite(&buf, uint32(2))    // rank
	mustWrite(&buf, uint64(cols)) // ne0
	mustWrite(&buf, uint64(rows)) // ne1
	mustWrite(&buf, uint32(8))    // Q8_0
	mustWrite(&buf, uint64(0))    // offset
	for buf.Len()%32 != 0 {       // align 32
		buf.WriteByte(0)
	}
	buf.Write(payload)
	return buf.Bytes()
}

// TestGGUFQ8_0RoundTrip checks GGUF Q8_0 payload -> MCF DTypeQ8 -> RowTo
// decodes to exactly the GGUF dequantized row (scale * int8).
func TestGGUFQ8_0RoundTrip(t *testing.T) {
	rows, cols := 3, 64
	blocks := rows * (cols / 32)
	scales := make([]uint16, blocks)
	data := make([]int8, blocks*32)
	for b := 0; b < blocks; b++ {
		scales[b] = uint16(0x3800 + b) // 0.5, 0.50097..., distinct exact-ish f16 values
		for i := 0; i < 32; i++ {
			data[b*32+i] = int8((b*7+i*3)%251 - 125)
		}
	}
	raw := ggufTestTensor(t, rows, cols, scales, data)

	f, err := mcf.ParseGGUF(bytes.NewReader(raw), int64(len(raw)))
	if err != nil {
		t.Fatalf("ParseGGUF: %v", err)
	}
	gt, ok := f.FindTensor("w")
	if !ok {
		t.Fatal("tensor w not found")
	}
	payload, err := f.TensorPayload(gt)
	if err != nil {
		t.Fatalf("TensorPayload: %v", err)
	}
	dt, shape, mcfPayload, err := mcf.TranscodeGGUFQuant(gt, payload)
	if err != nil {
		t.Fatalf("TranscodeGGUFQuant: %v", err)
	}
	if dt != mcf.DTypeQ8 || shape[0] != uint64(rows) || shape[1] != uint64(cols) {
		t.Fatalf("transcode = dtype %v shape %v", dt, shape)
	}

	m := Mat{R: rows, C: cols, Stride: cols, DType: dt, Raw: mcfPayload}
	row := make([]float32, cols)
	for r := 0; r < rows; r++ {
		m.RowTo(row, r)
		for c := 0; c < cols; c++ {
			b := r*(cols/32) + c/32
			want := fp16ToF32Table(scales[b]) * float32(data[b*32+c%32])
			got := row[c]
			if math.Abs(float64(got-want)) > 1e-6*math.Max(1, math.Abs(float64(want))) {
				t.Fatalf("row %d col %d: got %v want %v", r, c, got, want)
			}
		}
	}
}

// TestGGUFQ8_0RealFile is an opt-in check against a real GGUF Q8_0 container:
// set MANTLE_GGUF_Q8_0 to a *.gguf path. The first Q8_0 tensor is transcoded
// and its first row is compared against direct GGUF dequantization.
func TestGGUFQ8_0RealFile(t *testing.T) {
	path := os.Getenv("MANTLE_GGUF_Q8_0")
	if path == "" {
		t.Skip("set MANTLE_GGUF_Q8_0 to a GGUF Q8_0 file")
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Skipf("cannot read %s: %v", path, err)
	}
	f, err := mcf.ParseGGUF(bytes.NewReader(raw), int64(len(raw)))
	if err != nil {
		t.Fatalf("ParseGGUF: %v", err)
	}
	var gt mcf.GGUFTensor
	found := false
	for i := 0; i < f.TensorCount(); i++ {
		ti, err := f.Tensor(i)
		if err != nil {
			t.Fatal(err)
		}
		if ti.Type == 8 && len(ti.Dims) == 2 && ti.Dims[0]%32 == 0 {
			gt, found = ti, true
			break
		}
	}
	if !found {
		t.Fatal("no Q8_0 rank-2 tensor found")
	}
	payload, err := f.TensorPayload(gt)
	if err != nil {
		t.Fatalf("TensorPayload: %v", err)
	}
	dt, shape, mcfPayload, err := mcf.TranscodeGGUFQuant(gt, payload)
	if err != nil {
		t.Fatalf("TranscodeGGUFQuant: %v", err)
	}
	rows, cols := int(shape[0]), int(shape[1])
	if rows == 0 || cols == 0 {
		t.Fatal("empty tensor")
	}
	m := Mat{R: rows, C: cols, Stride: cols, DType: dt, Raw: mcfPayload}
	row := make([]float32, cols)
	m.RowTo(row, 0)
	// Reference: GGUF Q8_0 dequant of row 0: w[c] = f16(scale of block c/32) * int8.
	for c := 0; c < cols; c++ {
		b := c / 32
		scaleBits := binary.LittleEndian.Uint16(payload[b*34:])
		q := int8(payload[b*34+2+c%32])
		want := fp16ToF32Table(scaleBits) * float32(q)
		if got := row[c]; math.Abs(float64(got-want)) > 1e-6*math.Max(1, math.Abs(float64(want))) {
			t.Fatalf("col %d: got %v want %v", c, got, want)
		}
	}
	t.Logf("tensor %q %dx%d row 0 verified against GGUF dequant", gt.Name, rows, cols)
}
