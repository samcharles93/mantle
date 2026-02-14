package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"maps"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// writeSafetensors creates a minimal safetensors file for testing.
func writeSafetensors(t *testing.T, path string, tensors map[string]tensorHeader) {
	t.Helper()
	header := make(map[string]tensorHeader, len(tensors))
	maps.Copy(header, tensors)
	headerBytes, err := json.Marshal(header)
	if err != nil {
		t.Fatalf("marshal header: %v", err)
	}

	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("create file: %v", err)
	}
	defer func() { _ = f.Close() }()

	// Write 8-byte little-endian header length
	var lenBuf [8]byte
	binary.LittleEndian.PutUint64(lenBuf[:], uint64(len(headerBytes)))
	if _, err := f.Write(lenBuf[:]); err != nil {
		t.Fatalf("write header len: %v", err)
	}
	if _, err := f.Write(headerBytes); err != nil {
		t.Fatalf("write header: %v", err)
	}

	// Write tensor data (find max offset)
	var maxEnd int64
	for _, th := range tensors {
		if len(th.DataOffsets) == 2 && th.DataOffsets[1] > maxEnd {
			maxEnd = th.DataOffsets[1]
		}
	}
	if maxEnd > 0 {
		data := make([]byte, maxEnd)
		if _, err := f.Write(data); err != nil {
			t.Fatalf("write data: %v", err)
		}
	}
}

func TestOpenValidFile(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	path := filepath.Join(dir, "test.safetensors")

	writeSafetensors(t, path, map[string]tensorHeader{
		"weight": {
			DType:       "F32",
			Shape:       []int{2, 3},
			DataOffsets: []int64{0, 24}, // 2*3*4 = 24 bytes
		},
	})

	f, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if f.Path != path {
		t.Fatalf("expected path %q, got %q", path, f.Path)
	}
	if len(f.Tensors) != 1 {
		t.Fatalf("expected 1 tensor, got %d", len(f.Tensors))
	}

	info, ok := f.Tensor("weight")
	if !ok {
		t.Fatal("tensor 'weight' not found")
	}
	if info.DType != "F32" {
		t.Fatalf("expected dtype F32, got %q", info.DType)
	}
	if len(info.Shape) != 2 || info.Shape[0] != 2 || info.Shape[1] != 3 {
		t.Fatalf("unexpected shape: %v", info.Shape)
	}
}

func TestOpenNonexistentFile(t *testing.T) {
	t.Parallel()
	_, err := Open("/nonexistent/file.safetensors")
	if err == nil {
		t.Fatal("expected error for nonexistent file")
	}
}

func TestOpenTruncatedFile(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	path := filepath.Join(dir, "truncated.safetensors")

	// Write only 4 bytes (too short for header length)
	if err := os.WriteFile(path, []byte{0, 0, 0, 0}, 0o644); err != nil {
		t.Fatalf("write file: %v", err)
	}

	_, err := Open(path)
	if err == nil {
		t.Fatal("expected error for truncated file")
	}
}

func TestOpenInvalidJSON(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	path := filepath.Join(dir, "invalid.safetensors")

	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	// Write header length
	var lenBuf [8]byte
	binary.LittleEndian.PutUint64(lenBuf[:], 12)
	_, _ = f.Write(lenBuf[:])
	_, _ = f.Write([]byte("not valid js"))
	_ = f.Close()

	_, err = Open(path)
	if err == nil {
		t.Fatal("expected error for invalid JSON header")
	}
}

func TestInvalidDataOffsets(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	path := filepath.Join(dir, "bad_offsets.safetensors")

	// Create header with invalid data_offsets (only 1 element)
	header := map[string]any{
		"bad_tensor": map[string]any{
			"dtype":        "F32",
			"shape":        []int{1},
			"data_offsets": []int64{0},
		},
	}
	headerBytes, _ := json.Marshal(header)

	f, _ := os.Create(path)
	var lenBuf [8]byte
	binary.LittleEndian.PutUint64(lenBuf[:], uint64(len(headerBytes)))
	_, _ = f.Write(lenBuf[:])
	_, _ = f.Write(headerBytes)
	_ = f.Close()

	_, err := Open(path)
	if err == nil {
		t.Fatal("expected error for invalid data_offsets")
	}
}

func TestMetadataIgnored(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	path := filepath.Join(dir, "metadata.safetensors")

	// Manually create file with __metadata__ section
	header := map[string]any{
		"__metadata__": map[string]string{"format": "pt"},
		"tensor1": map[string]any{
			"dtype":        "F32",
			"shape":        []int{4},
			"data_offsets": []int64{0, 16},
		},
	}
	headerBytes, _ := json.Marshal(header)

	f, _ := os.Create(path)
	var lenBuf [8]byte
	binary.LittleEndian.PutUint64(lenBuf[:], uint64(len(headerBytes)))
	_, _ = f.Write(lenBuf[:])
	_, _ = f.Write(headerBytes)
	_, _ = f.Write(make([]byte, 16))
	_ = f.Close()

	sf, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if len(sf.Tensors) != 1 {
		t.Fatalf("expected 1 tensor (metadata should be excluded), got %d", len(sf.Tensors))
	}
}

func TestTensorNotFound(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	path := filepath.Join(dir, "test.safetensors")
	writeSafetensors(t, path, map[string]tensorHeader{
		"a": {DType: "F32", Shape: []int{1}, DataOffsets: []int64{0, 4}},
	})

	f, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	_, ok := f.Tensor("nonexistent")
	if ok {
		t.Fatal("expected tensor not found")
	}

	_, _, err = f.ReadTensor("nonexistent")
	if err == nil {
		t.Fatal("expected error for missing tensor")
	}
}

func TestReadTensorF32(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	path := filepath.Join(dir, "f32.safetensors")

	// Create a real F32 tensor with known data
	values := []float32{1.0, 2.0, 3.0, 4.0}
	data := make([]byte, 16)
	for i, v := range values {
		binary.LittleEndian.PutUint32(data[i*4:], math.Float32bits(v))
	}

	header := map[string]any{
		"test": map[string]any{
			"dtype":        "F32",
			"shape":        []int{4},
			"data_offsets": []int64{0, 16},
		},
	}
	headerBytes, _ := json.Marshal(header)

	f, _ := os.Create(path)
	var lenBuf [8]byte
	binary.LittleEndian.PutUint64(lenBuf[:], uint64(len(headerBytes)))
	_, _ = f.Write(lenBuf[:])
	_, _ = f.Write(headerBytes)
	_, _ = f.Write(data)
	_ = f.Close()

	sf, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}

	result, info, err := sf.ReadTensorF32("test")
	if err != nil {
		t.Fatalf("ReadTensorF32: %v", err)
	}
	if info.DType != "F32" {
		t.Fatalf("expected F32, got %q", info.DType)
	}
	if len(result) != 4 {
		t.Fatalf("expected 4 elements, got %d", len(result))
	}
	for i, v := range values {
		if result[i] != v {
			t.Fatalf("element %d: expected %f, got %f", i, v, result[i])
		}
	}
}

func TestReadTensorBF16(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	path := filepath.Join(dir, "bf16.safetensors")

	// BF16 for 1.0 is 0x3F80 (top 16 bits of float32 1.0)
	data := make([]byte, 4)
	binary.LittleEndian.PutUint16(data[0:], 0x3F80) // 1.0
	binary.LittleEndian.PutUint16(data[2:], 0x4000) // 2.0

	header := map[string]any{
		"test": map[string]any{
			"dtype":        "BF16",
			"shape":        []int{2},
			"data_offsets": []int64{0, 4},
		},
	}
	headerBytes, _ := json.Marshal(header)

	f, _ := os.Create(path)
	var lenBuf [8]byte
	binary.LittleEndian.PutUint64(lenBuf[:], uint64(len(headerBytes)))
	_, _ = f.Write(lenBuf[:])
	_, _ = f.Write(headerBytes)
	_, _ = f.Write(data)
	_ = f.Close()

	sf, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}

	result, _, err := sf.ReadTensorF32("test")
	if err != nil {
		t.Fatalf("ReadTensorF32: %v", err)
	}
	if len(result) != 2 {
		t.Fatalf("expected 2 elements, got %d", len(result))
	}
	if result[0] != 1.0 {
		t.Fatalf("element 0: expected 1.0, got %f", result[0])
	}
	if result[1] != 2.0 {
		t.Fatalf("element 1: expected 2.0, got %f", result[1])
	}
}

func TestReadTensorF16(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	path := filepath.Join(dir, "f16.safetensors")

	// F16 for 1.0 is 0x3C00
	data := make([]byte, 2)
	binary.LittleEndian.PutUint16(data[0:], 0x3C00) // 1.0

	header := map[string]any{
		"test": map[string]any{
			"dtype":        "F16",
			"shape":        []int{1},
			"data_offsets": []int64{0, 2},
		},
	}
	headerBytes, _ := json.Marshal(header)

	f, _ := os.Create(path)
	var lenBuf [8]byte
	binary.LittleEndian.PutUint64(lenBuf[:], uint64(len(headerBytes)))
	_, _ = f.Write(lenBuf[:])
	_, _ = f.Write(headerBytes)
	_, _ = f.Write(data)
	_ = f.Close()

	sf, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}

	result, _, err := sf.ReadTensorF32("test")
	if err != nil {
		t.Fatalf("ReadTensorF32: %v", err)
	}
	if len(result) != 1 {
		t.Fatalf("expected 1 element, got %d", len(result))
	}
	if result[0] != 1.0 {
		t.Fatalf("expected 1.0, got %f", result[0])
	}
}

func TestReadTensorUnsupportedDType(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	path := filepath.Join(dir, "unsupported.safetensors")

	header := map[string]any{
		"test": map[string]any{
			"dtype":        "I32",
			"shape":        []int{2},
			"data_offsets": []int64{0, 8},
		},
	}
	headerBytes, _ := json.Marshal(header)

	f, _ := os.Create(path)
	var lenBuf [8]byte
	binary.LittleEndian.PutUint64(lenBuf[:], uint64(len(headerBytes)))
	_, _ = f.Write(lenBuf[:])
	_, _ = f.Write(headerBytes)
	_, _ = f.Write(make([]byte, 8))
	_ = f.Close()

	sf, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}

	_, _, err = sf.ReadTensorF32("test")
	if err == nil {
		t.Fatal("expected error for unsupported dtype")
	}
}

func TestReadTensorSizeMismatch(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	path := filepath.Join(dir, "mismatch.safetensors")

	// Shape says 4 elements but data is only 8 bytes (2 F32 elements)
	header := map[string]any{
		"test": map[string]any{
			"dtype":        "F32",
			"shape":        []int{4},
			"data_offsets": []int64{0, 8},
		},
	}
	headerBytes, _ := json.Marshal(header)

	f, _ := os.Create(path)
	var lenBuf [8]byte
	binary.LittleEndian.PutUint64(lenBuf[:], uint64(len(headerBytes)))
	_, _ = f.Write(lenBuf[:])
	_, _ = f.Write(headerBytes)
	_, _ = f.Write(make([]byte, 8))
	_ = f.Close()

	sf, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}

	_, _, err = sf.ReadTensorF32("test")
	if err == nil {
		t.Fatal("expected error for size mismatch")
	}
}

func TestNumElements(t *testing.T) {
	t.Parallel()

	tests := []struct {
		shape    []int
		expected int
		wantErr  bool
	}{
		{[]int{2, 3}, 6, false},
		{[]int{1}, 1, false},
		{[]int{4, 5, 6}, 120, false},
		{[]int{}, 0, true},      // empty shape
		{[]int{0}, 0, true},     // zero dimension
		{[]int{-1}, 0, true},    // negative dimension
		{[]int{2, -1}, 0, true}, // negative dimension
	}

	for _, tc := range tests {
		n, err := numElements(tc.shape)
		if tc.wantErr {
			if err == nil {
				t.Errorf("numElements(%v): expected error", tc.shape)
			}
			continue
		}
		if err != nil {
			t.Errorf("numElements(%v): unexpected error: %v", tc.shape, err)
			continue
		}
		if n != tc.expected {
			t.Errorf("numElements(%v): expected %d, got %d", tc.shape, tc.expected, n)
		}
	}
}

func TestBf16ToF32(t *testing.T) {
	t.Parallel()

	tests := []struct {
		input    uint16
		expected float32
	}{
		{0x3F80, 1.0},
		{0x4000, 2.0},
		{0xBF80, -1.0},
		{0x0000, 0.0},
		{0x4040, 3.0},
	}

	for _, tc := range tests {
		result := bf16ToF32(tc.input)
		if result != tc.expected {
			t.Errorf("bf16ToF32(0x%04X): expected %f, got %f", tc.input, tc.expected, result)
		}
	}
}

func TestFp16ToFloat32(t *testing.T) {
	t.Parallel()

	tests := []struct {
		input    uint16
		expected float32
	}{
		{0x3C00, 1.0},  // 1.0
		{0x4000, 2.0},  // 2.0
		{0xBC00, -1.0}, // -1.0
		{0x0000, 0.0},  // +0
		{0x8000, math.Float32frombits(0x80000000)}, // -0
		{0x7C00, float32(math.Inf(1))},             // +inf
		{0xFC00, float32(math.Inf(-1))},            // -inf
	}

	for _, tc := range tests {
		result := fp16ToFloat32(tc.input)
		if math.IsInf(float64(tc.expected), 0) {
			if !math.IsInf(float64(result), 0) {
				t.Errorf("fp16ToFloat32(0x%04X): expected inf, got %f", tc.input, result)
			}
			continue
		}
		if result != tc.expected {
			t.Errorf("fp16ToFloat32(0x%04X): expected %f, got %f", tc.input, tc.expected, result)
		}
	}
}

func TestMultipleTensors(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	path := filepath.Join(dir, "multi.safetensors")

	writeSafetensors(t, path, map[string]tensorHeader{
		"weight": {DType: "F32", Shape: []int{2, 2}, DataOffsets: []int64{0, 16}},
		"bias":   {DType: "F32", Shape: []int{2}, DataOffsets: []int64{16, 24}},
	})

	f, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if len(f.Tensors) != 2 {
		t.Fatalf("expected 2 tensors, got %d", len(f.Tensors))
	}
	if _, ok := f.Tensor("weight"); !ok {
		t.Fatal("weight not found")
	}
	if _, ok := f.Tensor("bias"); !ok {
		t.Fatal("bias not found")
	}
}

func TestReadTensorInvertedOffsets(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	path := filepath.Join(dir, "inverted.safetensors")

	writeSafetensors(t, path, map[string]tensorHeader{
		"bad": {DType: "F32", Shape: []int{2}, DataOffsets: []int64{8, 0}}, // end < start
	})

	f, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}

	_, _, err = f.ReadTensor("bad")
	if err == nil {
		t.Fatal("expected error for inverted offsets")
	}
}
