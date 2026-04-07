package mcf

import (
	"encoding/binary"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"testing"
)

func TestOpenSafetensorsFileAllowsScalarTensor(t *testing.T) {
	t.Parallel()

	path := filepath.Join(t.TempDir(), "scalar.safetensors")
	header := map[string]any{
		"__metadata__": map[string]string{"format": "pt"},
		"scalar": map[string]any{
			"dtype":        "BF16",
			"shape":        []int{},
			"data_offsets": []int{0, 2},
		},
	}
	headerRaw, err := json.Marshal(header)
	if err != nil {
		t.Fatalf("marshal header: %v", err)
	}

	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("create safetensors: %v", err)
	}
	if err := binary.Write(f, binary.LittleEndian, uint64(len(headerRaw))); err != nil {
		t.Fatalf("write header len: %v", err)
	}
	if _, err := f.Write(headerRaw); err != nil {
		t.Fatalf("write header: %v", err)
	}
	if _, err := f.Write([]byte{0x00, 0x3f}); err != nil {
		t.Fatalf("write payload: %v", err)
	}
	if err := f.Close(); err != nil {
		t.Fatalf("close safetensors: %v", err)
	}

	sf, err := OpenSafetensorsFile(path)
	if err != nil {
		t.Fatalf("OpenSafetensorsFile: %v", err)
	}
	defer func() { _ = sf.Close() }()

	ti, ok := sf.Tensor("scalar")
	if !ok {
		t.Fatalf("scalar tensor not found")
	}
	if len(ti.Shape) != 0 {
		t.Fatalf("shape=%v want scalar rank-0", ti.Shape)
	}

	r, info, err := sf.TensorReader("scalar")
	if err != nil {
		t.Fatalf("TensorReader: %v", err)
	}
	if len(info.Shape) != 0 {
		t.Fatalf("reader shape=%v want scalar rank-0", info.Shape)
	}
	raw, err := io.ReadAll(r)
	if err != nil {
		t.Fatalf("ReadAll: %v", err)
	}
	if len(raw) != 2 {
		t.Fatalf("payload len=%d want 2", len(raw))
	}
}
