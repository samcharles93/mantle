package mcf

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"math/rand/v2"
	"os"
	"path/filepath"
	"sort"
	"testing"
)

// writeSafetensors writes tensors (name -> dtype, shape, bytes) as a single
// safetensors file.
func writeSafetensors(t *testing.T, path string, tensors map[string]struct {
	dtype string
	shape []int64
	data  []byte
}) {
	t.Helper()
	names := make([]string, 0, len(tensors))
	for n := range tensors {
		names = append(names, n)
	}
	sort.Strings(names)
	header := map[string]any{}
	var body bytes.Buffer
	for _, n := range names {
		tt := tensors[n]
		start := body.Len()
		body.Write(tt.data)
		header[n] = map[string]any{"dtype": tt.dtype, "shape": tt.shape, "data_offsets": []int{start, body.Len()}}
	}
	hj, err := json.Marshal(header)
	if err != nil {
		t.Fatal(err)
	}
	var out bytes.Buffer
	_ = binary.Write(&out, binary.LittleEndian, uint64(len(hj)))
	out.Write(hj)
	out.Write(body.Bytes())
	if err := os.WriteFile(path, out.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestPackImportsCompressedTensors(t *testing.T) {
	dir := t.TempDir()
	cfg := `{"model_type": "qwen3", "quantization_config": {"quant_method": "compressed-tensors", "format": "pack-quantized",
		"config_groups": {"g0": {"targets": ["Linear"], "weights": {"num_bits": 4, "type": "int", "symmetric": true, "strategy": "group", "group_size": 128}}}}}`
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(cfg), 0o644); err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewPCG(41, 42))
	const rows, cols = 8, 256
	in, codes, scales := ctFixture(rng, rows, cols, 4, 128, "BF16")
	shape := binary.LittleEndian.AppendUint64(binary.LittleEndian.AppendUint64(nil, rows), cols)
	norm := make([]byte, cols*2)
	writeSafetensors(t, filepath.Join(dir, "model.safetensors"), map[string]struct {
		dtype string
		shape []int64
		data  []byte
	}{
		"model.layers.0.mlp.up_proj.weight_packed": {"I32", []int64{rows, int64(in.PackedCols)}, in.Packed},
		"model.layers.0.mlp.up_proj.weight_scale":  {"BF16", []int64{rows, int64(in.ScaleCols)}, in.Scale},
		"model.layers.0.mlp.up_proj.weight_shape":  {"I64", []int64{2}, shape},
		"model.norm.weight":                        {"BF16", []int64{cols}, norm},
	})

	out := filepath.Join(dir, "out.mcf")
	if err := Pack(PackOptions{InputDir: dir, OutputPath: out, IncludeResources: true}); err != nil {
		t.Fatalf("Pack: %v", err)
	}
	f, err := Open(out)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = f.Close() }()
	if f.Header.Flags&FlagTensorDataAligned64 == 0 {
		t.Fatal("aligned64 flag not set")
	}
	if f.Header.Minor != 2 {
		t.Fatalf("header minor %d, want 2", f.Header.Minor)
	}
	idx, err := ParseTensorIndexSection(f.SectionData(f.Section(SectionTensorIndex)))
	if err != nil {
		t.Fatal(err)
	}
	var names []string
	gqIndex := -1
	for i := range idx.Count() {
		n, _ := idx.Name(i)
		names = append(names, n)
		if n == "model.layers.0.mlp.up_proj.weight" {
			gqIndex = i
		}
	}
	want := []string{"model.layers.0.mlp.up_proj.weight", "model.norm.weight"}
	if len(names) != len(want) || names[0] != want[0] || names[1] != want[1] {
		t.Fatalf("tensors %v, want %v", names, want)
	}
	e, _ := idx.Entry(gqIndex)
	if wantDT, _ := GQDType(4, 128); e.DType != wantDT {
		t.Fatalf("dtype %#x, want %#x", e.DType, wantDT)
	}
	qi, err := ParseQuantInfoSection(f.SectionData(f.Section(SectionQuantInfo)))
	if err != nil {
		t.Fatal(err)
	}
	recs := qi.Records()
	if len(recs) != 1 || recs[0].TensorIndex != uint32(gqIndex) || recs[0].BlockSize != 128 {
		t.Fatalf("quant records %+v, want one for index %d with block 128", recs, gqIndex)
	}

	dt, _ := GQDType(4, 128)
	l, _ := GQPayloadLayoutFor([]uint64{rows, cols}, dt)
	file, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	payload := file[e.DataOff : e.DataOff+e.DataSize]
	row := make([]float32, cols)
	for r := range rows {
		if err := DecodeGQRow(payload, l, r, cols, row); err != nil {
			t.Fatal(err)
		}
		for c, v := range row {
			if want := scales[r*in.ScaleCols+c/128] * float32(codes[r*cols+c]); v != want {
				t.Fatalf("r=%d c=%d: got %v want %v", r, c, v, want)
			}
		}
	}
}
