package mcf

import (
	"bytes"
	"encoding/binary"
	"testing"
)

// mustWrite writes v to buf; bytes.Buffer writes never fail, but errcheck
// requires the error to be checked.
func mustWrite(buf *bytes.Buffer, v any) {
	if err := binary.Write(buf, binary.LittleEndian, v); err != nil {
		panic(err)
	}
}

// buildTestGGUF assembles a minimal GGUF v3 container in memory.
type testGGTensor struct {
	name   string
	dims   []uint64
	ggtype uint32
	data   []byte
}

func buildTestGGUF(t *testing.T, kvs map[string]uint32, tensors []testGGTensor) []byte {
	t.Helper()
	var buf bytes.Buffer
	buf.WriteString(ggufMagic)
	mustWrite(&buf, uint32(3))
	mustWrite(&buf, uint64(len(tensors)))
	mustWrite(&buf, uint64(len(kvs)))
	// deterministic order for reproducible bytes
	keys := make([]string, 0, len(kvs))
	for k := range kvs {
		keys = append(keys, k)
	}
	for i := 0; i < len(keys); i++ {
		for j := i + 1; j < len(keys); j++ {
			if keys[j] < keys[i] {
				keys[i], keys[j] = keys[j], keys[i]
			}
		}
	}
	for _, k := range keys {
		mustWrite(&buf, uint64(len(k)))
		buf.WriteString(k)
		mustWrite(&buf, kvs[k]) // value type
		switch kvs[k] {
		case ggufValUint32:
			mustWrite(&buf, uint32(32))
		case ggufValString:
			s := "test"
			mustWrite(&buf, uint64(len(s)))
			buf.WriteString(s)
		default:
			t.Fatalf("unsupported test kv type %d", kvs[k])
		}
	}
	const align = 32
	offsets := make([]uint64, len(tensors))
	var data bytes.Buffer
	for i, tt := range tensors {
		offsets[i] = uint64(data.Len())
		data.Write(tt.data)
	}
	for i, tt := range tensors {
		mustWrite(&buf, uint64(len(tt.name)))
		buf.WriteString(tt.name)
		mustWrite(&buf, uint32(len(tt.dims)))
		for _, d := range tt.dims {
			mustWrite(&buf, d)
		}
		mustWrite(&buf, tt.ggtype)
		mustWrite(&buf, offsets[i])
	}
	for buf.Len()%align != 0 {
		buf.WriteByte(0)
	}
	buf.Write(data.Bytes())
	return buf.Bytes()
}

// f16bits encodes a float32 as IEEE half bits (values used in tests are exact).
func f16bits(f float32) uint16 {
	switch f {
	case 0:
		return 0
	case 0.5:
		return 0x3800
	case 0.25:
		return 0x3400
	case 1:
		return 0x3C00
	case 2:
		return 0x4000
	default:
		panic("test helper: non-exact f16 value")
	}
}

func TestParseGGUFAndTranscodeQ8_0(t *testing.T) {
	// One Q8_0 tensor: 2 rows x 64 cols = 4 blocks of 32.
	rows, cols := uint64(2), uint64(64)
	blocks := rows * (cols / 32)
	payload := make([]byte, blocks*34)
	scales := []float32{0.5, 0.25, 1, 2}
	for b := uint64(0); b < blocks; b++ {
		binary.LittleEndian.PutUint16(payload[b*34:], f16bits(scales[b]))
		for i := uint64(0); i < 32; i++ {
			payload[b*34+2+i] = byte(int8(int64(b)*32 + int64(i) - 64)) // distinct values
		}
	}
	raw := buildTestGGUF(t, map[string]uint32{"general.alignment": ggufValUint32, "general.name": ggufValString},
		[]testGGTensor{{name: "w", dims: []uint64{cols, rows}, ggtype: ggmlTypeQ8_0, data: payload}})

	f, err := ParseGGUF(bytes.NewReader(raw), int64(len(raw)))
	if err != nil {
		t.Fatalf("ParseGGUF: %v", err)
	}
	if f.Version != 3 {
		t.Fatalf("version = %d", f.Version)
	}
	if f.Alignment != 32 {
		t.Fatalf("alignment = %d", f.Alignment)
	}
	if got := f.Metadata["general.name"]; got != "test" {
		t.Fatalf("metadata general.name = %v", got)
	}
	if f.TensorCount() != 1 {
		t.Fatalf("tensor count = %d", f.TensorCount())
	}
	gt, ok := f.FindTensor("w")
	if !ok {
		t.Fatal("FindTensor(w) not found")
	}
	if gt.Type != ggmlTypeQ8_0 || gt.Dims[0] != cols || gt.Dims[1] != rows {
		t.Fatalf("tensor info = %+v", gt)
	}
	got, err := f.TensorPayload(gt)
	if err != nil {
		t.Fatalf("TensorPayload: %v", err)
	}
	if !bytes.Equal(got, payload) {
		t.Fatal("payload bytes differ")
	}

	dt, shape, mcfPayload, err := TranscodeGGUFQuant(gt, got)
	if err != nil {
		t.Fatalf("TranscodeGGUFQuant: %v", err)
	}
	if dt != DTypeQ8 {
		t.Fatalf("dtype = %v", dt)
	}
	if len(shape) != 2 || shape[0] != rows || shape[1] != cols {
		t.Fatalf("shape = %v", shape)
	}
	wantSize, err := QuantPayloadSize(shape, DTypeQ8)
	if err != nil {
		t.Fatal(err)
	}
	if uint64(len(mcfPayload)) != wantSize {
		t.Fatalf("mcf payload size = %d want %d", len(mcfPayload), wantSize)
	}
	// Region 1: f16 scales at the start.
	for b := uint64(0); b < blocks; b++ {
		gotScale := binary.LittleEndian.Uint16(mcfPayload[b*2:])
		if gotScale != f16bits(scales[b]) {
			t.Fatalf("block %d scale bits = %x want %x", b, gotScale, f16bits(scales[b]))
		}
	}
	// Region 2: data after align64(scales).
	dataOff := (blocks*2 + 63) &^ 63
	for b := uint64(0); b < blocks; b++ {
		for i := uint64(0); i < 32; i++ {
			want := payload[b*34+2+i]
			got := mcfPayload[dataOff+b*32+i]
			if got != want {
				t.Fatalf("data byte block %d elem %d = %d want %d", b, i, got, want)
			}
		}
	}
}

func TestParseGGUFErrors(t *testing.T) {
	raw := buildTestGGUF(t, nil, []testGGTensor{{name: "w", dims: []uint64{32, 1}, ggtype: ggmlTypeQ8_0, data: make([]byte, 34)}})

	if _, err := ParseGGUF(bytes.NewReader(raw), int64(len(raw))); err != nil {
		t.Fatalf("valid file rejected: %v", err)
	}
	bad := append([]byte(nil), raw...)
	bad[0] = 'X'
	if _, err := ParseGGUF(bytes.NewReader(bad), int64(len(bad))); err == nil {
		t.Fatal("bad magic accepted")
	}
	if _, err := ParseGGUF(bytes.NewReader(raw[:20]), 20); err == nil {
		t.Fatal("truncated file accepted")
	}
	// Corrupt the tensor offset field. With no metadata: header is 24 bytes,
	// then name (8+1), n_dims (4), dims (8), type (4), offset (8).
	badOff := append([]byte(nil), raw...)
	offPos := 24 + 8 + 1 + 4 + 8 + 4
	binary.LittleEndian.PutUint64(badOff[offPos:offPos+8], 1<<40)
	f, err := ParseGGUF(bytes.NewReader(badOff), int64(len(badOff)))
	if err != nil {
		t.Fatalf("parse with huge offset should succeed (payload read must fail): %v", err)
	}
	gt, _ := f.FindTensor("w")
	if _, err := f.TensorPayload(gt); err == nil {
		t.Fatal("out-of-bounds payload accepted")
	}
	// Unsupported transcode type.
	if _, _, _, err := TranscodeGGUFQuant(GGUFTensor{Name: "w", Dims: []uint64{32, 1}, Type: ggmlTypeQ4_K}, make([]byte, 144)); err == nil {
		t.Fatal("Q4_K transcode should be rejected")
	}
	// Rank-3 Q8_0 rejected.
	if _, _, _, err := TranscodeGGUFQuant(GGUFTensor{Name: "w", Dims: []uint64{32, 1, 1}, Type: ggmlTypeQ8_0}, make([]byte, 34)); err == nil {
		t.Fatal("rank-3 transcode should be rejected")
	}
	// cols not multiple of 32 rejected.
	if _, _, _, err := TranscodeGGUFQuant(GGUFTensor{Name: "w", Dims: []uint64{48, 1}, Type: ggmlTypeQ8_0}, nil); err == nil {
		t.Fatal("cols=48 transcode should be rejected")
	}
}

func TestGGMLTypeName(t *testing.T) {
	if GGMLTypeName(ggmlTypeQ8_0) != "Q8_0" || GGMLTypeName(999) != "type<999>" {
		t.Fatal("GGMLTypeName mismatch")
	}
	if _, ok := ggmlTypeBlocks[ggmlTypeQ8_0]; !ok || ggmlTypeBlocks[ggmlTypeQ8_0].BlockBytes != 34 {
		t.Fatal("Q8_0 block layout wrong")
	}
}
