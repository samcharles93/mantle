package mcf

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

func TestOpenReaderAtRoundTrip(t *testing.T) {
	t.Parallel()

	path := filepath.Join(t.TempDir(), "model.mcf")
	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("create file: %v", err)
	}

	w, err := NewWriter(f)
	if err != nil {
		t.Fatalf("new writer: %v", err)
	}
	if err := w.WriteSection(SectionModelInfo, 1, []byte("model-info")); err != nil {
		t.Fatalf("write model info: %v", err)
	}
	if err := w.WriteSection(SectionTensorData, 1, []byte{1, 2, 3, 4, 5, 6}); err != nil {
		t.Fatalf("write tensor data: %v", err)
	}
	if err := w.Finalise(); err != nil {
		t.Fatalf("finalise: %v", err)
	}
	if err := f.Close(); err != nil {
		t.Fatalf("close writer file: %v", err)
	}

	rf, err := os.Open(path)
	if err != nil {
		t.Fatalf("open file: %v", err)
	}
	defer func() { _ = rf.Close() }()

	st, err := rf.Stat()
	if err != nil {
		t.Fatalf("stat: %v", err)
	}
	mf, err := OpenReaderAt(rf, st.Size())
	if err != nil {
		t.Fatalf("open readerat: %v", err)
	}
	defer func() {
		if cerr := mf.Close(); cerr != nil {
			t.Fatalf("close mcf file: %v", cerr)
		}
	}()

	if mf.mmapped {
		t.Fatalf("OpenReaderAt should not mmap")
	}
	if mf.Header == nil {
		t.Fatalf("missing header")
	}
	if mf.Header.HeaderSize != mcfHeaderSize {
		t.Fatalf("header size mismatch: got %d want %d", mf.Header.HeaderSize, mcfHeaderSize)
	}

	modelSec := mf.Section(SectionModelInfo)
	if modelSec == nil {
		t.Fatalf("missing model info section")
	}
	got := mf.SectionData(modelSec)
	if !bytes.Equal(got, []byte("model-info")) {
		t.Fatalf("model info mismatch: got %q", string(got))
	}
}

func TestHeaderAndSectionEncodingLittleEndian(t *testing.T) {
	t.Parallel()

	h := MCFHeader{
		Magic:            [4]byte{'M', 'C', 'F', 0},
		Major:            0x1122,
		Minor:            0x3344,
		HeaderSize:       mcfHeaderSize,
		SectionCount:     7,
		SectionDirOffset: 0x0102030405060708,
		FileSize:         0x1112131415161718,
		Flags:            0x2122232425262728,
	}
	var hdrRaw [mcfHeaderSize]byte
	if !encodeHeader(hdrRaw[:], h) {
		t.Fatalf("encode header failed")
	}
	if hdrRaw[4] != 0x22 || hdrRaw[5] != 0x11 {
		t.Fatalf("major is not little-endian: %x", hdrRaw[4:6])
	}
	if hdrRaw[16] != 0x08 || hdrRaw[23] != 0x01 {
		t.Fatalf("section dir offset is not little-endian: %x", hdrRaw[16:24])
	}
	decodedH, ok := decodeHeader(hdrRaw[:])
	if !ok {
		t.Fatalf("decode header failed")
	}
	if decodedH != h {
		t.Fatalf("header round-trip mismatch: got %+v want %+v", decodedH, h)
	}

	s := MCFSection{
		Type:    0x11223344,
		Version: 0x55667788,
		Offset:  0x0102030405060708,
		Size:    0x1112131415161718,
	}
	var secRaw [mcfSectionSize]byte
	if !encodeSection(secRaw[:], s) {
		t.Fatalf("encode section failed")
	}
	if secRaw[0] != 0x44 || secRaw[3] != 0x11 {
		t.Fatalf("section type is not little-endian: %x", secRaw[0:4])
	}
	if secRaw[8] != 0x08 || secRaw[15] != 0x01 {
		t.Fatalf("section offset is not little-endian: %x", secRaw[8:16])
	}
	decodedS, ok := decodeSection(secRaw[:])
	if !ok {
		t.Fatalf("decode section failed")
	}
	if decodedS != s {
		t.Fatalf("section round-trip mismatch: got %+v want %+v", decodedS, s)
	}
}
