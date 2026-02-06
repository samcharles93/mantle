package mcfstore

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/samcharles93/mantle/pkg/mcf"
)

func TestOpenAndReadTensorF32(t *testing.T) {
	t.Parallel()

	modelPath := filepath.Join(t.TempDir(), "test.mcf")
	if err := writeTestMCF(modelPath, "weight", []uint64{2, 2}, []float32{1.5, -2.0, 3.25, 4.5}); err != nil {
		t.Fatalf("write mcf: %v", err)
	}

	f, err := Open(modelPath)
	if err != nil {
		t.Fatalf("open mcfstore: %v", err)
	}
	defer func() {
		if cerr := f.Close(); cerr != nil {
			t.Fatalf("close mcfstore: %v", cerr)
		}
	}()

	info, err := f.Tensor("weight")
	if err != nil {
		t.Fatalf("tensor metadata: %v", err)
	}
	if info.DType != mcf.DTypeF32 {
		t.Fatalf("dtype mismatch: got %v want %v", info.DType, mcf.DTypeF32)
	}
	if len(info.Shape) != 2 || info.Shape[0] != 2 || info.Shape[1] != 2 {
		t.Fatalf("shape mismatch: got %v", info.Shape)
	}

	vals, _, err := f.ReadTensorF32("weight")
	if err != nil {
		t.Fatalf("read tensor f32: %v", err)
	}
	want := []float32{1.5, -2.0, 3.25, 4.5}
	if len(vals) != len(want) {
		t.Fatalf("length mismatch: got %d want %d", len(vals), len(want))
	}
	for i := range vals {
		if vals[i] != want[i] {
			t.Fatalf("value mismatch at %d: got %v want %v", i, vals[i], want[i])
		}
	}
}

func TestTensorMissing(t *testing.T) {
	t.Parallel()

	modelPath := filepath.Join(t.TempDir(), "test.mcf")
	if err := writeTestMCF(modelPath, "weight", []uint64{1}, []float32{42}); err != nil {
		t.Fatalf("write mcf: %v", err)
	}
	f, err := Open(modelPath)
	if err != nil {
		t.Fatalf("open mcfstore: %v", err)
	}
	defer func() { _ = f.Close() }()

	_, err = f.Tensor("missing")
	if err == nil {
		t.Fatalf("expected missing tensor error")
	}
	if err != ErrTensorNotFound {
		t.Fatalf("unexpected error: %v", err)
	}
}

func writeTestMCF(path, tensorName string, shape []uint64, vals []float32) error {
	out, err := os.Create(path)
	if err != nil {
		return err
	}
	defer func() { _ = out.Close() }()

	w, err := mcf.NewWriter(out)
	if err != nil {
		return err
	}
	if err := w.WriteSection(mcf.SectionModelInfo, 1, []byte("{}")); err != nil {
		return err
	}

	sw, err := w.BeginSection(mcf.SectionTensorData, 1)
	if err != nil {
		return err
	}
	dataOff, err := sw.CurrentAbsOffset()
	if err != nil {
		return err
	}

	raw := make([]byte, 4*len(vals))
	for i, v := range vals {
		binary.LittleEndian.PutUint32(raw[i*4:], math.Float32bits(v))
	}
	if _, err := sw.Write(raw); err != nil {
		return err
	}
	if err := sw.End(); err != nil {
		return err
	}

	indexPayload, err := mcf.EncodeTensorIndexSection([]mcf.TensorIndexRecord{{
		Name:     tensorName,
		DType:    mcf.DTypeF32,
		Shape:    shape,
		DataOff:  dataOff,
		DataSize: uint64(len(raw)),
	}})
	if err != nil {
		return err
	}
	if err := w.WriteSection(mcf.SectionTensorIndex, mcf.TensorIndexVersion, indexPayload); err != nil {
		return err
	}

	return w.Finalise()
}
