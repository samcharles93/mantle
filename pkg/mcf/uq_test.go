package mcf

import (
	"bytes"
	"math"
	"math/rand"
	"testing"
)

func TestUQCodecRoundTrip(t *testing.T) {
	for _, dt := range []TensorDType{DTypeUQ2, DTypeUQ3, DTypeUQ4, DTypeUQ5, DTypeUQ6, DTypeUQ7, DTypeUQ8} {
		bits, _ := UQBits(dt)
		rows, cols := uint64(3), uint64(96) // 3 blocks/row, 1 super/row
		layout, err := UQPayloadLayoutFor([]uint64{rows, cols}, dt)
		if err != nil {
			t.Fatalf("%v: %v", dt, err)
		}
		wantSize, err := QuantPayloadSize([]uint64{rows, cols}, dt)
		if err != nil {
			t.Fatal(err)
		}
		if layout.TotalBytes != wantSize {
			t.Fatalf("%v: layout total %d != QuantPayloadSize %d", dt, layout.TotalBytes, wantSize)
		}

		cbElems := uint64(1) << uint(bits)
		codebooks := make([]float32, layout.SuperCount*cbElems)
		for i := range codebooks {
			codebooks[i] = float32(i%int(cbElems))*0.25 - 2 // spread of levels
		}
		scales := make([]float32, layout.TotalBlocks)
		for i := range scales {
			scales[i] = 0.01 * float32(i+1)
		}
		indices := make([]uint8, rows*cols)
		for i := range indices {
			indices[i] = uint8(i % int(cbElems))
		}

		payload, err := EncodeUQPayload([]uint64{rows, cols}, dt, codebooks, scales, indices)
		if err != nil {
			t.Fatalf("%v: encode: %v", dt, err)
		}
		if uint64(len(payload)) != wantSize {
			t.Fatalf("%v: payload size %d want %d", dt, len(payload), wantSize)
		}

		// Reference decode: w = blockScale(f16) * codebook(f16)[idx].
		dst := make([]float32, cols)
		for r := uint64(0); r < rows; r++ {
			if err := DecodeUQRow(payload, layout, int(r), int(cols), dst); err != nil {
				t.Fatalf("%v: decode row %d: %v", dt, r, err)
			}
			for c := uint64(0); c < cols; c++ {
				b := r*layout.BlocksPerRow + c/32
				super := r*layout.SuperBlocksPerRow + (c/32)/8
				idx := indices[r*cols+c]
				wantScale := fp16ToFloat32(float32ToFP16Bits(scales[b]))
				wantCB := fp16ToFloat32(float32ToFP16Bits(codebooks[super*cbElems+uint64(idx)]))
				want := wantScale * wantCB
				got := dst[c]
				if math.Abs(float64(got-want)) > 1e-6 {
					t.Fatalf("%v: row %d col %d: got %v want %v", dt, r, c, got, want)
				}
			}
		}
	}
}

func TestUQErrors(t *testing.T) {
	if !IsUQDType(DTypeUQ4) || IsUQDType(DTypeK4) {
		t.Fatal("IsUQDType wrong")
	}
	if b, ok := UQBits(DTypeUQ4); !ok || b != 4 {
		t.Fatal("UQBits wrong")
	}
	if _, ok := UQBits(DTypeF32); ok {
		t.Fatal("UQBits(F32) should fail")
	}
	if _, err := UQPayloadLayoutFor([]uint64{4, 4, 4}, DTypeUQ4); err == nil {
		t.Fatal("rank-3 layout should fail")
	}
	if _, err := UQPayloadLayoutFor([]uint64{2, 64}, DTypeK4); err == nil {
		t.Fatal("non-uq dtype should fail")
	}
	// index out of range
	layout, _ := UQPayloadLayoutFor([]uint64{1, 32}, DTypeUQ2)
	_, err := EncodeUQPayload([]uint64{1, 32}, DTypeUQ2, make([]float32, layout.SuperCount*4), make([]float32, 1), []uint8{4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0})
	if err == nil {
		t.Fatal("index 4 with 2 bits should fail")
	}
	// truncated payload decode
	payload, err := EncodeUQPayload([]uint64{1, 32}, DTypeUQ2, []float32{1, 2, 3, 4}, []float32{0.5}, make([]uint8, 32))
	if err != nil {
		t.Fatal(err)
	}
	if err := DecodeUQRow(payload[:len(payload)-1], layout, 0, 32, make([]float32, 32)); err == nil {
		t.Fatal("truncated payload should fail")
	}
}

// TestEncodeUQTensorDerivedCodebook checks the producer path with derived
// k-means codebooks: payload size matches the layout and decoded rows
// reconstruct the source within a sane relative error.
func TestEncodeUQTensorDerivedCodebook(t *testing.T) {
	rows, cols := 4, 256 // one super-block per row
	src := make([]float32, rows*cols)
	rng := rand.New(rand.NewSource(7))
	for i := range src {
		src[i] = float32(rng.NormFloat64() * 0.05)
	}
	dt, payload, err := EncodeUQTensor(src, rows, cols, 4, nil)
	if err != nil {
		t.Fatalf("EncodeUQTensor: %v", err)
	}
	if dt != DTypeUQ4 {
		t.Fatalf("dtype %v, want UQ4", dt)
	}
	layout, err := UQPayloadLayoutFor([]uint64{uint64(rows), uint64(cols)}, dt)
	if err != nil {
		t.Fatal(err)
	}
	if uint64(len(payload)) != layout.TotalBytes {
		t.Fatalf("payload %d bytes, layout says %d", len(payload), layout.TotalBytes)
	}
	dst := make([]float32, cols)
	var sumSq float64
	for _, v := range src {
		sumSq += float64(v) * float64(v)
	}
	rms := math.Sqrt(sumSq / float64(len(src)))
	// 4-bit non-uniform codebook on gaussian data: quantization noise should be
	// a small fraction of the tensor's variance
	var sumErrSq float64
	for r := range rows {
		if err := DecodeUQRow(payload, layout, r, cols, dst); err != nil {
			t.Fatalf("decode row %d: %v", r, err)
		}
		for c := range cols {
			d := float64(dst[c] - src[r*cols+c])
			sumErrSq += d * d
		}
	}
	rmse := math.Sqrt(sumErrSq / float64(len(src)))
	if rmse > 0.25*rms {
		t.Fatalf("rmse %g too large vs rms %g", rmse, rms)
	}
}

// TestEncodeUQTensorExplicitCodebook checks that provided codebooks are used
// verbatim and indices pick the nearest scaled level.
func TestEncodeUQTensorExplicitCodebook(t *testing.T) {
	rows, cols := 1, 256
	levels := make([]float32, 16) // 16 = 2^4, range [-8, 7]
	for i := range levels {
		levels[i] = float32(i) - 8
	}
	src := make([]float32, cols)
	for i := range src {
		src[i] = float32(i%16) - 8 // sweeps the codebook range exactly
	}
	cbs := [][]float32{levels}
	dt, payload, err := EncodeUQTensor(src, rows, cols, 4, cbs)
	if err != nil {
		t.Fatalf("EncodeUQTensor: %v", err)
	}
	layout, _ := UQPayloadLayoutFor([]uint64{1, uint64(cols)}, dt)
	dst := make([]float32, cols)
	if err := DecodeUQRow(payload, layout, 0, cols, dst); err != nil {
		t.Fatal(err)
	}
	for i := range src {
		if d := math.Abs(float64(dst[i] - src[i])); d > 0.51 { // levels are 1 apart
			t.Fatalf("elem %d: got %g want %g", i, dst[i], src[i])
		}
	}
	// determinism
	_, payload2, err := EncodeUQTensor(src, rows, cols, 4, cbs)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(payload, payload2) {
		t.Fatal("encode not deterministic")
	}
}

// TestUQDTypeForBits checks the bits->dtype mapping.
func TestUQDTypeForBits(t *testing.T) {
	for bits := 2; bits <= 8; bits++ {
		dt, err := UQDTypeForBits(bits)
		if err != nil {
			t.Fatal(err)
		}
		if b, ok := UQBits(dt); !ok || b != bits {
			t.Fatalf("bits %d -> dtype %v -> bits %d", bits, dt, b)
		}
	}
	if _, err := UQDTypeForBits(1); err == nil {
		t.Fatal("bits 1 should fail")
	}
	if _, err := UQDTypeForBits(9); err == nil {
		t.Fatal("bits 9 should fail")
	}
}
