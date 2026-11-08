package mcf

import (
	"math"
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
