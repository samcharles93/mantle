package mcf

import (
	"math"
	"math/rand/v2"
	"testing"
)

func TestGQDTypeRegistry(t *testing.T) {
	seen := map[TensorDType]bool{}
	for _, group := range []int{32, 64, 128} {
		for bits := 2; bits <= 8; bits++ {
			dt, err := GQDType(bits, group)
			if err != nil {
				t.Fatalf("GQDType(%d, %d): %v", bits, group, err)
			}
			if dt > 0xFF {
				t.Fatalf("dtype %#x does not fit QuantRecord.Method", dt)
			}
			if seen[dt] {
				t.Fatalf("dtype %#x assigned twice", dt)
			}
			seen[dt] = true
			b, g, ok := GQParams(dt)
			if !ok || b != bits || g != group {
				t.Fatalf("GQParams(%#x) = %d, %d, %v; want %d, %d", dt, b, g, ok, bits, group)
			}
			if !DTypeRequiresAligned64(dt) {
				t.Fatalf("%#x not marked aligned64", dt)
			}
			if IsUQDType(dt) {
				t.Fatalf("%#x overlaps uq*", dt)
			}
		}
	}
	for _, bad := range [][2]int{{1, 128}, {9, 128}, {4, 16}, {4, 256}} {
		if _, err := GQDType(bad[0], bad[1]); err == nil {
			t.Errorf("GQDType(%d, %d) accepted", bad[0], bad[1])
		}
	}
	for _, dt := range []TensorDType{DTypeQ4, DTypeK4, DTypeUQ4, 0x0057, 0x004F, 0x0077} {
		if IsGQDType(dt) {
			t.Errorf("%#x reported as gq*", dt)
		}
	}
}

func TestGQPayloadLayout(t *testing.T) {
	dt, _ := GQDType(3, 128)
	l, err := GQPayloadLayoutFor([]uint64{5, 300}, dt)
	if err != nil {
		t.Fatal(err)
	}
	// groups per row: ceil(300/128)=3 -> 15 scales = 30 bytes -> data at 64.
	// blocks per row: ceil(300/32)=10, 12 bytes each -> 5*10*12 = 600 bytes.
	want := GQPayloadLayout{Bits: 3, Group: 128, GroupsPerRow: 3, BlocksPerRow: 10, BlockDataBytes: 12,
		ScaleOff: 0, ScaleBytes: 30, DataOff: 64, DataBytes: 600, TotalBytes: 664}
	if l != want {
		t.Fatalf("layout %+v, want %+v", l, want)
	}
	size, err := QuantPayloadSize([]uint64{5, 300}, dt)
	if err != nil || size != want.TotalBytes {
		t.Fatalf("QuantPayloadSize = %d, %v; want %d", size, err, want.TotalBytes)
	}
}

// TestEncodeGQPayloadExact stores arbitrary in-range codes and bf16 scales and
// checks that every element decodes to exactly scale*code.
func TestEncodeGQPayloadExact(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 8))
	const rows, cols = 6, 200
	for _, group := range []int{32, 64, 128} {
		for bits := 2; bits <= 8; bits++ {
			dt, _ := GQDType(bits, group)
			l, _ := GQPayloadLayoutFor([]uint64{rows, cols}, dt)
			scales := make([]uint16, rows*int(l.GroupsPerRow))
			for i := range scales {
				scales[i] = BF16FromFloat32(float32(rng.NormFloat64() * 0.01))
			}
			lo, hi := -(1 << (bits - 1)), (1<<(bits-1))-1
			codes := make([]int8, rows*cols)
			for i := range codes {
				codes[i] = int8(lo + rng.IntN(hi-lo+1))
			}
			codes[0], codes[1] = int8(lo), int8(hi)
			payload, err := EncodeGQPayload([]uint64{rows, cols}, dt, scales, codes)
			if err != nil {
				t.Fatalf("bits=%d group=%d: %v", bits, group, err)
			}
			row := make([]float32, cols)
			for r := range rows {
				if err := DecodeGQRow(payload, l, r, cols, row); err != nil {
					t.Fatal(err)
				}
				for c, v := range row {
					want := BF16ToFloat32(scales[r*int(l.GroupsPerRow)+c/group]) * float32(codes[r*cols+c])
					if v != want {
						t.Fatalf("bits=%d group=%d r=%d c=%d: got %v want %v", bits, group, r, c, v, want)
					}
				}
			}
		}
	}
}

func TestEncodeGQPayloadRejectsBadInput(t *testing.T) {
	dt, _ := GQDType(2, 32)
	shape := []uint64{1, 32}
	codes := make([]int8, 32)
	codes[3] = 2 // outside [-2, 1]
	if _, err := EncodeGQPayload(shape, dt, []uint16{0x3f80}, codes); err == nil {
		t.Error("out-of-range code accepted")
	}
	if _, err := EncodeGQPayload(shape, dt, nil, make([]int8, 32)); err == nil {
		t.Error("missing scales accepted")
	}
	if _, err := EncodeGQPayload(shape, DTypeQ4, []uint16{0}, make([]int8, 32)); err == nil {
		t.Error("non-gq dtype accepted")
	}
}

func TestEncodeGQTensorRoundTrip(t *testing.T) {
	const rows, cols = 16, 384
	src := syntheticWeights(rows, cols, 9)
	for _, group := range []int{32, 128} {
		for _, bits := range []int{2, 3, 4, 8} {
			dt, payload, err := EncodeGQTensor(src, rows, cols, bits, group)
			if err != nil {
				t.Fatal(err)
			}
			l, _ := GQPayloadLayoutFor([]uint64{rows, cols}, dt)
			if uint64(len(payload)) != l.TotalBytes {
				t.Fatalf("payload %d bytes, want %d", len(payload), l.TotalBytes)
			}
			row := make([]float32, cols)
			var errSq, baseSq float64
			for r := range rows {
				if err := DecodeGQRow(payload, l, r, cols, row); err != nil {
					t.Fatal(err)
				}
				for g := 0; g < cols; g += group {
					x := src[r*cols+g : r*cols+g+group]
					for i, v := range row[g : g+group] {
						d := float64(v) - float64(x[i])
						errSq += float64(x[i]) * float64(x[i]) * d * d
					}
					baseSq += absmaxErrorBF16(x, (1<<(bits-1))-1)
				}
			}
			if errSq >= baseSq {
				t.Errorf("bits=%d group=%d: weighted error %.6g not below absmax %.6g", bits, group, errSq, baseSq)
			}
		}
	}
}

func TestQuantInfoValidatesGQ(t *testing.T) {
	dt, _ := GQDType(4, 64)
	good := QuantRecord{Method: uint8(dt), Domain: uint8(DomainWeights), BlockSize: 64}
	if err := validateQuantRecord(good); err != nil {
		t.Fatalf("valid gq record rejected: %v", err)
	}
	for _, bad := range []QuantRecord{
		{Method: uint8(dt), Domain: uint8(DomainWeights), BlockSize: 32},
		{Method: uint8(dt), Domain: uint8(DomainWeights), BlockSize: 64, SuperSize: 256},
		{Method: uint8(dt), Domain: uint8(DomainActivations), BlockSize: 64},
	} {
		if err := validateQuantRecord(bad); err == nil {
			t.Errorf("invalid gq record accepted: %+v", bad)
		}
	}
}

func TestBF16Helpers(t *testing.T) {
	for _, f := range []float32{0, 1, -2.5, 0.0078125, 3.0e-6} {
		if got := BF16ToFloat32(BF16FromFloat32(f)); math.Abs(float64(got-f)) > math.Abs(float64(f))/128 {
			t.Errorf("bf16 round trip %v -> %v", f, got)
		}
	}
}

func BenchmarkEncodeGQTensor4(b *testing.B) {
	const rows, cols = 256, 2048
	src := syntheticWeights(rows, cols, 10)
	b.SetBytes(int64(rows * cols * 4))
	for b.Loop() {
		if _, _, err := EncodeGQTensor(src, rows, cols, 4, 128); err != nil {
			b.Fatal(err)
		}
	}
}

// absmaxErrorBF16 is absmaxError with the scale rounded to bf16, as gq*
// stores it.
func absmaxErrorBF16(x []float32, hi int) float64 {
	var amax float32
	for _, v := range x {
		amax = max(amax, float32(math.Abs(float64(v))))
	}
	s := BF16ToFloat32(BF16FromFloat32(amax / float32(hi)))
	var e float64
	for _, v := range x {
		q := 0.0
		if s != 0 {
			q = math.Max(-float64(hi), math.Min(float64(hi), math.Round(float64(v/s))))
		}
		d := float64(v) - float64(s)*q
		e += float64(v) * float64(v) * d * d
	}
	return e
}
