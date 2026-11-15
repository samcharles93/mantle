package mcf

import (
	"bytes"
	"math"
	"math/rand/v2"
	"runtime"
	"testing"
)

// syntheticWeights returns a deterministic rows x cols matrix shaped like LLM
// weights: Gaussian rows with differing scales and a few large outliers.
func syntheticWeights(rows, cols int, seed uint64) []float32 {
	rng := rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15))
	out := make([]float32, rows*cols)
	for r := range rows {
		rowScale := 0.01 + 0.04*rng.Float64()
		for c := range cols {
			v := rng.NormFloat64() * rowScale
			if rng.IntN(200) == 0 {
				v *= 6
			}
			out[r*cols+c] = float32(v)
		}
	}
	return out
}

// absmaxError is the x^2-weighted squared error (the encoders' objective) of
// plain absmax rounding with codes in [-hi, hi], the scheme they replaced.
func absmaxError(x []float32, hi int) float64 {
	var amax float64
	for _, v := range x {
		amax = max(amax, math.Abs(float64(v)))
	}
	if amax == 0 {
		return 0
	}
	s := amax / float64(hi)
	var e float64
	for _, v := range x {
		q := math.Max(-float64(hi), math.Min(float64(hi), math.Round(float64(v)/s)))
		d := float64(v) - s*q
		e += float64(v) * float64(v) * d * d
	}
	return e
}

func TestBestScaleNeverWorseThanAbsmax(t *testing.T) {
	data := syntheticWeights(64, 256, 1)
	for _, bits := range []int{2, 3, 4, 6, 8} {
		for _, signed := range []bool{false, true} {
			lo, hi := QuantCodeRange(bits)
			var newErr, oldErr float64
			var q [32]int8
			for g := 0; g+32 <= len(data); g += 32 {
				x := data[g : g+32]
				s := bestScale(x, lo, hi, signed, gridForBits(bits))
				e := quantizeWithScale(x, q[:], s, lo, hi)
				old := absmaxError(x, -lo-boolToInt(bits < 8))
				if e > old*(1+1e-9) {
					t.Fatalf("bits=%d signed=%v group %d: error %.6g worse than absmax %.6g", bits, signed, g/32, e, old)
				}
				newErr += e
				oldErr += old
			}
			if bits < 8 && newErr >= 0.95*oldErr {
				t.Errorf("bits=%d signed=%v: total error %.6g not clearly below absmax %.6g", bits, signed, newErr, oldErr)
			}
		}
	}
}

func boolToInt(b bool) int {
	if b {
		return 1
	}
	return 0
}

func TestBestScaleZeroGroup(t *testing.T) {
	if s := bestScale(make([]float32, 32), -8, 7, true, gridForBits(4)); s != 0 {
		t.Fatalf("scale for zero group = %v, want 0", s)
	}
}

// unpackCodes is the inverse of packCodes for test assertions.
func unpackCodes(src []byte, bits int) [32]int8 {
	var out [32]int8
	bitPos := 0
	for i := range out {
		var u uint32
		for b := range bits {
			if src[bitPos>>3]>>uint(bitPos&7)&1 == 1 {
				u |= 1 << uint(b)
			}
			bitPos++
		}
		shift := 32 - bits
		out[i] = int8(int32(u<<uint(shift)) >> uint(shift))
	}
	return out
}

func TestPackCodesRoundTrip(t *testing.T) {
	for _, bits := range []int{2, 3, 4, 6, 8} {
		lo, hi := QuantCodeRange(bits)
		var q [32]int8
		for i := range q {
			q[i] = int8(lo + i%(hi-lo+1))
		}
		buf := make([]byte, 32*bits/8)
		packCodes(buf, &q, bits)
		if got := unpackCodes(buf, bits); got != q {
			t.Fatalf("bits=%d: round trip %v, want %v", bits, got, q)
		}
	}
}

func TestEncodeQTensorLayoutAndRange(t *testing.T) {
	const rows, cols = 16, 200 // cols not a multiple of 32
	src := syntheticWeights(rows, cols, 2)
	for _, bits := range []int{4, 8} {
		payload, err := EncodeQTensor(src, rows, cols, bits)
		if err != nil {
			t.Fatal(err)
		}
		dt, _ := QDTypeForBits(bits)
		want, _ := QuantPayloadSize([]uint64{rows, cols}, dt)
		if uint64(len(payload)) != want {
			t.Fatalf("bits=%d: payload %d bytes, want %d", bits, len(payload), want)
		}
		blocksPerRow := (cols + 31) / 32
		dataOff, _ := align64(uint64(rows*blocksPerRow) * 2)
		lo, hi := QuantCodeRange(bits)
		blockBytes := 32 * bits / 8
		sawLo := false
		for blk := range rows * blocksPerRow {
			codes := unpackCodes(payload[dataOff+uint64(blk*blockBytes):], bits)
			valid := min(32, cols-(blk%blocksPerRow)*32)
			for i, c := range codes {
				if int(c) < lo || int(c) > hi {
					t.Fatalf("bits=%d block %d: code %d outside [%d, %d]", bits, blk, c, lo, hi)
				}
				if i >= valid && c != 0 {
					t.Fatalf("bits=%d block %d: padding code %d at %d, want 0", bits, blk, c, i)
				}
				sawLo = sawLo || int(c) == lo
			}
		}
		if !sawLo {
			t.Errorf("bits=%d: most negative code %d never used", bits, lo)
		}
	}
}

func TestEncodeKTensorUsesFullSubScaleRange(t *testing.T) {
	const rows, cols = 8, 512
	src := syntheticWeights(rows, cols, 3)
	for _, bits := range []int{2, 3, 4, 6} {
		payload, err := EncodeKTensor(src, rows, cols, bits)
		if err != nil {
			t.Fatal(err)
		}
		dt, _ := KDTypeForBits(bits)
		want, _ := QuantPayloadSize([]uint64{rows, cols}, dt)
		if uint64(len(payload)) != want {
			t.Fatalf("bits=%d: payload %d bytes, want %d", bits, len(payload), want)
		}
		blocksPerRow := cols / 32
		subOff, _ := align64(uint64(rows*((blocksPerRow+7)/8)) * 2)
		var maxU uint8
		for _, u := range payload[subOff : subOff+uint64(rows*blocksPerRow)] {
			if u > 63 {
				t.Fatalf("bits=%d: sub-scale %d exceeds 6 bits", bits, u)
			}
			maxU = max(maxU, u)
		}
		if maxU < 60 {
			t.Errorf("bits=%d: largest sub-scale %d, want close to 63", bits, maxU)
		}
	}
}

func TestEncodeDeterministicAcrossWorkers(t *testing.T) {
	const rows, cols = 33, 96
	src := syntheticWeights(rows, cols, 4)
	encode := func() ([]byte, []byte) {
		q, err := EncodeQTensor(src, rows, cols, 4)
		if err != nil {
			t.Fatal(err)
		}
		k, err := EncodeKTensor(src, rows, cols, 3)
		if err != nil {
			t.Fatal(err)
		}
		return q, k
	}
	q1, k1 := encode()
	prev := runtime.GOMAXPROCS(1)
	q2, k2 := encode()
	runtime.GOMAXPROCS(prev)
	if !bytes.Equal(q1, q2) || !bytes.Equal(k1, k2) {
		t.Fatal("payload depends on worker count")
	}
}

func TestEncodeZeroTensor(t *testing.T) {
	src := make([]float32, 4*64)
	for _, enc := range []func() ([]byte, error){
		func() ([]byte, error) { return EncodeQTensor(src, 4, 64, 4) },
		func() ([]byte, error) { return EncodeKTensor(src, 4, 64, 4) },
	} {
		payload, err := enc()
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(payload, make([]byte, len(payload))) {
			t.Fatal("zero tensor produced non-zero payload")
		}
	}
}

func TestEncodeRejectsBadInput(t *testing.T) {
	src := make([]float32, 64)
	src[5] = float32(math.NaN())
	if _, err := EncodeQTensor(src, 2, 32, 4); err == nil {
		t.Error("NaN accepted")
	}
	if _, err := EncodeKTensor(make([]float32, 10), 2, 32, 4); err == nil {
		t.Error("short source accepted")
	}
	if _, err := EncodeKTensor(make([]float32, 64), 2, 32, 5); err == nil {
		t.Error("unsupported k* bit width accepted")
	}
	if _, err := EncodeQTensor(make([]float32, 64), 2, 32, 6); err == nil {
		t.Error("unsupported q* bit width accepted")
	}
}

func BenchmarkEncodeKTensor4(b *testing.B) {
	const rows, cols = 256, 2048
	src := syntheticWeights(rows, cols, 5)
	b.SetBytes(int64(rows * cols * 4))
	for b.Loop() {
		if _, err := EncodeKTensor(src, rows, cols, 4); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkEncodeQTensor4(b *testing.B) {
	const rows, cols = 256, 2048
	src := syntheticWeights(rows, cols, 6)
	b.SetBytes(int64(rows * cols * 4))
	for b.Loop() {
		if _, err := EncodeQTensor(src, rows, cols, 4); err != nil {
			b.Fatal(err)
		}
	}
}
