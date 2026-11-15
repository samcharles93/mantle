package mcf

import (
	"encoding/binary"
	"math"
	"math/rand/v2"
	"strings"
	"testing"
)

// ctPack packs signed codes the way compressed-tensors pack_to_int32 does:
// offset by 2^(b-1), element i of a row at bits [i*b, (i+1)*b) of the row's
// little-endian int32 words.
func ctPack(codes []int8, rows, cols, bits int) ([]byte, int) {
	words := (cols*bits + 31) / 32
	out := make([]byte, rows*words*4)
	for r := range rows {
		ws := make([]uint32, words)
		for c := range cols {
			u := uint32(int(codes[r*cols+c]) + 1<<(bits-1))
			for i := range bits {
				if u>>uint(i)&1 == 1 {
					p := c*bits + i
					ws[p/32] |= 1 << uint(p%32)
				}
			}
		}
		for w, v := range ws {
			binary.LittleEndian.PutUint32(out[(r*words+w)*4:], v)
		}
	}
	return out, words
}

func ctFixture(rng *rand.Rand, rows, cols, bits, srcGroup int, scaleDType string) (CompressedTensorsTensor, []int8, []float32) {
	lo, hi := -(1 << (bits - 1)), (1<<(bits-1))-1
	codes := make([]int8, rows*cols)
	for i := range codes {
		codes[i] = int8(lo + rng.IntN(hi-lo+1))
	}
	scaleCols := (cols + srcGroup - 1) / srcGroup
	scales := make([]float32, rows*scaleCols)
	var raw []byte
	for i := range scales {
		// Values exactly representable in bf16, so every scale dtype is exact.
		scales[i] = BF16ToFloat32(BF16FromFloat32(float32(rng.Float64()*0.02 + 0.001)))
		switch scaleDType {
		case "BF16":
			raw = binary.LittleEndian.AppendUint16(raw, BF16FromFloat32(scales[i]))
		case "F16":
			raw = binary.LittleEndian.AppendUint16(raw, float32ToFP16Bits(scales[i]))
		case "F32":
			raw = binary.LittleEndian.AppendUint32(raw, math.Float32bits(scales[i]))
		}
	}
	packed, words := ctPack(codes, rows, cols, bits)
	return CompressedTensorsTensor{
		Packed: packed, PackedCols: words, Scale: raw, ScaleDType: scaleDType,
		ScaleCols: scaleCols, Rows: rows, Cols: cols,
	}, codes, scales
}

func TestTranscodeCompressedTensorsExact(t *testing.T) {
	rng := rand.New(rand.NewPCG(31, 32))
	cases := []struct {
		bits, cols, srcGroup, wantGroup int
		scaleDType                      string
	}{
		{4, 256, 128, 128, "BF16"},
		{3, 320, 64, 64, "BF16"},  // 3-bit codes straddle int32 words
		{2, 96, 32, 32, "F16"},    // f16 scales that bf16 holds exactly
		{8, 128, 128, 128, "F32"}, // f32 scales
		{4, 512, 256, 128, "BF16"},
		{5, 384, 384, 128, "BF16"}, // per-channel scale (one per row)
		{6, 96, 96, 32, "BF16"},    // channel strategy, 96 columns
	}
	for _, tc := range cases {
		const rows = 5
		in, codes, scales := ctFixture(rng, rows, tc.cols, tc.bits, tc.srcGroup, tc.scaleDType)
		scheme := &CompressedTensorsScheme{Bits: map[int]bool{tc.bits: true}}
		dt, payload, err := TranscodeCompressedTensors(in, scheme)
		if err != nil {
			t.Fatalf("%+v: %v", tc, err)
		}
		bits, group, _ := GQParams(dt)
		if bits != tc.bits || group != tc.wantGroup {
			t.Fatalf("%+v: got gq%d-%d", tc, bits, group)
		}
		l, _ := GQPayloadLayoutFor([]uint64{rows, uint64(tc.cols)}, dt)
		row := make([]float32, tc.cols)
		for r := range rows {
			if err := DecodeGQRow(payload, l, r, tc.cols, row); err != nil {
				t.Fatal(err)
			}
			for c, v := range row {
				want := scales[r*in.ScaleCols+c/tc.srcGroup] * float32(codes[r*tc.cols+c])
				if v != want {
					t.Fatalf("%+v r=%d c=%d: got %v want %v", tc, r, c, v, want)
				}
			}
		}
	}
}

func TestTranscodeCompressedTensorsRejects(t *testing.T) {
	rng := rand.New(rand.NewPCG(33, 34))
	base, _, _ := ctFixture(rng, 2, 128, 4, 128, "BF16")
	four := &CompressedTensorsScheme{Bits: map[int]bool{4: true}}

	if _, _, err := TranscodeCompressedTensors(base, &CompressedTensorsScheme{Bits: map[int]bool{3: true}}); err == nil {
		t.Error("undeclared bit width accepted")
	}
	odd, _, _ := ctFixture(rng, 2, 96, 4, 48, "BF16")
	if _, _, err := TranscodeCompressedTensors(odd, four); err == nil {
		t.Error("group size 48 accepted")
	}
	inexact := base
	inexact.ScaleDType = "F32"
	inexact.Scale = binary.LittleEndian.AppendUint32(nil, math.Float32bits(0.0123456789))
	inexact.Scale = append(inexact.Scale, inexact.Scale...)
	if _, _, err := TranscodeCompressedTensors(inexact, four); err == nil || !strings.Contains(err.Error(), "bf16") {
		t.Errorf("inexact f32 scale: err = %v", err)
	}
	short := base
	short.Packed = short.Packed[:len(short.Packed)-4]
	if _, _, err := TranscodeCompressedTensors(short, four); err == nil {
		t.Error("truncated packed tensor accepted")
	}
}

func TestParseCompressedTensorsScheme(t *testing.T) {
	group := func(extra string) string {
		return `{"quantization_config": {"quant_method": "compressed-tensors", "format": "pack-quantized",
			"config_groups": {"g0": {"targets": ["Linear"], "input_activations": null,
			"weights": {"num_bits": 4, "type": "int", "symmetric": true, "strategy": "group", "group_size": 128, "dynamic": false, "actorder": null` + extra + `}}}}}`
	}
	s, err := ParseCompressedTensorsScheme([]byte(group("")))
	if err != nil || s == nil || !s.Bits[4] || len(s.Bits) != 1 {
		t.Fatalf("valid config: %+v, %v", s, err)
	}
	mixed := `{"quantization_config": {"quant_method": "compressed-tensors", "format": "mixed-precision",
		"config_groups": {
		  "a": {"format": "pack-quantized", "weights": {"num_bits": 2, "type": "int", "symmetric": true, "strategy": "group", "group_size": 128}},
		  "b": {"format": "pack-quantized", "weights": {"num_bits": 3, "type": "int", "symmetric": true, "strategy": "channel"}}}}}`
	s, err = ParseCompressedTensorsScheme([]byte(mixed))
	if err != nil || !s.Bits[2] || !s.Bits[3] {
		t.Fatalf("mixed config: %+v, %v", s, err)
	}
	if s, err := ParseCompressedTensorsScheme([]byte(`{"model_type": "qwen3"}`)); s != nil || err != nil {
		t.Fatalf("plain config: %+v, %v", s, err)
	}
	for name, cfg := range map[string]string{
		"asymmetric":  strings.Replace(group(""), `"symmetric": true`, `"symmetric": false`, 1),
		"float":       strings.Replace(group(""), `"type": "int"`, `"type": "float"`, 1),
		"tensor":      strings.Replace(group(""), `"strategy": "group"`, `"strategy": "tensor"`, 1),
		"dynamic":     strings.Replace(group(""), `"dynamic": false`, `"dynamic": true`, 1),
		"g_idx":       strings.Replace(group(""), `"actorder": null`, `"actorder": "group"`, 1),
		"activations": strings.Replace(group(""), `"input_activations": null`, `"input_activations": {"num_bits": 8}`, 1),
		"format":      strings.Replace(group(""), `"pack-quantized"`, `"int-quantized"`, 1),
		"one-bit":     strings.Replace(group(""), `"num_bits": 4`, `"num_bits": 1`, 1),
	} {
		if _, err := ParseCompressedTensorsScheme([]byte(cfg)); err == nil {
			t.Errorf("%s config accepted", name)
		}
	}
}

func TestCompressedTensorsModule(t *testing.T) {
	for name, want := range map[string]struct {
		module      string
		packed, has bool
	}{
		"model.layers.0.mlp.up_proj.weight_packed": {"model.layers.0.mlp.up_proj", true, true},
		"model.layers.0.mlp.up_proj.weight_scale":  {"model.layers.0.mlp.up_proj", false, true},
		"model.layers.0.mlp.up_proj.weight_shape":  {"model.layers.0.mlp.up_proj", false, true},
		"model.layers.0.mlp.up_proj.weight":        {"", false, false},
	} {
		m, p, ok := CompressedTensorsModule(name)
		if m != want.module || p != want.packed || ok != want.has {
			t.Errorf("%s: got %q %v %v", name, m, p, ok)
		}
	}
}
