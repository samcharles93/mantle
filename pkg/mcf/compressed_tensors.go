package mcf

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strings"
)

// compressed-tensors "pack-quantized" import. Symmetric integer checkpoints,
// as written by GSQ, RCO and llm-compressor, store each quantized linear
// layer as three tensors:
//
//	<module>.weight_packed  int32 [rows, ceil(cols*b/32)]: codes offset by
//	                        2^(b-1) to unsigned, packed densely LSB first
//	                        along each row, words little-endian
//	<module>.weight_scale   float [rows, ceil(cols/G)]
//	<module>.weight_shape   int [2] = (rows, cols)
//
// These transcode exactly into gq* tensors named <module>.weight.

const (
	ctPackedSuffix = ".weight_packed"
	ctScaleSuffix  = ".weight_scale"
	ctShapeSuffix  = ".weight_shape"
	ctZeroSuffix   = ".weight_zero_point"
	ctGIdxSuffix   = ".weight_g_idx"
)

// CompressedTensorsScheme is the validated weight scheme of a
// compressed-tensors checkpoint.
type CompressedTensorsScheme struct {
	// Bits lists every code width the config declares; RCO mixes several.
	Bits map[int]bool
}

type ctQuantArgs struct {
	NumBits   int             `json:"num_bits"`
	Type      string          `json:"type"`
	Symmetric bool            `json:"symmetric"`
	Strategy  string          `json:"strategy"`
	GroupSize *int            `json:"group_size"`
	Dynamic   json.RawMessage `json:"dynamic"`
	Actorder  json.RawMessage `json:"actorder"`
}

type ctGroup struct {
	Weights          *ctQuantArgs    `json:"weights"`
	InputActivations json.RawMessage `json:"input_activations"`
	Format           string          `json:"format"`
}

// ParseCompressedTensorsScheme reads quantization_config from an HF
// config.json. It returns nil, nil when the checkpoint is not
// compressed-tensors, and an error when it is but uses a scheme gq* cannot
// hold exactly.
func ParseCompressedTensorsScheme(configJSON []byte) (*CompressedTensorsScheme, error) {
	var cfg struct {
		QuantizationConfig *struct {
			QuantMethod  string             `json:"quant_method"`
			Format       string             `json:"format"`
			ConfigGroups map[string]ctGroup `json:"config_groups"`
		} `json:"quantization_config"`
	}
	if err := json.Unmarshal(configJSON, &cfg); err != nil {
		return nil, fmt.Errorf("mcf: config.json: %w", err)
	}
	qc := cfg.QuantizationConfig
	if qc == nil || qc.QuantMethod != "compressed-tensors" {
		return nil, nil
	}
	scheme := &CompressedTensorsScheme{Bits: map[int]bool{}}
	for name, g := range qc.ConfigGroups {
		if g.Weights == nil {
			continue
		}
		w := g.Weights
		format := g.Format
		if format == "" {
			format = qc.Format
		}
		switch {
		case format != "pack-quantized":
			return nil, fmt.Errorf("mcf: compressed-tensors group %s: format %q unsupported (want pack-quantized)", name, format)
		case w.Type != "int" || !w.Symmetric:
			return nil, fmt.Errorf("mcf: compressed-tensors group %s: only symmetric int weights are supported", name)
		case w.Strategy != "group" && w.Strategy != "channel":
			return nil, fmt.Errorf("mcf: compressed-tensors group %s: strategy %q unsupported (want group or channel)", name, w.Strategy)
		case isTruthyJSON(w.Dynamic):
			return nil, fmt.Errorf("mcf: compressed-tensors group %s: dynamic weight quantization unsupported", name)
		case strings.Trim(string(w.Actorder), `"`) == "group":
			return nil, fmt.Errorf("mcf: compressed-tensors group %s: group act-order (g_idx) unsupported", name)
		case len(g.InputActivations) > 0 && string(g.InputActivations) != "null":
			return nil, fmt.Errorf("mcf: compressed-tensors group %s: activation quantization unsupported (weight-only checkpoints only)", name)
		case w.NumBits < gqMinBits || w.NumBits > gqMaxBits:
			return nil, fmt.Errorf("mcf: compressed-tensors group %s: %d-bit weights unsupported", name, w.NumBits)
		}
		scheme.Bits[w.NumBits] = true
	}
	if len(scheme.Bits) == 0 {
		return nil, errors.New("mcf: compressed-tensors config has no weight quantization groups")
	}
	return scheme, nil
}

func isTruthyJSON(raw json.RawMessage) bool {
	s := strings.TrimSpace(string(raw))
	return s != "" && s != "null" && s != "false"
}

// CompressedTensorsModule returns the module name if name is a
// compressed-tensors companion tensor, and whether it is the packed weight
// that drives transcoding.
func CompressedTensorsModule(name string) (module string, packed, ok bool) {
	for _, suf := range []string{ctPackedSuffix, ctScaleSuffix, ctShapeSuffix, ctZeroSuffix, ctGIdxSuffix} {
		if m, found := strings.CutSuffix(name, suf); found {
			return m, suf == ctPackedSuffix, true
		}
	}
	return "", false, false
}

// CompressedTensorsTensor holds the raw parts of one quantized module.
type CompressedTensorsTensor struct {
	Packed     []byte // little-endian int32
	PackedCols int
	Scale      []byte
	ScaleDType string // safetensors dtype: BF16, F16 or F32
	ScaleCols  int
	Rows, Cols int
}

// TranscodeCompressedTensors converts one pack-quantized module into a gq*
// payload without re-quantizing. The code width comes from the packed width
// and must be one the scheme declares; the group size comes from the scale
// width and maps onto the largest gq* group (128, 64, 32) dividing it, with
// each source scale repeated across the gq* groups it covers.
func TranscodeCompressedTensors(t CompressedTensorsTensor, scheme *CompressedTensorsScheme) (TensorDType, []byte, error) {
	if t.Rows <= 0 || t.Cols <= 0 || t.ScaleCols <= 0 {
		return DTypeUnknown, nil, errors.New("mcf: compressed-tensors: invalid shape")
	}
	bits := 0
	for b := gqMinBits; b <= gqMaxBits; b++ {
		if (t.Cols*b+31)/32 == t.PackedCols && scheme.Bits[b] {
			if bits != 0 {
				return DTypeUnknown, nil, fmt.Errorf("mcf: compressed-tensors: packed width %d is ambiguous for %d columns", t.PackedCols, t.Cols)
			}
			bits = b
		}
	}
	if bits == 0 {
		return DTypeUnknown, nil, fmt.Errorf("mcf: compressed-tensors: packed width %d matches no declared bit width for %d columns", t.PackedCols, t.Cols)
	}
	srcGroup := (t.Cols + t.ScaleCols - 1) / t.ScaleCols
	if (t.Cols+srcGroup-1)/srcGroup != t.ScaleCols {
		return DTypeUnknown, nil, fmt.Errorf("mcf: compressed-tensors: %d scales per row do not tile %d columns", t.ScaleCols, t.Cols)
	}
	group := 0
	for _, g := range []int{128, 64, 32} {
		if srcGroup%g == 0 {
			group = g
			break
		}
	}
	if group == 0 {
		return DTypeUnknown, nil, fmt.Errorf("mcf: compressed-tensors: group size %d is not a multiple of 32", srcGroup)
	}
	if len(t.Packed) != t.Rows*t.PackedCols*4 {
		return DTypeUnknown, nil, errors.New("mcf: compressed-tensors: packed tensor size mismatch")
	}
	srcScales, err := ctScalesBF16(t.Scale, t.ScaleDType, t.Rows*t.ScaleCols)
	if err != nil {
		return DTypeUnknown, nil, err
	}

	dt, err := GQDType(bits, group)
	if err != nil {
		return DTypeUnknown, nil, err
	}
	groupsPerRow := (t.Cols + group - 1) / group
	scales := make([]uint16, t.Rows*groupsPerRow)
	for r := range t.Rows {
		for g := range groupsPerRow {
			scales[r*groupsPerRow+g] = srcScales[r*t.ScaleCols+g*group/srcGroup]
		}
	}

	codes := make([]int8, t.Rows*t.Cols)
	offset := 1 << (bits - 1)
	mask := uint32(1)<<uint(bits) - 1
	rowBytes := t.PackedCols * 4
	for r := range t.Rows {
		row := t.Packed[r*rowBytes : (r+1)*rowBytes]
		for c := range t.Cols {
			bit := c * bits
			// Words are little-endian, so the row is an LSB-first bit stream.
			var u uint32
			for i := range bits {
				p := bit + i
				u |= uint32(row[p>>3]>>uint(p&7)&1) << uint(i)
			}
			codes[r*t.Cols+c] = int8(int(u&mask) - offset)
		}
	}
	payload, err := EncodeGQPayload([]uint64{uint64(t.Rows), uint64(t.Cols)}, dt, scales, codes)
	if err != nil {
		return DTypeUnknown, nil, err
	}
	return dt, payload, nil
}

// ctScalesBF16 returns n scales as bf16 bits, failing if any value cannot be
// held exactly in bf16.
func ctScalesBF16(raw []byte, dtype string, n int) ([]uint16, error) {
	out := make([]uint16, n)
	switch strings.ToUpper(dtype) {
	case "BF16":
		if len(raw) != n*2 {
			return nil, errors.New("mcf: compressed-tensors: scale tensor size mismatch")
		}
		for i := range out {
			out[i] = binary.LittleEndian.Uint16(raw[i*2:])
		}
		return out, nil
	case "F16", "F32":
	default:
		return nil, fmt.Errorf("mcf: compressed-tensors: scale dtype %s unsupported", dtype)
	}
	elem := 2
	if strings.ToUpper(dtype) == "F32" {
		elem = 4
	}
	if len(raw) != n*elem {
		return nil, errors.New("mcf: compressed-tensors: scale tensor size mismatch")
	}
	for i := range out {
		var f float32
		if elem == 2 {
			f = fp16ToFloat32(binary.LittleEndian.Uint16(raw[i*2:]))
		} else {
			f = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		}
		h := BF16FromFloat32(f)
		if bf16ToF32(h) != f {
			return nil, fmt.Errorf("mcf: compressed-tensors: %s scale %g is not exactly representable in bf16", dtype, f)
		}
		out[i] = h
	}
	return out, nil
}
