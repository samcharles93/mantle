package mcf

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
)

const (
	resourceSectionVersion uint32 = 1
	tensorDataVersion      uint32 = 1
)

type PackOptions struct {
	// InputDir is the model directory (where config/tokenizer/vocab live).
	InputDir string

	// ModelSafetensors optionally overrides safetensors discovery.
	// If empty, we open InputDir and auto-detect single vs sharded.
	ModelSafetensors string

	// OutputPath is the .mcf file to create.
	OutputPath string

	// TensorAlign is the per-tensor alignment inside SectionTensorData. Typical: 64.
	// Set to 0 or 1 to disable padding between tensors.
	TensorAlign int

	// Cast controls float casting: "keep" (default), "f16", "bf16".
	Cast string

	// IncludeResources packs config/tokenizer/vocab/merges sections if present.
	IncludeResources bool

	// Optional explicit resource overrides. If empty, defaults to InputDir/<filename>.
	ConfigJSONPath           string
	GenerationConfigJSONPath string
	TokenizerJSONPath        string
	TokenizerConfigJSONPath  string
	VocabJSONPath            string
	MergesTXTPath            string
}

func Pack(opts PackOptions) error {
	if opts.InputDir == "" {
		return errors.New("mcf: pack: InputDir required")
	}
	if opts.OutputPath == "" {
		return errors.New("mcf: pack: OutputPath required")
	}
	if opts.TensorAlign == 0 {
		opts.TensorAlign = 64
	}
	if opts.Cast == "" {
		opts.Cast = "keep"
	}
	opts.Cast = strings.ToLower(strings.TrimSpace(opts.Cast))

	modelPath := opts.InputDir
	if opts.ModelSafetensors != "" {
		modelPath = opts.ModelSafetensors
		if !filepath.IsAbs(modelPath) {
			modelPath = filepath.Join(opts.InputDir, modelPath)
		}
	}

	st, err := OpenSafetensorsModel(modelPath)
	if err != nil {
		return err
	}
	defer func() { _ = st.Close() }()

	outF, err := os.Create(opts.OutputPath)
	if err != nil {
		return err
	}
	defer func() { _ = outF.Close() }()

	w, err := NewWriter(outF)
	if err != nil {
		return err
	}

	// Resource sections (optional)
	if opts.IncludeResources {
		if opts.ConfigJSONPath == "" {
			opts.ConfigJSONPath = filepath.Join(opts.InputDir, "config.json")
		}
		if opts.GenerationConfigJSONPath == "" {
			opts.GenerationConfigJSONPath = filepath.Join(opts.InputDir, "generation_config.json")
		}
		if opts.TokenizerJSONPath == "" {
			opts.TokenizerJSONPath = filepath.Join(opts.InputDir, "tokenizer.json")
		}
		if opts.TokenizerConfigJSONPath == "" {
			opts.TokenizerConfigJSONPath = filepath.Join(opts.InputDir, "tokenizer_config.json")
		}
		if opts.VocabJSONPath == "" {
			opts.VocabJSONPath = filepath.Join(opts.InputDir, "vocab.json")
		}
		if opts.MergesTXTPath == "" {
			opts.MergesTXTPath = filepath.Join(opts.InputDir, "merges.txt")
		}

		if err := writeOptionalFileSection(w, SectionHFConfigJSON, resourceSectionVersion, opts.ConfigJSONPath); err != nil {
			return err
		}
		if err := writeOptionalFileSection(w, SectionHFGenerationConfigJSON, resourceSectionVersion, opts.GenerationConfigJSONPath); err != nil {
			return err
		}
		if err := writeOptionalFileSection(w, SectionTokenizerJSON, resourceSectionVersion, opts.TokenizerJSONPath); err != nil {
			return err
		}
		if err := writeOptionalFileSection(w, SectionTokenizerConfigJSON, resourceSectionVersion, opts.TokenizerConfigJSONPath); err != nil {
			return err
		}
		if err := writeOptionalFileSection(w, SectionVocabJSON, resourceSectionVersion, opts.VocabJSONPath); err != nil {
			return err
		}
		if err := writeOptionalFileSection(w, SectionMergesTXT, resourceSectionVersion, opts.MergesTXTPath); err != nil {
			return err
		}
	}

	// ModelInfo (optional but nice). If config.json exists, we try to populate fields.
	mi, _ := loadModelInfoFromHFConfig(opts.InputDir)
	if mi == nil {
		mi = &ModelInfo{ModelName: filepath.Base(opts.InputDir)}
	}
	miBytes, err := EncodeModelInfo(mi)
	if err != nil {
		return err
	}
	if err := w.WriteSection(SectionModelInfo, 1, miBytes); err != nil {
		return err
	}

	// Tensor data (streaming)
	td, err := w.BeginSection(SectionTensorData, tensorDataVersion)
	if err != nil {
		return err
	}
	defer func() { _ = td.Close() }()

	align := opts.TensorAlign
	if align <= 1 {
		align = 0
	}
	if align == 64 {
		_ = w.AddFlags(FlagTensorDataAligned64)
	}

	copyBuf := make([]byte, 1<<20)
	outBuf := make([]byte, len(copyBuf))

	names := st.SortedTensorNames()
	recs := make([]TensorIndexRecord, 0, len(names))

	for _, name := range names {
		ref, ok := st.Tensor(name)
		if !ok {
			return fmt.Errorf("mcf: safetensors tensor disappeared: %s", name)
		}

		inDT, inElemSize, err := safetensorsDTypeInfo(ref.Info.DType)
		if err != nil {
			return fmt.Errorf("mcf: tensor %q: %w", name, err)
		}

		shapeU64, nElem, err := shapeToU64(ref.Info.Shape)
		if err != nil {
			return fmt.Errorf("mcf: tensor %q: %w", name, err)
		}

		inBytes := uint64(ref.Info.Size())
		wantIn := nElem * uint64(inElemSize)
		if wantIn != inBytes {
			return fmt.Errorf("mcf: tensor %q: dtype/shape mismatch (want %d bytes, have %d)", name, wantIn, inBytes)
		}

		if align != 0 {
			if err := td.Align(align); err != nil {
				return err
			}
		}

		off, err := td.CurrentAbsOffset()
		if err != nil {
			return err
		}

		r, _, err := st.TensorReader(name)
		if err != nil {
			return err
		}

		outDT := inDT
		var written uint64

		switch opts.Cast {
		case "keep":
			n, err := copyExact(td, r, inBytes, copyBuf)
			if err != nil {
				return fmt.Errorf("mcf: tensor %q: copy: %w", name, err)
			}
			written = n

		case "bf16":
			outDT, written, err = castToBF16(td, r, ref.Info.DType, nElem, copyBuf, outBuf)
			if err != nil {
				return fmt.Errorf("mcf: tensor %q: cast bf16: %w", name, err)
			}

		case "f16":
			outDT, written, err = castToF16(td, r, ref.Info.DType, nElem, copyBuf, outBuf)
			if err != nil {
				return fmt.Errorf("mcf: tensor %q: cast f16: %w", name, err)
			}

		default:
			return fmt.Errorf("mcf: unsupported --cast %q (use keep|f16|bf16)", opts.Cast)
		}

		recs = append(recs, TensorIndexRecord{
			Name:     name,
			DType:    outDT,
			Shape:    shapeU64,
			DataOff:  off,
			DataSize: written,
		})
	}

	if err := td.End(); err != nil {
		return err
	}

	idxBytes, err := EncodeTensorIndexSection(recs)
	if err != nil {
		return err
	}
	if err := w.WriteSection(SectionTensorIndex, TensorIndexVersion, idxBytes); err != nil {
		return err
	}

	return w.Finalise()
}

func writeOptionalFileSection(w *Writer, typ SectionType, version uint32, path string) error {
	if path == "" {
		return nil
	}
	f, err := os.Open(path)
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil
		}
		return err
	}
	defer func() { _ = f.Close() }()

	_, err = w.WriteSectionFromReader(typ, version, f)
	return err
}

func safetensorsDTypeInfo(dt string) (TensorDType, int, error) {
	switch strings.ToUpper(dt) {
	case "F32":
		return DTypeF32, 4, nil
	case "F16":
		return DTypeF16, 2, nil
	case "BF16":
		return DTypeBF16, 2, nil
	case "F64":
		return DTypeF64, 8, nil
	case "I8":
		return DTypeI8, 1, nil
	case "U8":
		return DTypeU8, 1, nil
	case "I16":
		return DTypeI16, 2, nil
	case "U16":
		return DTypeU16, 2, nil
	case "I32":
		return DTypeI32, 4, nil
	case "U32":
		return DTypeU32, 4, nil
	case "I64":
		return DTypeI64, 8, nil
	case "U64":
		return DTypeU64, 8, nil
	default:
		return DTypeUnknown, 0, fmt.Errorf("unsupported safetensors dtype %q", dt)
	}
}

func shapeToU64(shape []int64) ([]uint64, uint64, error) {
	if len(shape) == 0 {
		return nil, 0, errors.New("empty shape")
	}
	out := make([]uint64, len(shape))
	var n uint64 = 1
	for i, d := range shape {
		if d <= 0 {
			return nil, 0, fmt.Errorf("invalid dim %d", d)
		}
		ud := uint64(d)
		if ud != 0 && n > (^uint64(0))/ud {
			return nil, 0, errors.New("tensor too large")
		}
		n *= ud
		out[i] = ud
	}
	return out, n, nil
}

func copyExact(dst io.Writer, src io.Reader, n uint64, buf []byte) (uint64, error) {
	if n == 0 {
		return 0, nil
	}
	var total uint64
	for total < n {
		toRead := int(min(uint64(len(buf)), n-total))
		rn, err := io.ReadFull(src, buf[:toRead])
		if rn > 0 {
			if _, werr := dst.Write(buf[:rn]); werr != nil {
				return total, werr
			}
			total += uint64(rn)
		}
		if err != nil {
			return total, err
		}
	}
	return total, nil
}

// ---- Casting (streaming) ----

func castToBF16(dst io.Writer, src io.Reader, inDType string, nElem uint64, inBuf, outBuf []byte) (TensorDType, uint64, error) {
	switch strings.ToUpper(inDType) {
	case "BF16":
		w, err := copyExact(dst, src, nElem*2, inBuf)
		return DTypeBF16, w, err
	case "F32":
		w, err := convertF32ToBF16(dst, src, nElem, inBuf, outBuf)
		return DTypeBF16, w, err
	case "F16":
		w, err := convertF16ToBF16(dst, src, nElem, inBuf, outBuf)
		return DTypeBF16, w, err
	default:
		return DTypeUnknown, 0, fmt.Errorf("cannot cast %q to bf16", inDType)
	}
}

func castToF16(dst io.Writer, src io.Reader, inDType string, nElem uint64, inBuf, outBuf []byte) (TensorDType, uint64, error) {
	switch strings.ToUpper(inDType) {
	case "F16":
		w, err := copyExact(dst, src, nElem*2, inBuf)
		return DTypeF16, w, err
	case "F32":
		w, err := convertF32ToF16(dst, src, nElem, inBuf, outBuf)
		return DTypeF16, w, err
	case "BF16":
		w, err := convertBF16ToF16(dst, src, nElem, inBuf, outBuf)
		return DTypeF16, w, err
	default:
		return DTypeUnknown, 0, fmt.Errorf("cannot cast %q to f16", inDType)
	}
}

func convertF32ToBF16(dst io.Writer, src io.Reader, nElem uint64, inBuf, outBuf []byte) (uint64, error) {
	wantIn := nElem * 4
	var readTotal uint64
	var wroteTotal uint64

	var tail [4]byte
	tailN := 0

	for readTotal < wantIn {
		toRead := int(min(uint64(len(inBuf)), wantIn-readTotal))
		n, err := src.Read(inBuf[:toRead])
		if n > 0 {
			readTotal += uint64(n)
			b := inBuf[:n]

			// finish tail
			if tailN != 0 {
				need := 4 - tailN
				if len(b) < need {
					copy(tail[tailN:], b)
					tailN += len(b)
					goto next
				}
				copy(tail[tailN:], b[:need])
				u := binary.LittleEndian.Uint32(tail[:])
				u16 := bf16FromF32Bits(u)
				binary.LittleEndian.PutUint16(outBuf[:2], u16)
				if _, werr := dst.Write(outBuf[:2]); werr != nil {
					return wroteTotal, werr
				}
				wroteTotal += 2
				b = b[need:]
				tailN = 0
			}

			// bulk
			m := (len(b) / 4)
			proc := b[:m*4]
			outN := m * 2
			if outN > 0 {
				for i := 0; i < m; i++ {
					u := binary.LittleEndian.Uint32(proc[i*4 : i*4+4])
					binary.LittleEndian.PutUint16(outBuf[i*2:i*2+2], bf16FromF32Bits(u))
				}
				if _, werr := dst.Write(outBuf[:outN]); werr != nil {
					return wroteTotal, werr
				}
				wroteTotal += uint64(outN)
			}

			rem := b[m*4:]
			if len(rem) != 0 {
				copy(tail[:], rem)
				tailN = len(rem)
			}
		}
		if err != nil {
			if errors.Is(err, io.EOF) && readTotal == wantIn {
				break
			}
			if errors.Is(err, io.EOF) {
				return wroteTotal, io.ErrUnexpectedEOF
			}
			return wroteTotal, err
		}
	next:
	}
	if tailN != 0 {
		return wroteTotal, errors.New("f32->bf16: trailing partial element")
	}
	return wroteTotal, nil
}

func convertF32ToF16(dst io.Writer, src io.Reader, nElem uint64, inBuf, outBuf []byte) (uint64, error) {
	wantIn := nElem * 4
	var readTotal uint64
	var wroteTotal uint64

	var tail [4]byte
	tailN := 0

	for readTotal < wantIn {
		toRead := int(min(uint64(len(inBuf)), wantIn-readTotal))
		n, err := src.Read(inBuf[:toRead])
		if n > 0 {
			readTotal += uint64(n)
			b := inBuf[:n]

			if tailN != 0 {
				need := 4 - tailN
				if len(b) < need {
					copy(tail[tailN:], b)
					tailN += len(b)
					goto next
				}
				copy(tail[tailN:], b[:need])
				u := binary.LittleEndian.Uint32(tail[:])
				f := math.Float32frombits(u)
				u16 := float32ToFP16Bits(f)
				binary.LittleEndian.PutUint16(outBuf[:2], u16)
				if _, werr := dst.Write(outBuf[:2]); werr != nil {
					return wroteTotal, werr
				}
				wroteTotal += 2
				b = b[need:]
				tailN = 0
			}

			m := (len(b) / 4)
			proc := b[:m*4]
			outN := m * 2
			if outN > 0 {
				for i := 0; i < m; i++ {
					u := binary.LittleEndian.Uint32(proc[i*4 : i*4+4])
					f := math.Float32frombits(u)
					binary.LittleEndian.PutUint16(outBuf[i*2:i*2+2], float32ToFP16Bits(f))
				}
				if _, werr := dst.Write(outBuf[:outN]); werr != nil {
					return wroteTotal, werr
				}
				wroteTotal += uint64(outN)
			}

			rem := b[m*4:]
			if len(rem) != 0 {
				copy(tail[:], rem)
				tailN = len(rem)
			}
		}
		if err != nil {
			if errors.Is(err, io.EOF) && readTotal == wantIn {
				break
			}
			if errors.Is(err, io.EOF) {
				return wroteTotal, io.ErrUnexpectedEOF
			}
			return wroteTotal, err
		}
	next:
	}
	if tailN != 0 {
		return wroteTotal, errors.New("f32->f16: trailing partial element")
	}
	return wroteTotal, nil
}

func convertBF16ToF16(dst io.Writer, src io.Reader, nElem uint64, inBuf, outBuf []byte) (uint64, error) {
	wantIn := nElem * 2
	n, err := copyConvertU16(dst, src, wantIn, inBuf, outBuf, func(u uint16) uint16 {
		f := bf16ToF32(u)
		return float32ToFP16Bits(f)
	})
	return n, err
}

func convertF16ToBF16(dst io.Writer, src io.Reader, nElem uint64, inBuf, outBuf []byte) (uint64, error) {
	wantIn := nElem * 2
	n, err := copyConvertU16(dst, src, wantIn, inBuf, outBuf, func(u uint16) uint16 {
		f := fp16ToFloat32(u)
		return bf16FromF32Bits(math.Float32bits(f))
	})
	return n, err
}

func copyConvertU16(dst io.Writer, src io.Reader, wantIn uint64, inBuf, outBuf []byte, fn func(u uint16) uint16) (uint64, error) {
	var readTotal uint64
	var wroteTotal uint64

	var tail [2]byte
	tailN := 0

	for readTotal < wantIn {
		toRead := int(min(uint64(len(inBuf)), wantIn-readTotal))
		n, err := src.Read(inBuf[:toRead])
		if n > 0 {
			readTotal += uint64(n)
			b := inBuf[:n]

			if tailN != 0 {
				need := 2 - tailN
				if len(b) < need {
					copy(tail[tailN:], b)
					tailN += len(b)
					goto next
				}
				copy(tail[tailN:], b[:need])
				u := binary.LittleEndian.Uint16(tail[:])
				binary.LittleEndian.PutUint16(outBuf[:2], fn(u))
				if _, werr := dst.Write(outBuf[:2]); werr != nil {
					return wroteTotal, werr
				}
				wroteTotal += 2
				b = b[need:]
				tailN = 0
			}

			m := len(b) / 2
			proc := b[:m*2]
			outN := m * 2
			if outN > 0 {
				for i := 0; i < m; i++ {
					u := binary.LittleEndian.Uint16(proc[i*2 : i*2+2])
					binary.LittleEndian.PutUint16(outBuf[i*2:i*2+2], fn(u))
				}
				if _, werr := dst.Write(outBuf[:outN]); werr != nil {
					return wroteTotal, werr
				}
				wroteTotal += uint64(outN)
			}

			rem := b[m*2:]
			if len(rem) != 0 {
				copy(tail[:], rem)
				tailN = len(rem)
			}
		}
		if err != nil {
			if errors.Is(err, io.EOF) && readTotal == wantIn {
				break
			}
			if errors.Is(err, io.EOF) {
				return wroteTotal, io.ErrUnexpectedEOF
			}
			return wroteTotal, err
		}
	next:
	}

	if tailN != 0 {
		return wroteTotal, errors.New("u16 convert: trailing partial element")
	}
	return wroteTotal, nil
}

func bf16FromF32Bits(u uint32) uint16 {
	// round-to-nearest-even on the truncated 16 bits
	rnd := uint32(0x7FFF + ((u >> 16) & 1))
	return uint16((u + rnd) >> 16)
}

func bf16ToF32(u uint16) float32 {
	return math.Float32frombits(uint32(u) << 16)
}

// float32ToFP16Bits implements IEEE 754 binary16 rounding (nearest-even).
func float32ToFP16Bits(f float32) uint16 {
	u := math.Float32bits(f)
	sign := uint16((u >> 16) & 0x8000)
	exp := int((u >> 23) & 0xFF)
	frac := u & 0x7FFFFF

	switch exp {
	case 0xFF:
		if frac != 0 {
			return sign | 0x7E00 // NaN
		}
		return sign | 0x7C00 // Inf
	case 0:
		// Zero/subnormal float32 -> zero fp16 (good enough for packing).
		return sign
	}

	e := exp - 127 + 15
	if e >= 31 {
		return sign | 0x7C00 // overflow -> Inf
	}
	if e <= 0 {
		// subnormal fp16
		if e < -10 {
			return sign
		}
		m := frac | 0x800000
		shift := uint32(14 - e)
		// round-to-nearest-even
		round := uint32(1) << (shift - 1)
		m = m + round - 1 + ((m >> shift) & 1)
		return sign | uint16(m>>shift)
	}

	// normal fp16
	m := frac
	m = m + 0x0FFF + ((m >> 13) & 1)
	if (m & 0x800000) != 0 {
		m = 0
		e++
		if e >= 31 {
			return sign | 0x7C00
		}
	}
	return sign | uint16(e<<10) | uint16(m>>13)
}

func fp16ToFloat32(h uint16) float32 {
	sign := uint32(h>>15) & 0x1
	exp := uint32(h>>10) & 0x1F
	frac := uint32(h & 0x3FF)

	var f uint32
	switch exp {
	case 0:
		if frac == 0 {
			f = sign << 31
		} else {
			e := uint32(127 - 15 + 1)
			for (frac & 0x400) == 0 {
				frac <<= 1
				e--
			}
			frac &= 0x3FF
			f = (sign << 31) | (e << 23) | (frac << 13)
		}
	case 0x1F:
		f = (sign << 31) | 0x7F800000 | (frac << 13)
	default:
		e := exp + (127 - 15)
		f = (sign << 31) | (e << 23) | (frac << 13)
	}
	return math.Float32frombits(f)
}

// ---- ModelInfo from HF config ----

func loadModelInfoFromHFConfig(dir string) (*ModelInfo, error) {
	path := filepath.Join(dir, "config.json")
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		return nil, err
	}

	mi := &ModelInfo{
		ModelName: filepath.Base(dir),
		Extras:    make(map[string]any),
	}

	if mt, ok := m["model_type"].(string); ok {
		switch {
		case strings.Contains(strings.ToLower(mt), "llama"):
			mi.Arch = ArchLLaMA
		case strings.Contains(strings.ToLower(mt), "mistral"):
			mi.Arch = ArchMistral
		case strings.Contains(strings.ToLower(mt), "qwen"):
			mi.Arch = ArchQwen
		case strings.Contains(strings.ToLower(mt), "gemma"):
			mi.Arch = ArchGemma
		default:
			mi.Arch = ArchUnknown
		}
		mi.Extras["hf.model_type"] = mt
	}

	mi.VocabSize = readU32(m, "vocab_size")
	mi.HiddenSize = readU32(m, "hidden_size")
	mi.LayerCount = readU32(m, "num_hidden_layers")
	mi.HeadCount = readU32(m, "num_attention_heads")
	mi.HeadCountKV = readU32(m, "num_key_value_heads")
	mi.ContextLength = readU32(m, "max_position_embeddings")

	if rt, ok := m["rope_scaling"].(map[string]any); ok && len(rt) > 0 {
		mi.Extras["hf.rope_scaling"] = "present"
	}
	if theta, ok := readF32(m, "rope_theta"); ok {
		mi.RopeFreqBase = theta
	}

	// keep the raw config for later consumers if they want to look it up
	mi.Extras["hf.config_json_present"] = uint32(1)

	if len(mi.Extras) == 0 {
		mi.Extras = nil
	}
	return mi, nil
}

func readU32(m map[string]any, key string) uint32 {
	v, ok := m[key]
	if !ok {
		return 0
	}
	switch t := v.(type) {
	case float64:
		if t < 0 || t > math.MaxUint32 {
			return 0
		}
		return uint32(t)
	case json.Number:
		if i64, err := t.Int64(); err == nil && i64 >= 0 && i64 <= math.MaxUint32 {
			return uint32(i64)
		}
	}
	return 0
}

func readF32(m map[string]any, key string) (float32, bool) {
	v, ok := m[key]
	if !ok {
		return 0, false
	}
	switch t := v.(type) {
	case float64:
		if math.IsNaN(t) || math.IsInf(t, 0) {
			return 0, false
		}
		if t < -math.MaxFloat32 || t > math.MaxFloat32 {
			return 0, false
		}
		return float32(t), true
	}
	return 0, false
}
