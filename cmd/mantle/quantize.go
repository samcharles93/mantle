package main

import (
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/urfave/cli/v3"

	"github.com/samcharles93/mantle/internal/backend/simd"
	"github.com/samcharles93/mantle/internal/logger"
	"github.com/samcharles93/mantle/internal/mcfstore"
	"github.com/samcharles93/mantle/pkg/mcf"
)

const quantTensorDataVersion uint32 = 1

func quantizeCmd() *cli.Command {
	return &cli.Command{
		Name:  "quantize",
		Usage: "Quantise an existing .mcf",
		Flags: []cli.Flag{
			&cli.StringFlag{
				Name:     "input",
				Aliases:  []string{"in"},
				Usage:    "Input .mcf path",
				Required: true,
			},
			&cli.StringFlag{
				Name:    "output",
				Aliases: []string{"out"},
				Usage:   "Output .mcf path (default: <input-base>.quant.mcf)",
			},
			&cli.StringFlag{
				Name:  "method",
				Usage: "Quantization method: k6|k4|k3|k2|q8|q4",
				Value: "k4",
			},
			&cli.Float64Flag{
				Name:  "min-clip",
				Usage: "Optional override for MinClip (applies to all tensors)",
			},
			&cli.Float64Flag{
				Name:  "max-clip",
				Usage: "Optional override for MaxClip (applies to all tensors)",
			},
		},
		Action: func(ctx context.Context, cmd *cli.Command) error {
			log := logger.FromContext(ctx)

			inPath := cmd.String("input")
			outPath, defaulted, err := resolveQuantOut(inPath, cmd.String("output"))
			if err != nil {
				return fmt.Errorf("quantize: resolve output: %w", err)
			}
			if defaulted {
				log.Info("using default output path", "path", outPath)
			}
			method, err := parseQuantMethod(cmd.String("method"))
			if err != nil {
				return fmt.Errorf("quantize: %w", err)
			}

			var minClip, maxClip *float32
			if cmd.IsSet("min-clip") || cmd.IsSet("max-clip") {
				if !cmd.IsSet("min-clip") || !cmd.IsSet("max-clip") {
					return errors.New("quantize: both --min-clip and --max-clip must be set")
				}
				minVal := float32(cmd.Float64("min-clip"))
				maxVal := float32(cmd.Float64("max-clip"))
				minClip = &minVal
				maxClip = &maxVal
			}

			if samePath(inPath, outPath) {
				return errors.New("quantize: input and output paths must differ")
			}

			opts := quantizeOptions{
				Input:    inPath,
				Output:   outPath,
				Method:   method,
				MinClip:  minClip,
				MaxClip:  maxClip,
				Progress: 50,
				Logger:   log,
			}
			return runQuantize(opts)
		},
	}
}

type quantizeOptions struct {
	Input    string
	Output   string
	Method   quantMethod
	MinClip  *float32
	MaxClip  *float32
	Progress int
	Logger   logger.Logger
}

type quantMethod struct {
	Name      string
	DType     mcf.TensorDType
	Bits      int
	BlockSize int
	SuperSize int
	Family    string
}

func parseQuantMethod(s string) (quantMethod, error) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "k6":
		return quantMethod{Name: "k6", DType: mcf.DTypeK6, Bits: 6, BlockSize: 32, SuperSize: 256, Family: "k"}, nil
	case "k4":
		return quantMethod{Name: "k4", DType: mcf.DTypeK4, Bits: 4, BlockSize: 32, SuperSize: 256, Family: "k"}, nil
	case "k3":
		return quantMethod{Name: "k3", DType: mcf.DTypeK3, Bits: 3, BlockSize: 32, SuperSize: 256, Family: "k"}, nil
	case "k2":
		return quantMethod{Name: "k2", DType: mcf.DTypeK2, Bits: 2, BlockSize: 32, SuperSize: 256, Family: "k"}, nil
	case "q8":
		return quantMethod{Name: "q8", DType: mcf.DTypeQ8, Bits: 8, BlockSize: 32, SuperSize: 0, Family: "q"}, nil
	case "q4":
		return quantMethod{Name: "q4", DType: mcf.DTypeQ4, Bits: 4, BlockSize: 32, SuperSize: 0, Family: "q"}, nil
	case "int8", "int4":
		return quantMethod{}, errors.New("int* quantization is reserved for activations and is not implemented")
	default:
		return quantMethod{}, fmt.Errorf("unsupported method %q", s)
	}
}

func isQuantizableDType(dt mcf.TensorDType) bool {
	switch dt {
	case mcf.DTypeF32, mcf.DTypeF16, mcf.DTypeBF16:
		return true
	default:
		return false
	}
}

func resolveQuantOut(inPath, outPath string) (string, bool, error) {
	if inPath == "" {
		return "", false, errors.New("input path required")
	}
	if outPath != "" {
		return outPath, false, nil
	}
	base := filepath.Base(inPath)
	if strings.HasSuffix(strings.ToLower(base), ".mcf") {
		base = strings.TrimSuffix(base, filepath.Ext(base))
	}
	out := filepath.Join(filepath.Dir(inPath), base+".quant.mcf")
	return out, true, nil
}

func samePath(a, b string) bool {
	pa, errA := filepath.Abs(a)
	pb, errB := filepath.Abs(b)
	if errA != nil || errB != nil {
		return a == b
	}
	return pa == pb
}

func runQuantize(opts quantizeOptions) error {
	mf, err := mcf.Open(opts.Input)
	if err != nil {
		return err
	}
	defer func() { _ = mf.Close() }()

	store, err := mcfstore.Open(opts.Input)
	if err != nil {
		return err
	}
	defer func() { _ = store.Close() }()

	outF, err := os.Create(opts.Output)
	if err != nil {
		return err
	}
	defer func() { _ = outF.Close() }()

	w, err := mcf.NewWriter(outF)
	if err != nil {
		return err
	}

	if mcf.DTypeRequiresAligned64(opts.Method.DType) {
		if err := w.AddFlags(mcf.FlagTensorDataAligned64); err != nil {
			return err
		}
	}

	for i := range mf.Sections {
		s := &mf.Sections[i]
		typeID := mcf.SectionType(s.Type)
		switch typeID {
		case mcf.SectionTensorIndex, mcf.SectionTensorData, mcf.SectionQuantInfo:
			continue
		default:
			data := mf.SectionData(s)
			if data == nil {
				return errors.New("quantize: failed to read section payload")
			}
			if err := w.WriteSection(typeID, s.Version, data); err != nil {
				return err
			}
		}
	}

	indexSec := mf.Section(mcf.SectionTensorIndex)
	if indexSec == nil {
		return errors.New("quantize: missing tensor index section")
	}
	indexData := mf.SectionData(indexSec)
	if len(indexData) == 0 {
		return errors.New("quantize: empty tensor index section")
	}
	index, err := mcf.ParseTensorIndexSection(indexData)
	if err != nil {
		return err
	}

	names := make([]string, 0, index.Count())
	for i := 0; i < index.Count(); i++ {
		name, err := index.Name(i)
		if err != nil {
			return err
		}
		names = append(names, name)
	}
	sort.Strings(names)

	td, err := w.BeginSection(mcf.SectionTensorData, quantTensorDataVersion)
	if err != nil {
		return err
	}
	defer func() { _ = td.Close() }()

	recs := make([]mcf.TensorIndexRecord, 0, len(names))
	qrecs := make([]mcf.QuantRecord, 0, len(names))

	for i, name := range names {
		if opts.Progress > 0 && i%opts.Progress == 0 {
			opts.Logger.Debug("quantizing tensor", "progress", fmt.Sprintf("%d/%d", i, len(names)), "name", name)
		}

		info, err := store.Tensor(name)
		if err != nil {
			return err
		}
		if mcf.DTypeRequiresAligned64(info.DType) {
			return fmt.Errorf("quantize: tensor %s: already quantized", name)
		}
		shape, err := shapeToU64(info.Shape)
		if err != nil {
			return fmt.Errorf("quantize: tensor %s: %w", name, err)
		}

		shouldQuant := len(info.Shape) == 2 && isQuantizableDType(info.DType)
		var payload []byte
		var outDType mcf.TensorDType
		var minClip, maxClip float32

		if shouldQuant {
			data, _, err := store.ReadTensorF32(name)
			if err != nil {
				return fmt.Errorf("quantize: tensor %s: %w", name, err)
			}
			rows := info.Shape[0]
			cols := info.Shape[1]
			payload, minClip, maxClip, err = quantizeTensor(data, rows, cols, opts.Method, opts.MinClip, opts.MaxClip)
			if err != nil {
				return fmt.Errorf("quantize: tensor %s: %w", name, err)
			}
			outDType = opts.Method.DType
		} else {
			raw, _, err := store.ReadTensorRaw(name)
			if err != nil {
				return fmt.Errorf("quantize: tensor %s: %w", name, err)
			}
			payload = raw
			outDType = info.DType
		}

		if err := td.Align(64); err != nil {
			return err
		}
		off, err := td.CurrentAbsOffset()
		if err != nil {
			return err
		}
		if _, err := td.Write(payload); err != nil {
			return err
		}

		idx := uint32(len(recs))
		recs = append(recs, mcf.TensorIndexRecord{
			Name:     name,
			DType:    outDType,
			Shape:    shape,
			DataOff:  off,
			DataSize: uint64(len(payload)),
		})

		if shouldQuant {
			qrecs = append(qrecs, mcf.QuantRecord{
				TensorIndex: idx,
				Method:      uint8(opts.Method.DType),
				Domain:      uint8(mcf.DomainWeights),
				BlockSize:   uint16(opts.Method.BlockSize),
				SuperSize:   uint16(opts.Method.SuperSize),
				MinClip:     minClip,
				MaxClip:     maxClip,
			})
		}
	}

	if err := td.Close(); err != nil {
		return err
	}

	idxPayload, err := mcf.EncodeTensorIndexSection(recs)
	if err != nil {
		return err
	}
	if err := w.WriteSection(mcf.SectionTensorIndex, mcf.TensorIndexVersion, idxPayload); err != nil {
		return err
	}

	quantPayload, err := mcf.EncodeQuantInfoSection(qrecs)
	if err != nil {
		return err
	}
	if err := w.WriteSection(mcf.SectionQuantInfo, mcf.QuantInfoVersion, quantPayload); err != nil {
		return err
	}

	return w.Finalise()
}

func quantizeTensor(src []float32, rows, cols int, method quantMethod, minOverride, maxOverride *float32) ([]byte, float32, float32, error) {
	if rows <= 0 || cols <= 0 {
		return nil, 0, 0, errors.New("invalid tensor shape")
	}
	if rows > int(^uint(0)>>1)/cols {
		return nil, 0, 0, errors.New("tensor too large")
	}
	if len(src) != rows*cols {
		return nil, 0, 0, errors.New("tensor size mismatch")
	}

	minVal, maxVal, err := minMaxFinite(src)
	if err != nil {
		return nil, 0, 0, err
	}

	minClip := minVal
	maxClip := maxVal
	if minOverride != nil && maxOverride != nil {
		minClip = *minOverride
		maxClip = *maxOverride
	}
	if minClip > maxClip {
		return nil, 0, 0, errors.New("min-clip greater than max-clip")
	}

	if method.Family != "q" && method.Family != "k" {
		return nil, 0, 0, errors.New("unsupported quantization family")
	}

	if minOverride == nil || maxOverride == nil {
		maxAbs := maxAbs(minVal, maxVal)
		minClip = -maxAbs
		maxClip = maxAbs
	}

	switch method.Family {
	case "q":
		payload, err := quantizeQ(src, rows, cols, method.Bits, minClip, maxClip)
		return payload, minClip, maxClip, err
	case "k":
		payload, err := quantizeK(src, rows, cols, method.Bits, minClip, maxClip)
		return payload, minClip, maxClip, err
	default:
		return nil, 0, 0, errors.New("unsupported quantization family")
	}
}

func quantizeQ(src []float32, rows, cols, bits int, minClip, maxClip float32) ([]byte, error) {
	if bits != 4 && bits != 8 {
		return nil, errors.New("unsupported q* bit-width")
	}
	blockSize := 32
	blocksPerRow := (cols + blockSize - 1) / blockSize
	blockCount := rows * blocksPerRow

	scales := make([]uint16, blockCount)
	blockBytes := (blockSize * bits) / 8
	dataBytes := blockCount * blockBytes
	data := make([]byte, dataBytes)

	qMax := int32((1 << (bits - 1)) - 1)

	if bits == 8 {
		for r := range rows {
			rowBase := r * cols
			rowLimit := rowBase + cols
			for b := range blocksPerRow {
				blockIdx := r*blocksPerRow + b
				base := rowBase + b*blockSize
				maxAbs := float32(0)
				for i := range blockSize {
					idx := base + i
					v := float32(0)
					if idx < rowLimit {
						v = clamp(src[idx], minClip, maxClip)
					}
					if av := float32(math.Abs(float64(v))); av > maxAbs {
						maxAbs = av
					}
				}
				scale := float32(0)
				if maxAbs > 0 {
					scale = maxAbs / float32(qMax)
				}
				scales[blockIdx] = simd.Float32ToFloat16(scale)
				inv := float32(0)
				if scale != 0 {
					inv = float32(1.0) / scale
				}
				dataOff := blockIdx * blockBytes
				for i := range blockSize {
					idx := base + i
					v := float32(0)
					if idx < rowLimit {
						v = clamp(src[idx], minClip, maxClip)
					}
					q := int32(0)
					if scale != 0 {
						q = int32(math.Round(float64(v * inv)))
						if q > qMax {
							q = qMax
						} else if q < -qMax {
							q = -qMax
						}
					}
					data[dataOff+i] = byte(int8(q))
				}
			}
		}
	} else {
		for r := range rows {
			rowBase := r * cols
			rowLimit := rowBase + cols
			for b := range blocksPerRow {
				blockIdx := r*blocksPerRow + b
				base := rowBase + b*blockSize
				maxAbs := float32(0)
				for i := range blockSize {
					idx := base + i
					v := float32(0)
					if idx < rowLimit {
						v = clamp(src[idx], minClip, maxClip)
					}
					if av := float32(math.Abs(float64(v))); av > maxAbs {
						maxAbs = av
					}
				}
				scale := float32(0)
				if maxAbs > 0 {
					scale = maxAbs / float32(qMax)
				}
				scales[blockIdx] = simd.Float32ToFloat16(scale)
				inv := float32(0)
				if scale != 0 {
					inv = float32(1.0) / scale
				}
				dataOff := blockIdx * blockBytes
				for i := 0; i < blockSize; i += 2 {
					idx0 := base + i
					idx1 := base + i + 1
					q0 := int32(0)
					q1 := int32(0)
					if scale != 0 {
						if idx0 < rowLimit {
							v0 := clamp(src[idx0], minClip, maxClip)
							q0 = int32(math.Round(float64(v0 * inv)))
						}
						if idx1 < rowLimit {
							v1 := clamp(src[idx1], minClip, maxClip)
							q1 = int32(math.Round(float64(v1 * inv)))
						}
						if q0 > qMax {
							q0 = qMax
						} else if q0 < -qMax {
							q0 = -qMax
						}
						if q1 > qMax {
							q1 = qMax
						} else if q1 < -qMax {
							q1 = -qMax
						}
					}
					u0 := uint8(int8(q0)) & 0x0F
					u1 := uint8(int8(q1)) & 0x0F
					data[dataOff+(i/2)] = u0 | (u1 << 4)
				}
			}
		}
	}

	payload := make([]byte, 0, alignedLen(len(scales)*2)+len(data))
	payload = appendF16Slice(payload, scales)
	payload = alignPayload(payload, 64)
	payload = append(payload, data...)
	return payload, nil
}

func quantizeK(src []float32, rows, cols, bits int, minClip, maxClip float32) ([]byte, error) {
	if bits != 2 && bits != 3 && bits != 4 && bits != 6 {
		return nil, errors.New("unsupported k* bit-width")
	}
	blockSize := 32
	superBlocks := 8
	blocksPerRow := (cols + blockSize - 1) / blockSize
	superBlocksPerRow := (blocksPerRow + superBlocks - 1) / superBlocks
	blockCount := rows * blocksPerRow
	superCount := rows * superBlocksPerRow

	superScales := make([]uint16, superCount)
	subScales := make([]byte, blockCount)

	blockBytes := (blockSize * bits) / 8
	dataBytes := blockCount * blockBytes
	packer := newBitPacker(blockCount * blockSize * bits)
	qMax := int32((1 << (bits - 1)) - 1)

	for r := range rows {
		rowBase := r * cols
		rowLimit := rowBase + cols

		for s := range superBlocksPerRow {
			var blockScales [8]float32
			maxScale := float32(0)

			for b := range superBlocks {
				block := s*superBlocks + b
				if block >= blocksPerRow {
					blockScales[b] = 0
					continue
				}
				maxAbs := float32(0)
				base := rowBase + block*blockSize
				for i := range blockSize {
					idx := base + i
					v := float32(0)
					if idx < rowLimit {
						v = clamp(src[idx], minClip, maxClip)
					}
					if av := float32(math.Abs(float64(v))); av > maxAbs {
						maxAbs = av
					}
				}
				if maxAbs == 0 {
					blockScales[b] = 0
					continue
				}
				blockScales[b] = maxAbs / float32(qMax)
				if blockScales[b] > maxScale {
					maxScale = blockScales[b]
				}
			}

			superScale := maxScale
			superIdx := r*superBlocksPerRow + s
			superScales[superIdx] = simd.Float32ToFloat16(superScale)

			for b := range superBlocks {
				block := s*superBlocks + b
				if block >= blocksPerRow {
					continue
				}
				blockIdx := r*blocksPerRow + block
				u6 := uint8(0)
				if blockScales[b] > 0 && superScale > 0 {
					ratio := blockScales[b] / superScale
					u := int(math.Round(float64(ratio * 32.0)))
					if u < 1 {
						u = 1
					} else if u > 63 {
						u = 63
					}
					u6 = uint8(u)
				}
				subScales[blockIdx] = u6 & 0x3F

				var scale float32
				if u6 > 0 && superScale > 0 {
					scale = superScale * (float32(u6) / 32.0)
				}
				inv := float32(0)
				if scale != 0 {
					inv = 1 / scale
				}

				base := rowBase + block*blockSize
				for i := range blockSize {
					idx := base + i
					v := float32(0)
					if idx < rowLimit {
						v = clamp(src[idx], minClip, maxClip)
					}
					q := int32(0)
					if scale != 0 {
						q = int32(math.Round(float64(v * inv)))
						if q > qMax {
							q = qMax
						} else if q < -qMax {
							q = -qMax
						}
					}
					packer.write(signedToBits(q, bits), bits)
				}
			}
		}
	}

	data := packer.bytes()
	if len(data) != dataBytes {
		return nil, errors.New("quantize: packed data size mismatch")
	}

	payload := make([]byte, 0, alignedLen(len(superScales)*2)+alignedLen(len(subScales))+len(data))
	payload = appendF16Slice(payload, superScales)
	payload = alignPayload(payload, 64)
	payload = append(payload, subScales...)
	payload = alignPayload(payload, 64)
	payload = append(payload, data...)
	return payload, nil
}

type bitPacker struct {
	buf    []byte
	bitPos int
}

func newBitPacker(totalBits int) *bitPacker {
	return &bitPacker{buf: make([]byte, (totalBits+7)/8)}
}

func (p *bitPacker) write(v uint8, bits int) {
	for i := range bits {
		if (v>>uint(i))&1 == 1 {
			byteIdx := p.bitPos / 8
			bitIdx := uint(p.bitPos % 8)
			p.buf[byteIdx] |= 1 << bitIdx
		}
		p.bitPos++
	}
}

func (p *bitPacker) bytes() []byte {
	return p.buf
}

func signedToBits(v int32, bits int) uint8 {
	mask := uint8((1 << bits) - 1)
	return uint8(int8(v)) & mask
}

func alignPayload(buf []byte, align int) []byte {
	if align <= 1 {
		return buf
	}
	rem := len(buf) % align
	if rem == 0 {
		return buf
	}
	pad := align - rem
	return append(buf, make([]byte, pad)...)
}

func alignedLen(n int) int {
	if n <= 0 {
		return 0
	}
	return (n + 63) & ^63
}

func appendF16Slice(buf []byte, src []uint16) []byte {
	if len(src) == 0 {
		return buf
	}
	needed := len(src) * 2
	out := make([]byte, needed)
	for i, v := range src {
		out[i*2] = byte(v)
		out[i*2+1] = byte(v >> 8)
	}
	return append(buf, out...)
}

func minMaxFinite(src []float32) (float32, float32, error) {
	if len(src) == 0 {
		return 0, 0, errors.New("empty tensor")
	}
	minVal := float32(math.Inf(1))
	maxVal := float32(math.Inf(-1))
	for _, v := range src {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return 0, 0, errors.New("tensor contains NaN or Inf")
		}
		if v < minVal {
			minVal = v
		}
		if v > maxVal {
			maxVal = v
		}
	}
	return minVal, maxVal, nil
}

func maxAbs(minVal, maxVal float32) float32 {
	minAbs := float32(math.Abs(float64(minVal)))
	maxAbs := float32(math.Abs(float64(maxVal)))
	if minAbs > maxAbs {
		return minAbs
	}
	return maxAbs
}

func clamp(v, minVal, maxVal float32) float32 {
	if v < minVal {
		return minVal
	}
	if v > maxVal {
		return maxVal
	}
	return v
}

func shapeToU64(shape []int) ([]uint64, error) {
	if len(shape) == 0 {
		return nil, errors.New("empty shape")
	}
	out := make([]uint64, len(shape))
	for i, v := range shape {
		if v <= 0 {
			return nil, errors.New("invalid dim")
		}
		out[i] = uint64(v)
	}
	return out, nil
}
