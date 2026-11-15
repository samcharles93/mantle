package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/urfave/cli/v3"

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
				Usage: "Quantization method: k6|k4|k3|k2|q8|q4|uqN|gqN (gqN-32|gqN-64|gqN-128; gqN means gqN-128)",
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
			&cli.StringFlag{
				Name:  "recipe",
				Usage: "Per-tensor method choice: uniform (every tensor gets --method) or mixed (output head, and v_proj/down_proj in the first and last eighth of layers and every third layer between, get k6 for k2-k4, q8 for q4, or gq6 at the same group size for gq2-gq5)",
				Value: recipeUniform,
			},
			&cli.StringFlag{
				Name:  "assignments",
				Usage: `Path to a per-tensor quantization assignment JSON (RCO/SLQ search output): {"default": "k4", "tensors": {"model.layers.0.*": "k2"}}. Exact names win over globs; longest glob wins.`,
			},
			&cli.Uint64Flag{
				Name:  "budget-bytes",
				Usage: "Exact total budget in bytes for quantized tensor payloads; quantize fails if the result exceeds it (RCO/SLQ guarantee).",
			},
			&cli.StringFlag{
				Name:  "codebooks",
				Usage: `Path to a JSON file with per-tensor uq* codebooks (SLQ search output): {"tensors": {"<name>": [[...2^bits levels...] per super-block]}}. Tensors without an entry get a derived k-means codebook. Only used by --method uqN.`,
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

			recipe, err := parseRecipe(cmd.String("recipe"))
			if err != nil {
				return fmt.Errorf("quantize: %w", err)
			}
			if recipe != recipeUniform && cmd.IsSet("assignments") {
				return errors.New("quantize: --recipe and --assignments are mutually exclusive")
			}

			var assignments *quantAssignments
			if cmd.IsSet("assignments") {
				a, err := loadQuantAssignments(cmd.String("assignments"))
				if err != nil {
					return fmt.Errorf("quantize: %w", err)
				}
				assignments = a
			}
			var budget *uint64
			if cmd.IsSet("budget-bytes") {
				b := cmd.Uint64("budget-bytes")
				budget = &b
			}

			var codebooks map[string][][]float32
			if cbPath := cmd.String("codebooks"); cbPath != "" {
				cb, err := loadUQCodebooks(cbPath)
				if err != nil {
					return fmt.Errorf("quantize: %w", err)
				}
				codebooks = cb
			}

			opts := quantizeOptions{
				Input:       inPath,
				Output:      outPath,
				Method:      method,
				Recipe:      recipe,
				Assignments: assignments,
				BudgetBytes: budget,
				MinClip:     minClip,
				MaxClip:     maxClip,
				Codebooks:   codebooks,
				Progress:    50,
				Logger:      log,
			}
			return runQuantize(opts)
		},
	}
}

type quantizeOptions struct {
	Input       string
	Output      string
	Method      quantMethod
	Recipe      string
	Assignments *quantAssignments
	BudgetBytes *uint64
	MinClip     *float32
	MaxClip     *float32
	// Codebooks maps tensor name -> per-super-block codebooks for uq* methods
	// (SLQ search output). Nil or missing entries derive a k-means codebook.
	Codebooks map[string][][]float32
	Progress  int
	Logger    logger.Logger
}

// loadUQCodebooks reads the --codebooks JSON file:
// {"tensors": {"<name>": [[levels...] per super-block]}}.
func loadUQCodebooks(path string) (map[string][][]float32, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var doc struct {
		Tensors map[string][][]float32 `json:"tensors"`
	}
	if err := json.Unmarshal(raw, &doc); err != nil {
		return nil, fmt.Errorf("parse codebooks: %w", err)
	}
	if doc.Tensors == nil {
		return nil, errors.New("codebooks file has no \"tensors\" object")
	}
	return doc.Tensors, nil
}

// quantAssignments is a per-tensor quantization method assignment, as produced by
// an RCO/SLQ budget search. Exact tensor names win over globs; among globs the
// longest pattern wins. Unmatched tensors fall back to Default.
type quantAssignments struct {
	Default string            `json:"default"`
	Tensors map[string]string `json:"tensors"`
	Budget  uint64            `json:"budget_bytes"`

	defaultMethod quantMethod
	globs         []quantGlob
}

type quantGlob struct {
	prefix  string
	pattern string
	method  quantMethod
}

// loadQuantAssignments reads and validates an assignment file.
func loadQuantAssignments(path string) (*quantAssignments, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read assignments: %w", err)
	}
	var a quantAssignments
	if err := json.Unmarshal(raw, &a); err != nil {
		return nil, fmt.Errorf("parse assignments %s: %w", path, err)
	}
	defName := a.Default
	if defName == "" {
		defName = "k4"
	}
	def, err := parseQuantMethod(defName)
	if err != nil {
		return nil, fmt.Errorf("assignments default: %w", err)
	}
	a.defaultMethod = def
	// Deterministic glob order: longest prefix first, then pattern.
	a.globs = make([]quantGlob, 0, len(a.Tensors))
	for pattern, name := range a.Tensors {
		m, err := parseQuantMethod(name)
		if err != nil {
			return nil, fmt.Errorf("assignments %q: %w", pattern, err)
		}
		if !strings.Contains(pattern, "*") {
			a.globs = append(a.globs, quantGlob{prefix: pattern, pattern: pattern, method: m})
			continue
		}
		prefix := strings.TrimSuffix(pattern, "*")
		a.globs = append(a.globs, quantGlob{prefix: prefix, pattern: pattern, method: m})
	}
	sort.Slice(a.globs, func(i, j int) bool {
		if len(a.globs[i].prefix) != len(a.globs[j].prefix) {
			return len(a.globs[i].prefix) > len(a.globs[j].prefix)
		}
		return a.globs[i].pattern < a.globs[j].pattern
	})
	return &a, nil
}

// methodFor resolves the quantization method for a tensor name.
func (a *quantAssignments) methodFor(name string) quantMethod {
	if a == nil {
		return quantMethod{}
	}
	for _, g := range a.globs {
		if strings.Contains(g.pattern, "*") {
			if strings.HasPrefix(name, g.prefix) {
				return g.method
			}
			continue
		}
		if name == g.pattern {
			return g.method
		}
	}
	return a.defaultMethod
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
	bSize := int(mcf.QuantBlockSize)
	sSize := int(mcf.QuantSuperSize)

	switch strings.ToLower(strings.TrimSpace(s)) {
	case "k6":
		return quantMethod{Name: "k6", DType: mcf.DTypeK6, Bits: 6, BlockSize: bSize, SuperSize: sSize, Family: "k"}, nil
	case "k4":
		return quantMethod{Name: "k4", DType: mcf.DTypeK4, Bits: 4, BlockSize: bSize, SuperSize: sSize, Family: "k"}, nil
	case "k3":
		return quantMethod{Name: "k3", DType: mcf.DTypeK3, Bits: 3, BlockSize: bSize, SuperSize: sSize, Family: "k"}, nil
	case "k2":
		return quantMethod{Name: "k2", DType: mcf.DTypeK2, Bits: 2, BlockSize: bSize, SuperSize: sSize, Family: "k"}, nil
	case "q8":
		return quantMethod{Name: "q8", DType: mcf.DTypeQ8, Bits: 8, BlockSize: bSize, SuperSize: 0, Family: "q"}, nil
	case "q4":
		return quantMethod{Name: "q4", DType: mcf.DTypeQ4, Bits: 4, BlockSize: bSize, SuperSize: 0, Family: "q"}, nil
	case "int8", "int4":
		return quantMethod{}, errors.New("int* quantization is reserved for activations and is not implemented")
	default:
		if name := strings.ToLower(strings.TrimSpace(s)); strings.HasPrefix(name, "gq") {
			return parseGQMethod(name)
		}
		if strings.HasPrefix(strings.ToLower(s), "uq") {
			bits, err := strconv.Atoi(s[2:])
			if err != nil {
				return quantMethod{}, fmt.Errorf("unsupported method %q", s)
			}
			if bits > 4 {
				return quantMethod{}, fmt.Errorf("method %q is deprecated: a uq%d codebook costs more than it saves; use k6, q8 or gq%d", s, bits, bits)
			}
			dt, err := mcf.UQDTypeForBits(bits)
			if err != nil {
				return quantMethod{}, err
			}
			return quantMethod{Name: strings.ToLower(s), DType: dt, Bits: bits, BlockSize: bSize, SuperSize: sSize, Family: "u"}, nil
		}
		return quantMethod{}, fmt.Errorf("unsupported method %q", s)
	}
}

// parseGQMethod parses "gqN" (group 128) or "gqN-G".
func parseGQMethod(name string) (quantMethod, error) {
	spec := strings.TrimPrefix(name, "gq")
	group := 128
	if b, g, ok := strings.Cut(spec, "-"); ok {
		n, err := strconv.Atoi(g)
		if err != nil {
			return quantMethod{}, fmt.Errorf("unsupported method %q", name)
		}
		spec, group = b, n
	}
	bits, err := strconv.Atoi(spec)
	if err != nil {
		return quantMethod{}, fmt.Errorf("unsupported method %q", name)
	}
	dt, err := mcf.GQDType(bits, group)
	if err != nil {
		return quantMethod{}, err
	}
	return quantMethod{Name: mcf.GQName(dt), DType: dt, Bits: bits, BlockSize: group, SuperSize: 0, Family: "g"}, nil
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

	var recipePlan map[string]quantMethod
	if opts.Recipe == recipeMixed {
		recipePlan = planMixedRecipe(names, opts.Method)
		opts.Logger.Info("mixed recipe", "upgraded_tensors", len(recipePlan))
	}

	td, err := w.BeginSection(mcf.SectionTensorData, quantTensorDataVersion)
	if err != nil {
		return err
	}
	defer func() { _ = td.Close() }()

	recs := make([]mcf.TensorIndexRecord, 0, len(names))
	qrecs := make([]mcf.QuantRecord, 0, len(names))
	var quantBytes uint64

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
		method := opts.Method
		if opts.Assignments != nil {
			method = opts.Assignments.methodFor(name)
		}
		if up, ok := recipePlan[name]; ok {
			method = up
		}

		if shouldQuant {
			if mcf.DTypeRequiresAligned64(method.DType) {
				if err := w.AddFlags(mcf.FlagTensorDataAligned64); err != nil {
					return err
				}
			}
			data, _, err := store.ReadTensorF32(name)
			if err != nil {
				return fmt.Errorf("quantize: tensor %s: %w", name, err)
			}
			rows := info.Shape[0]
			cols := info.Shape[1]
			payload, minClip, maxClip, err = quantizeTensor(name, data, rows, cols, method, opts.MinClip, opts.MaxClip, opts.Codebooks[name])
			if err != nil {
				return fmt.Errorf("quantize: tensor %s: %w", name, err)
			}
			outDType = method.DType
			quantBytes += uint64(len(payload))
			if opts.BudgetBytes != nil {
				budget := *opts.BudgetBytes
				if opts.Assignments != nil && opts.Assignments.Budget != 0 {
					budget = opts.Assignments.Budget
				}
				if quantBytes > budget {
					return fmt.Errorf("quantize: quantized payloads total %d bytes after %s, exceeding budget %d bytes", quantBytes, name, budget)
				}
			}
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
				Method:      uint8(outDType),
				Domain:      uint8(mcf.DomainWeights),
				BlockSize:   uint16(method.BlockSize),
				SuperSize:   uint16(method.SuperSize),
				MinClip:     minClip,
				MaxClip:     maxClip,
			})
		}
	}

	if opts.BudgetBytes != nil {
		budget := *opts.BudgetBytes
		if opts.Assignments != nil && opts.Assignments.Budget != 0 {
			budget = opts.Assignments.Budget
		}
		if quantBytes > budget {
			return fmt.Errorf("quantize: quantized payloads total %d bytes, exceeding budget %d bytes", quantBytes, budget)
		}
		opts.Logger.Info("quantized payload budget", "used", quantBytes, "budget", budget)
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

func quantizeTensor(name string, src []float32, rows, cols int, method quantMethod, minOverride, maxOverride *float32, codebooks [][]float32) ([]byte, float32, float32, error) {
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

	if method.Family != "q" && method.Family != "k" && method.Family != "u" && method.Family != "g" {
		return nil, 0, 0, errors.New("unsupported quantization family")
	}

	if minOverride == nil || maxOverride == nil {
		maxAbs := maxAbs(minVal, maxVal)
		minClip = -maxAbs
		maxClip = maxAbs
	} else {
		for i, v := range src {
			src[i] = clamp(v, minClip, maxClip)
		}
	}

	switch method.Family {
	case "q":
		payload, err := mcf.EncodeQTensor(src, rows, cols, method.Bits)
		return payload, minClip, maxClip, err
	case "k":
		payload, err := mcf.EncodeKTensor(src, rows, cols, method.Bits)
		return payload, minClip, maxClip, err
	case "g":
		dt, payload, err := mcf.EncodeGQTensor(src, rows, cols, method.Bits, method.BlockSize)
		if err != nil {
			return nil, 0, 0, err
		}
		if dt != method.DType {
			return nil, 0, 0, fmt.Errorf("gq dtype mismatch: method %s, encoded %v", method.Name, dt)
		}
		return payload, minClip, maxClip, nil
	case "u":
		dt, payload, err := mcf.EncodeUQTensor(src, rows, cols, method.Bits, codebooks)
		if err != nil {
			return nil, 0, 0, err
		}
		if dt != method.DType {
			return nil, 0, 0, fmt.Errorf("uq dtype mismatch: method %s, encoded %v", method.Name, dt)
		}
		return payload, minClip, maxClip, nil
	default:
		return nil, 0, 0, errors.New("unsupported quantization family")
	}
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
		return []uint64{}, nil
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
