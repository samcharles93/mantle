package model

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"strings"

	"github.com/samcharles93/mantle/internal/mcfstore"
	"github.com/samcharles93/mantle/internal/tensor"
	"github.com/samcharles93/mantle/pkg/mcf"
)

type tensorPayload struct {
	DType mcf.TensorDType
	Shape []int
	Raw   []byte
}

type tensorSource interface {
	ReadTensor(name string) (tensorPayload, error)
	TensorShape(name string) ([]int, bool)
}

type mcfSource struct {
	mf *mcfstore.File
}

func (s mcfSource) ReadTensor(name string) (tensorPayload, error) {
	raw, info, err := s.mf.ReadTensorRaw(name)
	if err != nil {
		return tensorPayload{}, err
	}
	shape := make([]int, len(info.Shape))
	copy(shape, info.Shape)
	return tensorPayload{DType: info.DType, Shape: shape, Raw: raw}, nil
}

func (s mcfSource) TensorShape(name string) ([]int, bool) {
	info, err := s.mf.Tensor(name)
	if err != nil {
		return nil, false
	}
	shape := make([]int, len(info.Shape))
	copy(shape, info.Shape)
	return shape, true
}

func LoadModelMCF(mcfFile *mcfstore.File, configJSON []byte, maxContext int) (*Instance, error) {
	if mcfFile == nil {
		return nil, fmt.Errorf("mcf: nil file")
	}
	cfg, err := loadHFConfigBytes(configJSON)
	if err != nil {
		return nil, err
	}
	spec, err := detectArch(cfg)
	if err != nil {
		return nil, err
	}
	return loadModelFromSource(cfg, spec, mcfSource{mf: mcfFile}, maxContext)
}

func loadModelFromSource(cfg *hfConfig, spec *archSpec, src tensorSource, maxContext int) (*Instance, error) {
	if cfg == nil {
		return nil, fmt.Errorf("nil config")
	}
	if spec == nil {
		return nil, fmt.Errorf("nil arch spec")
	}
	names := spec.Names
	if !spec.UseLayerTypes && len(cfg.LayerTypes) > 0 {
		allowSliding := cfg.SlidingWindow > 0
		for i, lt := range cfg.LayerTypes {
			l := strings.ToLower(strings.TrimSpace(lt))
			if l == "" || l == "full_attention" || l == "attention" {
				continue
			}
			if allowSliding && l == "sliding_attention" {
				continue
			}
			return nil, fmt.Errorf("unsupported layer_types[%d]=%q for arch %s", i, lt, spec.Name)
		}
	}

	emb, err := loadMat(src, names.embedding)
	if err != nil {
		return nil, err
	}
	outNorm, err := loadVec(src, names.outputNorm)
	if err != nil {
		return nil, err
	}
	candidates := []string{names.embedding}
	if names.outputCandidates != nil {
		candidates = names.outputCandidates()
	}
	output, outputName, err := loadMatCandidates(src, candidates)
	if err != nil {
		return nil, err
	}
	if output == nil {
		return nil, fmt.Errorf("missing output projection (tried %v)", candidates)
	}
	_ = outputName

	blockCount, err := blockCountForConfig(cfg, spec)
	if err != nil {
		return nil, err
	}

	if maxContext <= 0 || maxContext > cfg.MaxPosition {
		maxContext = cfg.MaxPosition
	}

	headCount := cfg.NumAttentionHeads
	if headCount <= 0 {
		return nil, fmt.Errorf("num_attention_heads must be set")
	}
	if cfg.HiddenSize <= 0 {
		return nil, fmt.Errorf("hidden_size must be set")
	}
	if cfg.MaxPosition <= 0 {
		return nil, fmt.Errorf("max_position_embeddings must be set")
	}
	if cfg.VocabSize <= 0 {
		return nil, fmt.Errorf("vocab_size must be set")
	}

	headDim := cfg.HeadDim
	if headDim <= 0 {
		if cfg.HiddenSize%headCount != 0 {
			return nil, fmt.Errorf("hidden_size must be divisible by num_attention_heads when head_dim is unset")
		}
		headDim = cfg.HiddenSize / headCount
	}
	if headDim <= 0 {
		return nil, fmt.Errorf("invalid head_dim %d", headDim)
	}
	qDim := headCount * headDim

	headKVArr := buildHeadKV(cfg, spec, blockCount)
	if len(headKVArr) != blockCount {
		return nil, fmt.Errorf("head kv array length mismatch: %d != %d", len(headKVArr), blockCount)
	}

	ffnLength := cfg.IntermediateSize
	if ffnLength <= 0 {
		ffnLength = inferFFNLength(src, names, 0)
	}
	if ffnLength <= 0 {
		return nil, fmt.Errorf("intermediate_size missing and could not infer ffn length")
	}
	if cfg.MoEIntermediateSize > 0 {
		numShared := cfg.NumSharedExperts
		if numShared <= 0 {
			numShared = 1
		}
		sharedIntermediate := cfg.MoEIntermediateSize * numShared
		if sharedIntermediate > ffnLength {
			ffnLength = sharedIntermediate
		}
		if cfg.MoEIntermediateSize > ffnLength {
			ffnLength = cfg.MoEIntermediateSize
		}
	}

	rmsEps := rmsEpsilonForConfig(cfg)
	if rmsEps == 0 {
		return nil, fmt.Errorf("norm epsilon missing in config")
	}

	ropeBase := cfg.RopeTheta
	if ropeBase == 0 && cfg.RopeParameters != nil && cfg.RopeParameters.RopeTheta > 0 {
		ropeBase = cfg.RopeParameters.RopeTheta
	}
	if ropeBase == 0 {
		ropeBase = 10000
	}
	ropeScaling := ropeScalingForConfig(cfg)
	routeScale := cfg.RouteScale
	if routeScale == 0 {
		routeScale = 1
	}
	numDenseLayers := cfg.NumDenseLayers
	if numDenseLayers < 0 {
		numDenseLayers = 0
	}
	if numDenseLayers > blockCount {
		numDenseLayers = blockCount
	}
	layerTypes := []string(nil)
	if len(cfg.LayerTypes) == blockCount {
		layerTypes = make([]string, blockCount)
		copy(layerTypes, cfg.LayerTypes)
	}

	modelCfg := &ModelConfig{
		Arch: spec.Name,
		Config: Config{
			BlockCount:          blockCount,
			EmbeddingLength:     cfg.HiddenSize,
			FFNLength:           ffnLength,
			HeadCount:           headCount,
			HeadDim:             headDim,
			HeadCountKV:         headKVArr,
			RMSEpsilon:          rmsEps,
			RopeFreqBase:        ropeBase,
			RopeScaling:         ropeScaling,
			ContextLength:       cfg.MaxPosition,
			VocabSize:           cfg.VocabSize,
			ShortConvLCache:     cfg.ConvLCache,
			NumDenseLayers:      numDenseLayers,
			MoEIntermediateSize: cfg.MoEIntermediateSize,
			NumExperts:          cfg.NumExperts,
			NumExpertsPerTok:    cfg.NumExpertsPerTok,
			NumSharedExperts:    cfg.NumSharedExperts,
			RouteScale:          routeScale,
			SlidingWindow:       cfg.SlidingWindow,
			LayerTypes:          layerTypes,
			MuPEnabled:          cfg.MuPEnabled,
			AttentionBias:       cfg.AttentionBias,
		},
	}

	maxHeadKV := 0
	for _, v := range headKVArr {
		if v > maxHeadKV {
			maxHeadKV = v
		}
	}

	layers := make([]Layer, blockCount)
	for i := range blockCount {
		layer := &layers[i]
		layer.HeadKV = headKVArr[i]
		layer.IsRecurrent = layer.HeadKV == 0
		layer.AttnType = "full_attention"
		if len(layerTypes) == blockCount && layerTypes[i] != "" {
			layer.AttnType = layerTypes[i]
		}
		if layer.AttnType == "sliding_attention" && cfg.SlidingWindow > 0 {
			layer.AttnWindow = cfg.SlidingWindow
		}
		isMoELayer := spec.Name == "afmoe" && cfg.NumExperts > 0 && i >= numDenseLayers

		attnCandidates := make([]string, 0, 2)
		if names.attnNorm != nil {
			attnCandidates = append(attnCandidates, names.attnNorm(i))
		}
		if names.attnNormCandidates != nil {
			attnCandidates = names.attnNormCandidates(i)
		}
		if len(attnCandidates) == 0 {
			return nil, fmt.Errorf("layer %d: no attention norm candidates for arch %s", i, spec.Name)
		}
		attnNorm, usedAttn, err := loadVecCandidates(src, attnCandidates)
		if err != nil {
			return nil, err
		}
		if attnNorm == nil {
			return nil, fmt.Errorf("layer %d: missing attention norm (tried %v)", i, attnCandidates)
		}
		if usedAttn == "" {
			return nil, fmt.Errorf("layer %d: could not resolve attention norm tensor", i)
		}
		layer.AttnNorm = attnNorm
		if names.postAttnNormCandidates != nil {
			postAttnNorm, usedPost, err := loadVecCandidates(src, names.postAttnNormCandidates(i))
			if err != nil {
				return nil, err
			}
			if postAttnNorm == nil {
				return nil, fmt.Errorf(
					"layer %d: missing post-attention norm (tried %v)",
					i, names.postAttnNormCandidates(i),
				)
			}
			if usedPost == "" {
				return nil, fmt.Errorf("layer %d: could not resolve post-attention norm tensor", i)
			}
			layer.PostAttnNorm = postAttnNorm
		}

		ffnCandidates := make([]string, 0, 2)
		if names.ffnNorm != nil {
			ffnCandidates = append(ffnCandidates, names.ffnNorm(i))
		}
		if names.ffnNormCandidates != nil {
			ffnCandidates = names.ffnNormCandidates(i)
		}
		if len(ffnCandidates) == 0 {
			return nil, fmt.Errorf("layer %d: no ffn norm candidates for arch %s", i, spec.Name)
		}
		ffnNorm, usedFFN, err := loadVecCandidates(src, ffnCandidates)
		if err != nil {
			return nil, err
		}
		if ffnNorm == nil {
			return nil, fmt.Errorf("layer %d: missing ffn norm (tried %v)", i, ffnCandidates)
		}
		if usedFFN == "" {
			return nil, fmt.Errorf("layer %d: could not resolve ffn norm tensor", i)
		}
		layer.FfnNorm = ffnNorm
		if names.postFfnNormCandidates != nil {
			postFfnNorm, usedPost, err := loadVecCandidates(src, names.postFfnNormCandidates(i))
			if err != nil {
				return nil, err
			}
			if postFfnNorm == nil {
				return nil, fmt.Errorf(
					"layer %d: missing post-ffn norm (tried %v)",
					i, names.postFfnNormCandidates(i),
				)
			}
			if usedPost == "" {
				return nil, fmt.Errorf("layer %d: could not resolve post-ffn norm tensor", i)
			}
			layer.PostFfnNorm = postFfnNorm
		}
		if !isMoELayer {
			layer.FfnGate, err = loadMat(src, names.ffnGate(i))
			if err != nil {
				return nil, err
			}
			layer.FfnDown, err = loadMat(src, names.ffnDown(i))
			if err != nil {
				return nil, err
			}
			layer.FfnUp, err = loadMat(src, names.ffnUp(i))
			if err != nil {
				return nil, err
			}
		}

		if layer.IsRecurrent {
			if names.shortConvKernel == nil || names.shortConvInProj == nil || names.shortConvOutProj == nil {
				return nil, fmt.Errorf("layer %d: recurrent layer not supported for arch %s", i, spec.Name)
			}
			layer.ShortConvKernel, err = loadConvKernel(src, names.shortConvKernel(i))
			if err != nil {
				return nil, err
			}
			layer.ShortConvInProj, err = loadMat(src, names.shortConvInProj(i))
			if err != nil {
				return nil, err
			}
			layer.ShortConvOutProj, err = loadMat(src, names.shortConvOutProj(i))
			if err != nil {
				return nil, err
			}
			kernelLen := layer.ShortConvKernel.C
			if kernelLen < 1 {
				return nil, fmt.Errorf("invalid shortconv kernel length for layer %d", i)
			}
			layer.ShortConvState = shortConvState{
				buf:       make([]float32, (kernelLen-1)*cfg.HiddenSize),
				kernelLen: kernelLen,
			}
		} else {
			if spec.HasQKNorm {
				qNorm, used, err := loadVecCandidates(src, names.qNormCandidates(i))
				if err != nil {
					return nil, err
				}
				if qNorm == nil {
					return nil, fmt.Errorf("layer %d: missing q norm (tried %v)", i, names.qNormCandidates(i))
				}
				layer.AttnQNorm = qNorm

				kNorm, usedK, err := loadVecCandidates(src, names.kNormCandidates(i))
				if err != nil {
					return nil, err
				}
				if kNorm == nil {
					return nil, fmt.Errorf("layer %d: missing k norm (tried %v)", i, names.kNormCandidates(i))
				}
				layer.AttnKNorm = kNorm

				// Keep the selected names in error paths for easier debugging.
				if used == "" || usedK == "" {
					return nil, fmt.Errorf("layer %d: could not resolve q/k norm tensors", i)
				}
			}
			layer.Wq, err = loadMat(src, names.wq(i))
			if err != nil {
				return nil, err
			}
			layer.Wk, err = loadMat(src, names.wk(i))
			if err != nil {
				return nil, err
			}
			layer.Wv, err = loadMat(src, names.wv(i))
			if err != nil {
				return nil, err
			}
			layer.Wo, err = loadMat(src, names.wo(i))
			if err != nil {
				return nil, err
			}

			// Load optional biases.
			if names.wqBias != nil {
				layer.WqBias, _ = loadVec(src, names.wqBias(i))
			}
			if names.wkBias != nil {
				layer.WkBias, _ = loadVec(src, names.wkBias(i))
			}
			if names.wvBias != nil {
				layer.WvBias, _ = loadVec(src, names.wvBias(i))
			}
			hidden := cfg.HiddenSize
			if layer.Wq.C != hidden || layer.Wq.R != qDim {
				return nil, fmt.Errorf("layer %d: q_proj shape [%d %d] incompatible with hidden=%d head_dim=%d heads=%d", i, layer.Wq.R, layer.Wq.C, hidden, headDim, headCount)
			}
			wantKV := layer.HeadKV * headDim
			if layer.Wk.C != hidden || layer.Wk.R != wantKV {
				return nil, fmt.Errorf("layer %d: k_proj shape [%d %d] incompatible with hidden=%d head_dim=%d kv_heads=%d", i, layer.Wk.R, layer.Wk.C, hidden, headDim, layer.HeadKV)
			}
			if layer.Wv.C != hidden || layer.Wv.R != wantKV {
				return nil, fmt.Errorf("layer %d: v_proj shape [%d %d] incompatible with hidden=%d head_dim=%d kv_heads=%d", i, layer.Wv.R, layer.Wv.C, hidden, headDim, layer.HeadKV)
			}
			if layer.Wo.R != hidden || layer.Wo.C != qDim {
				return nil, fmt.Errorf("layer %d: o_proj shape [%d %d] incompatible with hidden=%d head_dim=%d heads=%d", i, layer.Wo.R, layer.Wo.C, hidden, headDim, headCount)
			}
			if names.attnGate != nil {
				layer.AttnGate, err = loadMat(src, names.attnGate(i))
				if err != nil {
					return nil, err
				}
				if layer.AttnGate.R != qDim || layer.AttnGate.C != hidden {
					return nil, fmt.Errorf(
						"layer %d: attn gate shape [%d %d] incompatible with hidden=%d qdim=%d",
						i, layer.AttnGate.R, layer.AttnGate.C, hidden, qDim,
					)
				}
			}
			if isMoELayer {
				moe, err := loadMoELayer(src, cfg, names, i)
				if err != nil {
					return nil, err
				}
				layer.MoE = moe
			}
			kvStride := layer.HeadKV * headDim

			// Initialize cache based on type
			cache := attnCache{kvStride: kvStride}

			// Key cache
			kt := modelCfg.Config.CacheTypeK
			if kt == "" {
				kt = CacheTypeF32 // Default for now, CLI will override
			}
			if kt == CacheTypeF16 {
				cache.k16 = make([]uint16, maxContext*kvStride)
			} else {
				cache.k = make([]float32, maxContext*kvStride)
			}

			// Value cache
			vt := modelCfg.Config.CacheTypeV
			if vt == "" {
				vt = CacheTypeF32
			}
			if vt == CacheTypeF16 {
				cache.v16 = make([]uint16, maxContext*kvStride)
			} else {
				cache.v = make([]float32, maxContext*kvStride)
			}

			layer.AttnCache = cache
		}
	}

	muPScale := float32(1)
	if modelCfg.Config.MuPEnabled && modelCfg.Config.EmbeddingLength > 0 {
		muPScale = float32(math.Sqrt(float64(modelCfg.Config.EmbeddingLength)))
	}

	m := &Instance{
		Config:        modelCfg,
		Embeddings:    emb,
		OutputNorm:    outNorm,
		Output:        output,
		Layers:        layers,
		MaxContext:    maxContext,
		Pos:           0,
		RMSEpsilon:    float32(rmsEps),
		HeadDim:       headDim,
		HeadCount:     headCount,
		MaxHeadKV:     maxHeadKV,
		muPScale:      muPScale,
		ropeLocalOnly: spec.RopeLocalOnly,
	}
	m.ops = defaultOps{}
	m.initScratch()
	m.UpdateRoPE()
	return m, nil
}

func readTensorF32(src tensorSource, name string) ([]float32, []int, error) {
	payload, err := src.ReadTensor(name)
	if err != nil {
		return nil, nil, fmt.Errorf("%s: %w", name, err)
	}
	data, err := decodeTensorF32(payload)
	if err != nil {
		return nil, nil, fmt.Errorf("%s: %w", name, err)
	}
	return data, payload.Shape, nil
}

func decodeTensorF32(p tensorPayload) ([]float32, error) {
	n, err := numElementsModel(p.Shape)
	if err != nil {
		return nil, err
	}
	switch p.DType {
	case mcf.DTypeF32:
		if len(p.Raw) != n*4 {
			return nil, fmt.Errorf("invalid f32 data size")
		}
		out := make([]float32, n)
		for i := 0; i < n; i++ {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(p.Raw[i*4:]))
		}
		return out, nil
	case mcf.DTypeBF16:
		if len(p.Raw) != n*2 {
			return nil, fmt.Errorf("invalid bf16 data size")
		}
		out := make([]float32, n)
		for i := 0; i < n; i++ {
			u := binary.LittleEndian.Uint16(p.Raw[i*2:])
			out[i] = bf16ToF32(u)
		}
		return out, nil
	case mcf.DTypeF16:
		if len(p.Raw) != n*2 {
			return nil, fmt.Errorf("invalid f16 data size")
		}
		out := make([]float32, n)
		for i := 0; i < n; i++ {
			u := binary.LittleEndian.Uint16(p.Raw[i*2:])
			out[i] = fp16ToF32(u)
		}
		return out, nil
	default:
		return nil, fmt.Errorf("unsupported dtype %d", p.DType)
	}
}

func numElementsModel(shape []int) (int, error) {
	if len(shape) == 0 {
		return 0, fmt.Errorf("empty shape")
	}
	n := 1
	maxInt := int(^uint(0) >> 1)
	for _, d := range shape {
		if d <= 0 {
			return 0, fmt.Errorf("invalid dim %d", d)
		}
		if n > maxInt/d {
			return 0, fmt.Errorf("tensor too large")
		}
		n *= d
	}
	return n, nil
}

func bf16ToF32(u uint16) float32 {
	return math.Float32frombits(uint32(u) << 16)
}

func fp16ToF32(h uint16) float32 {
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

func loadMat(src tensorSource, name string) (*tensor.Mat, error) {
	payload, err := src.ReadTensor(name)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", name, err)
	}
	shape := payload.Shape
	if len(shape) != 2 {
		return nil, fmt.Errorf("%s: expected 2D tensor", name)
	}
	r := shape[0]
	c := shape[1]
	if r <= 0 || c <= 0 {
		return nil, fmt.Errorf("%s: invalid shape %v", name, shape)
	}
	switch payload.DType {
	case mcf.DTypeF32:
		data, err := decodeTensorF32(payload)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", name, err)
		}
		if r*c != len(data) {
			return nil, fmt.Errorf("%s: size mismatch", name)
		}
		m := tensor.NewMatFromData(r, c, data)
		return &m, nil
	case mcf.DTypeBF16, mcf.DTypeF16:
		m, err := tensor.NewMatFromRaw(r, c, payload.DType, payload.Raw)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", name, err)
		}
		return &m, nil
	default:
		if mcf.DTypeRequiresAligned64(payload.DType) {
			shapeU64 := make([]uint64, len(shape))
			for i, v := range shape {
				shapeU64[i] = uint64(v)
			}
			want, err := mcf.QuantPayloadSize(shapeU64, payload.DType)
			if err != nil {
				return nil, fmt.Errorf("%s: %w", name, err)
			}
			if uint64(len(payload.Raw)) != want {
				return nil, fmt.Errorf("%s: quant payload size mismatch", name)
			}
			m := tensor.Mat{R: r, C: c, Stride: c, DType: payload.DType, Raw: payload.Raw}
			cache, err := tensor.BuildQuantCache(&m)
			if err != nil {
				return nil, fmt.Errorf("%s: quant cache: %w", name, err)
			}
			m.Quant = cache
			return &m, nil
		}
		data, err := decodeTensorF32(payload)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", name, err)
		}
		if r*c != len(data) {
			return nil, fmt.Errorf("%s: size mismatch", name)
		}
		m := tensor.NewMatFromData(r, c, data)
		return &m, nil
	}
}

func loadVec(src tensorSource, name string) ([]float32, error) {
	data, shape, err := readTensorF32(src, name)
	if err != nil {
		return nil, err
	}
	if len(shape) != 1 {
		return nil, fmt.Errorf("%s: expected 1D tensor", name)
	}
	return data, nil
}

func loadConvKernel(src tensorSource, name string) (*tensor.Mat, error) {
	data, shape, err := readTensorF32(src, name)
	if err != nil {
		return nil, err
	}
	if len(shape) != 3 {
		return nil, fmt.Errorf("%s: expected 3D tensor", name)
	}
	out := shape[0]
	in := shape[1]
	k := shape[2]
	if in != 1 {
		return nil, fmt.Errorf("%s: expected in=1, got %d", name, in)
	}
	if out*k != len(data) {
		return nil, fmt.Errorf("%s: size mismatch", name)
	}
	return &tensor.Mat{R: out, C: k, Stride: k, Data: data}, nil
}

func loadMatCandidates(src tensorSource, candidates []string) (*tensor.Mat, string, error) {
	for _, name := range candidates {
		if name == "" {
			continue
		}
		m, err := loadMat(src, name)
		if err == nil {
			return m, name, nil
		}
		if isTensorMissing(err) {
			continue
		}
		return nil, "", err
	}
	return nil, "", nil
}

func loadVecCandidates(src tensorSource, candidates []string) ([]float32, string, error) {
	for _, name := range candidates {
		if name == "" {
			continue
		}
		v, err := loadVec(src, name)
		if err == nil {
			return v, name, nil
		}
		if isTensorMissing(err) {
			continue
		}
		return nil, "", err
	}
	return nil, "", nil
}

func loadMoELayer(src tensorSource, cfg *hfConfig, names archNames, layer int) (*moeLayer, error) {
	if cfg == nil {
		return nil, fmt.Errorf("layer %d: nil config for moe", layer)
	}
	if cfg.NumExperts <= 0 {
		return nil, fmt.Errorf("layer %d: num_experts must be > 0 for moe", layer)
	}
	if cfg.MoEIntermediateSize <= 0 {
		return nil, fmt.Errorf("layer %d: moe_intermediate_size must be > 0 for moe", layer)
	}
	if names.moeRouter == nil || names.moeExpertUp == nil || names.moeExpertGate == nil || names.moeExpertDown == nil {
		return nil, fmt.Errorf("layer %d: moe tensor names are not defined for this arch", layer)
	}

	hidden := cfg.HiddenSize
	numExperts := cfg.NumExperts
	topK := cfg.NumExpertsPerTok
	if topK <= 0 {
		topK = 1
	}
	if topK > numExperts {
		return nil, fmt.Errorf("layer %d: num_experts_per_tok %d exceeds num_experts %d", layer, topK, numExperts)
	}
	numShared := cfg.NumSharedExperts
	if numShared <= 0 {
		numShared = 1
	}
	sharedIntermediate := cfg.MoEIntermediateSize * numShared

	router, err := loadMat(src, names.moeRouter(layer))
	if err != nil {
		return nil, err
	}
	if router.R != numExperts || router.C != hidden {
		return nil, fmt.Errorf(
			"layer %d: moe router shape [%d %d] incompatible with experts=%d hidden=%d",
			layer, router.R, router.C, numExperts, hidden,
		)
	}

	expertBias := make([]float32, numExperts)
	if names.moeExpertBias != nil {
		bias, err := loadVec(src, names.moeExpertBias(layer))
		if err != nil {
			if !isTensorMissing(err) {
				return nil, err
			}
		} else {
			if len(bias) != numExperts {
				return nil, fmt.Errorf(
					"layer %d: moe expert_bias len %d incompatible with experts=%d",
					layer, len(bias), numExperts,
				)
			}
			copy(expertBias, bias)
		}
	}

	if names.moeSharedUp == nil || names.moeSharedGate == nil || names.moeSharedDown == nil {
		return nil, fmt.Errorf("layer %d: moe shared expert tensor names are not defined for this arch", layer)
	}
	sharedUp, err := loadMat(src, names.moeSharedUp(layer))
	if err != nil {
		return nil, err
	}
	sharedGate, err := loadMat(src, names.moeSharedGate(layer))
	if err != nil {
		return nil, err
	}
	sharedDown, err := loadMat(src, names.moeSharedDown(layer))
	if err != nil {
		return nil, err
	}
	if sharedUp.C != hidden || sharedGate.C != hidden {
		return nil, fmt.Errorf("layer %d: moe shared expert input dims incompatible with hidden=%d", layer, hidden)
	}
	if sharedUp.R != sharedIntermediate || sharedGate.R != sharedIntermediate {
		return nil, fmt.Errorf(
			"layer %d: moe shared expert intermediate dims incompatible with shared_intermediate=%d",
			layer, sharedIntermediate,
		)
	}
	if sharedDown.R != hidden || sharedDown.C != sharedIntermediate {
		return nil, fmt.Errorf(
			"layer %d: moe shared expert down_proj shape [%d %d] incompatible with hidden=%d shared_intermediate=%d",
			layer, sharedDown.R, sharedDown.C, hidden, sharedIntermediate,
		)
	}

	experts := make([]moeExpert, numExperts)
	for expert := 0; expert < numExperts; expert++ {
		up, err := loadMat(src, names.moeExpertUp(layer, expert))
		if err != nil {
			return nil, err
		}
		gate, err := loadMat(src, names.moeExpertGate(layer, expert))
		if err != nil {
			return nil, err
		}
		down, err := loadMat(src, names.moeExpertDown(layer, expert))
		if err != nil {
			return nil, err
		}
		if up.C != hidden || gate.C != hidden {
			return nil, fmt.Errorf(
				"layer %d expert %d: moe expert input dims incompatible with hidden=%d",
				layer, expert, hidden,
			)
		}
		if up.R != cfg.MoEIntermediateSize || gate.R != cfg.MoEIntermediateSize {
			return nil, fmt.Errorf(
				"layer %d expert %d: moe expert intermediate dims incompatible with moe_intermediate_size=%d",
				layer, expert, cfg.MoEIntermediateSize,
			)
		}
		if down.R != hidden || down.C != cfg.MoEIntermediateSize {
			return nil, fmt.Errorf(
				"layer %d expert %d: moe expert down_proj shape [%d %d] incompatible with hidden=%d moe_intermediate_size=%d",
				layer, expert, down.R, down.C, hidden, cfg.MoEIntermediateSize,
			)
		}
		experts[expert] = moeExpert{Up: up, Gate: gate, Down: down}
	}

	routeScale := float32(cfg.RouteScale)
	if routeScale == 0 {
		routeScale = 1
	}

	return &moeLayer{
		Router:     router,
		ExpertBias: expertBias,
		Shared: moeShared{
			Up:           sharedUp,
			Gate:         sharedGate,
			Down:         sharedDown,
			Intermediate: sharedIntermediate,
		},
		Experts:    experts,
		TopK:       topK,
		RouteScale: routeScale,
	}, nil
}

func isTensorMissing(err error) bool {
	if err == nil {
		return false
	}
	if errors.Is(err, mcfstore.ErrTensorNotFound) {
		return true
	}
	msg := strings.ToLower(err.Error())
	return strings.Contains(msg, "tensor not found")
}

func blockCountForConfig(cfg *hfConfig, spec *archSpec) (int, error) {
	if cfg == nil || spec == nil {
		return 0, fmt.Errorf("nil config/spec")
	}
	if spec.UseLayerTypes && len(cfg.LayerTypes) > 0 {
		return len(cfg.LayerTypes), nil
	}
	if cfg.NumHiddenLayers > 0 {
		return cfg.NumHiddenLayers, nil
	}
	if len(cfg.LayerTypes) > 0 {
		return len(cfg.LayerTypes), nil
	}
	return 0, fmt.Errorf("could not determine layer count from config")
}

func buildHeadKV(cfg *hfConfig, spec *archSpec, blockCount int) []int {
	if blockCount <= 0 {
		return nil
	}
	headCount := cfg.NumAttentionHeads
	if headCount <= 0 {
		return nil
	}
	kvHeads := cfg.NumKeyValueHeads
	if kvHeads <= 0 {
		kvHeads = headCount
	}

	if spec.UseLayerTypes && len(cfg.LayerTypes) == blockCount {
		out := make([]int, blockCount)
		for i, t := range cfg.LayerTypes {
			if t == "full_attention" {
				out[i] = kvHeads
			} else {
				out[i] = 0
			}
		}
		return out
	}

	out := make([]int, blockCount)
	for i := range out {
		out[i] = kvHeads
	}
	return out
}

func rmsEpsilonForConfig(cfg *hfConfig) float64 {
	if cfg == nil {
		return 0
	}
	if cfg.RMSNormEps != 0 {
		return cfg.RMSNormEps
	}
	if cfg.LayerNormEps != 0 {
		return cfg.LayerNormEps
	}
	return cfg.NormEps
}

func inferFFNLength(src tensorSource, names archNames, layer int) int {
	name := names.ffnGate(layer)
	if shape, ok := src.TensorShape(name); ok && len(shape) >= 1 {
		return shape[0]
	}
	return 0
}

func (m *Instance) attention(layer *Layer, x []float32, pos int) []float32 {
	nHead := m.HeadCount
	headDim := m.HeadDim
	kvHeads := layer.HeadKV
	kvStride := layer.AttnCache.kvStride
	if kvHeads <= 0 {
		panic("attention layer without kv heads")
	}

	q := m.scratch.q[:nHead*headDim]
	k := m.scratch.k[:kvStride]
	v := m.scratch.v[:kvStride]
	attnOut := m.scratch.attnOut

	var qx *tensor.QuantVec
	if tensor.CanUseQuantVec(layer.Wq) || tensor.CanUseQuantVec(layer.Wk) || tensor.CanUseQuantVec(layer.Wv) {
		qx = tensor.PrepareQuantVec(x)
		defer tensor.ReleaseQuantVec(qx)
	}
	ensureOps(m.ops).MatVecWithQuant(q, layer.Wq, x, qx)
	ensureOps(m.ops).MatVecWithQuant(k, layer.Wk, x, qx)
	ensureOps(m.ops).MatVecWithQuant(v, layer.Wv, x, qx)

	if len(layer.WqBias) > 0 {
		tensor.Add(q, layer.WqBias)
	}
	if len(layer.WkBias) > 0 {
		tensor.Add(k, layer.WkBias)
	}
	if len(layer.WvBias) > 0 {
		tensor.Add(v, layer.WvBias)
	}

	if len(layer.AttnQNorm) > 0 {
		for h := range nHead {
			tensor.RMSNorm(q[h*headDim:(h+1)*headDim], q[h*headDim:(h+1)*headDim], layer.AttnQNorm, m.RMSEpsilon)
		}
	}
	if len(layer.AttnKNorm) > 0 {
		for h := range kvHeads {
			tensor.RMSNorm(k[h*headDim:(h+1)*headDim], k[h*headDim:(h+1)*headDim], layer.AttnKNorm, m.RMSEpsilon)
		}
	}

	applyRoPE := !m.ropeLocalOnly || layer.AttnType != "full_attention"
	if applyRoPE {
		tensor.ApplyRoPE(q, nHead, headDim, pos, m.ropeInvFreq, m.ropeAttnScale)
		tensor.ApplyRoPE(k, kvHeads, headDim, pos, m.ropeInvFreq, m.ropeAttnScale)
	}

	// Helper for F16 cache
	storeCache := func(src []float32, f32Dest []float32, f16Dest []uint16, offset int) {
		if f32Dest != nil {
			copy(f32Dest[offset:], src)
		} else if f16Dest != nil {
			tensor.Float32ToFloat16Slice(src, f16Dest[offset:])
		}
	}

	offset := pos * kvStride
	storeCache(k, layer.AttnCache.k, layer.AttnCache.k16, offset)
	storeCache(v, layer.AttnCache.v, layer.AttnCache.v16, offset)

	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	start := 0
	if layer.AttnWindow > 0 {
		start = pos - layer.AttnWindow + 1
		if start < 0 {
			start = 0
		}
	}
	ctx := attnContext{
		q:        q,
		cacheK:   layer.AttnCache.k,
		cacheV:   layer.AttnCache.v,
		cacheK16: layer.AttnCache.k16,
		cacheV16: layer.AttnCache.v16,
		attnOut:  attnOut,
		pos:      pos,
		start:    start,
		kvStride: kvStride,
		headDim:  headDim,
		nHead:    nHead,
		kvHeads:  kvHeads,
		scale:    scale,
	}

	pool := m.getAttnPool()
	workers := pool.size
	if workers <= 1 {
		runAttnHeads(&ctx, m.scratch.scores, 0, nHead)
	} else {
		chunk := (nHead + workers - 1) / workers
		done := <-pool.doneSlots
		activeWorkers := 0
		for i := 0; i < workers; i++ {
			rs := i * chunk
			re := rs + chunk
			if re > nHead {
				re = nHead
			}
			if rs >= re {
				break
			}
			activeWorkers++
			pool.tasks <- attnTask{
				ctx:  &ctx,
				rs:   rs,
				re:   re,
				done: done,
			}
		}
		for i := 0; i < activeWorkers; i++ {
			<-done
		}
		pool.doneSlots <- done
	}

	if layer.AttnGate != nil {
		gate := m.scratch.attnGate[:nHead*headDim]
		ensureOps(m.ops).MatVecWithQuant(gate, layer.AttnGate, x, qx)
		for i := range gate {
			attnOut[i] *= tensor.Sigmoid(gate[i])
		}
	}
	ensureOps(m.ops).MatVec(m.scratch.attnProj, layer.Wo, attnOut[:nHead*headDim])
	return m.scratch.attnProj
}

func (m *Instance) shortconv(layer *Layer, x []float32) []float32 {
	embd := m.Config.Config.EmbeddingLength
	ensureOps(m.ops).MatVec(m.scratch.scProj, layer.ShortConvInProj, x)
	b := m.scratch.scProj[:embd]
	c := m.scratch.scProj[embd : 2*embd]
	xg := m.scratch.scProj[2*embd:]

	bx := m.scratch.scBx
	for i := range embd {
		bx[i] = b[i] * xg[i]
	}

	kernel := layer.ShortConvKernel
	convOut := m.scratch.scConv
	kernelLen := kernel.C
	state := layer.ShortConvState.buf
	for i := range embd {
		row := kernel.Row(i)
		var sum float32
		for k := 0; k < kernelLen-1; k++ {
			sum += row[k] * state[k*embd+i]
		}
		sum += row[kernelLen-1] * bx[i]
		convOut[i] = sum
	}

	// update state: shift left and append current bx
	if kernelLen > 1 {
		if kernelLen == 2 {
			copy(state, bx)
		} else {
			copy(state, state[embd:])
			copy(state[(kernelLen-2)*embd:], bx)
		}
	}

	for i := range embd {
		m.scratch.tmp2[i] = c[i] * convOut[i]
	}
	ensureOps(m.ops).MatVec(m.scratch.tmp, layer.ShortConvOutProj, m.scratch.tmp2)
	return m.scratch.tmp
}

func (m *Instance) ffn(layer *Layer, x []float32) []float32 {
	var qx *tensor.QuantVec
	if tensor.CanUseQuantVec(layer.FfnUp) || tensor.CanUseQuantVec(layer.FfnGate) {
		qx = tensor.PrepareQuantVec(x)
		defer tensor.ReleaseQuantVec(qx)
	}
	ensureOps(m.ops).MatVecWithQuant(m.scratch.ffnUp, layer.FfnUp, x, qx)
	ensureOps(m.ops).MatVecWithQuant(m.scratch.ffnGate, layer.FfnGate, x, qx)
	for i := range m.scratch.ffnAct {
		m.scratch.ffnAct[i] = tensor.Silu(m.scratch.ffnGate[i]) * m.scratch.ffnUp[i]
	}
	ensureOps(m.ops).MatVec(m.scratch.tmp2, layer.FfnDown, m.scratch.ffnAct)
	return m.scratch.tmp2
}

func (m *Instance) initScratch() {
	embd := m.Config.Config.EmbeddingLength
	ffn := m.Config.Config.FFNLength
	numExperts := m.Config.Config.NumExperts
	topK := m.Config.Config.NumExpertsPerTok
	kv := m.MaxHeadKV * m.HeadDim
	if kv < 1 {
		kv = m.HeadDim
	}
	qDim := m.HeadCount * m.HeadDim
	if qDim < 1 {
		qDim = embd
	}
	m.scratch = scratchBuffers{
		x:         make([]float32, embd),
		tmp:       make([]float32, embd),
		tmp2:      make([]float32, embd),
		q:         make([]float32, qDim),
		k:         make([]float32, kv),
		v:         make([]float32, kv),
		attnOut:   make([]float32, qDim),
		attnProj:  make([]float32, embd),
		attnGate:  make([]float32, qDim),
		scores:    make([]float32, m.MaxContext),
		ffnUp:     make([]float32, ffn),
		ffnGate:   make([]float32, ffn),
		ffnAct:    make([]float32, ffn),
		moeAccum:  make([]float32, embd),
		routerRaw: make([]float32, numExperts),
		routerSel: make([]float32, numExperts),
		routerIdx: make([]int, topK),
		routerW:   make([]float32, topK),
		scProj:    make([]float32, embd*3),
		scBx:      make([]float32, embd),
		scConv:    make([]float32, embd),
		logits:    make([]float32, m.Config.Config.VocabSize),
	}
	m.initAttnPool()
}

func (m *Instance) UpdateRoPE() {
	headDim := m.HeadDim
	if headDim == 0 {
		headDim = m.Config.Config.HeadDim
	}
	if headDim == 0 {
		headDim = m.Config.Config.EmbeddingLength / m.Config.Config.HeadCount
	}
	ropeInvFreq := make([]float64, headDim/2)
	for i := 0; i < len(ropeInvFreq); i++ {
		power := float64(2*i) / float64(headDim)
		ropeInvFreq[i] = 1.0 / math.Pow(m.Config.Config.RopeFreqBase, power)
	}
	attnScale := 1.0
	if rs := m.Config.Config.RopeScaling; rs != nil {
		attnScale = applyRopeScaling(ropeInvFreq, m.Config.Config.RopeFreqBase, m.Config.Config.ContextLength, rs)
	}
	m.ropeInvFreq = ropeInvFreq
	m.ropeAttnScale = float32(attnScale)
}
