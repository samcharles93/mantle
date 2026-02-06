package simd

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"strings"

	"github.com/samcharles93/mantle/internal/mcfstore"
	"github.com/samcharles93/mantle/internal/model"
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
	cfg, err := model.LoadHFConfigBytes(configJSON)
	if err != nil {
		return nil, err
	}
	spec, err := model.DetectArch(cfg)
	if err != nil {
		return nil, err
	}
	return loadModelFromSource(cfg, spec, mcfSource{mf: mcfFile}, maxContext)
}

func loadModelFromSource(cfg *model.HFConfig, spec *model.ArchSpec, src tensorSource, maxContext int) (*Instance, error) {
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

	emb, err := loadMat(src, names.Embedding)
	if err != nil {
		return nil, err
	}
	outNorm, err := loadVec(src, names.OutputNorm)
	if err != nil {
		return nil, err
	}
	candidates := []string{names.Embedding}
	if names.OutputCandidates != nil {
		candidates = names.OutputCandidates()
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
	ropeLocalBase := cfg.RopeLocalBaseFreq
	if ropeLocalBase == 0 {
		ropeLocalBase = ropeBase
	}
	ropeScaling := model.RopeScalingForConfig(cfg)
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
	} else if cfg.GlobalAttnEveryN > 0 && cfg.SlidingWindow > 0 {
		layerTypes = make([]string, blockCount)
		for i := 0; i < blockCount; i++ {
			if (i+1)%cfg.GlobalAttnEveryN == 0 {
				layerTypes[i] = "full_attention"
			} else {
				layerTypes[i] = "sliding_attention"
			}
		}
	}

	modelCfg := &ModelConfig{
		Arch: spec.Name,
		Config: Config{
			BlockCount:             blockCount,
			EmbeddingLength:        cfg.HiddenSize,
			FFNLength:              ffnLength,
			HeadCount:              headCount,
			HeadDim:                headDim,
			HeadCountKV:            headKVArr,
			RMSEpsilon:             rmsEps,
			RopeFreqBase:           ropeBase,
			RopeFreqBaseLocal:      ropeLocalBase,
			RopeScaling:            toSIMDRopeScaling(ropeScaling),
			ContextLength:          cfg.MaxPosition,
			VocabSize:              cfg.VocabSize,
			ShortConvLCache:        cfg.ConvLCache,
			EmbeddingMultiplier:    cfg.EmbeddingMultiplier,
			LMHeadMultiplier:       cfg.LMHeadMultiplier,
			AttentionInMultiplier:  cfg.AttentionInMultiplier,
			AttentionOutMultiplier: cfg.AttentionOutMultiplier,
			SSMInMultiplier:        cfg.SSMInMultiplier,
			SSMOutMultiplier:       cfg.SSMOutMultiplier,
			SSMMultipliers:         cfg.SSMMultipliers,
			MambaRMSNorm:           cfg.MambaRMSNorm,
			MambaNormBeforeGate:    cfg.MambaNormBeforeGate,
			MambaChunkSize:         cfg.MambaChunkSize,
			TimeStepMin:            cfg.TimeStepMin,
			TimeStepMax:            cfg.TimeStepMax,
			TimeStepFloor:          cfg.TimeStepFloor,
			NumDenseLayers:         numDenseLayers,
			MoEIntermediateSize:    cfg.MoEIntermediateSize,
			NumExperts:             cfg.NumExperts,
			NumExpertsPerTok:       cfg.NumExpertsPerTok,
			NumSharedExperts:       cfg.NumSharedExperts,
			RouteScale:             routeScale,
			SlidingWindow:          cfg.SlidingWindow,
			LayerTypes:             layerTypes,
			MuPEnabled:             cfg.MuPEnabled,
			AttentionBias:          cfg.AttentionBias,
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
		if names.AttnNorm != nil {
			attnCandidates = append(attnCandidates, names.AttnNorm(i))
		}
		if names.AttnNormCandidates != nil {
			attnCandidates = names.AttnNormCandidates(i)
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
		if names.PostAttnNormCandidates != nil {
			postAttnNorm, usedPost, err := loadVecCandidates(src, names.PostAttnNormCandidates(i))
			if err != nil {
				return nil, err
			}
			if postAttnNorm == nil {
				return nil, fmt.Errorf(
					"layer %d: missing post-attention norm (tried %v)",
					i, names.PostAttnNormCandidates(i),
				)
			}
			if usedPost == "" {
				return nil, fmt.Errorf("layer %d: could not resolve post-attention norm tensor", i)
			}
			layer.PostAttnNorm = postAttnNorm
		}

		ffnCandidates := make([]string, 0, 2)
		if names.FfnNorm != nil {
			ffnCandidates = append(ffnCandidates, names.FfnNorm(i))
		}
		if names.FfnNormCandidates != nil {
			ffnCandidates = names.FfnNormCandidates(i)
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
		if names.PostFfnNormCandidates != nil {
			postFfnNorm, usedPost, err := loadVecCandidates(src, names.PostFfnNormCandidates(i))
			if err != nil {
				return nil, err
			}
			if postFfnNorm == nil {
				return nil, fmt.Errorf(
					"layer %d: missing post-ffn norm (tried %v)",
					i, names.PostFfnNormCandidates(i),
				)
			}
			if usedPost == "" {
				return nil, fmt.Errorf("layer %d: could not resolve post-ffn norm tensor", i)
			}
			layer.PostFfnNorm = postFfnNorm
		}
		if !isMoELayer {
			layer.FfnGate, err = loadMat(src, names.FfnGate(i))
			if err != nil {
				return nil, err
			}
			layer.FfnDown, err = loadMat(src, names.FfnDown(i))
			if err != nil {
				return nil, err
			}
			layer.FfnUp, err = loadMat(src, names.FfnUp(i))
			if err != nil {
				return nil, err
			}
		}

		if layer.IsRecurrent {
			if names.ShortConvKernel == nil || names.ShortConvInProj == nil || names.ShortConvOutProj == nil {
				return nil, fmt.Errorf("layer %d: recurrent layer not supported for arch %s", i, spec.Name)
			}
			layer.ShortConvKernel, err = loadConvKernel(src, names.ShortConvKernel(i))
			if err != nil {
				return nil, err
			}
			layer.ShortConvInProj, err = loadMat(src, names.ShortConvInProj(i))
			if err != nil {
				return nil, err
			}
			layer.ShortConvOutProj, err = loadMat(src, names.ShortConvOutProj(i))
			if err != nil {
				return nil, err
			}
			kernelLen := layer.ShortConvKernel.C
			if kernelLen < 1 {
				return nil, fmt.Errorf("invalid shortconv kernel length for layer %d", i)
			}
			layer.ShortConvState = ShortConvState{
				Buf:       make([]float32, (kernelLen-1)*cfg.HiddenSize),
				KernelLen: kernelLen,
			}
		} else {
			if spec.HasQKNorm {
				qNorm, used, err := loadVecCandidates(src, names.QNormCandidates(i))
				if err != nil {
					return nil, err
				}
				if qNorm == nil {
					return nil, fmt.Errorf("layer %d: missing q norm (tried %v)", i, names.QNormCandidates(i))
				}
				layer.AttnQNorm = qNorm

				kNorm, usedK, err := loadVecCandidates(src, names.KNormCandidates(i))
				if err != nil {
					return nil, err
				}
				if kNorm == nil {
					return nil, fmt.Errorf("layer %d: missing k norm (tried %v)", i, names.KNormCandidates(i))
				}
				layer.AttnKNorm = kNorm

				// Keep the selected names in error paths for easier debugging.
				if used == "" || usedK == "" {
					return nil, fmt.Errorf("layer %d: could not resolve q/k norm tensors", i)
				}
			}
			layer.Wq, err = loadMat(src, names.Wq(i))
			if err != nil {
				return nil, err
			}
			layer.Wk, err = loadMat(src, names.Wk(i))
			if err != nil {
				return nil, err
			}
			layer.Wv, err = loadMat(src, names.Wv(i))
			if err != nil {
				return nil, err
			}
			layer.Wo, err = loadMat(src, names.Wo(i))
			if err != nil {
				return nil, err
			}

			// Load optional biases.
			if names.WqBias != nil {
				layer.WqBias, _ = loadVec(src, names.WqBias(i))
			}
			if names.WkBias != nil {
				layer.WkBias, _ = loadVec(src, names.WkBias(i))
			}
			if names.WvBias != nil {
				layer.WvBias, _ = loadVec(src, names.WvBias(i))
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
			if names.AttnGate != nil {
				layer.AttnGate, err = loadMat(src, names.AttnGate(i))
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
			if names.MambaInProj != nil {
				mamba, err := loadMambaLayer(src, cfg, names, i, modelCfg)
				if err != nil {
					return nil, err
				}
				if mamba != nil {
					if modelCfg.Config.MambaRMSNorm && mamba.Norm == nil {
						return nil, fmt.Errorf("layer %d: mamba rms norm enabled but missing norm weights", i)
					}
					layer.Mamba = mamba
				}
			}
			kvStride := layer.HeadKV * headDim

			// Initialize cache based on type
			cacheLen := maxContext
			if layer.AttnType == "sliding_attention" && layer.AttnWindow > 0 {
				if layer.AttnWindow < cacheLen {
					cacheLen = layer.AttnWindow
				}
			}
			cache := AttnCache{KvStride: kvStride, CacheLen: cacheLen}

			// Key cache
			kt := modelCfg.Config.CacheTypeK
			if kt == "" {
				kt = CacheTypeF32 // Default for now, CLI will override
			}
			if kt == CacheTypeF16 {
				cache.K16 = make([]uint16, cacheLen*kvStride)
			} else {
				cache.K = make([]float32, cacheLen*kvStride)
			}

			// Value cache
			vt := modelCfg.Config.CacheTypeV
			if vt == "" {
				vt = CacheTypeF32
			}
			if vt == CacheTypeF16 {
				cache.V16 = make([]uint16, cacheLen*kvStride)
			} else {
				cache.V = make([]float32, cacheLen*kvStride)
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
		MuPScale:      muPScale,
		RopeLocalOnly: spec.RopeLocalOnly,
	}
	m.SetOps(DefaultOps{})
	initInstanceScratch(m)
	updateInstanceRoPE(m)
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

// bf16ToF32 and fp16ToF32 are defined in dtype.go.

func loadMat(src tensorSource, name string) (*Mat, error) {
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
		m := NewMatFromData(r, c, data)
		return &m, nil
	case mcf.DTypeBF16, mcf.DTypeF16:
		m, err := NewMatFromRaw(r, c, payload.DType, payload.Raw)
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
			m := Mat{R: r, C: c, Stride: c, DType: payload.DType, Raw: payload.Raw}
			cache, err := BuildQuantCache(&m)
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
		m := NewMatFromData(r, c, data)
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

func loadConvKernel(src tensorSource, name string) (*Mat, error) {
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
	return &Mat{R: out, C: k, Stride: k, Data: data}, nil
}

func loadMatCandidates(src tensorSource, candidates []string) (*Mat, string, error) {
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

func loadMoELayer(src tensorSource, cfg *model.HFConfig, names model.ArchNames, layer int) (*MoELayer, error) {
	if cfg == nil {
		return nil, fmt.Errorf("layer %d: nil config for moe", layer)
	}
	if cfg.NumExperts <= 0 {
		return nil, fmt.Errorf("layer %d: num_experts must be > 0 for moe", layer)
	}
	if cfg.MoEIntermediateSize <= 0 {
		return nil, fmt.Errorf("layer %d: moe_intermediate_size must be > 0 for moe", layer)
	}
	if names.MoERouter == nil || names.MoEExpertUp == nil || names.MoEExpertGate == nil || names.MoEExpertDown == nil {
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

	router, err := loadMat(src, names.MoERouter(layer))
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
	if names.MoEExpertBias != nil {
		bias, err := loadVec(src, names.MoEExpertBias(layer))
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

	if names.MoESharedUp == nil || names.MoESharedGate == nil || names.MoESharedDown == nil {
		return nil, fmt.Errorf("layer %d: moe shared expert tensor names are not defined for this arch", layer)
	}
	sharedUp, err := loadMat(src, names.MoESharedUp(layer))
	if err != nil {
		return nil, err
	}
	sharedGate, err := loadMat(src, names.MoESharedGate(layer))
	if err != nil {
		return nil, err
	}
	sharedDown, err := loadMat(src, names.MoESharedDown(layer))
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

	experts := make([]MoEExpert, numExperts)
	for expert := 0; expert < numExperts; expert++ {
		up, err := loadMat(src, names.MoEExpertUp(layer, expert))
		if err != nil {
			return nil, err
		}
		gate, err := loadMat(src, names.MoEExpertGate(layer, expert))
		if err != nil {
			return nil, err
		}
		down, err := loadMat(src, names.MoEExpertDown(layer, expert))
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
		experts[expert] = MoEExpert{Up: up, Gate: gate, Down: down}
	}

	routeScale := float32(cfg.RouteScale)
	if routeScale == 0 {
		routeScale = 1
	}

	return &MoELayer{
		Router:     router,
		ExpertBias: expertBias,
		Shared: MoEShared{
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

func loadMambaLayer(src tensorSource, cfg *model.HFConfig, names model.ArchNames, layer int, modelCfg *ModelConfig) (*MambaLayer, error) {
	if names.MambaInProj == nil || names.MambaOutProj == nil || names.MambaConv == nil || names.MambaALog == nil || names.MambaD == nil || names.MambaDTBias == nil {
		return nil, nil
	}
	if modelCfg == nil {
		return nil, fmt.Errorf("layer %d: nil model config for mamba", layer)
	}
	inProj, err := loadMat(src, names.MambaInProj(layer))
	if err != nil {
		return nil, err
	}
	outProj, err := loadMat(src, names.MambaOutProj(layer))
	if err != nil {
		return nil, err
	}
	conv, err := loadConvKernel(src, names.MambaConv(layer))
	if err != nil {
		return nil, err
	}
	convBias := []float32(nil)
	if names.MambaConvBias != nil {
		convBias, err = loadVec(src, names.MambaConvBias(layer))
		if err != nil && !isTensorMissing(err) {
			return nil, err
		}
	}
	if cfg.MambaConvBias && len(convBias) == 0 {
		return nil, fmt.Errorf("layer %d: mamba conv bias expected but missing", layer)
	}
	if len(convBias) > 0 && len(convBias) != conv.R {
		return nil, fmt.Errorf("layer %d: mamba conv bias length %d incompatible with channels=%d", layer, len(convBias), conv.R)
	}
	aLog, err := loadVec(src, names.MambaALog(layer))
	if err != nil {
		return nil, err
	}
	d, err := loadVec(src, names.MambaD(layer))
	if err != nil {
		return nil, err
	}
	dtBias, err := loadVec(src, names.MambaDTBias(layer))
	if err != nil {
		return nil, err
	}
	var norm []float32
	if names.MambaNorm != nil {
		norm, err = loadVec(src, names.MambaNorm(layer))
		if err != nil && !isTensorMissing(err) {
			return nil, err
		}
	}

	hidden := cfg.HiddenSize
	if inProj.C != hidden {
		return nil, fmt.Errorf("layer %d: mamba in_proj cols %d incompatible with hidden=%d", layer, inProj.C, hidden)
	}
	if outProj.R != hidden {
		return nil, fmt.Errorf("layer %d: mamba out_proj rows %d incompatible with hidden=%d", layer, outProj.R, hidden)
	}
	inner := outProj.C
	if inner <= 0 {
		return nil, fmt.Errorf("layer %d: invalid mamba inner dimension", layer)
	}
	headCount := len(aLog)
	if headCount == 0 || len(d) != headCount || len(dtBias) != headCount {
		return nil, fmt.Errorf("layer %d: mamba A/D/dt_bias head counts mismatch", layer)
	}
	headDim := inner / headCount
	if inner%headCount != 0 || headDim == 0 {
		return nil, fmt.Errorf("layer %d: mamba inner=%d not divisible by heads=%d", layer, inner, headCount)
	}

	groups := cfg.MambaNGroups
	if groups <= 0 {
		groups = 1
	}
	if headCount%groups != 0 {
		return nil, fmt.Errorf("layer %d: mamba heads=%d not divisible by groups=%d", layer, headCount, groups)
	}
	groupSize := headCount / groups
	convChannels := conv.R
	if convChannels <= 0 {
		return nil, fmt.Errorf("layer %d: invalid mamba conv channels", layer)
	}
	dState := cfg.MambaDState
	if dState <= 0 {
		remaining := convChannels - inner
		if remaining <= 0 || remaining%2 != 0 {
			return nil, fmt.Errorf("layer %d: mamba conv channels %d incompatible with inner=%d", layer, convChannels, inner)
		}
		dState = remaining / (2 * groups)
		if dState <= 0 {
			return nil, fmt.Errorf("layer %d: invalid inferred mamba d_state", layer)
		}
	} else {
		wantChannels := inner + 2*groups*dState
		if convChannels != wantChannels {
			return nil, fmt.Errorf("layer %d: mamba conv channels %d incompatible with inner=%d groups=%d d_state=%d", layer, convChannels, inner, groups, dState)
		}
	}

	if cfg.MambaNHeads > 0 && cfg.MambaNHeads != headCount {
		return nil, fmt.Errorf("layer %d: mamba head count %d != config mamba_n_heads %d", layer, headCount, cfg.MambaNHeads)
	}
	if cfg.MambaDHead > 0 && cfg.MambaDHead != headDim {
		return nil, fmt.Errorf("layer %d: mamba head_dim %d != config mamba_d_head %d", layer, headDim, cfg.MambaDHead)
	}

	dInProj := 2*inner + 2*groups*dState + headCount
	if inProj.R != dInProj {
		return nil, fmt.Errorf("layer %d: mamba in_proj rows %d incompatible with expected %d", layer, inProj.R, dInProj)
	}

	if modelCfg.Config.MambaInner == 0 {
		modelCfg.Config.MambaInner = inner
		modelCfg.Config.MambaHeadCount = headCount
		modelCfg.Config.MambaHeadDim = headDim
		modelCfg.Config.MambaDState = dState
		modelCfg.Config.MambaGroups = groups
		modelCfg.Config.MambaConvChannels = convChannels
		modelCfg.Config.MambaConvKernel = conv.C
	} else {
		if modelCfg.Config.MambaInner != inner ||
			modelCfg.Config.MambaHeadCount != headCount ||
			modelCfg.Config.MambaHeadDim != headDim ||
			modelCfg.Config.MambaDState != dState ||
			modelCfg.Config.MambaGroups != groups ||
			modelCfg.Config.MambaConvChannels != convChannels ||
			modelCfg.Config.MambaConvKernel != conv.C {
			return nil, fmt.Errorf("layer %d: inconsistent mamba dimensions across layers", layer)
		}
	}

	kernelLen := conv.C
	if kernelLen < 1 {
		return nil, fmt.Errorf("layer %d: invalid mamba conv kernel length", layer)
	}
	convState := make([]float32, (kernelLen-1)*convChannels)
	ssmState := make([]float32, headCount*headDim*dState)

	return &MambaLayer{
		InProj:       inProj,
		OutProj:      outProj,
		Conv:         conv,
		ConvBias:     convBias,
		ALog:         aLog,
		D:            d,
		DTBias:       dtBias,
		Norm:         norm,
		Inner:        inner,
		HeadCount:    headCount,
		HeadDim:      headDim,
		DState:       dState,
		Groups:       groups,
		GroupSize:    groupSize,
		ConvKernel:   kernelLen,
		ConvChannels: convChannels,
		ConvState:    convState,
		SSMState:     ssmState,
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

func blockCountForConfig(cfg *model.HFConfig, spec *model.ArchSpec) (int, error) {
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

func buildHeadKV(cfg *model.HFConfig, spec *model.ArchSpec, blockCount int) []int {
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

func rmsEpsilonForConfig(cfg *model.HFConfig) float64 {
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

func inferFFNLength(src tensorSource, names model.ArchNames, layer int) int {
	name := names.FfnGate(layer)
	if shape, ok := src.TensorShape(name); ok && len(shape) >= 1 {
		return shape[0]
	}
	return 0
}

func initInstanceScratch(m *Instance) {
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
	m.Scratch = ScratchBuffers{
		X:         make([]float32, embd),
		Tmp:       make([]float32, embd),
		Tmp2:      make([]float32, embd),
		Q:         make([]float32, qDim),
		K:         make([]float32, kv),
		V:         make([]float32, kv),
		AttnOut:   make([]float32, qDim),
		AttnProj:  make([]float32, embd),
		AttnGate:  make([]float32, qDim),
		Scores:    make([]float32, m.MaxContext),
		FfnUp:     make([]float32, ffn),
		FfnGate:   make([]float32, ffn),
		FfnAct:    make([]float32, ffn),
		MoeAccum:  make([]float32, embd),
		RouterRaw: make([]float32, numExperts),
		RouterSel: make([]float32, numExperts),
		RouterIdx: make([]int, topK),
		RouterW:   make([]float32, topK),
		ScProj:    make([]float32, embd*3),
		ScBx:      make([]float32, embd),
		ScConv:    make([]float32, embd),
		Logits:    make([]float32, m.Config.Config.VocabSize),
	}
	if m.Config.Config.MambaInner > 0 && m.Config.Config.MambaHeadCount > 0 {
		inner := m.Config.Config.MambaInner
		convCh := m.Config.Config.MambaConvChannels
		dState := m.Config.Config.MambaDState
		groups := m.Config.Config.MambaGroups
		heads := m.Config.Config.MambaHeadCount
		dInProj := 2*inner + 2*groups*dState + heads
		m.Scratch.MambaIn = make([]float32, embd)
		m.Scratch.MambaProj = make([]float32, dInProj)
		m.Scratch.MambaConv = make([]float32, convCh)
		m.Scratch.MambaZ = make([]float32, inner)
		m.Scratch.MambaX = make([]float32, convCh)
		m.Scratch.MambaB = make([]float32, groups*dState)
		m.Scratch.MambaC = make([]float32, groups*dState)
		m.Scratch.MambaDT = make([]float32, heads)
		m.Scratch.MambaY = make([]float32, inner)
		m.Scratch.MambaOut = make([]float32, embd)
	}
	m.initAttnPool()
}

func updateInstanceRoPE(m *Instance) {
	headDim := m.HeadDim
	if headDim == 0 {
		headDim = m.Config.Config.HeadDim
	}
	if headDim == 0 {
		headDim = m.Config.Config.EmbeddingLength / m.Config.Config.HeadCount
	}
	calc := func(base float64) ([]float64, float32) {
		if base <= 0 {
			base = 10000
		}
		ropeInvFreq := make([]float64, headDim/2)
		for i := 0; i < len(ropeInvFreq); i++ {
			power := float64(2*i) / float64(headDim)
			ropeInvFreq[i] = 1.0 / math.Pow(base, power)
		}
		attnScale := 1.0
		if rs := m.Config.Config.RopeScaling; rs != nil {
			attnScale = model.ApplyRopeScaling(ropeInvFreq, base, m.Config.Config.ContextLength, toModelRopeScaling(rs))
		}
		return ropeInvFreq, float32(attnScale)
	}

	ropeInvFreq, attnScale := calc(m.Config.Config.RopeFreqBase)
	m.RopeInvFreq = ropeInvFreq
	m.RopeAttnScale = attnScale

	localBase := m.Config.Config.RopeFreqBaseLocal
	if localBase == 0 || localBase == m.Config.Config.RopeFreqBase {
		m.RopeInvFreqLocal = nil
		m.RopeAttnScaleLocal = m.RopeAttnScale
		return
	}
	ropeInvFreqLocal, localScale := calc(localBase)
	m.RopeInvFreqLocal = ropeInvFreqLocal
	m.RopeAttnScaleLocal = localScale
}

func toSIMDRopeScaling(rs *model.RopeScaling) *RopeScaling {
	if rs == nil {
		return nil
	}
	return &RopeScaling{
		Type:            rs.Type,
		Factor:          rs.Factor,
		OrigMaxCtx:      rs.OrigMaxCtx,
		LowFactor:       rs.LowFactor,
		HighFactor:      rs.HighFactor,
		AttentionFactor: rs.AttentionFactor,
		BetaFast:        rs.BetaFast,
		BetaSlow:        rs.BetaSlow,
		MScale:          rs.MScale,
		MScaleAllDim:    rs.MScaleAllDim,
		Truncate:        rs.Truncate,
		HasTruncate:     rs.HasTruncate,
	}
}

func toModelRopeScaling(rs *RopeScaling) *model.RopeScaling {
	if rs == nil {
		return nil
	}
	return &model.RopeScaling{
		Type:            rs.Type,
		Factor:          rs.Factor,
		OrigMaxCtx:      rs.OrigMaxCtx,
		LowFactor:       rs.LowFactor,
		HighFactor:      rs.HighFactor,
		AttentionFactor: rs.AttentionFactor,
		BetaFast:        rs.BetaFast,
		BetaSlow:        rs.BetaSlow,
		MScale:          rs.MScale,
		MScaleAllDim:    rs.MScaleAllDim,
		Truncate:        rs.Truncate,
		HasTruncate:     rs.HasTruncate,
	}
}
