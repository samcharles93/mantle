package simd

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"strings"

	"github.com/samcharles93/mantle/internal/backend/core"
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

func LoadModelMCF(mcfFile *mcfstore.File, configJSON []byte, maxContext int, opts LoadModelOptions) (*Instance, error) {
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
	return loadModelFromSource(cfg, spec, mcfSource{mf: mcfFile}, maxContext, opts)
}

// safeMulInt multiplies two ints, returning an error if the result would overflow.
func safeMulInt(a, b int) (int, error) {
	if a < 0 || b < 0 {
		return 0, fmt.Errorf("negative dimension in size computation")
	}
	if a == 0 || b == 0 {
		return 0, nil
	}
	if a > math.MaxInt/b {
		return 0, fmt.Errorf("dimension too large in size computation")
	}
	return a * b, nil
}

func loadModelFromSource(cfg *model.HFConfig, spec *model.ArchSpec, src tensorSource, maxContext int, opts LoadModelOptions) (*Instance, error) {
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
		// Config omits num_attention_heads (e.g. Gemma3ForConditionalGeneration).
		// Infer from first-layer projection shapes via the shape-only index.
		headDimHint := cfg.HeadDim
		if headDimHint <= 0 && names.QNormCandidates != nil {
			for _, n := range names.QNormCandidates(0) {
				if s, ok := src.TensorShape(n); ok && len(s) == 1 && s[0] > 0 {
					headDimHint = s[0]
					break
				}
			}
		}
		if headDimHint > 0 && names.Wq != nil {
			if s, ok := src.TensorShape(names.Wq(0)); ok && len(s) == 2 && s[0]%headDimHint == 0 {
				headCount = s[0] / headDimHint
				if cfg.HeadDim == 0 {
					cfg.HeadDim = headDimHint
				}
				if cfg.NumKeyValueHeads == 0 && names.Wk != nil {
					if ks, ok := src.TensorShape(names.Wk(0)); ok && len(ks) == 2 && ks[0]%headDimHint == 0 {
						cfg.NumKeyValueHeads = ks[0] / headDimHint
					}
				}
			}
		}
		if headCount <= 0 {
			return nil, fmt.Errorf("num_attention_heads must be set")
		}
		cfg.NumAttentionHeads = headCount
	}
	if cfg.HiddenSize <= 0 {
		return nil, fmt.Errorf("hidden_size must be set")
	}
	if cfg.MaxPosition <= 0 {
		return nil, fmt.Errorf("max_position_embeddings must be set")
	}
	if cfg.VocabSize <= 0 {
		// Infer vocab_size from embedding tensor shape (rows = vocab size).
		if s, ok := src.TensorShape(names.Embedding); ok && len(s) >= 1 && s[0] > 0 {
			cfg.VocabSize = s[0]
		}
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
	layerCfgs, layerTypes, err := buildLayerLoadConfigs(cfg, spec, blockCount, headDim)
	if err != nil {
		return nil, err
	}
	headKVArr := make([]int, blockCount)
	maxHeadDim := headDim
	maxHeadKV := 0
	maxKVStride := 0
	maxRotaryDim := 0
	for i, lc := range layerCfgs {
		headKVArr[i] = lc.HeadKV
		if lc.HeadDim > maxHeadDim {
			maxHeadDim = lc.HeadDim
		}
		if lc.HeadKV > maxHeadKV {
			maxHeadKV = lc.HeadKV
		}
		if kvStride := lc.HeadKV * lc.HeadDim; kvStride > maxKVStride {
			maxKVStride = kvStride
		}
		if rotaryDim := len(lc.RopeInvFreq) * 2; rotaryDim > maxRotaryDim {
			maxRotaryDim = rotaryDim
		}
	}

	ffnLength := int(cfg.IntermediateSize)
	if len(cfg.FeedForwardLength) > 0 {
		for _, n := range cfg.FeedForwardLength {
			if n > ffnLength {
				ffnLength = n
			}
		}
	}
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
	rotaryDim := max(maxRotaryDim, model.RotaryDimForConfig(cfg))
	routeScale := cfg.RouteScale
	if routeScale == 0 {
		routeScale = 1
	}
	numDenseLayers := min(max(cfg.NumDenseLayers, 0), blockCount)

	modelCfg := &core.ModelConfig{
		Arch: spec.Name,
		Config: core.Config{
			BlockCount:             blockCount,
			EmbeddingLength:        cfg.HiddenSize,
			FFNLength:              ffnLength,
			HeadCount:              headCount,
			HeadDim:                maxHeadDim,
			RotaryDim:              rotaryDim,
			HeadCountKV:            headKVArr,
			RMSEpsilon:             rmsEps,
			RopeFreqBase:           ropeBase,
			RopeFreqBaseLocal:      ropeLocalBase,
			RopeScaling:            toSIMDRopeScaling(ropeScaling),
			ContextLength:          cfg.MaxPosition,
			VocabSize:              cfg.VocabSize,
			ShortConvLCache:        cfg.ConvLCache,
			DeltaKeyDim:            cfg.LinearNumKeyHeads * cfg.LinearKeyHeadDim,
			DeltaValueDim:          cfg.LinearNumValueHeads * cfg.LinearValueHeadDim,
			DeltaNumKeyHeads:       cfg.LinearNumKeyHeads,
			DeltaNumValueHeads:     cfg.LinearNumValueHeads,
			DeltaHeadKeyDim:        cfg.LinearKeyHeadDim,
			DeltaHeadValueDim:      cfg.LinearValueHeadDim,
			DeltaConvKernel:        cfg.LinearConvKernel,
			EmbeddingMultiplier:    cfg.EmbeddingMultiplier,
			LMHeadMultiplier:       cfg.LMHeadMultiplier,
			AttentionInMultiplier:  cfg.AttentionInMultiplier,
			AttentionOutMultiplier: cfg.AttentionOutMultiplier,
			AttnLogitSoftcap:       float32(cfg.AttnLogitSoftcapping),
			FinalLogitSoftcap:      float32(cfg.FinalLogitSoftcapping),
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
			HiddenAct:              cfg.HiddenActivation,
		},
	}
	if opts.CacheTypeK != "" {
		modelCfg.Config.CacheTypeK = opts.CacheTypeK
	}
	if opts.CacheTypeV != "" {
		modelCfg.Config.CacheTypeV = opts.CacheTypeV
	}

	layers := make([]core.Layer, blockCount)
	for i := range blockCount {
		layer := &layers[i]
		layer.HeadKV = layerCfgs[i].HeadKV
		layer.HeadDim = layerCfgs[i].HeadDim
		layer.KVHeadDim = layerCfgs[i].HeadDim
		layer.AttnScale = layerCfgs[i].AttnScale
		layer.ValueFromKey = layerCfgs[i].ValueFromKey
		layer.ApplyVNorm = layerCfgs[i].ApplyVNorm
		layer.SharedKVSource = layerCfgs[i].SharedKVSource
		layer.StoreFullKV = layerCfgs[i].StoreFullKV
		layer.RopeInvFreq = layerCfgs[i].RopeInvFreq
		layer.RopeAttnScale = layerCfgs[i].RopeAttnScale
		layer.LayerScale = 1
		layer.IsRecurrent = layer.HeadKV == 0
		layer.AttnType = layerCfgs[i].AttnType
		layer.AttnWindow = layerCfgs[i].AttnWindow
		if cfg.NoRopeLayerInterval > 0 && (i+1)%cfg.NoRopeLayerInterval == 0 {
			layer.NoRoPE = true
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
			if names.DeltaQKVProj != nil {
				delta, err := loadDeltaNetLayer(src, cfg, names, i)
				if err != nil {
					return nil, err
				}
				layer.DeltaNet = delta
			} else {
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
				layer.ShortConvState = core.ShortConvState{
					Buf:       make([]float32, (kernelLen-1)*cfg.HiddenSize),
					KernelLen: kernelLen,
				}
			}
		} else {
			qDim := headCount * layer.HeadDim
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
				if len(layer.AttnQNorm) != layer.HeadDim {
					return nil, fmt.Errorf("layer %d: q norm len %d incompatible with head_dim=%d", i, len(layer.AttnQNorm), layer.HeadDim)
				}
				if len(layer.AttnKNorm) != layer.HeadDim {
					return nil, fmt.Errorf("layer %d: k norm len %d incompatible with head_dim=%d", i, len(layer.AttnKNorm), layer.HeadDim)
				}
			}
			layer.Wq, layer.AttnGate, err = loadAttentionQAndGate(src, cfg, names.Wq(i), qDim, cfg.HiddenSize, layer.HeadDim)
			if err != nil {
				return nil, err
			}
			layer.Wk, err = loadMat(src, names.Wk(i))
			if err != nil {
				return nil, err
			}
			if !layer.ValueFromKey || names.Wv == nil {
				layer.Wv, err = loadMat(src, names.Wv(i))
				if err != nil {
					return nil, err
				}
			} else if _, ok := src.TensorShape(names.Wv(i)); ok {
				layer.Wv, err = loadMat(src, names.Wv(i))
				if err != nil && !isTensorMissing(err) {
					return nil, err
				}
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
			if layer.Wq.C != hidden {
				return nil, fmt.Errorf("layer %d: q_proj col dim %d incompatible with hidden=%d", i, layer.Wq.C, hidden)
			}
			if layer.Wq.R == 2*qDim && layer.AttnGate == nil {
				// Fused Q/gate tensor that could not be split at load time
				// (e.g. quantized dtype). The runtime will project through all
				// 2*qDim rows and split the output into Q and gate.
				layer.FusedQGate = true
			} else if layer.Wq.R != qDim {
				return nil, fmt.Errorf("layer %d: q_proj shape [%d %d] incompatible with hidden=%d head_dim=%d heads=%d", i, layer.Wq.R, layer.Wq.C, hidden, layer.HeadDim, headCount)
			}
			wantKV := layer.HeadKV * layer.HeadDim
			if layer.Wk.C != hidden || layer.Wk.R != wantKV {
				return nil, fmt.Errorf("layer %d: k_proj shape [%d %d] incompatible with hidden=%d head_dim=%d kv_heads=%d", i, layer.Wk.R, layer.Wk.C, hidden, layer.HeadDim, layer.HeadKV)
			}
			if layer.ValueFromKey {
				if layer.Wv != nil && (layer.Wv.C != hidden || layer.Wv.R != wantKV) {
					return nil, fmt.Errorf("layer %d: v_proj shape [%d %d] incompatible with hidden=%d head_dim=%d kv_heads=%d", i, layer.Wv.R, layer.Wv.C, hidden, layer.HeadDim, layer.HeadKV)
				}
			} else if layer.Wv == nil || layer.Wv.C != hidden || layer.Wv.R != wantKV {
				return nil, fmt.Errorf("layer %d: v_proj shape incompatible with hidden=%d head_dim=%d kv_heads=%d", i, hidden, layer.HeadDim, layer.HeadKV)
			}
			if layer.Wo.R != hidden || layer.Wo.C != qDim {
				return nil, fmt.Errorf("layer %d: o_proj shape [%d %d] incompatible with hidden=%d head_dim=%d heads=%d", i, layer.Wo.R, layer.Wo.C, hidden, layer.HeadDim, headCount)
			}
			if names.AttnGate != nil && layer.AttnGate == nil {
				layer.AttnGate, err = loadMat(src, names.AttnGate(i))
				if err != nil {
					return nil, err
				}
			}
			if layer.AttnGate != nil && (layer.AttnGate.R != qDim || layer.AttnGate.C != hidden) {
				return nil, fmt.Errorf(
					"layer %d: attn gate shape [%d %d] incompatible with hidden=%d qdim=%d",
					i, layer.AttnGate.R, layer.AttnGate.C, hidden, qDim,
				)
			}
			if isMoELayer {
				moe, err := loadMoELayer(src, cfg, names, i)
				if err != nil {
					return nil, err
				}
				layer.MoE = moe
			}
			if spec.Name == "gemma4" {
				if names.LayerScalar != nil {
					layerScalar, err := loadVec(src, names.LayerScalar(i))
					if err != nil {
						return nil, err
					}
					if len(layerScalar) != 1 {
						return nil, fmt.Errorf("layer %d: gemma4 layer scalar len %d incompatible with 1", i, len(layerScalar))
					}
					layer.LayerScale = layerScalar[0]
				}
				if cfg.HiddenSizePerLayerInput > 0 {
					ple := &core.Gemma4PLELayer{}
					ple.InputGate, err = loadMat(src, names.PerLayerInputGate(i))
					if err != nil {
						return nil, err
					}
					ple.Projection, err = loadMat(src, names.PerLayerInputProj(i))
					if err != nil {
						return nil, err
					}
					ple.PostNorm, err = loadVec(src, names.PostPerLayerInputNorm(i))
					if err != nil {
						return nil, err
					}
					layer.Gemma4PLE = ple
				}
				if cfg.EnableMoEBlock {
					gemma4MoE, err := loadGemma4MoELayer(src, cfg, names, i)
					if err != nil {
						return nil, err
					}
					layer.Gemma4MoE = gemma4MoE
				}
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
			kvStride := layer.HeadKV * layer.HeadDim

			// Initialize cache based on type
			cacheLen := maxContext
			if layer.AttnType == "sliding_attention" && layer.AttnWindow > 0 && !layer.StoreFullKV {
				if layer.AttnWindow < cacheLen {
					cacheLen = layer.AttnWindow
				}
			}
			cache := core.AttnCache{KvStride: kvStride, CacheLen: cacheLen}

			// Key cache type — backing slices are lazily allocated via EnsurePos.
			kt := modelCfg.Config.CacheTypeK
			if kt == "" {
				kt = core.CacheTypeF32
			}
			switch kt {
			case core.CacheTypeF16:
				cache.K16 = make([]uint16, 0)
			case core.CacheTypeQ8_0:
				cache.KQ8 = make([]int8, 0)
				cache.KQ8S = make([]float32, 0)
			default:
				cache.K = make([]float32, 0)
			}

			// Value cache type
			vt := modelCfg.Config.CacheTypeV
			if vt == "" {
				vt = core.CacheTypeF32
			}
			switch vt {
			case core.CacheTypeF16:
				cache.V16 = make([]uint16, 0)
			case core.CacheTypeQ8_0:
				cache.VQ8 = make([]int8, 0)
				cache.VQ8S = make([]float32, 0)
			default:
				cache.V = make([]float32, 0)
			}

			layer.AttnCache = cache
		}
	}

	gemma4PerLayer, err := loadGemma4PerLayerInputModel(src, cfg, names, blockCount)
	if err != nil {
		return nil, err
	}

	muPScale := float32(1)
	if modelCfg.Config.MuPEnabled && modelCfg.Config.EmbeddingLength > 0 {
		muPScale = float32(math.Sqrt(float64(modelCfg.Config.EmbeddingLength)))
	}

	coreInst := &core.Instance{
		Config:         modelCfg,
		Embeddings:     emb,
		OutputNorm:     outNorm,
		Output:         output,
		Layers:         layers,
		Gemma4PerLayer: gemma4PerLayer,
		MaxContext:     maxContext,
		Pos:            0,
		RMSEpsilon:     float32(rmsEps),
		HeadDim:        maxHeadDim,
		HeadCount:      headCount,
		MaxHeadKV:      maxHeadKV,
		MaxQDim:        headCount * maxHeadDim,
		MaxKVStride:    maxKVStride,
		MuPScale:       muPScale,
		RopeLocalOnly:  spec.RopeLocalOnly,
		TilingConfig:   core.TilingConfig(opts.TilingConfig),
	}
	m := (*Instance)(coreInst)
	m.SetHostCapabilities(opts.HostCaps)
	m.BindDefaultOps()
	initInstanceScratch(m)
	updateInstanceRoPE(m)
	adjustGemmaNorms(m, cfg)
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
		for i := range n {
			out[i] = math.Float32frombits(binary.LittleEndian.Uint32(p.Raw[i*4:]))
		}
		return out, nil
	case mcf.DTypeBF16:
		if len(p.Raw) != n*2 {
			return nil, fmt.Errorf("invalid bf16 data size")
		}
		out := make([]float32, n)
		for i := range n {
			u := binary.LittleEndian.Uint16(p.Raw[i*2:])
			out[i] = core.BF16ToFloat32(u)
		}
		return out, nil
	case mcf.DTypeF16:
		if len(p.Raw) != n*2 {
			return nil, fmt.Errorf("invalid f16 data size")
		}
		out := make([]float32, n)
		for i := range n {
			u := binary.LittleEndian.Uint16(p.Raw[i*2:])
			out[i] = core.FP16ToFloat32(u)
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

func loadInterleavedHeadBlocksFromPayload(name string, payload tensorPayload, headDim, headCount, hidden int) (*core.Mat, *core.Mat, error) {
	shape := payload.Shape
	if len(shape) != 2 {
		return nil, nil, fmt.Errorf("%s: expected 2D tensor", name)
	}
	totalRows := 2 * headDim * headCount
	if shape[0] != totalRows || shape[1] != hidden {
		return nil, nil, fmt.Errorf("%s: unexpected fused q/gate shape %v", name, shape)
	}

	switch payload.DType {
	case mcf.DTypeF32:
		data, err := decodeTensorF32(payload)
		if err != nil {
			return nil, nil, fmt.Errorf("%s: %w", name, err)
		}
		rowsPerMat := headDim * headCount
		qData := make([]float32, rowsPerMat*hidden)
		gData := make([]float32, rowsPerMat*hidden)
		dstQ := 0
		dstG := 0
		for h := range headCount {
			baseRow := 2 * h * headDim
			qStart := baseRow * hidden
			qEnd := (baseRow + headDim) * hidden
			copy(qData[dstQ:dstQ+headDim*hidden], data[qStart:qEnd])
			dstQ += headDim * hidden

			gStart := qEnd
			gEnd := (baseRow + 2*headDim) * hidden
			copy(gData[dstG:dstG+headDim*hidden], data[gStart:gEnd])
			dstG += headDim * hidden
		}
		qMat := core.NewMatFromData(rowsPerMat, hidden, qData)
		gMat := core.NewMatFromData(rowsPerMat, hidden, gData)
		return &qMat, &gMat, nil
	case mcf.DTypeBF16, mcf.DTypeF16:
		rowBytes, err := safeMulInt(hidden, 2)
		if err != nil {
			return nil, nil, fmt.Errorf("%s: %w", name, err)
		}
		blockBytes := headDim * rowBytes
		qRaw := make([]byte, headCount*blockBytes)
		gRaw := make([]byte, headCount*blockBytes)
		dstQ := 0
		dstG := 0
		for h := range headCount {
			baseRow := 2 * h * headDim
			qStart := baseRow * rowBytes
			qEnd := qStart + blockBytes
			copy(qRaw[dstQ:dstQ+blockBytes], payload.Raw[qStart:qEnd])
			dstQ += blockBytes

			gStart := qEnd
			gEnd := gStart + blockBytes
			copy(gRaw[dstG:dstG+blockBytes], payload.Raw[gStart:gEnd])
			dstG += blockBytes
		}
		qMat, err := core.NewMatFromRaw(headDim*headCount, hidden, payload.DType, qRaw)
		if err != nil {
			return nil, nil, fmt.Errorf("%s: %w", name, err)
		}
		gMat, err := core.NewMatFromRaw(headDim*headCount, hidden, payload.DType, gRaw)
		if err != nil {
			return nil, nil, fmt.Errorf("%s: %w", name, err)
		}
		return &qMat, &gMat, nil
	default:
		return nil, nil, fmt.Errorf("%s: fused q/gate split unsupported for dtype %d", name, payload.DType)
	}
}

func loadAttentionQAndGate(src tensorSource, cfg *model.HFConfig, qName string, qDim, hidden, headDim int) (*core.Mat, *core.Mat, error) {
	if cfg != nil && cfg.AttnOutputGate {
		if shape, ok := src.TensorShape(qName); ok && len(shape) == 2 && shape[0] == 2*qDim && shape[1] == hidden {
			// Fused Q+Gate projection with interleaved per-head layout.
			// Read the raw payload to check the dtype before committing to a split path.
			payload, err := src.ReadTensor(qName)
			if err != nil {
				return nil, nil, fmt.Errorf("%s: %w", qName, err)
			}
			// Quantized dtypes use block-structured payloads that interleave
			// scales and data across all rows. Row-level splitting is not
			// possible without re-encoding, so fall through to loading the
			// full fused mat and splitting the output at inference time.
			if !mcf.DTypeRequiresAligned64(payload.DType) {
				headCount := qDim / headDim
				return loadInterleavedHeadBlocksFromPayload(qName, payload, headDim, headCount, hidden)
			}
			// Non-quantized fused Q+Gate: use interleaved head block splitting.
			headCount := qDim / headDim
			return loadInterleavedHeadBlocksFromPayload(qName, payload, headDim, headCount, hidden)
		}
	}
	wq, err := loadMat(src, qName)
	if err != nil {
		return nil, nil, err
	}
	return wq, nil, nil
}

// bf16ToF32 and fp16ToF32 are defined in dtype.go.

func loadMat(src tensorSource, name string) (*core.Mat, error) {
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
		m := core.NewMatFromData(r, c, data)
		return &m, nil
	case mcf.DTypeBF16, mcf.DTypeF16:
		m, err := core.NewMatFromRaw(r, c, payload.DType, payload.Raw)
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
			m := core.Mat{R: r, C: c, Stride: c, DType: payload.DType, Raw: payload.Raw}
			if quantCacheBuildEnabledForLoad() {
				cache, err := BuildQuantCache(&m)
				if err != nil {
					return nil, fmt.Errorf("%s: quant cache: %w", name, err)
				}
				m.Quant = cache
			}
			return &m, nil
		}
		data, err := decodeTensorF32(payload)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", name, err)
		}
		if r*c != len(data) {
			return nil, fmt.Errorf("%s: size mismatch", name)
		}
		m := core.NewMatFromData(r, c, data)
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

func loadConvKernel(src tensorSource, name string) (*core.Mat, error) {
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
	return &core.Mat{R: out, C: k, Stride: k, Data: data}, nil
}

func loadMatCandidates(src tensorSource, candidates []string) (*core.Mat, string, error) {
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

func loadMoELayer(src tensorSource, cfg *model.HFConfig, names model.ArchNames, layer int) (*core.MoELayer, error) {
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

	experts := make([]core.MoEExpert, numExperts)
	for expert := range numExperts {
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
		experts[expert] = core.MoEExpert{Up: up, Gate: gate, Down: down}
	}

	routeScale := float32(cfg.RouteScale)
	if routeScale == 0 {
		routeScale = 1
	}

	return &core.MoELayer{
		Router:     router,
		ExpertBias: expertBias,
		Shared: core.MoEShared{
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

func loadDeltaNetLayer(src tensorSource, cfg *model.HFConfig, names model.ArchNames, layer int) (*core.DeltaNetLayer, error) {
	if names.DeltaQKVProj == nil || names.DeltaAProj == nil || names.DeltaBProj == nil || names.DeltaZProj == nil ||
		names.DeltaOutProj == nil || names.DeltaConv == nil || names.DeltaNorm == nil || names.DeltaALog == nil || names.DeltaDTBias == nil {
		return nil, fmt.Errorf("layer %d: gated deltanet tensor names are incomplete", layer)
	}
	qkv, err := loadMat(src, names.DeltaQKVProj(layer))
	if err != nil {
		return nil, err
	}
	aProj, err := loadMat(src, names.DeltaAProj(layer))
	if err != nil {
		return nil, err
	}
	bProj, err := loadMat(src, names.DeltaBProj(layer))
	if err != nil {
		return nil, err
	}
	zProj, err := loadMat(src, names.DeltaZProj(layer))
	if err != nil {
		return nil, err
	}
	outProj, err := loadMat(src, names.DeltaOutProj(layer))
	if err != nil {
		return nil, err
	}
	conv, err := loadConvKernel(src, names.DeltaConv(layer))
	if err != nil {
		return nil, err
	}
	norm, err := loadVec(src, names.DeltaNorm(layer))
	if err != nil {
		return nil, err
	}
	aLog, err := loadVec(src, names.DeltaALog(layer))
	if err != nil {
		return nil, err
	}
	dtBias, err := loadVec(src, names.DeltaDTBias(layer))
	if err != nil {
		return nil, err
	}

	keyHeads := cfg.LinearNumKeyHeads
	valueHeads := cfg.LinearNumValueHeads
	keyDim := cfg.LinearKeyHeadDim
	valueDim := cfg.LinearValueHeadDim
	hidden := cfg.HiddenSize
	if keyHeads <= 0 || valueHeads <= 0 || keyDim <= 0 || valueDim <= 0 {
		return nil, fmt.Errorf("layer %d: linear attention config is incomplete", layer)
	}
	if valueHeads%keyHeads != 0 {
		return nil, fmt.Errorf("layer %d: linear attention value_heads=%d not divisible by key_heads=%d", layer, valueHeads, keyHeads)
	}
	totalKey := keyHeads * keyDim
	totalValue := valueHeads * valueDim
	convDim := 2*totalKey + totalValue
	if qkv.C != hidden || qkv.R != convDim {
		return nil, fmt.Errorf("layer %d: delta qkv shape [%d %d] incompatible with hidden=%d conv_dim=%d", layer, qkv.R, qkv.C, hidden, convDim)
	}
	if zProj.C != hidden || zProj.R != totalValue {
		return nil, fmt.Errorf("layer %d: delta z_proj shape [%d %d] incompatible with hidden=%d value_dim=%d", layer, zProj.R, zProj.C, hidden, totalValue)
	}
	if aProj.C != hidden || aProj.R != valueHeads {
		return nil, fmt.Errorf("layer %d: delta a_proj shape [%d %d] incompatible with hidden=%d value_heads=%d", layer, aProj.R, aProj.C, hidden, valueHeads)
	}
	if bProj.C != hidden || bProj.R != valueHeads {
		return nil, fmt.Errorf("layer %d: delta b_proj shape [%d %d] incompatible with hidden=%d value_heads=%d", layer, bProj.R, bProj.C, hidden, valueHeads)
	}
	if outProj.R != hidden || outProj.C != totalValue {
		return nil, fmt.Errorf("layer %d: delta out_proj shape [%d %d] incompatible with hidden=%d value_dim=%d", layer, outProj.R, outProj.C, hidden, totalValue)
	}
	if conv.R != convDim {
		return nil, fmt.Errorf("layer %d: delta conv channels %d incompatible with conv_dim=%d", layer, conv.R, convDim)
	}
	if len(norm) != valueDim {
		return nil, fmt.Errorf("layer %d: delta norm length %d incompatible with head_value_dim=%d", layer, len(norm), valueDim)
	}
	if len(aLog) != valueHeads || len(dtBias) != valueHeads {
		return nil, fmt.Errorf("layer %d: delta A_log/dt_bias lengths incompatible with value_heads=%d", layer, valueHeads)
	}

	return &core.DeltaNetLayer{
		QKVProj:        qkv,
		AProj:          aProj,
		BProj:          bProj,
		ZProj:          zProj,
		OutProj:        outProj,
		Conv:           conv,
		Norm:           norm,
		ALog:           aLog,
		DTBias:         dtBias,
		NumKeyHeads:    keyHeads,
		NumValueHeads:  valueHeads,
		HeadKeyDim:     keyDim,
		HeadValueDim:   valueDim,
		KeyDim:         totalKey,
		ValueDim:       totalValue,
		ConvState:      make([]float32, max(conv.C-1, 0)*convDim),
		RecurrentState: make([]float32, valueHeads*keyDim*valueDim),
	}, nil
}

func loadMambaLayer(src tensorSource, cfg *model.HFConfig, names model.ArchNames, layer int, modelCfg *core.ModelConfig) (*core.MambaLayer, error) {
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
	convLen, err := safeMulInt(kernelLen-1, convChannels)
	if err != nil {
		return nil, fmt.Errorf("layer %d: mamba conv state too large: %w", layer, err)
	}
	convState := make([]float32, convLen)
	hc, err := safeMulInt(headCount, headDim)
	if err != nil {
		return nil, fmt.Errorf("layer %d: mamba ssm state too large (head dims): %w", layer, err)
	}
	ssmLen, err := safeMulInt(hc, dState)
	if err != nil {
		return nil, fmt.Errorf("layer %d: mamba ssm state too large (d_state): %w", layer, err)
	}
	ssmState := make([]float32, ssmLen)

	return &core.MambaLayer{
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
	if len(cfg.FeedForwardLength) > 0 {
		return len(cfg.FeedForwardLength), nil
	}
	if len(cfg.LayerTypes) > 0 {
		return len(cfg.LayerTypes), nil
	}
	return 0, fmt.Errorf("could not determine layer count from config")
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
	for i := range m.Layers {
		if gm := m.Layers[i].Gemma4MoE; gm != nil {
			if len(gm.Experts) > numExperts {
				numExperts = len(gm.Experts)
			}
			if gm.TopK > topK {
				topK = gm.TopK
			}
			if need := 2 * gm.Intermediate; need > ffn {
				ffn = need
			}
		}
		if gp := m.Layers[i].Gemma4PLE; gp != nil && gp.InputGate != nil && gp.InputGate.R > ffn {
			ffn = gp.InputGate.R
		}
	}
	deltaQKV := 2*m.Config.Config.DeltaKeyDim + m.Config.Config.DeltaValueDim
	deltaValue := m.Config.Config.DeltaValueDim
	deltaHeads := m.Config.Config.DeltaNumValueHeads
	kv := m.MaxKVStride
	if kv < 1 {
		kv = m.HeadDim
	}
	qDim := m.MaxQDim
	if qDim < 1 {
		qDim = embd
	}
	perLayerTotal := 0
	if m.Gemma4PerLayer != nil {
		perLayerTotal = m.Gemma4PerLayer.LayerCount * m.Gemma4PerLayer.HiddenSize
	}
	m.Scratch = ScratchBuffers{
		X:             make([]float32, embd),
		Tmp:           make([]float32, embd),
		Tmp2:          make([]float32, embd),
		Q:             make([]float32, qDim),
		K:             make([]float32, kv),
		V:             make([]float32, kv),
		AttnOut:       make([]float32, qDim),
		AttnProj:      make([]float32, embd),
		AttnGate:      make([]float32, 2*qDim),
		Scores:        make([]float32, m.MaxContext),
		FfnUp:         make([]float32, ffn),
		FfnGate:       make([]float32, ffn),
		FfnAct:        make([]float32, ffn),
		MoeAccum:      make([]float32, embd),
		RouterRaw:     make([]float32, numExperts),
		RouterSel:     make([]float32, numExperts),
		RouterIdx:     make([]int, topK),
		RouterW:       make([]float32, topK),
		RouterTop:     make([]float32, topK),
		ScProj:        make([]float32, embd*3),
		ScBx:          make([]float32, embd),
		ScConv:        make([]float32, embd),
		DeltaQKV:      make([]float32, max(deltaQKV, 1)),
		DeltaA:        make([]float32, max(deltaHeads, 1)),
		DeltaB:        make([]float32, max(deltaHeads, 1)),
		DeltaZ:        make([]float32, max(deltaValue, 1)),
		DeltaQ:        make([]float32, max(m.Config.Config.DeltaKeyDim, 1)),
		DeltaK:        make([]float32, max(m.Config.Config.DeltaKeyDim, 1)),
		DeltaV:        make([]float32, max(deltaValue, 1)),
		DeltaOut:      make([]float32, max(deltaValue, 1)),
		Logits:        make([]float32, m.Config.Config.VocabSize),
		PerLayerTok:   make([]float32, perLayerTotal),
		PerLayerProj:  make([]float32, perLayerTotal),
		PerLayerInput: make([]float32, perLayerTotal),
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
	_ = m.GetAttnPool()
}

func updateInstanceRoPE(m *Instance) {
	headDim := m.HeadDim
	if headDim == 0 {
		headDim = m.Config.Config.HeadDim
	}
	if headDim == 0 {
		headDim = m.Config.Config.EmbeddingLength / m.Config.Config.HeadCount
	}
	rotaryDim := m.Config.Config.RotaryDim
	if rotaryDim <= 0 || rotaryDim > headDim {
		rotaryDim = headDim
	}
	if rotaryDim%2 != 0 {
		rotaryDim--
	}
	if rotaryDim <= 0 {
		rotaryDim = headDim
	}
	calc := func(base float64) ([]float64, float32) {
		if base <= 0 {
			base = 10000
		}
		ropeInvFreq := make([]float64, rotaryDim/2)
		for i := range ropeInvFreq {
			power := float64(2*i) / float64(rotaryDim)
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

func adjustGemmaNorms(m *Instance, cfg *model.HFConfig) {
	if m == nil || cfg == nil {
		return
	}
	if !needsOneCenteredRMSNorm(cfg) {
		return
	}

	// Helper to add 1.0 to a slice
	addOne := func(s []float32) {
		for i := range s {
			s[i] += 1.0
		}
	}

	addOne(m.OutputNorm)

	for i := range m.Layers {
		l := &m.Layers[i]
		addOne(l.AttnNorm)
		addOne(l.PostAttnNorm)
		addOne(l.FfnNorm)
		addOne(l.PostFfnNorm)
		addOne(l.AttnQNorm)
		addOne(l.AttnKNorm)
	}
}

func needsOneCenteredRMSNorm(cfg *model.HFConfig) bool {
	if cfg == nil {
		return false
	}

	modelTags := append([]string{cfg.ModelType}, cfg.Architectures...)
	oneCentered := false
	for _, raw := range modelTags {
		tag := strings.ToLower(strings.TrimSpace(raw))
		switch {
		case strings.Contains(tag, "gemma4"), strings.Contains(tag, "gemma3n"), strings.Contains(tag, "qwen3_5"):
			return false
		case tag == "gemma", strings.Contains(tag, "gemma2"), strings.Contains(tag, "gemma3"):
			oneCentered = true
		}
	}
	return oneCentered
}
