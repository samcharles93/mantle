package simd

import (
	"fmt"
	"math"
	"strings"

	"github.com/samcharles93/mantle/internal/backend/core"
	"github.com/samcharles93/mantle/internal/model"
	"github.com/samcharles93/mantle/pkg/mcf"
)

func usesGemmaAttentionFeatures(spec *model.ArchSpec) bool {
	if spec == nil {
		return false
	}
	switch spec.Name {
	case "gemma4", "gemma3n_text":
		return true
	default:
		return false
	}
}

func applyGemmaAttentionConfig(lc *layerLoadConfig, cfg *model.HFConfig, spec *model.ArchSpec, layer int) error {
	if !usesGemmaAttentionFeatures(spec) {
		return nil
	}

	lc.ApplyVNorm = true
	lc.AttnScale = 1
	if spec.Name == "gemma4" && lc.AttnType == "full_attention" && cfg.GlobalHeadDim > 0 {
		lc.HeadDim = cfg.GlobalHeadDim
	}
	if spec.Name == "gemma4" && lc.AttnType == "full_attention" && cfg.AttentionKEqualV {
		lc.ValueFromKey = true
		if cfg.NumGlobalKeyValueHeads > 0 {
			lc.HeadKV = cfg.NumGlobalKeyValueHeads
		}
	}
	invFreq, attnScale, err := model.LayerRoPEForConfig(cfg, lc.AttnType, lc.HeadDim)
	if err != nil {
		return fmt.Errorf("layer %d rope: %w", layer, err)
	}
	lc.RopeInvFreq = invFreq
	lc.RopeAttnScale = attnScale
	return nil
}

func applyGemmaSharedKVConfig(out []layerLoadConfig, layerTypes []string, cfg *model.HFConfig, spec *model.ArchSpec) error {
	if !usesGemmaAttentionFeatures(spec) || cfg.NumKVSharedLayers <= 0 || len(layerTypes) != len(out) {
		return nil
	}

	firstShared := len(out) - cfg.NumKVSharedLayers
	if firstShared <= 0 {
		return nil
	}

	prevLayers := layerTypes[:firstShared]
	for i := range prevLayers {
		if i == lastIndexOfLayerType(prevLayers, prevLayers[i]) {
			out[i].StoreFullKV = true
		}
	}
	for i := firstShared; i < len(out); i++ {
		layerType := layerTypes[i]
		for j := firstShared - 1; j >= 0; j-- {
			if prevLayers[j] == layerType {
				out[i].SharedKVSource = j
				break
			}
		}
		if out[i].SharedKVSource < 0 {
			return fmt.Errorf("layer %d: no shared kv source for type %q", i, layerType)
		}
	}
	return nil
}

func loadGemma4PerLayerInputModel(src tensorSource, cfg *model.HFConfig, names model.ArchNames, blockCount int) (*core.Gemma4PerLayerInputModel, error) {
	if cfg.HiddenSizePerLayerInput <= 0 {
		return nil, nil
	}
	if names.PerLayerEmbedding == "" || names.PerLayerModelProjection == "" || names.PerLayerProjectionNorm == "" {
		return nil, fmt.Errorf("gemma4 per-layer input tensor names are not defined")
	}

	embeddings, err := loadMat(src, names.PerLayerEmbedding)
	if err != nil {
		return nil, err
	}
	projection, err := loadMat(src, names.PerLayerModelProjection)
	if err != nil {
		return nil, err
	}
	projectionNorm, err := loadVec(src, names.PerLayerProjectionNorm)
	if err != nil {
		return nil, err
	}
	totalDim := blockCount * cfg.HiddenSizePerLayerInput
	if embeddings.C != totalDim {
		return nil, fmt.Errorf("gemma4 per-layer embeddings width %d incompatible with layers=%d hidden=%d", embeddings.C, blockCount, cfg.HiddenSizePerLayerInput)
	}
	if projection.R != totalDim || projection.C != cfg.HiddenSize {
		return nil, fmt.Errorf("gemma4 per-layer projection shape [%d %d] incompatible with total=%d hidden=%d", projection.R, projection.C, totalDim, cfg.HiddenSize)
	}
	if len(projectionNorm) != cfg.HiddenSizePerLayerInput {
		return nil, fmt.Errorf("gemma4 per-layer projection norm len %d incompatible with hidden=%d", len(projectionNorm), cfg.HiddenSizePerLayerInput)
	}

	return &core.Gemma4PerLayerInputModel{
		Embeddings:      embeddings,
		Projection:      projection,
		ProjectionNorm:  projectionNorm,
		HiddenSize:      cfg.HiddenSizePerLayerInput,
		LayerCount:      blockCount,
		EmbeddingScale:  model.RoundFloat32ToBF16(float32(math.Sqrt(float64(cfg.HiddenSizePerLayerInput)))),
		ProjectionScale: model.RoundFloat32ToBF16(float32(1.0 / math.Sqrt(float64(cfg.HiddenSize)))),
		InputScale:      model.RoundFloat32ToBF16(float32(math.Pow(2, -0.5))),
	}, nil
}

func loadGemma4MoELayer(src tensorSource, cfg *model.HFConfig, names model.ArchNames, layer int) (*core.Gemma4MoELayer, error) {
	if !cfg.EnableMoEBlock {
		return nil, nil
	}
	if cfg.NumExperts <= 0 || cfg.TopKExperts <= 0 || cfg.MoEIntermediateSize <= 0 {
		return nil, fmt.Errorf("layer %d: gemma4 moe config is incomplete", layer)
	}

	routerProj, err := loadMat(src, names.Gemma4RouterProj(layer))
	if err != nil {
		return nil, err
	}
	routerScale, err := loadVec(src, names.Gemma4RouterScale(layer))
	if err != nil {
		return nil, err
	}
	perExpertScale, err := loadVec(src, names.Gemma4RouterPerExpertScale(layer))
	if err != nil {
		return nil, err
	}
	preNorm2, err := loadVec(src, names.Gemma4PreFfnNorm2(layer))
	if err != nil {
		return nil, err
	}
	postNorm1, err := loadVec(src, names.Gemma4PostFfnNorm1(layer))
	if err != nil {
		return nil, err
	}
	postNorm2, err := loadVec(src, names.Gemma4PostFfnNorm2(layer))
	if err != nil {
		return nil, err
	}
	if len(routerScale) != cfg.HiddenSize {
		return nil, fmt.Errorf("layer %d: gemma4 router scale len %d incompatible with hidden=%d", layer, len(routerScale), cfg.HiddenSize)
	}
	if len(perExpertScale) != cfg.NumExperts {
		return nil, fmt.Errorf("layer %d: gemma4 router per-expert scale len %d incompatible with experts=%d", layer, len(perExpertScale), cfg.NumExperts)
	}

	gateUpExperts, err := loadMatStack3D(src, names.Gemma4ExpertsGateUp(layer), cfg.NumExperts, 2*cfg.MoEIntermediateSize, cfg.HiddenSize)
	if err != nil {
		return nil, err
	}
	downExperts, err := loadMatStack3D(src, names.Gemma4ExpertsDown(layer), cfg.NumExperts, cfg.HiddenSize, cfg.MoEIntermediateSize)
	if err != nil {
		return nil, err
	}
	experts := make([]core.Gemma4MoEExpert, cfg.NumExperts)
	for i := range cfg.NumExperts {
		experts[i] = core.Gemma4MoEExpert{
			GateUp: gateUpExperts[i],
			Down:   downExperts[i],
		}
	}

	return &core.Gemma4MoELayer{
		RouterProj:           routerProj,
		RouterScale:          routerScale,
		RouterPerExpertScale: perExpertScale,
		PreNorm2:             preNorm2,
		PostNorm1:            postNorm1,
		PostNorm2:            postNorm2,
		Experts:              experts,
		TopK:                 cfg.TopKExperts,
		Intermediate:         cfg.MoEIntermediateSize,
		ScalarRootSize:       model.RoundFloat32ToBF16(float32(1.0 / math.Sqrt(float64(cfg.HiddenSize)))),
	}, nil
}

func loadMatStack3D(src tensorSource, name string, outer, rows, cols int) ([]*core.Mat, error) {
	payload, err := src.ReadTensor(name)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", name, err)
	}
	if len(payload.Shape) != 3 {
		return nil, fmt.Errorf("%s: expected 3D tensor, got %v", name, payload.Shape)
	}
	if payload.Shape[0] != outer || payload.Shape[1] != rows || payload.Shape[2] != cols {
		return nil, fmt.Errorf("%s: shape %v incompatible with [%d %d %d]", name, payload.Shape, outer, rows, cols)
	}

	mats := make([]*core.Mat, outer)
	switch payload.DType {
	case mcf.DTypeF32:
		data, err := decodeTensorF32(payload)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", name, err)
		}
		chunk := rows * cols
		for i := range outer {
			start := i * chunk
			end := start + chunk
			mat := core.NewMatFromData(rows, cols, data[start:end])
			mats[i] = &mat
		}
		return mats, nil
	case mcf.DTypeBF16, mcf.DTypeF16:
		elemSize := 2
		chunkBytes := rows * cols * elemSize
		for i := range outer {
			start := i * chunkBytes
			end := start + chunkBytes
			mat, err := core.NewMatFromRaw(rows, cols, payload.DType, payload.Raw[start:end])
			if err != nil {
				return nil, fmt.Errorf("%s[%d]: %w", name, i, err)
			}
			mats[i] = &mat
		}
		return mats, nil
	default:
		return nil, fmt.Errorf("%s: unsupported expert tensor dtype %d", name, payload.DType)
	}
}

func prepareGemma4PerLayerInputs(m *Instance, tok int, x []float32) []float32 {
	_, projected := computeGemma4PerLayerInputs(m, tok, x)
	return projected
}

func computeGemma4PerLayerInputs(m *Instance, tok int, x []float32) (raw, projected []float32) {
	if m == nil || m.Gemma4PerLayer == nil {
		return nil, nil
	}
	perLayer := m.Gemma4PerLayer
	totalDim := perLayer.LayerCount * perLayer.HiddenSize
	proj := m.Scratch.PerLayerProj[:totalDim]
	tokBuf := m.Scratch.PerLayerTok[:totalDim]
	inputs := m.Scratch.PerLayerInput[:totalDim]

	m.Ops().MatVec(proj, perLayer.Projection, x)
	for i := range proj {
		proj[i] *= perLayer.ProjectionScale
	}
	perLayer.Embeddings.RowTo(tokBuf, tok)
	if perLayer.EmbeddingScale != 1 {
		for i := range tokBuf {
			tokBuf[i] *= perLayer.EmbeddingScale
		}
	}
	for layerIdx := range perLayer.LayerCount {
		start := layerIdx * perLayer.HiddenSize
		end := start + perLayer.HiddenSize
		rmsNormWeighted(inputs[start:end], proj[start:end], perLayer.ProjectionNorm, m.RMSEpsilon)
		Add(inputs[start:end], tokBuf[start:end])
		if perLayer.InputScale != 1 {
			s := perLayer.InputScale
			for i := start; i < end; i++ {
				inputs[i] *= s
			}
		}
	}
	return tokBuf, inputs
}

func usesGemma4BF16Rounding(layer *Layer) bool {
	if layer == nil {
		return false
	}
	return layer.Gemma4MoE != nil || layer.Gemma4PLE != nil || layer.LayerScale != 1
}

func roundBF16Value(v float32) float32 {
	return model.RoundFloat32ToBF16(v)
}

func gemma4SiluExact(x float32) float32 {
	return x / float32(1.0+math.Exp(float64(-x)))
}

func gemma4GeluTanhExact(x float32) float32 {
	const sqrt2OverPi = 0.7978845608028654
	inner := sqrt2OverPi * (x + 0.044715*x*x*x)
	return 0.5 * x * (1 + float32(math.Tanh(float64(inner))))
}

func roundBF16SliceInPlace(x []float32) {
	for i := range x {
		x[i] = roundBF16Value(x[i])
	}
}

func roundBF16SliceTo(dst, src []float32) []float32 {
	dst = dst[:len(src)]
	for i, v := range src {
		dst[i] = roundBF16Value(v)
	}
	return dst
}

func markHostStateDirty(ds DeviceStateOps, x []float32) {
	if ds != nil {
		ds.HostStateDirty(x)
	}
}

func gemma4DenseFFN(m *Instance, layer *Layer, normed []float32) []float32 {
	if layer == nil || layer.FfnUp == nil || layer.FfnGate == nil || layer.FfnDown == nil {
		return FFN(m, layer, normed)
	}

	ops := m.Ops()
	normed = roundBF16SliceTo(m.Scratch.AttnOut[:len(normed)], normed)
	ops.MatVec(m.Scratch.FfnUp[:layer.FfnUp.R], layer.FfnUp, normed)
	ops.MatVec(m.Scratch.FfnGate[:layer.FfnGate.R], layer.FfnGate, normed)
	roundBF16SliceInPlace(m.Scratch.FfnUp[:layer.FfnUp.R])
	roundBF16SliceInPlace(m.Scratch.FfnGate[:layer.FfnGate.R])

	useGelu := strings.Contains(m.Config.Config.HiddenAct, "gelu")
	for i := 0; i < layer.FfnUp.R; i++ {
		gate := m.Scratch.FfnGate[i]
		up := m.Scratch.FfnUp[i]
		var act float32
		if useGelu {
			act = roundBF16Value(gemma4GeluTanhExact(gate))
		} else {
			act = roundBF16Value(gemma4SiluExact(gate))
		}
		m.Scratch.FfnAct[i] = roundBF16Value(act * up)
	}

	ops.MatVec(m.Scratch.Tmp2[:layer.FfnDown.R], layer.FfnDown, m.Scratch.FfnAct[:layer.FfnUp.R])
	roundBF16SliceInPlace(m.Scratch.Tmp2[:layer.FfnDown.R])
	return m.Scratch.Tmp2[:layer.FfnDown.R]
}

func runGemma4FFNBlock(m *Instance, layer *Layer, x, normed, perLayerInput []float32, ds DeviceStateOps, trace *gemma4FFNTrace) error {
	ops := m.Ops()
	denseOut := gemma4DenseFFN(m, layer, normed)
	syncDeviceSlice(ops, denseOut)
	if err := consumeFastPathError(ops); err != nil {
		return fmt.Errorf("gemma4 dense ffn sync failed: %w", err)
	}

	if layer.Gemma4MoE != nil {
		ops.RMSNorm(m.Scratch.Tmp2, denseOut, layer.Gemma4MoE.PostNorm1, m.RMSEpsilon)
		roundBF16SliceInPlace(m.Scratch.Tmp2[:len(x)])
		roundedX := roundBF16SliceTo(m.Scratch.AttnProj[:len(x)], x)
		ops.RMSNorm(m.Scratch.Tmp, roundedX, layer.Gemma4MoE.PreNorm2, m.RMSEpsilon)
		roundBF16SliceInPlace(m.Scratch.Tmp[:len(x)])
		expertOut := gemma4Experts(m, layer.Gemma4MoE, m.Scratch.Tmp)
		rmsNormWeighted(m.Scratch.MoeAccum[:len(x)], expertOut, layer.Gemma4MoE.PostNorm2, m.RMSEpsilon)
		roundBF16SliceInPlace(m.Scratch.MoeAccum[:len(x)])
		Add(m.Scratch.Tmp2[:len(x)], m.Scratch.MoeAccum[:len(x)])
		ops.RMSNorm(m.Scratch.Tmp, m.Scratch.Tmp2[:len(x)], layer.PostFfnNorm, m.RMSEpsilon)
		roundBF16SliceInPlace(m.Scratch.Tmp[:len(x)])
		if trace != nil {
			trace.FfnOut = cloneVec(m.Scratch.Tmp[:len(x)])
		}
		addResidual(ds, x, m.Scratch.Tmp)
		syncDeviceSlice(ops, x)
		if err := consumeFastPathError(ops); err != nil {
			return fmt.Errorf("gemma4 moe residual bf16 sync failed: %w", err)
		}
		roundBF16SliceInPlace(x)
		markHostStateDirty(ds, x)
	} else {
		ffnOut := denseOut
		if len(layer.PostFfnNorm) > 0 {
			ops.RMSNorm(m.Scratch.Tmp2, denseOut, layer.PostFfnNorm, m.RMSEpsilon)
			ffnOut = m.Scratch.Tmp2
		}
		roundBF16SliceInPlace(ffnOut[:len(x)])
		if trace != nil {
			trace.FfnOut = cloneVec(ffnOut[:len(x)])
		}
		addResidual(ds, x, ffnOut[:len(x)])
		syncDeviceSlice(ops, x)
		if err := consumeFastPathError(ops); err != nil {
			return fmt.Errorf("gemma4 ffn residual bf16 sync failed: %w", err)
		}
		roundBF16SliceInPlace(x)
		markHostStateDirty(ds, x)
	}
	if trace != nil {
		trace.PostFfnHidden = cloneVec(x)
	}

	if layer.Gemma4PLE != nil && perLayerInput != nil {
		syncDeviceSlice(ops, x)
		if err := consumeFastPathError(ops); err != nil {
			return fmt.Errorf("gemma4 per-layer input sync failed: %w", err)
		}
		roundedX := roundBF16SliceTo(m.Scratch.AttnProj[:len(x)], x)
		ops.MatVec(m.Scratch.FfnGate[:len(perLayerInput)], layer.Gemma4PLE.InputGate, roundedX)
		roundBF16SliceInPlace(m.Scratch.FfnGate[:len(perLayerInput)])
		useGelu := strings.Contains(m.Config.Config.HiddenAct, "gelu")
		for i := range perLayerInput {
			if useGelu {
				act := roundBF16Value(gemma4GeluTanhExact(m.Scratch.FfnGate[i]))
				m.Scratch.FfnAct[i] = roundBF16Value(act * perLayerInput[i])
			} else {
				act := roundBF16Value(gemma4SiluExact(m.Scratch.FfnGate[i]))
				m.Scratch.FfnAct[i] = roundBF16Value(act * perLayerInput[i])
			}
		}
		ops.MatVec(m.Scratch.Tmp2, layer.Gemma4PLE.Projection, m.Scratch.FfnAct[:len(perLayerInput)])
		roundBF16SliceInPlace(m.Scratch.Tmp2[:len(x)])
		ops.RMSNorm(m.Scratch.Tmp, m.Scratch.Tmp2[:len(x)], layer.Gemma4PLE.PostNorm, m.RMSEpsilon)
		roundBF16SliceInPlace(m.Scratch.Tmp[:len(x)])
		if trace != nil {
			trace.PerLayerResidual = cloneVec(m.Scratch.Tmp[:len(x)])
		}
		addResidual(ds, x, m.Scratch.Tmp[:len(x)])
		syncDeviceSlice(ops, x)
		if err := consumeFastPathError(ops); err != nil {
			return fmt.Errorf("gemma4 ple residual bf16 sync failed: %w", err)
		}
		roundBF16SliceInPlace(x)
		markHostStateDirty(ds, x)
	}

	if layer.LayerScale != 0 && layer.LayerScale != 1 {
		syncDeviceSlice(ops, x)
		if err := consumeFastPathError(ops); err != nil {
			return fmt.Errorf("gemma4 layer scale sync failed: %w", err)
		}
		for i := range x {
			x[i] *= layer.LayerScale
		}
		roundBF16SliceInPlace(x)
		markHostStateDirty(ds, x)
	}
	return nil
}

func gemma4Experts(m *Instance, moe *Gemma4MoELayer, x []float32) []float32 {
	accum := m.Scratch.MoeAccum
	for i := range x {
		accum[i] = 0
	}

	rmsNormNoWeightTo(m.Scratch.Tmp2[:len(x)], x, m.RMSEpsilon)
	for i := range x {
		m.Scratch.Tmp2[i] *= moe.RouterScale[i] * moe.ScalarRootSize
	}
	m.Ops().MatVec(m.Scratch.RouterRaw, moe.RouterProj, m.Scratch.Tmp2[:len(x)])
	Softmax(m.Scratch.RouterRaw[:len(moe.Experts)])
	selectGemma4TopK(m.Scratch.RouterRaw[:len(moe.Experts)], moe.TopK, moe.RouterPerExpertScale, m.Scratch.RouterIdx, m.Scratch.RouterTop, m.Scratch.RouterW)

	for j := 0; j < moe.TopK; j++ {
		expertID := m.Scratch.RouterIdx[j]
		if expertID < 0 || expertID >= len(moe.Experts) {
			continue
		}
		weight := m.Scratch.RouterW[j]
		if weight == 0 {
			continue
		}
		expert := moe.Experts[expertID]
		m.Ops().MatVec(m.Scratch.FfnGate[:2*moe.Intermediate], expert.GateUp, x)
		roundBF16SliceInPlace(m.Scratch.FfnGate[:2*moe.Intermediate])
		gate := m.Scratch.FfnGate[:moe.Intermediate]
		up := m.Scratch.FfnGate[moe.Intermediate : 2*moe.Intermediate]
		useGelu := strings.Contains(m.Config.Config.HiddenAct, "gelu")
		for i := 0; i < moe.Intermediate; i++ {
			if useGelu {
				act := roundBF16Value(gemma4GeluTanhExact(gate[i]))
				m.Scratch.FfnAct[i] = roundBF16Value(act * up[i])
			} else {
				act := roundBF16Value(gemma4SiluExact(gate[i]))
				m.Scratch.FfnAct[i] = roundBF16Value(act * up[i])
			}
		}
		m.Ops().MatVec(m.Scratch.Tmp[:len(x)], expert.Down, m.Scratch.FfnAct[:moe.Intermediate])
		roundBF16SliceInPlace(m.Scratch.Tmp[:len(x)])
		for i := range x {
			accum[i] += weight * m.Scratch.Tmp[i]
		}
	}
	return accum[:len(x)]
}

func selectGemma4TopK(probabilities []float32, k int, perExpertScale []float32, idxOut []int, topProb []float32, weightOut []float32) {
	if k > len(probabilities) {
		k = len(probabilities)
	}
	for i := range k {
		idxOut[i] = -1
		topProb[i] = float32(-math.MaxFloat32)
	}
	for idx, prob := range probabilities {
		pos := k
		for j := range k {
			if prob > topProb[j] {
				pos = j
				break
			}
		}
		if pos == k {
			continue
		}
		for j := k - 1; j > pos; j-- {
			idxOut[j] = idxOut[j-1]
			topProb[j] = topProb[j-1]
		}
		idxOut[pos] = idx
		topProb[pos] = prob
	}
	var denom float32
	for j := range k {
		if idxOut[j] >= 0 {
			denom += topProb[j]
		}
	}
	if denom == 0 {
		denom = 1
	}
	for j := range k {
		id := idxOut[j]
		if id < 0 {
			weightOut[j] = 0
			continue
		}
		weightOut[j] = (topProb[j] / denom) * perExpertScale[id]
	}
}

func rmsNormNoWeightTo(dst, src []float32, eps float32) {
	var sum float32
	for _, v := range src {
		sum += v * v
	}
	scale := float32(1.0 / math.Sqrt(float64(sum/float32(len(src))+eps)))
	for i, v := range src {
		dst[i] = v * scale
	}
}

func rmsNormWeighted(dst, src, weight []float32, eps float32) {
	var sum float32
	for _, v := range src {
		sum += v * v
	}
	scale := float32(1.0 / math.Sqrt(float64(sum/float32(len(src))+eps)))
	for i, v := range src {
		dst[i] = v * scale * weight[i]
	}
}
