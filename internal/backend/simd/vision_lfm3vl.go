package simd

import (
	"fmt"
	"math"

	"github.com/samcharles93/mantle/internal/backend/core"
	"github.com/samcharles93/mantle/internal/model"
)

// VisionTower is the LFM3-VL vision encoder: a linear patchifier, an RMSNorm,
// a stack of gated depthwise-2D-conv / bidirectional-attention layers with
// axial 2D RoPE, and the multimodal projector that maps merged vision tokens
// into language-model embeddings. It implements core.VisionEncoder and runs
// on the host (SIMD) even when the text decoder runs on a device.
type VisionTower struct {
	cfg        *core.VisionConfig
	Patchifier *core.Mat
	EmbedNorm  []float32
	Layers     []visionLayer

	ProjNormW, ProjNormB []float32
	Proj1, Proj2         *core.Mat
	Proj1Bias, Proj2Bias []float32

	invFreq []float64 // axial RoPE inverse frequencies, headDim/4 entries
}

type visionLayer struct {
	IsAttn bool

	// attention
	Q, K, V, O   *core.Mat
	QNorm, KNorm []float32 // per-head-dim RMSNorm weights

	// gated depthwise 2D conv
	ConvIn     *core.Mat
	ConvKernel []float32 // [C][k][k]
	ConvOut    *core.Mat

	OpNorm, FfnNorm []float32
	W1, W2, W3      *core.Mat
}

// visionFrame is one attention segment: the patches of a single image frame.
type visionFrame struct {
	start, end int // patch index range [start, end)
	h, w       int // patch grid dims
}

func (v *VisionTower) Config() *core.VisionConfig { return v.cfg }

// visionConfigFromHFConfig builds the runtime vision config from the merged
// HF config plus the packed processor_config.json (image preprocessing).
func visionConfigFromHFConfig(cfg *model.HFConfig, proc *model.ProcessorConfig) (*core.VisionConfig, error) {
	vc := cfg.VisionConfig
	if vc == nil {
		return nil, nil
	}
	if vc.HiddenSize <= 0 || vc.NumAttentionHeads <= 0 || vc.NumHiddenLayers <= 0 {
		return nil, fmt.Errorf("vision_config: invalid vision tower dimensions")
	}
	headDim := vc.HiddenSize / vc.NumAttentionHeads
	ropeTheta := 1e6
	if vc.RopeParameters != nil && vc.RopeParameters.RopeTheta > 0 {
		ropeTheta = vc.RopeParameters.RopeTheta
	}
	merge := vc.SpatialMergeSize
	if merge <= 0 {
		merge = 1
	}
	temporal := vc.TemporalPatchSize
	if temporal <= 0 {
		temporal = 1
	}
	down := cfg.DownsampleFactor
	if down <= 0 {
		down = merge
	}
	out := &core.VisionConfig{
		HiddenSize:            vc.HiddenSize,
		InChannels:            vc.InChannels,
		PatchSize:             vc.PatchSize,
		ConvKernelSize:        vc.ConvKernelSize,
		IntermediateSize:      vc.IntermediateSize,
		NumLayers:             vc.NumHiddenLayers,
		NumHeads:              vc.NumAttentionHeads,
		NumKVHeads:            vc.NumKeyValueHeads,
		HeadDim:               headDim,
		LayerTypes:            vc.LayerTypes,
		NormEps:               vc.NormEps,
		MergeSize:             merge,
		TemporalPatch:         temporal,
		RopeTheta:             ropeTheta,
		DownsampleFactor:      down,
		ProjectorHiddenSize:   cfg.ProjectorHiddenSize,
		ProjectorHiddenAct:    cfg.ProjectorHiddenAct,
		ProjectorLayerNormEps: cfg.ProjectorLayerNormEps,
		ProjectorBias:         cfg.ProjectorBias,
		TextHiddenSize:        cfg.HiddenSize,
		ImageTokenID:          cfg.ImageTokenID,
		VisionStartTokenID:    cfg.VisionStartTokenID,
		VisionEndTokenID:      cfg.VisionEndTokenID,
		ImageMean:             []float64{0.5, 0.5, 0.5},
		ImageStd:              []float64{0.5, 0.5, 0.5},
		RescaleFactor:         1.0 / 255.0,
		MinPixels:             65536,
		MaxPixels:             16777216,
	}
	if proc != nil && proc.ImageProcessor != nil {
		ip := proc.ImageProcessor
		if len(ip.ImageMean) == 3 {
			out.ImageMean = ip.ImageMean
		}
		if len(ip.ImageStd) == 3 {
			out.ImageStd = ip.ImageStd
		}
		if ip.RescaleFactor != 0 {
			out.RescaleFactor = ip.RescaleFactor
		}
		if ip.Size != nil {
			if ip.Size.ShortestEdge > 0 {
				out.MinPixels = ip.Size.ShortestEdge
			}
			if ip.Size.LongestEdge > 0 {
				out.MaxPixels = ip.Size.LongestEdge
			}
		}
		if ip.PatchSize > 0 {
			out.PatchSize = ip.PatchSize
		}
		if ip.MergeSize > 0 {
			out.MergeSize = ip.MergeSize
		}
		if ip.TemporalPatch > 0 {
			out.TemporalPatch = ip.TemporalPatch
		}
	}
	if out.PatchSize <= 0 || out.InChannels <= 0 {
		return nil, fmt.Errorf("vision_config: patch_size/in_channels missing")
	}
	if len(out.LayerTypes) == 0 {
		out.LayerTypes = make([]string, out.NumLayers)
		for i := range out.LayerTypes {
			out.LayerTypes[i] = "full_attention"
		}
	}
	if len(out.LayerTypes) != out.NumLayers {
		return nil, fmt.Errorf("vision_config: layer_types has %d entries, want %d", len(out.LayerTypes), out.NumLayers)
	}
	return out, nil
}

// loadVisionTower reads the vision tower and projector tensors from the
// container. Names follow the HF LFM3-VL checkpoint layout.
func loadVisionTower(src tensorSource, cfg *model.HFConfig, vc *core.VisionConfig) (*VisionTower, error) {
	vt := &VisionTower{cfg: vc}
	var err error
	if vt.Patchifier, err = loadMat(src, "model.vision_tower.patchifier.weight"); err != nil {
		return nil, err
	}
	if vt.EmbedNorm, err = loadVec(src, "model.vision_tower.embedding_norm.weight"); err != nil {
		return nil, err
	}
	vt.Layers = make([]visionLayer, vc.NumLayers)
	for i := range vt.Layers {
		l := &vt.Layers[i]
		prefix := fmt.Sprintf("model.vision_tower.layers.%d", i)
		if l.OpNorm, err = loadVec(src, prefix+".operator_norm.weight"); err != nil {
			return nil, err
		}
		if l.FfnNorm, err = loadVec(src, prefix+".ffn_norm.weight"); err != nil {
			return nil, err
		}
		if l.W1, err = loadMat(src, prefix+".feed_forward.w1.weight"); err != nil {
			return nil, err
		}
		if l.W2, err = loadMat(src, prefix+".feed_forward.w2.weight"); err != nil {
			return nil, err
		}
		if l.W3, err = loadMat(src, prefix+".feed_forward.w3.weight"); err != nil {
			return nil, err
		}
		isAttn := i < len(vc.LayerTypes) && vc.LayerTypes[i] == "full_attention"
		l.IsAttn = isAttn
		if isAttn {
			if l.Q, err = loadMat(src, prefix+".self_attn.q_proj.weight"); err != nil {
				return nil, err
			}
			if l.K, err = loadMat(src, prefix+".self_attn.k_proj.weight"); err != nil {
				return nil, err
			}
			if l.V, err = loadMat(src, prefix+".self_attn.v_proj.weight"); err != nil {
				return nil, err
			}
			if l.O, err = loadMat(src, prefix+".self_attn.out_proj.weight"); err != nil {
				return nil, err
			}
			if l.QNorm, err = loadVec(src, prefix+".self_attn.q_layernorm.weight"); err != nil {
				return nil, err
			}
			if l.KNorm, err = loadVec(src, prefix+".self_attn.k_layernorm.weight"); err != nil {
				return nil, err
			}
		} else {
			if l.ConvIn, err = loadMat(src, prefix+".conv.in_proj.weight"); err != nil {
				return nil, err
			}
			kdata, kshape, err := readTensorF32(src, prefix+".conv.spatial_conv.weight")
			if err != nil {
				return nil, err
			}
			if len(kshape) != 4 || kshape[1] != 1 || kshape[2] != vc.ConvKernelSize || kshape[3] != vc.ConvKernelSize {
				return nil, fmt.Errorf("%s: expected shape [%d 1 %d %d], got %v",
					prefix+".conv.spatial_conv.weight", vc.HiddenSize, vc.ConvKernelSize, vc.ConvKernelSize, kshape)
			}
			l.ConvKernel = kdata
			if l.ConvOut, err = loadMat(src, prefix+".conv.out_proj.weight"); err != nil {
				return nil, err
			}
		}
	}
	if vt.ProjNormW, err = loadVec(src, "model.multi_modal_projector.layer_norm.weight"); err != nil {
		return nil, err
	}
	if vc.ProjectorBias {
		if vt.ProjNormB, err = loadVec(src, "model.multi_modal_projector.layer_norm.bias"); err != nil {
			return nil, err
		}
	}
	if vt.Proj1, err = loadMat(src, "model.multi_modal_projector.linear_1.weight"); err != nil {
		return nil, err
	}
	if vc.ProjectorBias {
		if vt.Proj1Bias, err = loadVec(src, "model.multi_modal_projector.linear_1.bias"); err != nil {
			return nil, err
		}
	}
	if vt.Proj2, err = loadMat(src, "model.multi_modal_projector.linear_2.weight"); err != nil {
		return nil, err
	}
	if vc.ProjectorBias {
		if vt.Proj2Bias, err = loadVec(src, "model.multi_modal_projector.linear_2.bias"); err != nil {
			return nil, err
		}
	}

	// Axial RoPE: inverse frequencies over half the head dimension per axis.
	spatialDim := vc.HeadDim / 2
	vt.invFreq = make([]float64, spatialDim/2)
	for i := range vt.invFreq {
		vt.invFreq[i] = 1.0 / math.Pow(vc.RopeTheta, float64(2*i)/float64(spatialDim))
	}
	return vt, nil
}

// Encode runs the vision tower + projector over preprocessed patches.
// pixelValues holds numPatches rows of cfg.PatchDim() floats in merge-block
// major order (see ProcessImages); grids holds one (t, h, w) patch grid per
// image. It returns one embedding row per image token per image.
func (v *VisionTower) Encode(pixelValues []float32, grids [][3]int) ([][][]float32, error) {
	cfg := v.cfg
	hidden := cfg.HiddenSize
	patchDim := cfg.PatchDim()
	if patchDim <= 0 || len(pixelValues)%patchDim != 0 {
		return nil, fmt.Errorf("vision: pixel payload size %d not a multiple of patch dim %d", len(pixelValues), patchDim)
	}
	numPatches := len(pixelValues) / patchDim

	total := 0
	for _, g := range grids {
		total += g[0] * g[1] * g[2]
	}
	if total != numPatches {
		return nil, fmt.Errorf("vision: grids cover %d patches, payload has %d", total, numPatches)
	}

	// Patchify projection + embedding norm.
	h := make([]float32, numPatches*hidden)
	tmp := make([]float32, hidden)
	for p := range numPatches {
		MatVec(tmp, v.Patchifier, pixelValues[p*patchDim:(p+1)*patchDim])
		RMSNorm(h[p*hidden:(p+1)*hidden], tmp, v.EmbedNorm, float32(cfg.NormEps))
	}

	// Axial 2D RoPE cos/sin tables, block-major over merge blocks, repeated t.
	headDim := cfg.HeadDim
	cosT := make([]float32, numPatches*headDim)
	sinT := make([]float32, numPatches*headDim)
	pos := 0
	for _, g := range grids {
		t, gh, gw := g[0], g[1], g[2]
		m := cfg.MergeSize
		hpos := make([]int, gh*gw)
		wpos := make([]int, gh*gw)
		idx := 0
		for bh := 0; bh < gh/m; bh++ {
			for bw := 0; bw < gw/m; bw++ {
				for mh := 0; mh < m; mh++ {
					for mw := 0; mw < m; mw++ {
						hpos[idx] = bh*m + mh
						wpos[idx] = bw*m + mw
						idx++
					}
				}
			}
		}
		for range t {
			for i := 0; i < gh*gw; i++ {
				hw := make([]float64, headDim/2)
				for j := range v.invFreq {
					hw[j] = float64(hpos[i]) * v.invFreq[j]
					hw[len(v.invFreq)+j] = float64(wpos[i]) * v.invFreq[j]
				}
				for d := 0; d < headDim/2; d++ {
					c := float32(math.Cos(hw[d]))
					s := float32(math.Sin(hw[d]))
					cosT[pos*headDim+d] = c
					sinT[pos*headDim+d] = s
					cosT[pos*headDim+headDim/2+d] = c
					sinT[pos*headDim+headDim/2+d] = s
				}
				pos++
			}
		}
	}

	// Attention frame segments: each temporal frame of each grid is one segment.
	var frames []visionFrame
	off := 0
	for _, g := range grids {
		t, gh, gw := g[0], g[1], g[2]
		for range t {
			frames = append(frames, visionFrame{start: off, end: off + gh*gw, h: gh, w: gw})
			off += gh * gw
		}
	}

	normed := make([]float32, hidden)
	for li := range v.Layers {
		l := &v.Layers[li]
		if l.IsAttn {
			v.attnLayer(l, h, cosT, sinT, frames, normed)
		} else {
			v.convLayer(l, h, frames, normed)
		}
		// FFN (pre-norm residual MLP).
		ffnIn := make([]float32, numPatches*hidden)
		for p := range numPatches {
			RMSNorm(ffnIn[p*hidden:(p+1)*hidden], h[p*hidden:(p+1)*hidden], l.FfnNorm, float32(cfg.NormEps))
		}
		v.mlp(l, h, ffnIn)
	}

	// Projector: LayerNorm -> merge downsample_factor^2 consecutive tokens ->
	// linear_1 -> gelu -> linear_2, split per image.
	down := cfg.DownsampleFactor
	group := down * down
	if group <= 0 {
		group = 1
	}
	projIn := make([]float32, numPatches*hidden)
	for p := range numPatches {
		layerNorm(h[p*hidden:(p+1)*hidden], projIn[p*hidden:(p+1)*hidden], v.ProjNormW, v.ProjNormB, float32(cfg.ProjectorLayerNormEps))
	}
	merged := numPatches / group
	projHidden := cfg.ProjectorHiddenSize
	out := make([]float32, merged*cfg.TextHiddenSize)
	vec := make([]float32, hidden*group)
	tmp1 := make([]float32, projHidden)
	for i := range merged {
		copy(vec, projIn[i*group*hidden:(i+1)*group*hidden])
		MatVec(tmp1, v.Proj1, vec)
		for j := range tmp1 {
			if len(v.Proj1Bias) > 0 {
				tmp1[j] += v.Proj1Bias[j]
			}
			tmp1[j] = geluExact(tmp1[j])
		}
		MatVec(out[i*cfg.TextHiddenSize:(i+1)*cfg.TextHiddenSize], v.Proj2, tmp1)
		for j := range cfg.TextHiddenSize {
			if len(v.Proj2Bias) > 0 {
				out[i*cfg.TextHiddenSize+j] += v.Proj2Bias[j]
			}
		}
	}
	res := make([][][]float32, len(grids))
	row := 0
	for gi, g := range grids {
		nTok := cfg.TokensPerImage(g)
		rows := make([][]float32, nTok)
		for i := range nTok {
			rows[i] = out[(row+i)*cfg.TextHiddenSize : (row+i+1)*cfg.TextHiddenSize]
		}
		res[gi] = rows
		row += nTok
	}
	return res, nil
}

// attnLayer runs one bidirectional-attention vision layer over all frames.
func (v *VisionTower) attnLayer(l *visionLayer, h []float32, cosT, sinT []float32, frames []visionFrame, normed []float32) {
	cfg := v.cfg
	hidden := cfg.HiddenSize
	heads := cfg.NumHeads
	kvHeads := cfg.NumKVHeads
	headDim := cfg.HeadDim
	groups := heads / kvHeads
	n := len(h) / hidden
	eps := float32(cfg.NormEps)

	q := make([]float32, n*heads*headDim)
	k := make([]float32, n*kvHeads*headDim)
	vv := make([]float32, n*kvHeads*headDim)
	for p := range n {
		RMSNorm(normed, h[p*hidden:(p+1)*hidden], l.OpNorm, eps)
		MatVec(q[p*heads*headDim:(p+1)*heads*headDim], l.Q, normed)
		MatVec(k[p*kvHeads*headDim:(p+1)*kvHeads*headDim], l.K, normed)
		MatVec(vv[p*kvHeads*headDim:(p+1)*kvHeads*headDim], l.V, normed)
	}
	// Per-head RMSNorm on q and k.
	for p := range n {
		for hi := range heads {
			base := p*heads*headDim + hi*headDim
			RMSNorm(q[base:base+headDim], q[base:base+headDim], l.QNorm, eps)
		}
		for hi := range kvHeads {
			base := p*kvHeads*headDim + hi*headDim
			RMSNorm(k[base:base+headDim], k[base:base+headDim], l.KNorm, eps)
		}
	}
	// Axial RoPE.
	applyAxialRoPE(q, cosT, sinT, n, heads, headDim)
	applyAxialRoPE(k, cosT, sinT, n, kvHeads, headDim)

	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	attnOut := make([]float32, n*heads*headDim)
	var scores []float32
	for _, f := range frames {
		length := f.end - f.start
		if cap(scores) < length*length {
			scores = make([]float32, length*length)
		}
		scores = scores[:length*length]
		for hi := range heads {
			kvh := hi / groups
			for qi := 0; qi < length; qi++ {
				qBase := (f.start+qi)*heads*headDim + hi*headDim
				row := scores[qi*length : (qi+1)*length]
				for ki := 0; ki < length; ki++ {
					kBase := (f.start+ki)*kvHeads*headDim + kvh*headDim
					var dot float32
					for d := range headDim {
						dot += q[qBase+d] * k[kBase+d]
					}
					row[ki] = dot * scale
				}
				core.Softmax(row)
				outBase := (f.start+qi)*heads*headDim + hi*headDim
				for ki := 0; ki < length; ki++ {
					vBase := (f.start+ki)*kvHeads*headDim + kvh*headDim
					w := row[ki]
					for d := range headDim {
						attnOut[outBase+d] += w * vv[vBase+d]
					}
				}
			}
		}
	}
	// Output projection + residual.
	res := make([]float32, hidden)
	for p := range n {
		MatVec(res, l.O, attnOut[p*heads*headDim:(p+1)*heads*headDim])
		for d := range hidden {
			h[p*hidden+d] += res[d]
		}
	}
}

// convLayer runs one gated depthwise 2D convolution vision layer. Tokens are
// stored in merge-block-major order; the convolution runs on the raster grid.
func (v *VisionTower) convLayer(l *visionLayer, h []float32, frames []visionFrame, normed []float32) {
	cfg := v.cfg
	hidden := cfg.HiddenSize
	n := len(h) / hidden
	m := cfg.MergeSize
	kSize := cfg.ConvKernelSize
	pad := kSize / 2
	eps := float32(cfg.NormEps)

	in := make([]float32, n*3*hidden)
	for p := range n {
		RMSNorm(normed, h[p*hidden:(p+1)*hidden], l.OpNorm, eps)
		MatVec(in[p*3*hidden:(p+1)*3*hidden], l.ConvIn, normed)
	}
	convTok := make([]float32, n*hidden)
	raster := make([]float32, hidden*0) // grown per frame below
	convImg := make([]float32, 0)
	res := make([]float32, hidden)
	for _, f := range frames {
		C, H, W := hidden, f.h, f.w
		if len(raster) < C*H*W {
			raster = make([]float32, C*H*W)
			convImg = make([]float32, C*H*W)
		}
		raster = raster[:C*H*W]
		convImg = convImg[:C*H*W]
		// gated = B * x where B = chunk 0, C = chunk 1, x = chunk 2
		for i := 0; i < H*W; i++ {
			p := f.start + i
			y, x := blockMajorToRaster(i, H, W, m)
			bBase := p * 3 * hidden
			xBase := p*3*hidden + 2*hidden
			for c := range C {
				raster[c*H*W+y*W+x] = in[bBase+c] * in[xBase+c]
			}
		}
		// depthwise conv2d with zero padding
		for c := range C {
			kc := l.ConvKernel[c*kSize*kSize:]
			for y := 0; y < H; y++ {
				for x := 0; x < W; x++ {
					var acc float32
					for ky := 0; ky < kSize; ky++ {
						iy := y + ky - pad
						if iy < 0 || iy >= H {
							continue
						}
						for kx := 0; kx < kSize; kx++ {
							ix := x + kx - pad
							if ix < 0 || ix >= W {
								continue
							}
							acc += kc[ky*kSize+kx] * raster[c*H*W+iy*W+ix]
						}
					}
					convImg[c*H*W+y*W+x] = acc
				}
			}
		}
		// back to block-major token order; multiply by the C chunk; out_proj
		for i := 0; i < H*W; i++ {
			p := f.start + i
			y, x := blockMajorToRaster(i, H, W, m)
			cBase := p*3*hidden + hidden
			row := convTok[p*hidden : (p+1)*hidden]
			for c := range C {
				row[c] = convImg[c*H*W+y*W+x] * in[cBase+c]
			}
		}
	}
	for p := range n {
		MatVec(res, l.ConvOut, convTok[p*hidden:(p+1)*hidden])
		for d := range hidden {
			h[p*hidden+d] += res[d]
		}
	}
}

// blockMajorToRaster maps a merge-block-major token index to its (y, x) raster
// position in an H x W patch grid with merge size m.
func blockMajorToRaster(i, H, W, m int) (y, x int) {
	bi := i / (m * m)
	wi := i % (m * m)
	bh := bi / (W / m)
	bw := bi % (W / m)
	mh := wi / m
	mw := wi % m
	return bh*m + mh, bw*m + mw
}

func (v *VisionTower) mlp(l *visionLayer, h, ffnIn []float32) {
	hidden := v.cfg.HiddenSize
	ffn := l.W1.R
	n := len(h) / hidden
	up := make([]float32, ffn)
	gate := make([]float32, ffn)
	down := make([]float32, hidden)
	for p := range n {
		x := ffnIn[p*hidden : (p+1)*hidden]
		MatVec(up, l.W1, x)
		MatVec(gate, l.W3, x)
		for i := range up {
			s := up[i]
			up[i] = gate[i] * s / (1 + float32(math.Exp(-float64(s))))
		}
		MatVec(down, l.W2, up)
		for d := range hidden {
			h[p*hidden+d] += down[d]
		}
	}
}

// applyAxialRoPE applies the recomposed axial 2D rotary embedding in place:
// x_embed = x*cos + rotate_half(x)*sin with cos/sin duplicated over both halves.
func applyAxialRoPE(x []float32, cosT, sinT []float32, n, heads, headDim int) {
	half := headDim / 2
	for p := range n {
		cos := cosT[p*headDim : (p+1)*headDim]
		sin := sinT[p*headDim : (p+1)*headDim]
		for hi := range heads {
			base := p*heads*headDim + hi*headDim
			for d := 0; d < half; d++ {
				x1 := x[base+d]
				x2 := x[base+half+d]
				x[base+d] = x1*cos[d] - x2*sin[d]
				x[base+half+d] = x2*cos[half+d] + x1*sin[half+d]
			}
		}
	}
}

func layerNorm(src, dst, weight, bias []float32, eps float32) {
	n := len(src)
	var mean float32
	for _, v := range src {
		mean += v
	}
	mean /= float32(n)
	var varAcc float32
	for _, v := range src {
		d := v - mean
		varAcc += d * d
	}
	varAcc /= float32(n)
	scale := 1.0 / float32(math.Sqrt(float64(varAcc+eps)))
	for i, v := range src {
		b := float32(0)
		if len(bias) > i {
			b = bias[i]
		}
		dst[i] = (v-mean)*scale*weight[i] + b
	}
}

func geluExact(x float32) float32 {
	return 0.5 * x * (1 + float32(math.Erf(float64(x)/math.Sqrt2)))
}

// VisionEncoder implements core.VisionEncoderProvider for the SIMD instance.
func (m *Instance) VisionEncoder() core.VisionEncoder { return m.Vision }
