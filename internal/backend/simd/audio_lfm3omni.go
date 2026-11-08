package simd

import (
	"fmt"
	"math"

	"github.com/samcharles93/mantle/internal/audio"
	"github.com/samcharles93/mantle/internal/backend/core"
	"github.com/samcharles93/mantle/internal/model"
)

// AudioTower is the LFM3-Omni audio encoder: a stride-8 (mel, time) conv
// downsampler, an LFM2 stack with causal sliding-window attention, and the
// audio projector that maps tower tokens into language-model embeddings. It
// implements core.AudioEncoder and runs on the host (SIMD) even when the text
// decoder runs on a device.
type AudioTower struct {
	cfg *core.AudioConfig

	// downsample: conv_in (1->C), conv_mid/lin_mid, conv_end/lin_end (depthwise
	// C->C), lin_out (C*NumMelBins/8 -> hidden) — HF Lfm3OmniDownsampler.
	convInW  []float32 // [C][9], conv2d 3x3 weight (in=1)
	convInB  []float32
	convMidW []float32 // [C][9] depthwise 3x3
	convMidB []float32
	linMid   *core.Mat // [C][C] 1x1 conv
	linMidB  []float32
	convEndW []float32 // [C][9] depthwise 3x3
	convEndB []float32
	linEnd   *core.Mat // [C][C] 1x1 conv
	linEndB  []float32
	linOut   *core.Mat // [hidden][C*NumMelBins/8] conv1d 1x1
	linOutB  []float32

	embedNorm []float32
	layers    []audioLayer

	// audio projector: LayerNorm -> linear_1 -> GELU -> linear_2.
	projNormW, projNormB []float32
	proj1, proj2         *core.Mat
	proj1B, proj2B       []float32

	invFreq []float64 // RoPE inverse frequencies, HeadDim/2 entries
}

type audioLayer struct {
	isAttn          bool
	opNorm, ffnNorm []float32
	q, k, v, o      *core.Mat
	qNorm, kNorm    []float32 // per-head-dim RMSNorm weights
	convIn          *core.Mat // [3*hidden, hidden]
	convW           []float32 // [hidden][kernel] depthwise causal conv
	convOut         *core.Mat
	w1, w2, w3      *core.Mat
}

func (a *AudioTower) Config() *core.AudioConfig { return a.cfg }

// audioConfigFromHFConfig builds the runtime audio config from the merged HF
// config plus the packed processor_config.json (feature_extractor block).
func audioConfigFromHFConfig(cfg *model.HFConfig, proc *model.ProcessorConfig) (*core.AudioConfig, error) {
	aec := cfg.AudioEncoderConfig
	if aec == nil {
		return nil, nil
	}
	tc := aec.TextConfig
	if tc == nil {
		return nil, fmt.Errorf("audio_encoder_config: missing nested text_config (lfm stack)")
	}
	if aec.HiddenSize <= 0 || aec.ConvChannels <= 0 || aec.NumMelBins <= 0 {
		return nil, fmt.Errorf("audio_encoder_config: invalid tower dimensions")
	}
	if tc.HiddenSize != aec.HiddenSize {
		return nil, fmt.Errorf("audio_encoder_config: lfm hidden_size %d != hidden_size %d", tc.HiddenSize, aec.HiddenSize)
	}
	if tc.NumAttentionHeads <= 0 || tc.NumHiddenLayers <= 0 {
		return nil, fmt.Errorf("audio_encoder_config: invalid lfm dimensions")
	}
	headDim := tc.HiddenSize / tc.NumAttentionHeads
	if headDim <= 0 || headDim%2 != 0 {
		return nil, fmt.Errorf("audio_encoder_config: invalid head_dim %d", headDim)
	}
	// LFM2 block_auto_adjust_ff_dim: int(2*inter/3), times the multiplier,
	// rounded up to a multiple of block_multiple_of (Lfm2MLP).
	ffnDim := tc.IntermediateSize
	if tc.BlockAutoAdjustFFDim {
		ffnDim = int(2 * tc.IntermediateSize / 3)
		if tc.BlockFFNDimMultiplier != 0 {
			ffnDim = int(tc.BlockFFNDimMultiplier * float64(ffnDim))
		}
		if tc.BlockMultipleOf > 1 {
			ffnDim = tc.BlockMultipleOf * ((ffnDim + tc.BlockMultipleOf - 1) / tc.BlockMultipleOf)
		}
	}
	if ffnDim <= 0 {
		return nil, fmt.Errorf("audio_encoder_config: invalid ffn dim %d", ffnDim)
	}
	convL := tc.ConvLCache
	if convL <= 0 {
		convL = 3
	}
	ropeTheta := 1e6
	if tc.RopeParameters != nil && tc.RopeParameters.RopeTheta > 0 {
		ropeTheta = tc.RopeParameters.RopeTheta
	}
	layerTypes := tc.LayerTypes
	if len(layerTypes) == 0 {
		layerTypes = make([]string, tc.NumHiddenLayers)
		for i := range layerTypes {
			layerTypes[i] = "full_attention"
		}
	}
	if len(layerTypes) != tc.NumHiddenLayers {
		return nil, fmt.Errorf("audio_encoder_config: layer_types has %d entries, want %d", len(layerTypes), tc.NumHiddenLayers)
	}
	fe := audio.DefaultFeatureExtractorConfig()
	if proc != nil && proc.FeatureExtractor != nil {
		fe = *proc.FeatureExtractor
	}
	if fe.FeatureSize <= 0 || fe.SamplingRate <= 0 || fe.NFFT <= 0 || fe.HopLength <= 0 || fe.EncoderStride <= 0 {
		return nil, fmt.Errorf("audio feature_extractor: invalid parameters")
	}
	if fe.FeatureSize != aec.NumMelBins {
		return nil, fmt.Errorf("audio feature_extractor: feature_size %d != num_mel_bins %d", fe.FeatureSize, aec.NumMelBins)
	}
	sw := aec.SlidingWindow
	if sw <= 0 {
		sw = 1 << 30 // effectively full causal attention
	}
	out := &core.AudioConfig{
		FeatureExtractor:   fe,
		ConvChannels:       aec.ConvChannels,
		HiddenSize:         aec.HiddenSize,
		NumMelBins:         aec.NumMelBins,
		SlidingWindow:      sw,
		NumLayers:          tc.NumHiddenLayers,
		NumHeads:           tc.NumAttentionHeads,
		NumKVHeads:         tc.NumKeyValueHeads,
		HeadDim:            headDim,
		FFNDim:             ffnDim,
		LayerTypes:         layerTypes,
		NormEps:            tc.NormEps,
		ConvKernelSize:     convL,
		RopeTheta:          ropeTheta,
		TextHiddenSize:     cfg.HiddenSize,
		AudioTokenID:       cfg.AudioTokenID,
		AudioOutTokenID:    cfg.AudioOutTokenID,
		VisionStartTokenID: cfg.VisionStartTokenID,
		VisionEndTokenID:   cfg.VisionEndTokenID,
	}
	if out.TextHiddenSize <= 0 {
		return nil, fmt.Errorf("audio config: text hidden size missing (text_config merge)")
	}
	if out.AudioTokenID <= 0 {
		return nil, fmt.Errorf("audio config: audio_token_id missing")
	}
	return out, nil
}

// loadAudioTower reads the audio tower and projector tensors from the
// container. Names follow the HF LFM3-Omni checkpoint layout
// (model.audio_encoder.*, model.audio_projector.*).
func loadAudioTower(src tensorSource, cfg *model.HFConfig, ac *core.AudioConfig) (*AudioTower, error) {
	at := &AudioTower{cfg: ac}
	var err error
	C := ac.ConvChannels
	hidden := ac.HiddenSize

	loadConv2d := func(name string, cOut, cInGroups, k int) ([]float32, []float32, error) {
		wdata, wshape, err := readTensorF32(src, name+".weight")
		if err != nil {
			return nil, nil, err
		}
		if len(wshape) != 4 || wshape[0] != cOut || wshape[2] != k || wshape[3] != k || wshape[1] != cInGroups {
			return nil, nil, fmt.Errorf("%s: expected shape [%d %d %d %d], got %v", name+".weight", cOut, cInGroups, k, k, wshape)
		}
		b, err := loadVec(src, name+".bias")
		if err != nil {
			return nil, nil, err
		}
		if len(b) != cOut {
			return nil, nil, fmt.Errorf("%s: bias has %d entries, want %d", name+".bias", len(b), cOut)
		}
		return wdata, b, nil
	}
	if at.convInW, at.convInB, err = loadConv2d("model.audio_encoder.downsample.conv_in", C, 1, 3); err != nil {
		return nil, err
	}
	if at.convMidW, at.convMidB, err = loadConv2d("model.audio_encoder.downsample.conv_mid", C, 1, 3); err != nil {
		return nil, err
	}
	if at.convEndW, at.convEndB, err = loadConv2d("model.audio_encoder.downsample.conv_end", C, 1, 3); err != nil {
		return nil, err
	}
	loadLin2d := func(name string) (*core.Mat, []float32, error) {
		// 1x1 Conv2d weight: 4D [C][C][1][1]; flatten to a [C][C] matrix.
		wdata, wshape, err := readTensorF32(src, name+".weight")
		if err != nil {
			return nil, nil, err
		}
		if len(wshape) != 4 || wshape[0] != C || wshape[1] != C || wshape[2] != 1 || wshape[3] != 1 {
			return nil, nil, fmt.Errorf("%s: expected shape [%d %d 1 1], got %v", name+".weight", C, C, wshape)
		}
		m := core.NewMatFromData(C, C, wdata)
		b, err := loadVec(src, name+".bias")
		if err != nil {
			return nil, nil, err
		}
		return &m, b, nil
	}
	if at.linMid, at.linMidB, err = loadLin2d("model.audio_encoder.downsample.lin_mid"); err != nil {
		return nil, err
	}
	if at.linEnd, at.linEndB, err = loadLin2d("model.audio_encoder.downsample.lin_end"); err != nil {
		return nil, err
	}
	linOutIn := C * ac.NumMelBins / 8
	// Conv1d k=1 weight: 3D [hidden][linOutIn][1]; flatten to a matrix.
	linOutData, linOutShape, err := readTensorF32(src, "model.audio_encoder.downsample.lin_out.weight")
	if err != nil {
		return nil, err
	}
	if len(linOutShape) != 3 || linOutShape[0] != hidden || linOutShape[1] != linOutIn || linOutShape[2] != 1 {
		return nil, fmt.Errorf("model.audio_encoder.downsample.lin_out.weight: expected [%d %d 1], got %v", hidden, linOutIn, linOutShape)
	}
	linOutMat := core.NewMatFromData(hidden, linOutIn, linOutData)
	at.linOut = &linOutMat
	if at.linOutB, err = loadVec(src, "model.audio_encoder.downsample.lin_out.bias"); err != nil {
		return nil, err
	}

	if at.embedNorm, err = loadVec(src, "model.audio_encoder.lfm.embedding_norm.weight"); err != nil {
		return nil, err
	}
	if len(at.embedNorm) != hidden {
		return nil, fmt.Errorf("audio lfm embedding_norm: %d entries, want %d", len(at.embedNorm), hidden)
	}

	at.layers = make([]audioLayer, ac.NumLayers)
	for i := range at.layers {
		l := &at.layers[i]
		prefix := fmt.Sprintf("model.audio_encoder.lfm.layers.%d", i)
		if l.opNorm, err = loadVec(src, prefix+".operator_norm.weight"); err != nil {
			return nil, err
		}
		if l.ffnNorm, err = loadVec(src, prefix+".ffn_norm.weight"); err != nil {
			return nil, err
		}
		if l.w1, err = loadMat(src, prefix+".feed_forward.w1.weight"); err != nil {
			return nil, err
		}
		if l.w2, err = loadMat(src, prefix+".feed_forward.w2.weight"); err != nil {
			return nil, err
		}
		if l.w3, err = loadMat(src, prefix+".feed_forward.w3.weight"); err != nil {
			return nil, err
		}
		if l.w1.R != ac.FFNDim || l.w1.C != hidden || l.w2.R != hidden || l.w2.C != ac.FFNDim || l.w3.R != ac.FFNDim || l.w3.C != hidden {
			return nil, fmt.Errorf("%s: feed_forward shapes [%d %d]/[%d %d]/[%d %d], want [%d %d]/[%d %d]/[%d %d]",
				prefix, l.w1.R, l.w1.C, l.w2.R, l.w2.C, l.w3.R, l.w3.C, ac.FFNDim, hidden, hidden, ac.FFNDim, ac.FFNDim, hidden)
		}
		l.isAttn = i < len(ac.LayerTypes) && ac.LayerTypes[i] == "full_attention"
		if l.isAttn {
			if l.q, err = loadMat(src, prefix+".self_attn.q_proj.weight"); err != nil {
				return nil, err
			}
			if l.k, err = loadMat(src, prefix+".self_attn.k_proj.weight"); err != nil {
				return nil, err
			}
			if l.v, err = loadMat(src, prefix+".self_attn.v_proj.weight"); err != nil {
				return nil, err
			}
			if l.o, err = loadMat(src, prefix+".self_attn.out_proj.weight"); err != nil {
				return nil, err
			}
			if l.qNorm, err = loadVec(src, prefix+".self_attn.q_layernorm.weight"); err != nil {
				return nil, err
			}
			if l.kNorm, err = loadVec(src, prefix+".self_attn.k_layernorm.weight"); err != nil {
				return nil, err
			}
			wantQ := ac.NumHeads * ac.HeadDim
			wantKV := ac.NumKVHeads * ac.HeadDim
			if l.q.R != wantQ || l.q.C != hidden || l.o.R != hidden || l.o.C != wantQ ||
				l.k.R != wantKV || l.k.C != hidden || l.v.R != wantKV || l.v.C != hidden {
				return nil, fmt.Errorf("%s: self_attn shape mismatch", prefix)
			}
			if len(l.qNorm) != ac.HeadDim || len(l.kNorm) != ac.HeadDim {
				return nil, fmt.Errorf("%s: q/k_layernorm %d/%d entries, want %d", prefix, len(l.qNorm), len(l.kNorm), ac.HeadDim)
			}
		} else {
			if l.convIn, err = loadMat(src, prefix+".conv.in_proj.weight"); err != nil {
				return nil, err
			}
			if l.convIn.R != 3*hidden || l.convIn.C != hidden {
				return nil, fmt.Errorf("%s: conv.in_proj shape [%d %d], want [%d %d]", prefix, l.convIn.R, l.convIn.C, 3*hidden, hidden)
			}
			kdata, kshape, err := readTensorF32(src, prefix+".conv.conv.weight")
			if err != nil {
				return nil, err
			}
			if len(kshape) != 3 || kshape[0] != hidden || kshape[1] != 1 || kshape[2] != ac.ConvKernelSize {
				return nil, fmt.Errorf("%s: conv.conv.weight expected [%d 1 %d], got %v", prefix, hidden, ac.ConvKernelSize, kshape)
			}
			l.convW = kdata
			if l.convOut, err = loadMat(src, prefix+".conv.out_proj.weight"); err != nil {
				return nil, err
			}
			if l.convOut.R != hidden || l.convOut.C != hidden {
				return nil, fmt.Errorf("%s: conv.out_proj shape [%d %d], want [%d %d]", prefix, l.convOut.R, l.convOut.C, hidden, hidden)
			}
		}
	}

	if at.projNormW, err = loadVec(src, "model.audio_projector.layer_norm.weight"); err != nil {
		return nil, err
	}
	if at.projNormB, err = loadVec(src, "model.audio_projector.layer_norm.bias"); err != nil {
		return nil, err
	}
	if len(at.projNormW) != hidden || len(at.projNormB) != hidden {
		return nil, fmt.Errorf("audio_projector.layer_norm: %d/%d entries, want %d", len(at.projNormW), len(at.projNormB), hidden)
	}
	if at.proj1, err = loadMat(src, "model.audio_projector.linear_1.weight"); err != nil {
		return nil, err
	}
	if at.proj1.R != ac.TextHiddenSize || at.proj1.C != hidden {
		return nil, fmt.Errorf("audio_projector.linear_1: [%d %d], want [%d %d]", at.proj1.R, at.proj1.C, ac.TextHiddenSize, hidden)
	}
	if at.proj1B, err = loadVec(src, "model.audio_projector.linear_1.bias"); err != nil {
		return nil, err
	}
	if at.proj2, err = loadMat(src, "model.audio_projector.linear_2.weight"); err != nil {
		return nil, err
	}
	if at.proj2.R != ac.TextHiddenSize || at.proj2.C != ac.TextHiddenSize {
		return nil, fmt.Errorf("audio_projector.linear_2: [%d %d], want [%d %d]", at.proj2.R, at.proj2.C, ac.TextHiddenSize, ac.TextHiddenSize)
	}
	if at.proj2B, err = loadVec(src, "model.audio_projector.linear_2.bias"); err != nil {
		return nil, err
	}
	if len(at.proj1B) != ac.TextHiddenSize || len(at.proj2B) != ac.TextHiddenSize {
		return nil, fmt.Errorf("audio_projector biases: %d/%d entries, want %d", len(at.proj1B), len(at.proj2B), ac.TextHiddenSize)
	}

	// Full 1D RoPE inverse frequencies over the head dim (Lfm2RotaryEmbedding
	// default rope: 1 / theta^(2i/headDim), i in [0, headDim/2)).
	at.invFreq = make([]float64, ac.HeadDim/2)
	for i := range at.invFreq {
		at.invFreq[i] = 1.0 / math.Pow(ac.RopeTheta, float64(2*i)/float64(ac.HeadDim))
	}
	return at, nil
}

// Encode runs the downsampler + LFM2 stack + projector over one log-mel matrix
// per clip (logMels[clip][melBin][frame], as produced by
// audio.FeatureExtractor.LogMel). It returns per clip one projected embedding
// row per audio token: [clip][token][TextHiddenSize].
func (a *AudioTower) Encode(logMels [][][]float64) ([][][]float32, error) {
	ac := a.cfg
	if ac.NumMelBins <= 0 || ac.HiddenSize <= 0 {
		return nil, fmt.Errorf("audio: invalid audio config")
	}
	out := make([][][]float32, len(logMels))
	for ci, mel := range logMels {
		if len(mel) != ac.NumMelBins {
			return nil, fmt.Errorf("audio: clip %d has %d mel bins, want %d", ci, len(mel), ac.NumMelBins)
		}
		frames := 0
		if len(mel) > 0 {
			frames = len(mel[0])
		}
		for _, row := range mel {
			if len(row) != frames {
				return nil, fmt.Errorf("audio: clip %d has ragged mel rows", ci)
			}
		}
		seq, err := a.downsample(mel, frames)
		if err != nil {
			return nil, fmt.Errorf("audio: clip %d downsample: %w", ci, err)
		}
		encoded := a.lfm(seq)
		want := ac.NumEncoderTokens(frames)
		if len(encoded) != want {
			return nil, fmt.Errorf("audio: clip %d encoded %d tokens, want ceil(%d/%d)=%d", ci, len(encoded), frames, ac.FeatureExtractor.EncoderStride, want)
		}
		rows := make([][]float32, len(encoded))
		for t := range encoded {
			rows[t] = a.project(encoded[t])
		}
		out[ci] = rows
	}
	return out, nil
}

// downsample runs the HF Lfm3OmniDownsampler over one clip's log-mel:
// three stride-2 3x3 convs over the (mel, time) plane (128 bins and 8 frames
// become one vector) with SiLU after each conv/lin except the final lin_out.
// Input mel is [NumMelBins][frames]; the result is [ceil(frames/8)][hidden].
func (a *AudioTower) downsample(mel [][]float64, frames int) ([][]float32, error) {
	ac := a.cfg
	C := ac.ConvChannels
	if frames <= 0 {
		return nil, fmt.Errorf("empty mel clip")
	}
	// x: [1][NumMelBins][frames] laid out channel-major [c][h][w], w fastest.
	x := make([]float32, ac.NumMelBins*frames)
	for m := range ac.NumMelBins {
		for t := range frames {
			x[m*frames+t] = float32(mel[m][t])
		}
	}
	h, w := ac.NumMelBins, frames
	x = conv2dS2P1K3(x, 1, h, w, a.convInW, 1, C, a.convInB)
	siluInPlace(x)
	h, w = (h+1)/2, (w+1)/2
	x = conv2dS2P1K3(x, C, h, w, a.convMidW, C, C, a.convMidB) // depthwise
	h, w = (h+1)/2, (w+1)/2
	x = lin1x1(x, C, h, w, a.linMid, a.linMidB)
	siluInPlace(x)
	x = conv2dS2P1K3(x, C, h, w, a.convEndW, C, C, a.convEndB) // depthwise
	h, w = (h+1)/2, (w+1)/2
	x = lin1x1(x, C, h, w, a.linEnd, a.linEndB)
	siluInPlace(x)

	// lin_out: Conv1d(k=1) over flatten(1,2) of [C][h][w]: the flattened input
	// index is c*h + hh (hh fastest), and w stays the conv1d length dim.
	seq := make([][]float32, w)
	vec := make([]float32, C*h)
	tmp := make([]float32, ac.HiddenSize)
	for t := 0; t < w; t++ {
		for c := range C {
			base := c * h * w
			for hh := range h {
				vec[c*h+hh] = x[base+hh*w+t]
			}
		}
		MatVec(tmp, a.linOut, vec)
		row := make([]float32, ac.HiddenSize)
		for o := range tmp {
			row[o] = tmp[o] + a.linOutB[o]
		}
		seq[t] = row
	}
	return seq, nil
}

// conv2dS2P1K3 runs a 2D convolution with stride 2, padding 1, 3x3 kernel over
// in [cIn][H][W] (channel-major, W fastest) producing [cOut][ceil(H/2)][ceil(W/2)].
// groups == cIn means depthwise (weight per channel). Weight layout is
// [cOut][cIn/groups][3][3].
func conv2dS2P1K3(in []float32, cIn, H, W int, weight []float32, groups, cOut int, bias []float32) []float32 {
	g := cIn / groups
	hOut := (H + 1) / 2
	wOut := (W + 1) / 2
	out := make([]float32, cOut*hOut*wOut)
	og := cOut / groups
	for co := range cOut {
		ci0 := (co / og) * g
		wbase := co * g * 9
		for oy := range hOut {
			for ox := range wOut {
				var acc float32
				if bias != nil {
					acc = bias[co]
				}
				for ci := range g {
					c := ci0 + ci
					wb := wbase + ci*9
					for ky := range 3 {
						iy := oy*2 + ky - 1
						if iy < 0 || iy >= H {
							continue
						}
						for kx := range 3 {
							ix := ox*2 + kx - 1
							if ix < 0 || ix >= W {
								continue
							}
							acc += weight[wb+ky*3+kx] * in[c*H*W+iy*W+ix]
						}
					}
				}
				out[co*hOut*wOut+oy*wOut+ox] = acc
			}
		}
	}
	return out
}

// lin1x1 applies a 1x1 conv (pointwise linear) over in [C][H][W] (channel-
// major, W fastest) with weight mat [C][C] and bias.
func lin1x1(in []float32, C, H, W int, w *core.Mat, bias []float32) []float32 {
	out := make([]float32, len(in))
	x := make([]float32, C)
	tmp := make([]float32, C)
	pixels := H * W
	for p := range pixels {
		for c := range C {
			x[c] = in[c*pixels+p]
		}
		MatVec(tmp, w, x)
		for c := range C {
			out[c*pixels+p] = tmp[c] + bias[c]
		}
	}
	return out
}

func siluInPlace(x []float32) {
	for i, v := range x {
		x[i] = v / (1 + float32(math.Exp(-float64(v))))
	}
}

// lfm runs the LFM2 stack over the downsampled sequence: per layer a
// pre-norm operator (conv or causal sliding-window attention) plus a pre-norm
// SwiGLU FFN, then the final embedding_norm (Lfm2Model applies it after all
// layers). seq is [tokens][hidden] and is returned as a new slice.
func (a *AudioTower) lfm(seq [][]float32) [][]float32 {
	ac := a.cfg
	hidden := ac.HiddenSize
	n := len(seq)
	h := make([]float32, n*hidden)
	for t := range seq {
		copy(h[t*hidden:(t+1)*hidden], seq[t])
	}
	normed := make([]float32, hidden)
	eps := float32(ac.NormEps)
	for li := range a.layers {
		l := &a.layers[li]
		if l.isAttn {
			a.attnLayer(l, h, n, normed, eps)
		} else {
			a.convLayer(l, h, n, normed, eps)
		}
		a.mlp(l, h, n, normed, eps)
	}
	out := make([][]float32, n)
	for t := range n {
		row := make([]float32, hidden)
		RMSNorm(row, h[t*hidden:(t+1)*hidden], a.embedNorm, eps)
		out[t] = row
	}
	return out
}

// attnLayer runs one causal sliding-window full-attention layer over the whole
// sequence (HF create_sliding_window_causal_mask: query t attends keys
// [t-window+1, t]). Q/K get a per-head-dim RMSNorm and full 1D RoPE.
func (a *AudioTower) attnLayer(l *audioLayer, h []float32, n int, normed []float32, eps float32) {
	ac := a.cfg
	hidden := ac.HiddenSize
	heads := ac.NumHeads
	kvHeads := ac.NumKVHeads
	headDim := ac.HeadDim
	groups := heads / kvHeads
	window := ac.SlidingWindow

	q := make([]float32, n*heads*headDim)
	k := make([]float32, n*kvHeads*headDim)
	v := make([]float32, n*kvHeads*headDim)
	for t := range n {
		RMSNorm(normed, h[t*hidden:(t+1)*hidden], l.opNorm, eps)
		MatVec(q[t*heads*headDim:(t+1)*heads*headDim], l.q, normed)
		MatVec(k[t*kvHeads*headDim:(t+1)*kvHeads*headDim], l.k, normed)
		MatVec(v[t*kvHeads*headDim:(t+1)*kvHeads*headDim], l.v, normed)
	}
	// Per-head-dim RMSNorm on q and k, then full RoPE (rotate_half pairing).
	half := headDim / 2
	for t := range n {
		for hi := range heads {
			base := t*heads*headDim + hi*headDim
			RMSNorm(q[base:base+headDim], q[base:base+headDim], l.qNorm, eps)
		}
		for hi := range kvHeads {
			base := t*kvHeads*headDim + hi*headDim
			RMSNorm(k[base:base+headDim], k[base:base+headDim], l.kNorm, eps)
		}
		for hi := range heads {
			base := t*heads*headDim + hi*headDim
			for d := range half {
				ang := float64(t) * a.invFreq[d]
				c := float32(math.Cos(ang))
				s := float32(math.Sin(ang))
				q1, q2 := q[base+d], q[base+half+d]
				q[base+d] = q1*c - q2*s
				q[base+half+d] = q2*c + q1*s
			}
		}
		for hi := range kvHeads {
			base := t*kvHeads*headDim + hi*headDim
			for d := range half {
				ang := float64(t) * a.invFreq[d]
				c := float32(math.Cos(ang))
				s := float32(math.Sin(ang))
				k1, k2 := k[base+d], k[base+half+d]
				k[base+d] = k1*c - k2*s
				k[base+half+d] = k2*c + k1*s
			}
		}
	}

	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	attnOut := make([]float32, n*heads*headDim)
	scores := make([]float32, min(window, n))
	for t := range n {
		lo := max(t-window+1, 0)
		length := t - lo + 1
		row := scores[:length]
		for hi := range heads {
			kvh := hi / groups
			qBase := t*heads*headDim + hi*headDim
			for i, j := 0, lo; j <= t; i, j = i+1, j+1 {
				kBase := j*kvHeads*headDim + kvh*headDim
				var dot float32
				for d := range headDim {
					dot += q[qBase+d] * k[kBase+d]
				}
				row[i] = dot * scale
			}
			core.Softmax(row)
			outBase := t*heads*headDim + hi*headDim
			for i, j := 0, lo; j <= t; i, j = i+1, j+1 {
				vBase := j*kvHeads*headDim + kvh*headDim
				w := row[i]
				for d := range headDim {
					attnOut[outBase+d] += w * v[vBase+d]
				}
			}
		}
	}
	res := make([]float32, hidden)
	for t := range n {
		MatVec(res, l.o, attnOut[t*heads*headDim:(t+1)*heads*headDim])
		for d := range hidden {
			h[t*hidden+d] += res[d]
		}
	}
}

// convLayer runs one LFM2ShortConv layer: in_proj -> chunks (B, C, x) ->
// gated B*x -> causal depthwise conv (kernel ConvKernelSize, left zero pad)
// -> *C -> out_proj, with a pre-norm on the input.
func (a *AudioTower) convLayer(l *audioLayer, h []float32, n int, normed []float32, eps float32) {
	ac := a.cfg
	hidden := ac.HiddenSize
	kSize := ac.ConvKernelSize
	pad := kSize - 1

	in := make([]float32, n*3*hidden)
	for t := range n {
		RMSNorm(normed, h[t*hidden:(t+1)*hidden], l.opNorm, eps)
		MatVec(in[t*3*hidden:(t+1)*3*hidden], l.convIn, normed)
	}
	// gated = B * x (chunks 0 and 2 of the in_proj output).
	gated := make([]float32, n*hidden)
	for t := range n {
		bBase := t * 3 * hidden
		xBase := bBase + 2*hidden
		for c := range hidden {
			gated[t*hidden+c] = in[bBase+c] * in[xBase+c]
		}
	}
	// causal depthwise conv (left zero-pad), then multiply by the C chunk.
	convTok := make([]float32, n*hidden)
	for t := range n {
		cBase := t*3*hidden + hidden
		for c := range hidden {
			var acc float32
			for kk := range kSize {
				src := t + kk - pad
				if src < 0 {
					continue
				}
				acc += l.convW[c*kSize+kk] * gated[src*hidden+c]
			}
			convTok[t*hidden+c] = acc * in[cBase+c]
		}
	}
	res := make([]float32, hidden)
	for t := range n {
		MatVec(res, l.convOut, convTok[t*hidden:(t+1)*hidden])
		for d := range hidden {
			h[t*hidden+d] += res[d]
		}
	}
}

// mlp runs the pre-norm SwiGLU FFN of one layer: w2(silu(w1(x)) * w3(x)).
func (a *AudioTower) mlp(l *audioLayer, h []float32, n int, normed []float32, eps float32) {
	hidden := a.cfg.HiddenSize
	ffn := a.cfg.FFNDim
	up := make([]float32, ffn)
	gate := make([]float32, ffn)
	down := make([]float32, hidden)
	for t := range n {
		RMSNorm(normed, h[t*hidden:(t+1)*hidden], l.ffnNorm, eps)
		MatVec(up, l.w1, normed)
		MatVec(gate, l.w3, normed)
		for i := range up {
			s := up[i]
			up[i] = gate[i] * (s / (1 + float32(math.Exp(-float64(s)))))
		}
		MatVec(down, l.w2, up)
		for d := range hidden {
			h[t*hidden+d] += down[d]
		}
	}
}

// project maps one encoded tower token through the audio projector:
// LayerNorm -> linear_1 -> GELU -> linear_2 (all with biases).
func (a *AudioTower) project(x []float32) []float32 {
	ac := a.cfg
	hidden := ac.HiddenSize
	outDim := ac.TextHiddenSize
	normed := make([]float32, hidden)
	layerNorm(x, normed, a.projNormW, a.projNormB, 1e-5) // nn.LayerNorm default eps
	tmp := make([]float32, outDim)
	out := make([]float32, outDim)
	MatVec(tmp, a.proj1, normed)
	for i := range tmp {
		tmp[i] = geluExact(tmp[i] + a.proj1B[i])
	}
	MatVec(out, a.proj2, tmp)
	for i := range out {
		out[i] += a.proj2B[i]
	}
	return out
}

// AudioEncoder implements core.AudioEncoderProvider for the SIMD instance.
func (m *Instance) AudioEncoder() core.AudioEncoder { return m.Audio }
