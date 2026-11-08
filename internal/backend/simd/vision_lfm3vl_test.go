package simd

import (
	"math"
	"math/rand"
	"testing"

	"github.com/samcharles93/mantle/internal/backend/core"
)

// TestVisionTowerSynthetic runs a tiny randomly-initialized LFM3-VL-style
// vision tower end to end and checks the output contract: one row per image
// token, finite values, and determinism.
func TestVisionTowerSynthetic(t *testing.T) {
	const (
		hidden   = 32
		heads    = 4
		kvHeads  = 2
		headDim  = hidden / heads
		patch    = 4
		merge    = 2
		temporal = 2
		inCh     = 3
		interm   = 48
		layers   = 4
		down     = 2
		textH    = 24
	)
	vc := &core.VisionConfig{
		HiddenSize:            hidden,
		InChannels:            inCh,
		PatchSize:             patch,
		ConvKernelSize:        3,
		IntermediateSize:      interm,
		NumLayers:             layers,
		NumHeads:              heads,
		NumKVHeads:            kvHeads,
		HeadDim:               headDim,
		LayerTypes:            []string{"conv", "full_attention", "conv", "full_attention"},
		NormEps:               1e-5,
		MergeSize:             merge,
		TemporalPatch:         temporal,
		RopeTheta:             1e6,
		DownsampleFactor:      down,
		ProjectorHiddenSize:   64,
		ProjectorHiddenAct:    "gelu",
		ProjectorLayerNormEps: 1e-6,
		ProjectorBias:         true,
		TextHiddenSize:        textH,
	}
	rng := rand.New(rand.NewSource(42))
	randMat := func(r, c int) *core.Mat {
		m := core.NewMat(r, c)
		for i := range m.Data {
			m.Data[i] = float32(rng.NormFloat64() * 0.1)
		}
		return &m
	}
	randVec := func(n int) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = float32(1 + rng.NormFloat64()*0.1)
		}
		return v
	}
	vt := &VisionTower{cfg: vc}
	vt.Patchifier = randMat(hidden, vc.PatchDim())
	vt.EmbedNorm = randVec(hidden)
	for i := range layers {
		l := visionLayer{
			IsAttn:  vc.LayerTypes[i] == "full_attention",
			OpNorm:  randVec(hidden),
			FfnNorm: randVec(hidden),
			W1:      randMat(interm, hidden),
			W2:      randMat(hidden, interm),
			W3:      randMat(interm, hidden),
		}
		if l.IsAttn {
			l.Q = randMat(heads*headDim, hidden)
			l.K = randMat(kvHeads*headDim, hidden)
			l.V = randMat(kvHeads*headDim, hidden)
			l.O = randMat(hidden, heads*headDim)
			l.QNorm = randVec(headDim)
			l.KNorm = randVec(headDim)
		} else {
			l.ConvIn = randMat(3*hidden, hidden)
			l.ConvKernel = make([]float32, hidden*9)
			for j := range l.ConvKernel {
				l.ConvKernel[j] = float32(rng.NormFloat64() * 0.1)
			}
			l.ConvOut = randMat(hidden, hidden)
		}
		vt.Layers = append(vt.Layers, l)
	}
	vt.ProjNormW = randVec(hidden)
	vt.ProjNormB = randVec(hidden)
	vt.Proj1 = randMat(64, hidden*down*down)
	vt.Proj1Bias = randVec(64)
	vt.Proj2 = randMat(textH, 64)
	vt.Proj2Bias = randVec(textH)
	spatialDim := headDim / 2
	vt.invFreq = make([]float64, spatialDim/2)
	for i := range vt.invFreq {
		vt.invFreq[i] = 1.0 / math.Pow(vc.RopeTheta, float64(2*i)/float64(spatialDim))
	}

	// grid (1, 4, 4) patches with merge 2 -> 16/4 = 4 image tokens
	grid := [3]int{1, 4, 4}
	nPatches := 16
	pix := make([]float32, nPatches*vc.PatchDim())
	for i := range pix {
		pix[i] = float32(rng.NormFloat64())
	}
	out1, err := vt.Encode(pix, [][3]int{grid})
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	if len(out1) != 1 || len(out1[0]) != 4 || len(out1[0][0]) != textH {
		t.Fatalf("bad output shape: %d images, %d rows, %d dims", len(out1), len(out1[0]), len(out1[0][0]))
	}
	for _, v := range out1[0][0] {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("non-finite feature value %g", v)
		}
	}
	out2, err := vt.Encode(pix, [][3]int{grid})
	if err != nil {
		t.Fatalf("Encode(2): %v", err)
	}
	for i := range out1[0][0] {
		if out1[0][0][i] != out2[0][0][i] {
			t.Fatalf("non-deterministic output at %d: %g vs %g", i, out1[0][0][i], out2[0][0][i])
		}
	}
}

// TestSmartResize checks the HF smart_resize rounding rules.
func TestSmartResize(t *testing.T) {
	cases := []struct{ h, w, factor, minP, maxP, wantH, wantW int }{
		{256, 256, 32, 65536, 16777216, 256, 256},   // in range, divisible: unchanged
		{100, 70, 32, 65536, 16777216, 320, 224},    // upscale to min_pixels (h/w swapped by caller dims)
		{1000, 1000, 32, 65536, 16777216, 992, 992}, // downscale to max_pixels
		{33, 33, 32, 65536, 16777216, 256, 256},     // upscale small
	}
	for _, c := range cases {
		// note: smartResize takes (h, w); case 2 is h=100, w=70
		h, w := smartResize(c.h, c.w, c.factor, c.minP, c.maxP)
		if h != c.wantH || w != c.wantW {
			t.Errorf("smartResize(%d,%d) = %d,%d, want %d,%d", c.h, c.w, h, w, c.wantH, c.wantW)
		}
	}
}
