package simd

import (
	"testing"

	core "github.com/samcharles93/mantle/internal/backend/core"
)

type gemma4PerLayerDeviceOps struct {
	DefaultOps

	beginTokenCalled    bool
	endTokenCalled      bool
	deviceMatVecCalls   int
	deviceMatVecSawInit bool
	batchedNormCalls    int
	matVecCalls         int
	projOut             []float32
	deviceScaleCalls    int
}

func (o *gemma4PerLayerDeviceOps) BeginToken(_ []float32) {
	o.beginTokenCalled = true
}

func (o *gemma4PerLayerDeviceOps) EndToken(_ []float32) {
	o.endTokenCalled = true
}

func (o *gemma4PerLayerDeviceOps) HostStateDirty(_ []float32) {}

func (o *gemma4PerLayerDeviceOps) SyncHostState(_ []float32) {}

func (o *gemma4PerLayerDeviceOps) DeviceAdd(_, _ []float32) bool { return false }

func (o *gemma4PerLayerDeviceOps) DeviceRMSNorm(_, _, _ []float32, _ float32) bool { return false }

func (o *gemma4PerLayerDeviceOps) RMSNormBatched(dst, src, weight []float32, eps float32, headDim, nHeads int) bool {
	o.batchedNormCalls++
	for head := range nHeads {
		start := head * headDim
		end := start + headDim
		rmsNormWeighted(dst[start:end], src[start:end], weight[:headDim], eps)
	}
	return true
}

func (o *gemma4PerLayerDeviceOps) DeviceMatVec(dst []float32, _ *Mat, _ []float32) bool {
	o.deviceMatVecCalls++
	o.deviceMatVecSawInit = o.beginTokenCalled
	copy(dst, o.projOut)
	return true
}

func (o *gemma4PerLayerDeviceOps) DeviceMatVecNoCopy(_ *Mat, _ []float32) bool { return false }

func (o *gemma4PerLayerDeviceOps) DeviceArgMaxLastResult() (int, bool) { return 0, false }

func (o *gemma4PerLayerDeviceOps) DeviceLogitSoftcap(_ []float32, _ float32) bool { return false }

func (o *gemma4PerLayerDeviceOps) DeviceScaleBF16InPlace(x []float32, scale float32) bool {
	o.deviceScaleCalls++
	for i := range x {
		x[i] = roundBF16Value(x[i] * scale)
	}
	return true
}

func (o *gemma4PerLayerDeviceOps) MatVec(dst []float32, _ *Mat, _ []float32) {
	o.matVecCalls++
	copy(dst, o.projOut)
}

func TestPrepareTokenRuntimeStateUsesDeviceProjectionAfterBeginToken(t *testing.T) {
	projOut := []float32{1, 2, 3, 4}
	perLayerEmb := core.NewMatFromData(1, 4, []float32{10, 20, 30, 40})
	projMat := core.NewMat(4, 2)
	tokEmb := core.NewMatFromData(1, 2, []float32{0.25, -0.5})

	ops := &gemma4PerLayerDeviceOps{projOut: projOut}
	m := &Instance{
		Config:     &ModelConfig{Config: Config{VocabSize: 1}},
		Embeddings: &tokEmb,
		Gemma4PerLayer: &Gemma4PerLayerInputModel{
			Embeddings:      &perLayerEmb,
			Projection:      &projMat,
			ProjectionNorm:  []float32{1, 1},
			HiddenSize:      2,
			LayerCount:      2,
			EmbeddingScale:  1,
			ProjectionScale: 1,
			InputScale:      1,
		},
		MaxContext: 8,
		RMSEpsilon: 1e-6,
		Scratch: ScratchBuffers{
			X:             make([]float32, 2),
			PerLayerTok:   make([]float32, 4),
			PerLayerProj:  make([]float32, 4),
			PerLayerInput: make([]float32, 4),
		},
	}
	m.SetOps(ops)

	state, cleanup, err := prepareTokenRuntimeState(m, 0)
	if err != nil {
		t.Fatalf("prepareTokenRuntimeState returned error: %v", err)
	}
	defer cleanup()

	if !ops.beginTokenCalled {
		t.Fatal("BeginToken was not called")
	}
	if !ops.deviceMatVecSawInit {
		t.Fatal("DeviceMatVec ran before BeginToken initialized device state")
	}
	if ops.deviceMatVecCalls != 1 {
		t.Fatalf("DeviceMatVec calls = %d, want 1", ops.deviceMatVecCalls)
	}
	if ops.matVecCalls != 0 {
		t.Fatalf("MatVec fallback calls = %d, want 0", ops.matVecCalls)
	}
	if ops.batchedNormCalls != 1 {
		t.Fatalf("RMSNormBatched calls = %d, want 1", ops.batchedNormCalls)
	}

	want := make([]float32, 4)
	rawTok := []float32{10, 20, 30, 40}
	for layerIdx := range 2 {
		start := layerIdx * 2
		end := start + 2
		rmsNormWeighted(want[start:end], projOut[start:end], []float32{1, 1}, m.RMSEpsilon)
		Add(want[start:end], rawTok[start:end])
	}

	if len(state.perLayerInputs) != len(want) {
		t.Fatalf("perLayerInputs len = %d, want %d", len(state.perLayerInputs), len(want))
	}
	for i := range want {
		if got := state.perLayerInputs[i]; got != want[i] {
			t.Fatalf("perLayerInputs[%d] = %v, want %v", i, got, want[i])
		}
	}
}

func TestApplyRoundedLayerScaleUsesDeviceFastPath(t *testing.T) {
	ops := &gemma4PerLayerDeviceOps{}
	m := &Instance{}
	m.SetOps(ops)
	layer := &Layer{LayerScale: 0.5}
	x := []float32{1.1, -2.2, 3.3}

	if err := applyRoundedLayerScale(m, layer, x, nil); err != nil {
		t.Fatalf("applyRoundedLayerScale returned error: %v", err)
	}
	if ops.deviceScaleCalls != 1 {
		t.Fatalf("DeviceScaleBF16InPlace calls = %d, want 1", ops.deviceScaleCalls)
	}
	want := []float32{
		roundBF16Value(1.1 * 0.5),
		roundBF16Value(-2.2 * 0.5),
		roundBF16Value(3.3 * 0.5),
	}
	for i := range want {
		if x[i] != want[i] {
			t.Fatalf("x[%d] = %v, want %v", i, x[i], want[i])
		}
	}
}
