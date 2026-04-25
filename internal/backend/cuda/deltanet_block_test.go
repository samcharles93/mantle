//go:build cuda

package cuda

import (
	"math"
	"math/rand"
	"testing"

	model "github.com/samcharles93/mantle/internal/backend/core"
	"github.com/samcharles93/mantle/internal/backend/cuda/native"
	"github.com/samcharles93/mantle/pkg/mcf"
)

func TestDeltaNetBlockParity(t *testing.T) {
	count, err := native.DeviceCount()
	if err != nil {
		t.Fatalf("DeviceCount: %v", err)
	}
	if count < 1 {
		t.Skip("no cuda device available")
	}

	cases := []struct {
		name  string
		quant bool
	}{
		{name: "quant_k4", quant: true},
		{name: "f32", quant: false},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			runDeltaNetBlockParity(t, tc.quant)
		})
	}
}

func runDeltaNetBlockParity(t *testing.T, quant bool) {
	t.Helper()

	stream, err := native.NewStream()
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	defer func() {
		if err := stream.Destroy(); err != nil {
			t.Fatalf("stream destroy: %v", err)
		}
	}()

	blas, err := native.NewBlasHandle(stream)
	if err != nil {
		t.Fatalf("NewBlasHandle: %v", err)
	}
	defer func() {
		if err := blas.Destroy(); err != nil {
			t.Fatalf("blas destroy: %v", err)
		}
	}()

	ops := NewOps(stream, blas)
	defer func() {
		if err := ops.Close(); err != nil {
			t.Fatalf("ops close: %v", err)
		}
	}()

	rng := rand.New(rand.NewSource(42))
	const (
		embd         = 32
		numKeyHeads  = 2
		numValHeads  = 4
		headKeyDim   = 16
		headValueDim = 16
		convKernel   = 4
	)
	keyDim := numKeyHeads * headKeyDim
	valueDim := numValHeads * headValueDim
	convChannels := 2*keyDim + valueDim

	makeQuantMat := func(rows, cols int, scale float32) *model.Mat {
		qvals := make([]int8, rows*((cols+31)/32)*32)
		for i := range qvals {
			qvals[i] = int8(rng.Intn(15) - 7)
		}
		payload := buildK4Payload(rows, cols, scale, qvals)
		mat := &model.Mat{R: rows, C: cols, Stride: cols, DType: mcf.DTypeK4, Raw: payload}
		cache, err := model.BuildQuantCache(mat)
		if err != nil {
			t.Fatalf("BuildQuantCache [%d x %d]: %v", rows, cols, err)
		}
		mat.Quant = cache
		return mat
	}

	makeF32Mat := func(rows, cols int, scale float32) *model.Mat {
		data := make([]float32, rows*cols)
		for i := range data {
			data[i] = (rng.Float32()*2 - 1) * scale
		}
		return &model.Mat{R: rows, C: cols, Stride: cols, DType: mcf.DTypeF32, Data: data}
	}

	makeVec := func(n int, scale float32) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = (rng.Float32()*2 - 1) * scale
		}
		return v
	}

	convData := make([]float32, convChannels*convKernel)
	for i := range convData {
		convData[i] = (rng.Float32()*2 - 1) * 0.08
	}

	var qkvProj, aProj, bProj, zProj, outProj *model.Mat
	if quant {
		qkvProj = makeQuantMat(convChannels, embd, 0.05)
		aProj = makeQuantMat(numValHeads, embd, 0.05)
		bProj = makeQuantMat(numValHeads, embd, 0.05)
		zProj = makeQuantMat(valueDim, embd, 0.05)
		outProj = makeQuantMat(embd, valueDim, 0.05)
	} else {
		qkvProj = makeF32Mat(convChannels, embd, 0.05)
		aProj = makeF32Mat(numValHeads, embd, 0.05)
		bProj = makeF32Mat(numValHeads, embd, 0.05)
		zProj = makeF32Mat(valueDim, embd, 0.05)
		outProj = makeF32Mat(embd, valueDim, 0.05)
	}

	dl := &model.DeltaNetLayer{
		QKVProj:        qkvProj,
		AProj:          aProj,
		BProj:          bProj,
		ZProj:          zProj,
		OutProj:        outProj,
		Conv:           &model.Mat{R: convChannels, C: convKernel, Stride: convKernel, DType: mcf.DTypeF32, Data: convData},
		Norm:           makeVec(headValueDim, 0.2),
		ALog:           makeVec(numValHeads, 0.2),
		DTBias:         makeVec(numValHeads, 0.1),
		NumKeyHeads:    numKeyHeads,
		NumValueHeads:  numValHeads,
		HeadKeyDim:     headKeyDim,
		HeadValueDim:   headValueDim,
		KeyDim:         keyDim,
		ValueDim:       valueDim,
		ConvState:      make([]float32, convChannels*(convKernel-1)),
		RecurrentState: make([]float32, numValHeads*headKeyDim*headValueDim),
	}
	x := makeVec(embd, 0.25)
	cfg := model.DeltaNetConfig{RMSEpsilon: 1e-6}

	refLayer := cloneDeltaNetLayer(dl)
	want := deltaNetBlockRef(refLayer, x, cfg)
	if len(want) != embd {
		t.Fatalf("reference output len = %d, want %d", len(want), embd)
	}

	got := make([]float32, embd)
	if ok := ops.DeltaNetBlock(dl, x, got, cfg); !ok {
		t.Fatalf("DeltaNetBlock returned false: %v", ops.ConsumeFastPathError())
	}
	if err := ops.FlushBlockResult(); err != nil {
		t.Fatalf("FlushBlockResult: %v", err)
	}

	for i := range want {
		if !approxEqualTol(want[i], got[i], 1e-4, 1e-4) {
			t.Fatalf("out[%d] = %g, want %g", i, got[i], want[i])
		}
	}
}

func cloneDeltaNetLayer(src *model.DeltaNetLayer) *model.DeltaNetLayer {
	if src == nil {
		return nil
	}
	cloneMat := func(m *model.Mat) *model.Mat {
		if m == nil {
			return nil
		}
		cp := *m
		if len(m.Data) > 0 {
			cp.Data = append([]float32(nil), m.Data...)
		}
		if len(m.Raw) > 0 {
			cp.Raw = append([]byte(nil), m.Raw...)
		}
		cp.Quant = m.Quant
		return &cp
	}
	return &model.DeltaNetLayer{
		QKVProj:        cloneMat(src.QKVProj),
		AProj:          cloneMat(src.AProj),
		BProj:          cloneMat(src.BProj),
		ZProj:          cloneMat(src.ZProj),
		OutProj:        cloneMat(src.OutProj),
		Conv:           cloneMat(src.Conv),
		Norm:           append([]float32(nil), src.Norm...),
		ALog:           append([]float32(nil), src.ALog...),
		DTBias:         append([]float32(nil), src.DTBias...),
		NumKeyHeads:    src.NumKeyHeads,
		NumValueHeads:  src.NumValueHeads,
		HeadKeyDim:     src.HeadKeyDim,
		HeadValueDim:   src.HeadValueDim,
		KeyDim:         src.KeyDim,
		ValueDim:       src.ValueDim,
		ConvState:      append([]float32(nil), src.ConvState...),
		RecurrentState: append([]float32(nil), src.RecurrentState...),
	}
}

func deltaNetBlockRef(dl *model.DeltaNetLayer, x []float32, cfg model.DeltaNetConfig) []float32 {
	convR := dl.Conv.R
	mixed := make([]float32, convR)
	matVecRef(mixed, dl.QKVProj, x)

	aRaw := make([]float32, dl.NumValueHeads)
	bRaw := make([]float32, dl.NumValueHeads)
	zRaw := make([]float32, dl.ValueDim)
	matVecRef(aRaw, dl.AProj, x)
	matVecRef(bRaw, dl.BProj, x)
	matVecRef(zRaw, dl.ZProj, x)

	convOut := make([]float32, convR)
	mambaDepthwiseConvRef(convOut, mixed, dl.Conv, nil, dl.ConvState)
	for i := range convOut {
		convOut[i] = siluRef(convOut[i])
	}

	qBuf := make([]float32, dl.KeyDim)
	kBuf := make([]float32, dl.KeyDim)
	vBuf := make([]float32, dl.ValueDim)
	copy(qBuf, convOut[:dl.KeyDim])
	copy(kBuf, convOut[dl.KeyDim:2*dl.KeyDim])
	copy(vBuf, convOut[2*dl.KeyDim:2*dl.KeyDim+dl.ValueDim])

	out := make([]float32, dl.ValueDim)
	groupSize := 1
	if dl.NumKeyHeads > 0 && dl.NumValueHeads > dl.NumKeyHeads {
		groupSize = dl.NumValueHeads / dl.NumKeyHeads
	}
	scale := float32(1.0 / math.Sqrt(float64(dl.HeadKeyDim)))
	eps := float32(1e-6)

	for hv := 0; hv < dl.NumValueHeads; hv++ {
		hk := hv
		if groupSize > 1 {
			hk = hv / groupSize
		}
		qHead := append([]float32(nil), qBuf[hk*dl.HeadKeyDim:(hk+1)*dl.HeadKeyDim]...)
		kHead := append([]float32(nil), kBuf[hk*dl.HeadKeyDim:(hk+1)*dl.HeadKeyDim]...)
		vHead := vBuf[hv*dl.HeadValueDim : (hv+1)*dl.HeadValueDim]
		zHead := zRaw[hv*dl.HeadValueDim : (hv+1)*dl.HeadValueDim]
		outHead := out[hv*dl.HeadValueDim : (hv+1)*dl.HeadValueDim]
		state := dl.RecurrentState[hv*dl.HeadKeyDim*dl.HeadValueDim : (hv+1)*dl.HeadKeyDim*dl.HeadValueDim]

		deltaNetL2NormRef(qHead, eps)
		deltaNetL2NormRef(kHead, eps)
		g := -float32(math.Exp(float64(dl.ALog[hv]))) * softplusRef(aRaw[hv]+dl.DTBias[hv])
		beta := sigmoidRef(bRaw[hv])
		decay := float32(math.Exp(float64(g)))

		for i := range state {
			state[i] *= decay
		}

		delta := make([]float32, dl.HeadValueDim)
		for v := 0; v < dl.HeadValueDim; v++ {
			var kvMem float32
			for k := 0; k < dl.HeadKeyDim; k++ {
				kvMem += state[k*dl.HeadValueDim+v] * kHead[k]
			}
			delta[v] = (vHead[v] - kvMem) * beta
		}

		for k := 0; k < dl.HeadKeyDim; k++ {
			base := k * dl.HeadValueDim
			kk := kHead[k]
			for v := 0; v < dl.HeadValueDim; v++ {
				state[base+v] += kk * delta[v]
			}
		}

		for v := 0; v < dl.HeadValueDim; v++ {
			var sum float32
			for k := 0; k < dl.HeadKeyDim; k++ {
				sum += state[k*dl.HeadValueDim+v] * (qHead[k] * scale)
			}
			outHead[v] = sum
		}

		rmsNormGatedRef(outHead, outHead, zHead, dl.Norm, cfg.RMSEpsilon, true)
	}

	final := make([]float32, dl.OutProj.R)
	matVecRef(final, dl.OutProj, out)
	return final
}

func deltaNetL2NormRef(x []float32, eps float32) {
	var sum float32
	for _, v := range x {
		sum += v * v
	}
	scale := float32(1.0 / math.Sqrt(float64(sum+eps)))
	for i := range x {
		x[i] *= scale
	}
}

func sigmoidRef(x float32) float32 {
	return float32(1.0 / (1.0 + math.Exp(float64(-x))))
}

func TestDeltaNetBlockFastPathOptOut(t *testing.T) {
	count, err := native.DeviceCount()
	if err != nil {
		t.Fatalf("DeviceCount: %v", err)
	}
	if count < 1 {
		t.Skip("no cuda device available")
	}

	t.Setenv("MANTLE_CUDA_DELTANET", "0")

	stream, err := native.NewStream()
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	defer func() {
		if err := stream.Destroy(); err != nil {
			t.Fatalf("stream destroy: %v", err)
		}
	}()

	blas, err := native.NewBlasHandle(stream)
	if err != nil {
		t.Fatalf("NewBlasHandle: %v", err)
	}
	defer func() {
		if err := blas.Destroy(); err != nil {
			t.Fatalf("blas destroy: %v", err)
		}
	}()

	ops := NewOps(stream, blas)
	defer func() {
		if err := ops.Close(); err != nil {
			t.Fatalf("ops close: %v", err)
		}
	}()

	const (
		embd         = 16
		numKeyHeads  = 1
		numValHeads  = 2
		headKeyDim   = 8
		headValueDim = 8
		convKernel   = 4
	)
	keyDim := numKeyHeads * headKeyDim
	valueDim := numValHeads * headValueDim
	convChannels := 2*keyDim + valueDim

	dl := &model.DeltaNetLayer{
		QKVProj:        &model.Mat{R: convChannels, C: embd, Stride: embd, DType: mcf.DTypeF32, Data: make([]float32, convChannels*embd)},
		AProj:          &model.Mat{R: numValHeads, C: embd, Stride: embd, DType: mcf.DTypeF32, Data: make([]float32, numValHeads*embd)},
		BProj:          &model.Mat{R: numValHeads, C: embd, Stride: embd, DType: mcf.DTypeF32, Data: make([]float32, numValHeads*embd)},
		ZProj:          &model.Mat{R: valueDim, C: embd, Stride: embd, DType: mcf.DTypeF32, Data: make([]float32, valueDim*embd)},
		OutProj:        &model.Mat{R: embd, C: valueDim, Stride: valueDim, DType: mcf.DTypeF32, Data: make([]float32, embd*valueDim)},
		Conv:           &model.Mat{R: convChannels, C: convKernel, Stride: convKernel, DType: mcf.DTypeF32, Data: make([]float32, convChannels*convKernel)},
		Norm:           make([]float32, headValueDim),
		ALog:           make([]float32, numValHeads),
		DTBias:         make([]float32, numValHeads),
		NumKeyHeads:    numKeyHeads,
		NumValueHeads:  numValHeads,
		HeadKeyDim:     headKeyDim,
		HeadValueDim:   headValueDim,
		KeyDim:         keyDim,
		ValueDim:       valueDim,
		ConvState:      make([]float32, convChannels*(convKernel-1)),
		RecurrentState: make([]float32, numValHeads*headKeyDim*headValueDim),
	}
	x := make([]float32, embd)
	out := make([]float32, embd)
	cfg := model.DeltaNetConfig{RMSEpsilon: 1e-6}

	if ok := ops.DeltaNetBlock(dl, x, out, cfg); ok {
		t.Fatalf("DeltaNetBlock returned true under MANTLE_CUDA_DELTANET=0; expected fast-path bail")
	}
	if err := ops.ConsumeFastPathError(); err != nil {
		t.Fatalf("ConsumeFastPathError after opt-out = %v; want nil", err)
	}
}

func BenchmarkDeltaNetRecurrent(b *testing.B) {
	count, err := native.DeviceCount()
	if err != nil {
		b.Fatalf("DeviceCount: %v", err)
	}
	if count < 1 {
		b.Skip("no cuda device available")
	}

	const (
		embd         = 2048
		numKeyHeads  = 16
		numValHeads  = 32
		headKeyDim   = 128
		headValueDim = 128
		convKernel   = 4
	)
	keyDim := numKeyHeads * headKeyDim
	valueDim := numValHeads * headValueDim
	convChannels := 2*keyDim + valueDim

	rng := rand.New(rand.NewSource(7))

	makeQuantMat := func(rows, cols int, scale float32) *model.Mat {
		qvals := make([]int8, rows*((cols+31)/32)*32)
		for i := range qvals {
			qvals[i] = int8(rng.Intn(15) - 7)
		}
		payload := buildK4Payload(rows, cols, scale, qvals)
		mat := &model.Mat{R: rows, C: cols, Stride: cols, DType: mcf.DTypeK4, Raw: payload}
		cache, err := model.BuildQuantCache(mat)
		if err != nil {
			b.Fatalf("BuildQuantCache [%d x %d]: %v", rows, cols, err)
		}
		mat.Quant = cache
		return mat
	}
	makeVec := func(n int, scale float32) []float32 {
		v := make([]float32, n)
		for i := range v {
			v[i] = (rng.Float32()*2 - 1) * scale
		}
		return v
	}
	convData := make([]float32, convChannels*convKernel)
	for i := range convData {
		convData[i] = (rng.Float32()*2 - 1) * 0.08
	}

	dl := &model.DeltaNetLayer{
		QKVProj:        makeQuantMat(convChannels, embd, 0.05),
		AProj:          makeQuantMat(numValHeads, embd, 0.05),
		BProj:          makeQuantMat(numValHeads, embd, 0.05),
		ZProj:          makeQuantMat(valueDim, embd, 0.05),
		OutProj:        makeQuantMat(embd, valueDim, 0.05),
		Conv:           &model.Mat{R: convChannels, C: convKernel, Stride: convKernel, DType: mcf.DTypeF32, Data: convData},
		Norm:           makeVec(headValueDim, 0.2),
		ALog:           makeVec(numValHeads, 0.2),
		DTBias:         makeVec(numValHeads, 0.1),
		NumKeyHeads:    numKeyHeads,
		NumValueHeads:  numValHeads,
		HeadKeyDim:     headKeyDim,
		HeadValueDim:   headValueDim,
		KeyDim:         keyDim,
		ValueDim:       valueDim,
		ConvState:      make([]float32, convChannels*(convKernel-1)),
		RecurrentState: make([]float32, numValHeads*headKeyDim*headValueDim),
	}
	x := makeVec(embd, 0.25)
	cfg := model.DeltaNetConfig{RMSEpsilon: 1e-6}

	b.Run("cuda", func(b *testing.B) {
		stream, err := native.NewStream()
		if err != nil {
			b.Fatalf("NewStream: %v", err)
		}
		defer func() { _ = stream.Destroy() }()
		blas, err := native.NewBlasHandle(stream)
		if err != nil {
			b.Fatalf("NewBlasHandle: %v", err)
		}
		defer func() { _ = blas.Destroy() }()
		ops := NewOps(stream, blas)
		defer func() { _ = ops.Close() }()

		out := make([]float32, embd)
		if ok := ops.DeltaNetBlock(dl, x, out, cfg); !ok {
			b.Fatalf("DeltaNetBlock warmup returned false: %v", ops.ConsumeFastPathError())
		}
		if err := ops.FlushBlockResult(); err != nil {
			b.Fatalf("FlushBlockResult warmup: %v", err)
		}

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if ok := ops.DeltaNetBlock(dl, x, out, cfg); !ok {
				b.Fatalf("DeltaNetBlock returned false at i=%d: %v", i, ops.ConsumeFastPathError())
			}
			if err := ops.FlushBlockResult(); err != nil {
				b.Fatalf("FlushBlockResult: %v", err)
			}
		}
	})

	b.Run("simd_ref", func(b *testing.B) {
		refLayer := cloneDeltaNetLayer(dl)
		_ = deltaNetBlockRef(refLayer, x, cfg)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = deltaNetBlockRef(refLayer, x, cfg)
		}
	})
}
