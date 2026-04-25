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

func TestMambaBlockParity(t *testing.T) {
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
			runMambaBlockParity(t, tc.quant)
		})
	}
}

func runMambaBlockParity(t *testing.T, quant bool) {
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
		embd       = 32
		inner      = 64
		headCount  = 4
		headDim    = 16
		dState     = 8
		groups     = 2
		groupSize  = 2
		convKernel = 4
	)
	convChannels := inner + 2*groups*dState
	dInProj := 2*inner + 2*groups*dState + headCount

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

	var inProj, outProj *model.Mat
	if quant {
		inProj = makeQuantMat(dInProj, embd, 0.05)
		outProj = makeQuantMat(embd, inner, 0.05)
	} else {
		inProj = makeF32Mat(dInProj, embd, 0.05)
		outProj = makeF32Mat(embd, inner, 0.05)
	}

	ml := &model.MambaLayer{
		InProj:       inProj,
		OutProj:      outProj,
		Conv:         &model.Mat{R: convChannels, C: convKernel, Stride: convKernel, DType: mcf.DTypeF32, Data: convData},
		ALog:         makeVec(headCount, 0.2),
		D:            makeVec(headCount, 0.15),
		DTBias:       makeVec(headCount, 0.1),
		Norm:         makeVec(inner, 0.2),
		Inner:        inner,
		HeadCount:    headCount,
		HeadDim:      headDim,
		DState:       dState,
		Groups:       groups,
		GroupSize:    groupSize,
		ConvKernel:   convKernel,
		ConvChannels: convChannels,
		ConvState:    make([]float32, convChannels*(convKernel-1)),
		SSMState:     make([]float32, headCount*headDim*dState),
	}
	x := makeVec(embd, 0.25)
	cfg := MambaConfig{
		SSMInMultiplier:     0.75,
		SSMOutMultiplier:    1.25,
		TimeStepMin:         1e-4,
		TimeStepMax:         1.0,
		TimeStepFloor:       1e-5,
		MambaRMSNorm:        true,
		MambaNormBeforeGate: true,
		RMSEpsilon:          1e-5,
	}

	refLayer := cloneMambaLayer(ml)
	want := mambaBlockRef(refLayer, x, cfg)
	if len(want) != embd {
		t.Fatalf("reference output len = %d, want %d", len(want), embd)
	}

	ops.ResetConvStates()
	got := make([]float32, inner)
	if ok := ops.MambaBlock(ml, x, got, cfg); !ok {
		t.Fatalf("MambaBlock returned false: %v", ops.ConsumeFastPathError())
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

func cloneMambaLayer(src *model.MambaLayer) *model.MambaLayer {
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
	return &model.MambaLayer{
		InProj:       cloneMat(src.InProj),
		OutProj:      cloneMat(src.OutProj),
		Conv:         cloneMat(src.Conv),
		ConvBias:     append([]float32(nil), src.ConvBias...),
		ALog:         append([]float32(nil), src.ALog...),
		D:            append([]float32(nil), src.D...),
		DTBias:       append([]float32(nil), src.DTBias...),
		Norm:         append([]float32(nil), src.Norm...),
		Inner:        src.Inner,
		HeadCount:    src.HeadCount,
		HeadDim:      src.HeadDim,
		DState:       src.DState,
		Groups:       src.Groups,
		GroupSize:    src.GroupSize,
		ConvKernel:   src.ConvKernel,
		ConvChannels: src.ConvChannels,
		ConvState:    append([]float32(nil), src.ConvState...),
		SSMState:     append([]float32(nil), src.SSMState...),
	}
}

func mambaBlockRef(ml *model.MambaLayer, x []float32, cfg MambaConfig) []float32 {
	in := x
	if cfg.SSMInMultiplier != 0 && cfg.SSMInMultiplier != 1 {
		scaled := make([]float32, len(x))
		for i := range x {
			scaled[i] = x[i] * cfg.SSMInMultiplier
		}
		in = scaled
	}

	dInProj := 2*ml.Inner + 2*ml.Groups*ml.DState + ml.HeadCount
	proj := make([]float32, dInProj)
	matVecRef(proj, ml.InProj, in)

	z := append([]float32(nil), proj[:ml.Inner]...)
	convIn := append([]float32(nil), proj[ml.Inner:ml.Inner+ml.ConvChannels]...)
	dt := append([]float32(nil), proj[ml.Inner+ml.ConvChannels:ml.Inner+ml.ConvChannels+ml.HeadCount]...)

	convOut := make([]float32, ml.ConvChannels)
	mambaDepthwiseConvRef(convOut, convIn, ml.Conv, ml.ConvBias, ml.ConvState)
	for i := range convOut {
		convOut[i] = siluRef(convOut[i])
	}

	xSplit := append([]float32(nil), convOut[:ml.Inner]...)
	bSplit := append([]float32(nil), convOut[ml.Inner:ml.Inner+ml.Groups*ml.DState]...)
	cSplit := append([]float32(nil), convOut[ml.Inner+ml.Groups*ml.DState:]...)
	for i := range dt {
		dt[i] = softplusRef(dt[i] + ml.DTBias[i])
		dt[i] = clampTimeStepRef(dt[i], cfg.TimeStepMin, cfg.TimeStepMax, cfg.TimeStepFloor)
	}

	y := make([]float32, ml.Inner)
	mambaScanRef(y, ml, xSplit, dt, bSplit, cSplit)
	if cfg.MambaRMSNorm && ml.Norm != nil {
		rmsNormGatedRef(y, y, z, ml.Norm, cfg.RMSEpsilon, cfg.MambaNormBeforeGate)
	} else {
		for i := range y {
			y[i] *= siluRef(z[i])
		}
	}

	out := make([]float32, ml.OutProj.R)
	matVecRef(out, ml.OutProj, y)
	if cfg.SSMOutMultiplier != 0 && cfg.SSMOutMultiplier != 1 {
		for i := range out {
			out[i] *= cfg.SSMOutMultiplier
		}
	}
	return out
}

func matVecRef(dst []float32, w *model.Mat, x []float32) {
	if w == nil {
		panic("nil mat")
	}
	if w.Quant != nil {
		quantMatVecRef(dst, w, x)
		return
	}
	if len(w.Data) == 0 {
		panic("mat has neither Quant nor Data")
	}
	for r := 0; r < w.R; r++ {
		row := w.Data[r*w.Stride : r*w.Stride+w.C]
		var sum float32
		for c := 0; c < w.C; c++ {
			sum += row[c] * x[c]
		}
		dst[r] = sum
	}
}

func quantMatVecRef(dst []float32, w *model.Mat, x []float32) {
	if w == nil || w.Quant == nil {
		panic("missing quant mat")
	}
	for r := 0; r < w.R; r++ {
		var sum float32
		rowBase := r * w.Quant.BlocksPerRow
		for b := 0; b < w.Quant.BlocksPerRow; b++ {
			colBase := b * 32
			n := w.C - colBase
			if n <= 0 {
				break
			}
			if n > 32 {
				n = 32
			}
			scale := w.Quant.Scales[rowBase+b]
			qb := w.Quant.Q[(rowBase+b)*32 : (rowBase+b+1)*32]
			for i := 0; i < n; i++ {
				sum += float32(qb[i]) * x[colBase+i] * scale
			}
		}
		dst[r] = sum
	}
}

func mambaDepthwiseConvRef(out, in []float32, kernel *model.Mat, bias []float32, state []float32) {
	kernelLen := kernel.C
	channels := kernel.R
	for c := 0; c < channels; c++ {
		row := kernel.Data[c*kernelLen : (c+1)*kernelLen]
		sum := float32(0)
		if len(bias) == channels {
			sum = bias[c]
		}
		for k := 0; k < kernelLen-1; k++ {
			sum += row[k] * state[k*channels+c]
		}
		sum += row[kernelLen-1] * in[c]
		out[c] = sum
	}
	if kernelLen > 1 {
		if kernelLen == 2 {
			copy(state, in)
		} else {
			copy(state, state[channels:])
			copy(state[(kernelLen-2)*channels:], in)
		}
	}
}

func mambaScanRef(out []float32, ml *model.MambaLayer, x, dt, b, c []float32) {
	for h := 0; h < ml.HeadCount; h++ {
		group := h / ml.GroupSize
		a := -float32(math.Exp(float64(ml.ALog[h])))
		dtH := dt[h]
		dA := float32(math.Exp(float64(a * dtH)))
		bGroup := b[group*ml.DState : (group+1)*ml.DState]
		cGroup := c[group*ml.DState : (group+1)*ml.DState]
		for p := 0; p < ml.HeadDim; p++ {
			xhp := x[h*ml.HeadDim+p]
			stateBase := (h*ml.HeadDim + p) * ml.DState
			var sum float32
			for n := 0; n < ml.DState; n++ {
				idx := stateBase + n
				ml.SSMState[idx] = ml.SSMState[idx]*dA + dtH*bGroup[n]*xhp
				sum += cGroup[n] * ml.SSMState[idx]
			}
			out[h*ml.HeadDim+p] = sum + ml.D[h]*xhp
		}
	}
}

func rmsNormGatedRef(dst, src, gate, weight []float32, eps float32, normBeforeGate bool) {
	if normBeforeGate {
		rmsNormRef(dst, src, weight, eps)
		for i := range dst {
			dst[i] *= siluRef(gate[i])
		}
		return
	}
	for i := range src {
		dst[i] = src[i] * siluRef(gate[i])
	}
	rmsNormRef(dst, dst, weight, eps)
}

func rmsNormRef(dst, src, weight []float32, eps float32) {
	var ss float32
	for i := range src {
		ss += src[i] * src[i]
	}
	scale := float32(1 / math.Sqrt(float64(ss/float32(len(src))+eps)))
	for i := range src {
		dst[i] = src[i] * scale * weight[i]
	}
}

func siluRef(x float32) float32 {
	return x / (1 + float32(math.Exp(float64(-x))))
}

func softplusRef(x float32) float32 {
	if x > 20 {
		return x
	}
	if x < -20 {
		return float32(math.Exp(float64(x)))
	}
	return float32(math.Log1p(math.Exp(float64(x))))
}

func clampTimeStepRef(v, minV, maxV, floorV float32) float32 {
	if floorV > 0 && v < floorV {
		v = floorV
	}
	if minV > 0 && v < minV {
		v = minV
	}
	if maxV > 0 && v > maxV {
		v = maxV
	}
	return v
}

func approxEqualTol(a, b, absTol, relTol float32) bool {
	diff := float32(math.Abs(float64(a - b)))
	limit := maxFloat32(absTol, relTol*maxFloat32(float32(math.Abs(float64(a))), float32(math.Abs(float64(b)))))
	return diff <= limit
}

func maxFloat32(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}

func buildK4Payload(rows, cols int, scale float32, qvals []int8) []byte {
	blocksPerRow := (cols + 31) / 32
	superBlocksPerRow := (blocksPerRow + 7) / 8
	totalBlocks := rows * blocksPerRow
	totalSuper := rows * superBlocksPerRow

	superScales := make([]byte, totalSuper*2)
	f16 := model.Float32ToFloat16(scale)
	for i := 0; i < totalSuper; i++ {
		superScales[i*2] = byte(f16)
		superScales[i*2+1] = byte(f16 >> 8)
	}
	buf := append([]byte{}, superScales...)
	buf = align64(buf)

	subScales := make([]byte, totalBlocks)
	for i := range subScales {
		subScales[i] = 32
	}
	buf = append(buf, subScales...)
	buf = align64(buf)

	data := make([]byte, totalBlocks*16)
	padded := make([]int8, totalBlocks*32)
	copy(padded, qvals)
	for block := 0; block < totalBlocks; block++ {
		start := block * 32
		packQ4Block(data[block*16:(block+1)*16], padded[start:start+32])
	}
	buf = append(buf, data...)
	return buf
}

func align64(buf []byte) []byte {
	rem := len(buf) % 64
	if rem == 0 {
		return buf
	}
	return append(buf, make([]byte, 64-rem)...)
}

func packQ4Block(dst []byte, values []int8) {
	for i := 0; i < 16; i++ {
		lo := encodeQ4Nibble(values[2*i])
		hi := encodeQ4Nibble(values[2*i+1])
		dst[i] = lo | (hi << 4)
	}
}

func encodeQ4Nibble(v int8) byte {
	if v >= 0 {
		return byte(v)
	}
	return byte(int(v) + 16)
}

func TestMambaBlockFastPathOptOut(t *testing.T) {
	count, err := native.DeviceCount()
	if err != nil {
		t.Fatalf("DeviceCount: %v", err)
	}
	if count < 1 {
		t.Skip("no cuda device available")
	}

	t.Setenv("MANTLE_CUDA_MAMBA", "0")

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
		embd       = 16
		inner      = 32
		headCount  = 2
		headDim    = 16
		dState     = 4
		groups     = 1
		groupSize  = 2
		convKernel = 4
	)
	convChannels := inner + 2*groups*dState
	dInProj := 2*inner + 2*groups*dState + headCount

	ml := &model.MambaLayer{
		InProj:       &model.Mat{R: dInProj, C: embd, Stride: embd, DType: mcf.DTypeF32, Data: make([]float32, dInProj*embd)},
		OutProj:      &model.Mat{R: embd, C: inner, Stride: inner, DType: mcf.DTypeF32, Data: make([]float32, embd*inner)},
		Conv:         &model.Mat{R: convChannels, C: convKernel, Stride: convKernel, DType: mcf.DTypeF32, Data: make([]float32, convChannels*convKernel)},
		ALog:         make([]float32, headCount),
		D:            make([]float32, headCount),
		DTBias:       make([]float32, headCount),
		Norm:         make([]float32, inner),
		Inner:        inner,
		HeadCount:    headCount,
		HeadDim:      headDim,
		DState:       dState,
		Groups:       groups,
		GroupSize:    groupSize,
		ConvKernel:   convKernel,
		ConvChannels: convChannels,
		ConvState:    make([]float32, convChannels*(convKernel-1)),
		SSMState:     make([]float32, headCount*headDim*dState),
	}
	x := make([]float32, embd)
	out := make([]float32, inner)
	cfg := MambaConfig{RMSEpsilon: 1e-5}

	if ok := ops.MambaBlock(ml, x, out, cfg); ok {
		t.Fatalf("MambaBlock returned true under MANTLE_CUDA_MAMBA=0; expected fast-path bail")
	}
	if err := ops.ConsumeFastPathError(); err != nil {
		t.Fatalf("ConsumeFastPathError after opt-out = %v; want nil", err)
	}
}

func BenchmarkMambaBlock(b *testing.B) {
	count, err := native.DeviceCount()
	if err != nil {
		b.Fatalf("DeviceCount: %v", err)
	}
	if count < 1 {
		b.Skip("no cuda device available")
	}

	const (
		embd       = 2560
		inner      = 5120
		headCount  = 80
		headDim    = 64
		dState     = 128
		groups     = 8
		groupSize  = 10
		convKernel = 4
	)
	convChannels := inner + 2*groups*dState
	dInProj := 2*inner + 2*groups*dState + headCount

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

	ml := &model.MambaLayer{
		InProj:       makeQuantMat(dInProj, embd, 0.05),
		OutProj:      makeQuantMat(embd, inner, 0.05),
		Conv:         &model.Mat{R: convChannels, C: convKernel, Stride: convKernel, DType: mcf.DTypeF32, Data: convData},
		ALog:         makeVec(headCount, 0.2),
		D:            makeVec(headCount, 0.15),
		DTBias:       makeVec(headCount, 0.1),
		Norm:         makeVec(inner, 0.2),
		Inner:        inner,
		HeadCount:    headCount,
		HeadDim:      headDim,
		DState:       dState,
		Groups:       groups,
		GroupSize:    groupSize,
		ConvKernel:   convKernel,
		ConvChannels: convChannels,
		ConvState:    make([]float32, convChannels*(convKernel-1)),
		SSMState:     make([]float32, headCount*headDim*dState),
	}
	x := makeVec(embd, 0.25)
	cfg := MambaConfig{
		SSMInMultiplier:     0.75,
		SSMOutMultiplier:    1.25,
		TimeStepMin:         1e-4,
		TimeStepMax:         1.0,
		TimeStepFloor:       1e-5,
		MambaRMSNorm:        true,
		MambaNormBeforeGate: true,
		RMSEpsilon:          1e-5,
	}

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

		out := make([]float32, inner)
		ops.ResetConvStates()
		if ok := ops.MambaBlock(ml, x, out, cfg); !ok {
			b.Fatalf("MambaBlock warmup returned false: %v", ops.ConsumeFastPathError())
		}
		if err := ops.FlushBlockResult(); err != nil {
			b.Fatalf("FlushBlockResult warmup: %v", err)
		}

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if ok := ops.MambaBlock(ml, x, out, cfg); !ok {
				b.Fatalf("MambaBlock returned false at i=%d: %v", i, ops.ConsumeFastPathError())
			}
			if err := ops.FlushBlockResult(); err != nil {
				b.Fatalf("FlushBlockResult: %v", err)
			}
		}
	})

	b.Run("simd_ref", func(b *testing.B) {
		refLayer := cloneMambaLayer(ml)
		_ = mambaBlockRef(refLayer, x, cfg)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = mambaBlockRef(refLayer, x, cfg)
		}
	})
}
