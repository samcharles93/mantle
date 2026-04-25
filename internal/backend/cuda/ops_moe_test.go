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

func ffnProjectRef(up, gate, down *model.Mat, x, scratchUp, scratchGate, scratchAct, scratchTmp []float32) []float32 {
	intermediate := up.R
	upBuf := scratchUp[:intermediate]
	gateBuf := scratchGate[:intermediate]
	actBuf := scratchAct[:intermediate]
	out := scratchTmp[:down.R]

	matVecRef(upBuf, up, x)
	matVecRef(gateBuf, gate, x)
	for i := range actBuf {
		actBuf[i] = siluRef(gateBuf[i]) * upBuf[i]
	}
	matVecRef(out, down, actBuf)
	return out
}

func selectTopKRef(selScores, rawScores []float32, k int, routeScale float32, idxOut []int, wOut []float32) {
	if k > len(selScores) {
		k = len(selScores)
	}
	if k <= 0 {
		return
	}

	type pair struct {
		idx   int
		score float32
	}
	pairs := make([]pair, len(selScores))
	for i, s := range selScores {
		pairs[i] = pair{i, s}
	}
	for i := 0; i < k; i++ {
		best := i
		for j := i + 1; j < len(pairs); j++ {
			if pairs[j].score > pairs[best].score || (pairs[j].score == pairs[best].score && pairs[j].idx < pairs[best].idx) {
				best = j
			}
		}
		pairs[i], pairs[best] = pairs[best], pairs[i]
	}

	for i := 0; i < k; i++ {
		idxOut[i] = pairs[i].idx
	}

	var denom float32
	for j := 0; j < k; j++ {
		id := idxOut[j]
		if id >= 0 && id < len(rawScores) {
			denom += rawScores[id]
		}
	}
	if denom == 0 {
		denom = 1
	}
	for j := 0; j < k; j++ {
		id := idxOut[j]
		if id < 0 || id >= len(rawScores) {
			wOut[j] = 0
			continue
		}
		wOut[j] = (rawScores[id] / denom) * routeScale
	}
}

func moeBlockRef(ml *model.MoELayer, x []float32, scratch [][]float32) []float32 {
	accum := scratch[0][:ml.Shared.Down.R]
	raw := scratch[1][:ml.Router.R]
	sel := scratch[2][:ml.Router.R]
	idx := make([]int, ml.TopK)
	weights := make([]float32, ml.TopK)
	upBuf := scratch[3][:ml.Shared.Up.R]
	gateBuf := scratch[4][:ml.Shared.Up.R]
	actBuf := scratch[5][:ml.Shared.Up.R]
	tmpBuf := scratch[6]

	for i := range accum {
		accum[i] = 0
	}

	sharedOut := ffnProjectRef(ml.Shared.Up, ml.Shared.Gate, ml.Shared.Down, x, upBuf, gateBuf, actBuf, tmpBuf)
	for i := range accum {
		accum[i] += sharedOut[i]
	}

	matVecRef(raw, ml.Router, x)
	for i := range raw {
		raw[i] = sigmoidRef(raw[i])
		bias := float32(0)
		if i < len(ml.ExpertBias) {
			bias = ml.ExpertBias[i]
		}
		sel[i] = raw[i] + bias
	}

	selectTopKRef(sel, raw, ml.TopK, ml.RouteScale, idx, weights)

	for j := 0; j < ml.TopK; j++ {
		if idx[j] < 0 || idx[j] >= len(ml.Experts) {
			continue
		}
		w := weights[j]
		if w == 0 {
			continue
		}
		ex := ml.Experts[idx[j]]
		out := ffnProjectRef(ex.Up, ex.Gate, ex.Down, x, upBuf, gateBuf, actBuf, tmpBuf)
		for i := range accum {
			accum[i] += w * out[i]
		}
	}

	return accum
}

func TestMoEBlockParity(t *testing.T) {
	count, err := native.DeviceCount()
	if err != nil {
		t.Fatalf("DeviceCount: %v", err)
	}
	if count < 1 {
		t.Skip("no cuda device available")
	}

	stream, err := native.NewStream()
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	defer func() { _ = stream.Destroy() }()

	blas, err := native.NewBlasHandle(stream)
	if err != nil {
		t.Fatalf("NewBlasHandle: %v", err)
	}
	defer func() { _ = blas.Destroy() }()

	ops := NewOps(stream, blas)
	defer func() { _ = ops.Close() }()

	rng := rand.New(rand.NewSource(0xACC01))

	const (
		embd         = 32
		numExperts   = 8
		intermediate = 64
		topK         = 2
	)
	const routeScale float32 = 2.5

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

	experts := make([]model.MoEExpert, numExperts)
	for j := range experts {
		experts[j] = model.MoEExpert{
			Up:   makeF32Mat(intermediate, embd, 0.08),
			Gate: makeF32Mat(intermediate, embd, 0.08),
			Down: makeF32Mat(embd, intermediate, 0.08),
		}
	}

	ml := &model.MoELayer{
		Router:     makeF32Mat(numExperts, embd, 0.08),
		ExpertBias: makeVec(numExperts, 0.1),
		Shared: model.MoEShared{
			Up:           makeF32Mat(intermediate, embd, 0.08),
			Gate:         makeF32Mat(intermediate, embd, 0.08),
			Down:         makeF32Mat(embd, intermediate, 0.08),
			Intermediate: intermediate,
		},
		Experts:    experts,
		TopK:       topK,
		RouteScale: routeScale,
	}

	x := makeVec(embd, 0.25)
	cfg := model.MoEConfig{}

	scratch := make([][]float32, 7)
	scratch[0] = make([]float32, embd)
	scratch[1] = make([]float32, numExperts)
	scratch[2] = make([]float32, numExperts)
	scratch[3] = make([]float32, intermediate)
	scratch[4] = make([]float32, intermediate)
	scratch[5] = make([]float32, intermediate)
	scratch[6] = make([]float32, int(math.Max(float64(embd), float64(intermediate))))

	want := moeBlockRef(ml, x, scratch)

	got := make([]float32, embd)
	if ok := ops.MoEBlock(ml, x, got, cfg); !ok {
		t.Fatalf("MoEBlock returned false: %v", ops.ConsumeFastPathError())
	}

	for i := range want {
		if !approxEqualTol(want[i], got[i], 1e-4, 1e-4) {
			t.Errorf("out[%d] = %g, want %g (diff=%g)", i, got[i], want[i], got[i]-want[i])
		}
	}
}

func TestMoEBlockFastPathOptOut(t *testing.T) {
	count, err := native.DeviceCount()
	if err != nil {
		t.Fatalf("DeviceCount: %v", err)
	}
	if count < 1 {
		t.Skip("no cuda device available")
	}

	t.Setenv("MANTLE_CUDA_MOE", "0")

	stream, err := native.NewStream()
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	defer func() { _ = stream.Destroy() }()

	blas, err := native.NewBlasHandle(stream)
	if err != nil {
		t.Fatalf("NewBlasHandle: %v", err)
	}
	defer func() { _ = blas.Destroy() }()

	ops := NewOps(stream, blas)
	defer func() { _ = ops.Close() }()

	const embd = 16
	const numExperts = 4
	const intermediate = 32

	experts := make([]model.MoEExpert, numExperts)
	for j := range experts {
		experts[j] = model.MoEExpert{
			Up:   &model.Mat{R: intermediate, C: embd, Stride: embd, DType: mcf.DTypeF32, Data: make([]float32, intermediate*embd)},
			Gate: &model.Mat{R: intermediate, C: embd, Stride: embd, DType: mcf.DTypeF32, Data: make([]float32, intermediate*embd)},
			Down: &model.Mat{R: embd, C: intermediate, Stride: intermediate, DType: mcf.DTypeF32, Data: make([]float32, embd*intermediate)},
		}
	}

	ml := &model.MoELayer{
		Router:     &model.Mat{R: numExperts, C: embd, Stride: embd, DType: mcf.DTypeF32, Data: make([]float32, numExperts*embd)},
		ExpertBias: make([]float32, numExperts),
		Shared: model.MoEShared{
			Up:           &model.Mat{R: intermediate, C: embd, Stride: embd, DType: mcf.DTypeF32, Data: make([]float32, intermediate*embd)},
			Gate:         &model.Mat{R: intermediate, C: embd, Stride: embd, DType: mcf.DTypeF32, Data: make([]float32, intermediate*embd)},
			Down:         &model.Mat{R: embd, C: intermediate, Stride: intermediate, DType: mcf.DTypeF32, Data: make([]float32, embd*intermediate)},
			Intermediate: intermediate,
		},
		Experts:    experts,
		TopK:       2,
		RouteScale: 1.0,
	}

	x := make([]float32, embd)
	out := make([]float32, embd)
	cfg := model.MoEConfig{}

	if ok := ops.MoEBlock(ml, x, out, cfg); ok {
		t.Fatalf("MoEBlock returned true under MANTLE_CUDA_MOE=0; expected fast-path bail")
	}
	if err := ops.ConsumeFastPathError(); err != nil {
		t.Fatalf("ConsumeFastPathError after opt-out = %v; want nil", err)
	}
}
