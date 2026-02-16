package simd

import (
	"math"
	"testing"
)

func TestRunAttnHeadsMatchesReferenceFull(t *testing.T) {
	const (
		nHead   = 4
		kvHeads = 2
		headDim = 8
		pos     = 5
		start   = 0
	)
	kvStride := kvHeads * headDim
	q := make([]float32, nHead*headDim)
	cacheK := make([]float32, (pos+1)*kvStride)
	cacheV := make([]float32, (pos+1)*kvStride)

	fillTestData(q, 0.1)
	fillTestData(cacheK, 0.2)
	fillTestData(cacheV, 0.3)

	ctx := AttnContext{
		Q:        q,
		CacheK:   cacheK,
		CacheV:   cacheV,
		AttnOut:  make([]float32, nHead*headDim),
		Pos:      pos,
		Start:    start,
		KvStride: kvStride,
		HeadDim:  headDim,
		NHead:    nHead,
		KvHeads:  kvHeads,
		Scale:    float32(1.0 / math.Sqrt(float64(headDim))),
		CacheLen: pos + 1,
	}

	RunAttnHeads(&ctx, make([]float32, pos+1), 0, nHead)

	want := referenceAttention(&ctx)
	compareSlices(t, ctx.AttnOut, want, 1e-5)
}

func TestRunAttnHeadsMatchesReferenceSlidingWindow(t *testing.T) {
	const (
		nHead   = 6
		kvHeads = 3
		headDim = 8
		pos     = 9
		window  = 4
	)
	start := pos - window + 1
	kvStride := kvHeads * headDim
	q := make([]float32, nHead*headDim)
	cacheK := make([]float32, (pos+1)*kvStride)
	cacheV := make([]float32, (pos+1)*kvStride)

	fillTestData(q, 0.05)
	fillTestData(cacheK, 0.07)
	fillTestData(cacheV, 0.09)

	ctx := AttnContext{
		Q:        q,
		CacheK:   cacheK,
		CacheV:   cacheV,
		AttnOut:  make([]float32, nHead*headDim),
		Pos:      pos,
		Start:    start,
		KvStride: kvStride,
		HeadDim:  headDim,
		NHead:    nHead,
		KvHeads:  kvHeads,
		Scale:    float32(1.0 / math.Sqrt(float64(headDim))),
		CacheLen: pos + 1,
	}

	RunAttnHeads(&ctx, make([]float32, window), 0, nHead)

	want := referenceAttention(&ctx)
	compareSlices(t, ctx.AttnOut, want, 1e-5)
}

func TestRunAttnHeadsUsesContextSoftmaxOps(t *testing.T) {
	const (
		nHead   = 2
		kvHeads = 1
		headDim = 4
		pos     = 3
	)
	kvStride := kvHeads * headDim
	ctx := AttnContext{
		Q:        make([]float32, nHead*headDim),
		CacheK:   make([]float32, (pos+1)*kvStride),
		CacheV:   make([]float32, (pos+1)*kvStride),
		AttnOut:  make([]float32, nHead*headDim),
		Pos:      pos,
		Start:    0,
		KvStride: kvStride,
		HeadDim:  headDim,
		NHead:    nHead,
		KvHeads:  kvHeads,
		Scale:    float32(1.0 / math.Sqrt(float64(headDim))),
		CacheLen: pos + 1,
	}
	fillTestData(ctx.Q, 0.03)
	fillTestData(ctx.CacheK, 0.05)
	fillTestData(ctx.CacheV, 0.07)

	rec := &softmaxRecorderOps{}
	ctx.Ops = rec

	RunAttnHeads(&ctx, make([]float32, pos+1), 0, nHead)

	if rec.calls == 0 {
		t.Fatalf("expected ctx.Ops.Softmax to be used")
	}
}

type softmaxRecorderOps struct {
	DefaultOps
	calls int
}

func (s *softmaxRecorderOps) Softmax(x []float32) {
	s.calls++
	Softmax(x)
}

func BenchmarkRunAttnHeadsFull(b *testing.B) {
	benchRunAttnHeads(b, 16, 4, 64, 256, 0)
}

func BenchmarkRunAttnHeadsSliding(b *testing.B) {
	benchRunAttnHeads(b, 16, 4, 64, 256, 128)
}

func benchRunAttnHeads(b *testing.B, nHead, kvHeads, headDim, pos, window int) {
	if window <= 0 || window > pos+1 {
		window = pos + 1
	}
	start := pos - window + 1
	kvStride := kvHeads * headDim

	q := make([]float32, nHead*headDim)
	cacheK := make([]float32, (pos+1)*kvStride)
	cacheV := make([]float32, (pos+1)*kvStride)
	fillTestData(q, 0.01)
	fillTestData(cacheK, 0.02)
	fillTestData(cacheV, 0.03)

	ctx := AttnContext{
		Q:        q,
		CacheK:   cacheK,
		CacheV:   cacheV,
		AttnOut:  make([]float32, nHead*headDim),
		Pos:      pos,
		Start:    start,
		KvStride: kvStride,
		HeadDim:  headDim,
		NHead:    nHead,
		KvHeads:  kvHeads,
		Scale:    float32(1.0 / math.Sqrt(float64(headDim))),
		CacheLen: pos + 1,
	}
	scores := make([]float32, window)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		RunAttnHeads(&ctx, scores, 0, nHead)
	}
}

func referenceAttention(ctx *AttnContext) []float32 {
	out := make([]float32, len(ctx.AttnOut))
	winLen := ctx.Pos - ctx.Start + 1
	scores := make([]float32, winLen)

	for h := 0; h < ctx.NHead; h++ {
		kvHead := h * ctx.KvHeads / ctx.NHead
		qh := ctx.Q[h*ctx.HeadDim : (h+1)*ctx.HeadDim]
		for t := ctx.Start; t <= ctx.Pos; t++ {
			koff := t*ctx.KvStride + kvHead*ctx.HeadDim
			kv := ctx.CacheK[koff : koff+ctx.HeadDim]
			scores[t-ctx.Start] = dotRef(qh, kv) * ctx.Scale
		}
		softmaxRef(scores)
		oh := out[h*ctx.HeadDim : (h+1)*ctx.HeadDim]
		for d := range ctx.HeadDim {
			var sum float32
			for t := ctx.Start; t <= ctx.Pos; t++ {
				voff := t*ctx.KvStride + kvHead*ctx.HeadDim + d
				sum += scores[t-ctx.Start] * ctx.CacheV[voff]
			}
			oh[d] = sum
		}
	}
	return out
}

func dotRef(a, b []float32) float32 {
	var sum float32
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

func softmaxRef(x []float32) {
	if len(x) == 0 {
		return
	}
	maxv := x[0]
	for i := 1; i < len(x); i++ {
		if x[i] > maxv {
			maxv = x[i]
		}
	}
	var sum float64
	for i := range x {
		v := math.Exp(float64(x[i] - maxv))
		x[i] = float32(v)
		sum += v
	}
	if sum == 0 {
		return
	}
	inv := float32(1.0 / sum)
	for i := range x {
		x[i] *= inv
	}
}

func fillTestData(x []float32, scale float32) {
	for i := range x {
		x[i] = scale * float32((i%29)-14)
	}
}

func compareSlices(t *testing.T, got, want []float32, tol float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length mismatch: got %d want %d", len(got), len(want))
	}
	for i := range got {
		g := got[i]
		w := want[i]
		if g < w-tol || g > w+tol {
			t.Fatalf("mismatch at %d: got %v want %v±%v", i, g, w, tol)
		}
	}
}

func TestAttentionSkipsStoreKVOnInnerProjectionFastPath(t *testing.T) {
	ops := &attentionFastPathOps{
		useQKVFastPath:             true,
		useInnerProjectionFastPath: true,
	}
	m, layer := newAttentionFastPathFixture(ops)

	out := Attention(m, layer, []float32{0.1, 0.2}, 0)
	if out == nil {
		t.Fatal("expected output")
	}
	if ops.storeKVCalls != 0 {
		t.Fatalf("StoreKV calls: got %d want 0", ops.storeKVCalls)
	}
}

func TestAttentionSkipsStoreKVOnInnerFastPath(t *testing.T) {
	ops := &attentionFastPathOps{
		useQKVFastPath:     true,
		useInnerFastPath:   true,
		innerFastPathValue: []float32{2, 3},
	}
	m, layer := newAttentionFastPathFixture(ops)

	out := Attention(m, layer, []float32{0.1, 0.2}, 0)
	if out == nil {
		t.Fatal("expected output")
	}
	if ops.storeKVCalls != 0 {
		t.Fatalf("StoreKV calls: got %d want 0", ops.storeKVCalls)
	}
	compareSlices(t, out, []float32{2, 3}, 1e-6)
}

func TestAttentionStoresKVOnFallbackPath(t *testing.T) {
	ops := &attentionFastPathOps{
		useQKVFastPath: true,
	}
	m, layer := newAttentionFastPathFixture(ops)

	out := Attention(m, layer, []float32{0.1, 0.2}, 0)
	if out == nil {
		t.Fatal("expected output")
	}
	if ops.storeKVCalls != 1 {
		t.Fatalf("StoreKV calls: got %d want 1", ops.storeKVCalls)
	}
}

type attentionFastPathOps struct {
	DefaultOps
	storeKVCalls                int
	useQKVFastPath              bool
	useInnerFastPath            bool
	useInnerProjectionFastPath  bool
	innerFastPathValue          []float32
	innerProjectionFastPathFill float32
}

func (o *attentionFastPathOps) MatVecQKV(q, k, v []float32, _ *Mat, _ *Mat, _ *Mat, _ []float32) bool {
	if !o.useQKVFastPath {
		return false
	}
	for i := range q {
		q[i] = float32(i + 1)
	}
	for i := range k {
		k[i] = float32(i + 2)
	}
	for i := range v {
		v[i] = float32(i + 3)
	}
	return true
}

func (o *attentionFastPathOps) AttentionInner(attnOut []float32, _ *Layer, _ []float32, _ []float32, _ []float32, _ int, _ int, _ int, _ int, _ int, _ int, _ float32) bool {
	if !o.useInnerFastPath {
		return false
	}
	fill := o.innerFastPathValue
	if len(fill) == 0 {
		fill = []float32{1, 1}
	}
	copy(attnOut, fill)
	return true
}

func (o *attentionFastPathOps) AttentionInnerProjection(projOut []float32, _ *Layer, _ []float32, _ []float32, _ []float32, _ int, _ int, _ int, _ int, _ int, _ int, _ float32, _ float32) bool {
	if !o.useInnerProjectionFastPath {
		return false
	}
	v := o.innerProjectionFastPathFill
	if v == 0 {
		v = 1
	}
	for i := range projOut {
		projOut[i] = v
	}
	return true
}

func (o *attentionFastPathOps) StoreKV(layerIndex, pos, kvStride int, kDst, vDst []float32, kDst16, vDst16 []uint16, kDstQ8, vDstQ8 []int8, kDstQ8S, vDstQ8S []float32, k, v []float32) {
	o.storeKVCalls++
	o.DefaultOps.StoreKV(layerIndex, pos, kvStride, kDst, vDst, kDst16, vDst16, kDstQ8, vDstQ8, kDstQ8S, vDstQ8S, k, v)
}

func newAttentionFastPathFixture(ops Ops) (*Instance, *Layer) {
	wo := NewMatFromData(2, 2, []float32{
		1, 0,
		0, 1,
	})
	layer := &Layer{
		HeadKV: 1,
		Wo:     &wo,
		AttnCache: AttnCache{
			K:        make([]float32, 0, 32),
			V:        make([]float32, 0, 32),
			KvStride: 2,
			CacheLen: 16,
		},
	}
	m := &Instance{
		HeadCount:          1,
		HeadDim:            2,
		RMSEpsilon:         1e-6,
		RopeInvFreq:        []float64{1.0},
		RopeAttnScale:      1.0,
		MaxContext:         16,
		Scratch:            ScratchBuffers{},
		RopeLocalOnly:      false,
		MuPScale:           1,
		RopeAttnScaleLocal: 1.0,
	}
	m.Scratch.Q = make([]float32, 2)
	m.Scratch.K = make([]float32, 2)
	m.Scratch.V = make([]float32, 2)
	m.Scratch.AttnOut = make([]float32, 2)
	m.Scratch.AttnProj = make([]float32, 2)
	m.Scratch.Scores = make([]float32, 1)
	m.SetOps(ops)
	return m, layer
}
