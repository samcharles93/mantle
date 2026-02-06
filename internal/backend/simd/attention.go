package simd

import (
	"math"

	"simd/archsimd"
)

// Attention performs multi-head attention with optional RoPE, KV caching, and sliding window.
// Implements the full attention mechanism including Q/K/V projections, attention computation,
// and output projection.
func Attention(m *Instance, layer *Layer, x []float32, pos int) []float32 {
	nHead := m.HeadCount
	headDim := m.HeadDim
	kvHeads := layer.HeadKV
	kvStride := layer.AttnCache.KvStride
	if kvHeads <= 0 {
		panic("attention layer without kv heads")
	}

	q := m.Scratch.Q[:nHead*headDim]
	k := m.Scratch.K[:kvStride]
	v := m.Scratch.V[:kvStride]
	attnOut := m.Scratch.AttnOut

	var qx *QuantVec
	if CanUseQuantVec(layer.Wq) || CanUseQuantVec(layer.Wk) || CanUseQuantVec(layer.Wv) {
		qx = PrepareQuantVec(x)
		defer ReleaseQuantVec(qx)
	}
	m.Ops().MatVecWithQuant(q, layer.Wq, x, qx)
	m.Ops().MatVecWithQuant(k, layer.Wk, x, qx)
	m.Ops().MatVecWithQuant(v, layer.Wv, x, qx)

	if len(layer.WqBias) > 0 {
		Add(q, layer.WqBias)
	}
	if len(layer.WkBias) > 0 {
		Add(k, layer.WkBias)
	}
	if len(layer.WvBias) > 0 {
		Add(v, layer.WvBias)
	}

	if len(layer.AttnQNorm) > 0 {
		for h := range nHead {
			RMSNorm(q[h*headDim:(h+1)*headDim], q[h*headDim:(h+1)*headDim], layer.AttnQNorm, m.RMSEpsilon)
		}
	}
	if len(layer.AttnKNorm) > 0 {
		for h := range kvHeads {
			RMSNorm(k[h*headDim:(h+1)*headDim], k[h*headDim:(h+1)*headDim], layer.AttnKNorm, m.RMSEpsilon)
		}
	}

	applyRoPE := !m.RopeLocalOnly || layer.AttnType != "full_attention"
	if applyRoPE {
		invFreq := m.RopeInvFreq
		attnScale := m.RopeAttnScale
		if layer.AttnType == "sliding_attention" && len(m.RopeInvFreqLocal) > 0 {
			invFreq = m.RopeInvFreqLocal
			attnScale = m.RopeAttnScaleLocal
		}
		ApplyRoPE(q, nHead, headDim, pos, invFreq, attnScale)
		ApplyRoPE(k, kvHeads, headDim, pos, invFreq, attnScale)
	}

	// Helper for F16 cache
	storeCache := func(src []float32, f32Dest []float32, f16Dest []uint16, offset int) {
		if f32Dest != nil {
			copy(f32Dest[offset:], src)
		} else if f16Dest != nil {
			Float32ToFloat16Slice(src, f16Dest[offset:])
		}
	}

	cacheLen := layer.AttnCache.CacheLen
	cachePos := pos
	if cacheLen > 0 {
		cachePos = pos % cacheLen
	}
	offset := cachePos * kvStride
	storeCache(k, layer.AttnCache.K, layer.AttnCache.K16, offset)
	storeCache(v, layer.AttnCache.V, layer.AttnCache.V16, offset)

	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	start := 0
	if layer.AttnWindow > 0 {
		start = pos - layer.AttnWindow + 1
		if start < 0 {
			start = 0
		}
	}
	ctx := AttnContext{
		Q:        q,
		CacheK:   layer.AttnCache.K,
		CacheV:   layer.AttnCache.V,
		CacheK16: layer.AttnCache.K16,
		CacheV16: layer.AttnCache.V16,
		AttnOut:  attnOut,
		Pos:      pos,
		Start:    start,
		KvStride: kvStride,
		HeadDim:  headDim,
		NHead:    nHead,
		KvHeads:  kvHeads,
		Scale:    scale,
		CacheLen: layer.AttnCache.CacheLen,
	}

	pool := m.GetAttnPool()
	workers := pool.Size
	if workers <= 1 {
		RunAttnHeads(&ctx, m.Scratch.Scores, 0, nHead)
	} else {
		chunk := (nHead + workers - 1) / workers
		done := <-pool.DoneSlots
		activeWorkers := 0
		for i := 0; i < workers; i++ {
			rs := i * chunk
			re := rs + chunk
			if re > nHead {
				re = nHead
			}
			if rs >= re {
				break
			}
			activeWorkers++
			pool.Tasks <- AttnTask{
				Ctx:  &ctx,
				Rs:   rs,
				Re:   re,
				Done: done,
			}
		}
		for i := 0; i < activeWorkers; i++ {
			<-done
		}
		pool.DoneSlots <- done
	}

	if layer.AttnGate != nil {
		gate := m.Scratch.AttnGate[:nHead*headDim]
		m.Ops().MatVecWithQuant(gate, layer.AttnGate, x, qx)

		// Vectorized Sigmoid with multiplication
		n := len(gate)
		i := 0
		if cpu.HasAVX2 {
			for ; i+8 <= n; i += 8 {
				vout := archsimd.LoadFloat32x8Slice(attnOut[i:])
				vgate := archsimd.LoadFloat32x8Slice(gate[i:])
				vout = vout.Mul(fastSigmoidVec(vgate))
				vout.StoreSlice(attnOut[i:])
			}
		}
		for ; i < n; i++ {
			attnOut[i] *= Sigmoid(gate[i])
		}
	}
	m.Ops().MatVec(m.Scratch.AttnProj, layer.Wo, attnOut[:nHead*headDim])
	return m.Scratch.AttnProj
}
