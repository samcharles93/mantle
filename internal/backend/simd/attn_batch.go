package simd

import (
	"math"
	"simd/archsimd"
)

// This file holds the per-block batched kernels driven by batchPrefillChunk. Each
// mirrors the corresponding sequential block (attn.go's Attention, ffn.go's FFN,
// shortconv.go's ShortConv) for every layer shape batchEligible accepts — dense
// or sliding attention with q/k norms, an output gate and sandwich norms, or a
// recurrent short-conv block — applied to [n,width] blocks instead of one token
// at a time.

// batchLayerRoPE resolves the RoPE tables for a layer exactly as Attention does:
// the layer override wins, then the model tables, then the local (sliding)
// tables. apply is false when the layer or the model disables RoPE for this
// attention type.
func batchLayerRoPE(m *Instance, layer *Layer) (apply bool, invFreq []float64, attnScale float32) {
	apply = !layer.NoRoPE && (!m.RopeLocalOnly || layer.AttnType != "full_attention")
	invFreq = layer.RopeInvFreq
	attnScale = layer.RopeAttnScale
	if len(invFreq) == 0 {
		invFreq = m.RopeInvFreq
		attnScale = m.RopeAttnScale
		if layer.AttnType == "sliding_attention" && len(m.RopeInvFreqLocal) > 0 {
			invFreq = m.RopeInvFreqLocal
			attnScale = m.RopeAttnScaleLocal
		}
	}
	return apply, invFreq, attnScale
}

// attentionBlock runs the attention block for every row and leaves the Wo
// output in p.proj. Positions are startPos..startPos+n-1; a sliding-window
// layer starts each row's window at max(pos-window+1, 0), exactly as the
// per-token path does (a non-causal block keeps the full window).
func (p *batchPrefillChunk) attentionBlock(layer *Layer) {
	m := p.m
	n := p.n
	headDim := m.HeadDim
	nHead := m.HeadCount
	kvHeads := layer.HeadKV
	qDim := p.qDim
	kvStride := p.kvStride

	p.gemmRows(p.q, p.norm, layer.Wq)
	p.gemmRows(p.k, p.norm, layer.Wk)
	p.gemmRows(p.v, p.norm, layer.Wv)

	for r := range n {
		if len(layer.AttnQNorm) > 0 {
			applyHeadRMSNorm(p.ops, p.q[r*qDim:(r+1)*qDim], layer.AttnQNorm, m.RMSEpsilon, headDim, nHead)
		}
		if len(layer.AttnKNorm) > 0 {
			applyHeadRMSNorm(p.ops, p.k[r*kvStride:(r+1)*kvStride], layer.AttnKNorm, m.RMSEpsilon, headDim, kvHeads)
		}
	}

	if apply, invFreq, attnScale := batchLayerRoPE(m, layer); apply {
		for r := range n {
			pos := p.startPos + r
			p.ops.ApplyRoPE(p.q[r*qDim:(r+1)*qDim], nHead, headDim, pos, invFreq, attnScale)
			p.ops.ApplyRoPE(p.k[r*kvStride:(r+1)*kvStride], kvHeads, headDim, pos, invFreq, attnScale)
		}
	}

	scale := layer.AttnScale
	if scale == 0 {
		scale = float32(1.0 / math.Sqrt(float64(headDim)))
	}
	var softcap float32
	if m.Config != nil {
		softcap = m.Config.Config.AttnLogitSoftcap
	}

	cache := &layer.AttnCache
	cacheLen := cache.CacheLen
	storeRow := func(r int) {
		cachePos := (p.startPos + r) % cacheLen
		p.ops.StoreKV(-1, cachePos, kvStride,
			cache.K, cache.V, cache.K16, cache.V16,
			cache.KQ8, cache.VQ8, cache.KQ8S, cache.VQ8S,
			p.k[r*kvStride:(r+1)*kvStride], p.v[r*kvStride:(r+1)*kvStride])
	}
	attendRow := func(r int) {
		pos := p.startPos + r
		start := 0
		if p.nonCausal {
			pos = p.startPos + n - 1
		} else if layer.AttnWindow > 0 {
			start = max(pos-layer.AttnWindow+1, 0)
		}
		cache.EnsurePos(pos % cacheLen)
		ctx := AttnContext{
			Q:         p.q[r*qDim : (r+1)*qDim],
			CacheK:    cache.K,
			CacheV:    cache.V,
			CacheK16:  cache.K16,
			CacheV16:  cache.V16,
			CacheKQ8:  cache.KQ8,
			CacheVQ8:  cache.VQ8,
			CacheKQ8S: cache.KQ8S,
			CacheVQ8S: cache.VQ8S,
			AttnOut:   p.attn[r*qDim : (r+1)*qDim],
			Ops:       p.ops,
			Pos:       pos,
			Start:     start,
			KvStride:  kvStride,
			HeadDim:   headDim,
			NHead:     nHead,
			KvHeads:   kvHeads,
			Scale:     scale,
			Softcap:   softcap,
			CacheLen:  cacheLen,
		}
		RunAttnHeads(&ctx, m.Scratch.Scores, 0, nHead)
	}

	// Position t lives in cache slot t%cacheLen. Storing every row before
	// attending is only equivalent to the sequential path while the batch does
	// not wrap the ring: once startPos+n exceeds cacheLen, a later store
	// overwrites a slot an earlier query still reads. When it wraps, interleave
	// store and attend per row exactly as the sequential decode path does.
	if p.startPos+n > cacheLen && !p.nonCausal {
		for r := range n {
			storeRow(r)
			attendRow(r)
		}
	} else {
		for r := range n {
			storeRow(r)
		}
		for r := range n {
			attendRow(r)
		}
	}

	// Attention output gate (LFM3-style): one gate projection over the normed
	// input per row, then sigmoid scaling of the attention output before Wo.
	// Per-head gates (rows == head count) scale a whole head; otherwise the
	// gate is per element, matching applyAttnOutputGate in attn.go.
	if layer.AttnGate != nil {
		gateR := layer.AttnGate.R
		p.gemmRows(p.attnGate, p.norm, layer.AttnGate)
		if gateR == nHead && nHead != qDim {
			for r := range n {
				row := p.attn[r*qDim : (r+1)*qDim]
				g := p.attnGate[r*gateR : (r+1)*gateR]
				for h := range nHead {
					s := Sigmoid(g[h])
					for i := h * headDim; i < (h+1)*headDim; i++ {
						row[i] *= s
					}
				}
			}
		} else {
			for r := range n {
				row := p.attn[r*qDim : (r+1)*qDim]
				g := p.attnGate[r*gateR : (r+1)*gateR]
				i := 0
				if cpu.HasAVX2 {
					for ; i+8 <= qDim; i += 8 {
						vout := archsimd.LoadFloat32x8(row[i:])
						vgate := archsimd.LoadFloat32x8(g[i:])
						vout = vout.Mul(fastSigmoidVec(vgate))
						vout.Store(row[i:])
					}
				}
				for ; i < qDim; i++ {
					row[i] *= Sigmoid(g[i])
				}
			}
		}
	}

	p.gemmRows(p.proj, p.attn, layer.Wo)
}

// addBlockResidual adds the block output left in p.proj to the residual rows,
// applying the sandwich (post-block) norm first when the layer has one, the
// same order runDecoderLayers uses: norm the block output, then add.
func (p *batchPrefillChunk) addBlockResidual(postNorm []float32) {
	hidden := p.hidden
	if len(postNorm) > 0 {
		for r := range p.n {
			p.ops.RMSNorm(p.norm[r*hidden:(r+1)*hidden], p.proj[r*hidden:(r+1)*hidden], postNorm, p.m.RMSEpsilon)
			Add(p.x[r*hidden:(r+1)*hidden], p.norm[r*hidden:(r+1)*hidden])
		}
		return
	}
	for r := range p.n {
		Add(p.x[r*hidden:(r+1)*hidden], p.proj[r*hidden:(r+1)*hidden])
	}
}

// ffnBlock runs the dense gated FFN for every row and leaves the down projection
// in p.proj.
func (p *batchPrefillChunk) ffnBlock(layer *Layer) {
	ffn := p.ffn
	p.gemmRows(p.up, p.norm, layer.FfnUp)
	p.gemmRows(p.gate, p.norm, layer.FfnGate)
	for r := range p.n {
		activateFFNRow(
			p.act[r*ffn:(r+1)*ffn],
			p.gate[r*ffn:(r+1)*ffn],
			p.up[r*ffn:(r+1)*ffn],
			p.useGelu,
		)
	}
	p.gemmRows(p.proj, p.act, layer.FfnDown)
}

// activateFFNRow computes act = fn(gate) * up for one row: the activation loop
// of FFN (ffn.go) applied row by row, keeping the same vector/scalar split so
// every element takes the identical path to the sequential single-token FFN.
func activateFFNRow(act, gate, up []float32, useGelu bool) {
	n := len(act)
	i := 0
	if useGelu {
		if cpu.HasAVX2 {
			for ; i+8 <= n; i += 8 {
				vgate := archsimd.LoadFloat32x8(gate[i:])
				vup := archsimd.LoadFloat32x8(up[i:])
				vact := fastGeluVec(vgate).Mul(vup)
				vact.Store(act[i:])
			}
		}
		for ; i < n; i++ {
			act[i] = Gelu(gate[i]) * up[i]
		}
		return
	}
	if cpu.HasAVX2 {
		for ; i+8 <= n; i += 8 {
			vgate := archsimd.LoadFloat32x8(gate[i:])
			vup := archsimd.LoadFloat32x8(up[i:])
			vact := fastSiluVec(vgate).Mul(vup)
			vact.Store(act[i:])
		}
	}
	for ; i < n; i++ {
		act[i] = Silu(gate[i]) * up[i]
	}
}
