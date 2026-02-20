package simd

import instance "github.com/samcharles93/mantle/internal/backend/core"

// Ops defines the matvec operations used by the runtime.
type Ops interface {
	MatVec(dst []float32, w *instance.Mat, x []float32)
	MatVecWithQuant(dst []float32, w *instance.Mat, x []float32, qx *instance.QuantVec)
	RMSNorm(dst, src, weight []float32, eps float32)
	Softmax(x []float32)
	ApplyRoPE(x []float32, nHead, headDim, pos int, invFreq []float64, attentionFactor float32)
	StoreKV(layerIndex, pos, kvStride int, kDst, vDst []float32, kDst16, vDst16 []uint16, kDstQ8, vDstQ8 []int8, kDstQ8S, vDstQ8S []float32, k, v []float32)
}

// DefaultOps provides default CPU-based operations.
type DefaultOps struct {
	// scratch is a reusable buffer for attention. It is not thread-safe and
	// must not be used concurrently.
	scratch []float32
}

func (o *DefaultOps) MatVec(dst []float32, w *instance.Mat, x []float32) {
	MatVec(dst, w, x)
}

func (o *DefaultOps) MatVecWithQuant(dst []float32, w *instance.Mat, x []float32, qx *instance.QuantVec) {
	MatVecWithQuant(dst, w, x, qx)
}

func (o *DefaultOps) FlashAttention(attnOut []float32, layer *instance.Layer, q, k, v []float32, pos, start, nHead, headDim, kvHeads, kvStride int, scale, softcap float32) bool {
	return o.IncrementalAttention(attnOut, layer, q, k, v, pos, start, nHead, headDim, kvHeads, kvStride, scale, softcap)
}

func (o *DefaultOps) FlashAttentionMultiHead(attnOut []float32, layer *instance.Layer, q, k, v []float32, pos, start, nHead, headDim, kvHeads, kvStride int, scale, softcap float32) bool {
	return o.IncrementalAttention(attnOut, layer, q, k, v, pos, start, nHead, headDim, kvHeads, kvStride, scale, softcap)
}

func (o *DefaultOps) IncrementalAttention(attnOut []float32, layer *instance.Layer, q, k, v []float32, pos, start, nHead, headDim, kvHeads, kvStride int, scale, softcap float32) bool {
	if layer == nil || kvStride <= 0 || nHead <= 0 || headDim <= 0 || kvHeads <= 0 {
		return false
	}
	if len(attnOut) < nHead*headDim || len(q) < nHead*headDim || len(k) < kvStride || len(v) < kvStride {
		return false
	}
	if start < 0 || start > pos {
		return false
	}

	cacheLen := layer.AttnCache.CacheLen
	cachePos := pos
	if cacheLen > 0 {
		cachePos = pos % cacheLen
	}
	layer.AttnCache.EnsurePos(cachePos)
	o.StoreKV(-1, cachePos, kvStride,
		layer.AttnCache.K, layer.AttnCache.V,
		layer.AttnCache.K16, layer.AttnCache.V16,
		layer.AttnCache.KQ8, layer.AttnCache.VQ8,
		layer.AttnCache.KQ8S, layer.AttnCache.VQ8S,
		k, v)

	winLen := pos - start + 1
	if winLen <= 0 {
		return false
	}
	ctx := AttnContext{
		Q:         q,
		CacheK:    layer.AttnCache.K,
		CacheV:    layer.AttnCache.V,
		CacheK16:  layer.AttnCache.K16,
		CacheV16:  layer.AttnCache.V16,
		CacheKQ8:  layer.AttnCache.KQ8,
		CacheVQ8:  layer.AttnCache.VQ8,
		CacheKQ8S: layer.AttnCache.KQ8S,
		CacheVQ8S: layer.AttnCache.VQ8S,
		AttnOut:   attnOut[:nHead*headDim],
		Pos:       pos,
		Start:     start,
		KvStride:  kvStride,
		HeadDim:   headDim,
		NHead:     nHead,
		KvHeads:   kvHeads,
		Scale:     scale,
		Softcap:   softcap,
		CacheLen:  cacheLen,
		Ops:       o,
	}
	if cap(o.scratch) < winLen {
		o.scratch = make([]float32, winLen)
	} else {
		o.scratch = o.scratch[:winLen]
	}
	RunAttnHeads(&ctx, o.scratch, 0, nHead)
	return true
}

func (o *DefaultOps) RMSNorm(dst, src, weight []float32, eps float32) {
	RMSNorm(dst, src, weight, eps)
}

func (o *DefaultOps) Softmax(x []float32) {
	Softmax(x)
}

func (o *DefaultOps) ApplyRoPE(x []float32, nHead, headDim, pos int, invFreq []float64, attentionFactor float32) {
	ApplyRoPE(x, nHead, headDim, pos, invFreq, attentionFactor)
}

func (o *DefaultOps) StoreKV(_ int, pos, kvStride int, kDst, vDst []float32, kDst16, vDst16 []uint16, kDstQ8, vDstQ8 []int8, kDstQ8S, vDstQ8S []float32, k, v []float32) {
	if kvStride <= 0 {
		return
	}
	offset := pos * kvStride
	if kDst != nil {
		copy(kDst[offset:], k)
	} else if kDst16 != nil {
		Float32ToFloat16Slice(k, kDst16[offset:])
	} else if kDstQ8 != nil {
		storeQ8(k, kDstQ8[offset:offset+kvStride], kDstQ8S, pos)
	}
	if vDst != nil {
		copy(vDst[offset:], v)
	} else if vDst16 != nil {
		Float32ToFloat16Slice(v, vDst16[offset:])
	} else if vDstQ8 != nil {
		storeQ8(v, vDstQ8[offset:offset+kvStride], vDstQ8S, pos)
	}
}

// storeQ8 quantizes src to int8 with per-position scaling.
func storeQ8(src []float32, dst []int8, scales []float32, pos int) {
	var maxAbs float32
	for _, v := range src {
		if v < 0 {
			if -v > maxAbs {
				maxAbs = -v
			}
		} else if v > maxAbs {
			maxAbs = v
		}
	}
	if maxAbs == 0 {
		scales[pos] = 0
		for i := range dst {
			dst[i] = 0
		}
		return
	}
	scale := maxAbs / 127.0
	scales[pos] = scale
	invScale := 127.0 / maxAbs
	for i, v := range src {
		q := int32(v*invScale + 0.5)
		if v < 0 {
			q = int32(v*invScale - 0.5)
		}
		if q > 127 {
			q = 127
		} else if q < -127 {
			q = -127
		}
		dst[i] = int8(q)
	}
}
