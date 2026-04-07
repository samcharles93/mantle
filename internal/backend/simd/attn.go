package simd

import (
	"fmt"
	"math"

	"simd/archsimd"

	instance "github.com/samcharles93/mantle/internal/backend/core"
)

type attentionQKVFastPath interface {
	MatVecQKV(q, k, v []float32, wq, wk, wv *instance.Mat, x []float32) bool
}

type attentionInnerFastPath interface {
	AttentionInner(attnOut []float32, layer *Layer, q, k, v []float32, pos, start, nHead, headDim, kvHeads, kvStride int, scale, softcap float32) bool
}

type attentionInnerProjectionFastPath interface {
	AttentionInnerProjection(projOut []float32, layer, kvLayer *Layer, q, k, v []float32, pos, start, nHead, headDim, kvHeads, kvStride int, scale, epsilon, softcap float32) bool
}

type flashAttentionFastPath interface {
	FlashAttention(attnOut []float32, layer *Layer, q, k, v []float32, pos, start, nHead, headDim, kvHeads, kvStride int, scale, softcap float32) bool
}

type flashAttentionMultiHeadFastPath interface {
	FlashAttentionMultiHead(attnOut []float32, layer *Layer, q, k, v []float32, pos, start, nHead, headDim, kvHeads, kvStride int, scale, softcap float32) bool
}

type deviceMatVecFastPath interface {
	DeviceMatVec(dst []float32, w *instance.Mat, x []float32) bool
}

type incrementalAttentionFastPath interface {
	IncrementalAttention(attnOut []float32, layer *Layer, q, k, v []float32, pos, start, nHead, headDim, kvHeads, kvStride int, scale, softcap float32) bool
}

type batchedNormOps interface {
	RMSNormBatched(dst, src, weight []float32, eps float32, headDim, nHeads int) bool
}

// Attention performs multi-head attention with optional RoPE, KV caching, and sliding window.
// Implements the full attention mechanism including Q/K/V projections, attention computation,
// and output projection.
func Attention(m *Instance, layer *Layer, x []float32, pos int) []float32 {
	ops := m.Ops()
	nHead := m.HeadCount
	headDim := layer.HeadDim
	if headDim <= 0 {
		headDim = m.HeadDim
	}
	kvHeads := layer.HeadKV
	kvLayer := layer
	if layer.SharedKVSource >= 0 && layer.SharedKVSource < len(m.Layers) {
		kvLayer = &m.Layers[layer.SharedKVSource]
		kvHeads = kvLayer.HeadKV
	}
	kvStride := kvLayer.AttnCache.KvStride
	if kvHeads <= 0 {
		panic("attention layer without kv heads")
	}

	q := m.Scratch.Q[:nHead*headDim]
	k := m.Scratch.K[:kvStride]
	v := m.Scratch.V[:kvStride]
	attnOut := m.Scratch.AttnOut
	if cap(attnOut) < nHead*headDim {
		panic(fmt.Sprintf("attnOut capacity too small: cap=%d, needed=%d", cap(attnOut), nHead*headDim))
	}

	var qx *QuantVec
	defer func() {
		if qx != nil {
			ReleaseQuantVec(qx)
		}
	}()

	if kvLayer == layer && !layer.ValueFromKey && !layer.ApplyVNorm {
		if fp, ok := ops.(attentionQKVFastPath); ok && fp.MatVecQKV(q, k, v, layer.Wq, layer.Wk, layer.Wv, x) {
			qx = nil
		} else {
			needSync := true
			syncOnce := func() {
				if needSync {
					syncDeviceSlice(ops, x)
					needSync = false
				}
			}
			var dm deviceMatVecFastPath
			if d, ok := ops.(deviceMatVecFastPath); ok {
				dm = d
			}
			ensureQX := func(w *instance.Mat) {
				syncOnce()
				if qx == nil && CanUseQuantVec(w) {
					qx = PrepareQuantVec(x)
				}
			}
			project := func(dst []float32, w *instance.Mat) {
				if dm != nil && dm.DeviceMatVec(dst, w, x) {
					return
				}
				syncOnce()
				ensureQX(w)
				ops.MatVecWithQuant(dst, w, x, qx)
			}

			if layer.FusedQGate {
				qDim := nHead * headDim
				fused := m.Scratch.AttnGate[:2*qDim]
				project(fused, layer.Wq)
				copy(q, fused[:qDim])
			} else {
				project(q, layer.Wq)
			}
			project(k, layer.Wk)
			project(v, layer.Wv)
		}
	} else {
		qx = nil
		needSync := true
		syncOnce := func() {
			if needSync {
				syncDeviceSlice(ops, x)
				needSync = false
			}
		}
		var dm deviceMatVecFastPath
		if d, ok := ops.(deviceMatVecFastPath); ok {
			dm = d
		}
		ensureQX := func(w *instance.Mat) {
			syncOnce()
			if qx == nil && CanUseQuantVec(w) {
				qx = PrepareQuantVec(x)
			}
		}
		project := func(dst []float32, w *instance.Mat) {
			if dm != nil && dm.DeviceMatVec(dst, w, x) {
				return
			}
			syncOnce()
			ensureQX(w)
			ops.MatVecWithQuant(dst, w, x, qx)
		}

		if layer.FusedQGate {
			qDim := nHead * headDim
			fused := m.Scratch.AttnGate[:2*qDim]
			project(fused, layer.Wq)
			copy(q, fused[:qDim])
		} else {
			project(q, layer.Wq)
		}
		if kvLayer == layer {
			project(k, layer.Wk)
			if layer.ValueFromKey {
				copy(v, k)
				if layer.Wv != nil {
					project(v, layer.Wv)
				}
			} else {
				project(v, layer.Wv)
			}
		}
	}

	if len(layer.WqBias) > 0 {
		Add(q, layer.WqBias)
	}
	if kvLayer == layer && len(layer.WkBias) > 0 {
		Add(k, layer.WkBias)
	}
	if kvLayer == layer && len(layer.WvBias) > 0 {
		Add(v, layer.WvBias)
	}

	if len(layer.AttnQNorm) > 0 {
		batched := false
		if bp, ok := ops.(batchedNormOps); ok {
			batched = bp.RMSNormBatched(q, q, layer.AttnQNorm, m.RMSEpsilon, headDim, nHead)
		}
		if !batched {
			for h := range nHead {
				ops.RMSNorm(q[h*headDim:(h+1)*headDim], q[h*headDim:(h+1)*headDim], layer.AttnQNorm, m.RMSEpsilon)
			}
		}
	}
	if kvLayer == layer && len(layer.AttnKNorm) > 0 {
		batched := false
		if bp, ok := ops.(batchedNormOps); ok {
			batched = bp.RMSNormBatched(k, k, layer.AttnKNorm, m.RMSEpsilon, headDim, kvHeads)
		}
		if !batched {
			for h := range kvHeads {
				ops.RMSNorm(k[h*headDim:(h+1)*headDim], k[h*headDim:(h+1)*headDim], layer.AttnKNorm, m.RMSEpsilon)
			}
		}
	}
	if kvLayer == layer && layer.ApplyVNorm {
		for h := range kvHeads {
			rmsNormNoWeightTo(v[h*headDim:(h+1)*headDim], v[h*headDim:(h+1)*headDim], m.RMSEpsilon)
		}
	}

	applyRoPE := !layer.NoRoPE && (!m.RopeLocalOnly || layer.AttnType != "full_attention")
	if applyRoPE {
		invFreq := layer.RopeInvFreq
		attnScale := layer.RopeAttnScale
		if len(invFreq) == 0 {
			invFreq = m.RopeInvFreq
			attnScale = m.RopeAttnScale
			if layer.AttnType == "sliding_attention" && len(m.RopeInvFreqLocal) > 0 {
				invFreq = m.RopeInvFreqLocal
				attnScale = m.RopeAttnScaleLocal
			}
		}
		ops.ApplyRoPE(q, nHead, headDim, pos, invFreq, attnScale)
		if kvLayer == layer {
			ops.ApplyRoPE(k, kvHeads, headDim, pos, invFreq, attnScale)
		}
	}

	scale := layer.AttnScale
	if scale == 0 {
		scale = float32(1.0 / math.Sqrt(float64(headDim)))
	}
	start := 0
	if layer.AttnWindow > 0 {
		start = max(pos-layer.AttnWindow+1, 0)
	}

	var softcap float32
	if m.Config != nil {
		softcap = m.Config.Config.AttnLogitSoftcap
	}

	flashAttentionEnabled := m.Config != nil && m.Config.Config.FlashAttention && kvLayer == layer

	// FlashAttention fast path (multi-head version) - only if enabled in config
	if flashAttentionEnabled {
		// For incremental processing with KV caching, we need to adapt FlashAttention
		// FlashAttention is most effective when processing full sequences, but we can still
		// leverage parts of it for improved performance

		// First, check if we have a full sequence FlashAttention implementation available
		if fp, ok := ops.(flashAttentionMultiHeadFastPath); ok && fp.FlashAttentionMultiHead(m.Scratch.AttnProj, layer, q, k, v, pos, start, nHead, headDim, kvHeads, kvStride, scale, softcap) {
			return m.Scratch.AttnProj
		}

		// FlashAttention fast path (single-head version)
		if fp, ok := ops.(flashAttentionFastPath); ok && fp.FlashAttention(attnOut[:nHead*headDim], layer, q, k, v, pos, start, nHead, headDim, kvHeads, kvStride, scale, softcap) {
			// If FlashAttention handles the computation, skip the traditional path
			var gate []float32
			if layer.FusedQGate {
				qDim := nHead * headDim
				gate = m.Scratch.AttnGate[qDim : 2*qDim]
			} else {
				gate = m.Scratch.AttnGate[:nHead*headDim]
				if dm, ok := ops.(deviceMatVecFastPath); !ok || !dm.DeviceMatVec(gate, layer.AttnGate, x) {
					// x may still be device-backed from fast-path norm/projections.
					// Ensure host visibility before CPU gate matvec fallback.
					syncDeviceSlice(ops, x)
					if qx == nil && CanUseQuantVec(layer.AttnGate) {
						qx = PrepareQuantVec(x)
					}
					ops.MatVecWithQuant(gate, layer.AttnGate, x, qx)
				}
			}

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
			ops.MatVec(m.Scratch.AttnProj, layer.Wo, attnOut[:nHead*headDim])
			return m.Scratch.AttnProj
		}
	}

	// Alternative: Enhanced attention with optimized kernels for current processing pattern
	// This is where we can add custom optimized implementations for the incremental attention

	// Check for incremental attention fast path
	if kvLayer == layer {
		if fp, ok := ops.(incrementalAttentionFastPath); ok && fp.IncrementalAttention(attnOut[:nHead*headDim], layer, q, k, v, pos, start, nHead, headDim, kvHeads, kvStride, scale, softcap) {
			// If incremental attention handles the computation, skip the traditional path
			if layer.AttnGate != nil || layer.FusedQGate {
				var gate []float32
				if layer.FusedQGate {
					qDim := nHead * headDim
					gate = m.Scratch.AttnGate[qDim : 2*qDim]
				} else {
					gate = m.Scratch.AttnGate[:nHead*headDim]
					if dm, ok := ops.(deviceMatVecFastPath); !ok || !dm.DeviceMatVec(gate, layer.AttnGate, x) {
						// x may still be device-backed from fast-path norm/projections.
						// Ensure host visibility before CPU gate matvec fallback.
						syncDeviceSlice(ops, x)
						if qx == nil && CanUseQuantVec(layer.AttnGate) {
							qx = PrepareQuantVec(x)
						}
						ops.MatVecWithQuant(gate, layer.AttnGate, x, qx)
					}
				}

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
			ops.MatVec(m.Scratch.AttnProj, layer.Wo, attnOut[:nHead*headDim])
			return m.Scratch.AttnProj
		}
	}

	// Combined attention inner + projection fast path (no gate)
	if layer.AttnGate == nil && !layer.FusedQGate {
		if fp, ok := ops.(attentionInnerProjectionFastPath); ok {
			projOK := fp.AttentionInnerProjection(m.Scratch.AttnProj, layer, kvLayer, q, k, v, pos, start, nHead, headDim, kvHeads, kvStride, scale, m.RMSEpsilon, softcap)
			if projOK {
				return m.Scratch.AttnProj
			}
		}
	}

	usedInnerFastPath := false
	if kvLayer == layer {
		if fp, ok := ops.(attentionInnerFastPath); ok {
			usedInnerFastPath = fp.AttentionInner(attnOut[:nHead*headDim], layer, q, k, v, pos, start, nHead, headDim, kvHeads, kvStride, scale, softcap)
		}
	}

	if !usedInnerFastPath {
		cacheRef := &kvLayer.AttnCache
		cacheLen := cacheRef.CacheLen
		cachePos := pos
		if cacheLen > 0 {
			cachePos = pos % cacheLen
		}

		// Ensure host-side cache is allocated if we are falling back to CPU.
		// This is necessary because CUDA fast paths might have skipped host-side allocation.
		cacheRef.EnsurePos(cachePos)

		if kvLayer == layer {
			ops.StoreKV(-1, cachePos, kvStride,
				cacheRef.K, cacheRef.V,
				cacheRef.K16, cacheRef.V16,
				cacheRef.KQ8, cacheRef.VQ8,
				cacheRef.KQ8S, cacheRef.VQ8S,
				k, v)
		}

		ctx := AttnContext{
			Q:         q,
			CacheK:    cacheRef.K,
			CacheV:    cacheRef.V,
			CacheK16:  cacheRef.K16,
			CacheV16:  cacheRef.V16,
			CacheKQ8:  cacheRef.KQ8,
			CacheVQ8:  cacheRef.VQ8,
			CacheKQ8S: cacheRef.KQ8S,
			CacheVQ8S: cacheRef.VQ8S,
			AttnOut:   attnOut,
			Pos:       pos,
			Start:     start,
			KvStride:  kvStride,
			HeadDim:   headDim,
			NHead:     nHead,
			KvHeads:   kvHeads,
			Scale:     scale,
			Softcap:   softcap,
			CacheLen:  cacheRef.CacheLen,
			Ops:       ops,
		}

		RunAttnHeads(&ctx, m.Scratch.Scores, 0, nHead)
	}

	if layer.AttnGate != nil || layer.FusedQGate {
		var gate []float32
		if layer.FusedQGate {
			qDim := nHead * headDim
			gate = m.Scratch.AttnGate[qDim : 2*qDim]
		} else {
			gate = m.Scratch.AttnGate[:nHead*headDim]
			if dm, ok := ops.(deviceMatVecFastPath); !ok || !dm.DeviceMatVec(gate, layer.AttnGate, x) {
				// x may still be device-backed from fast-path norm/projections.
				// Ensure host visibility before CPU gate matvec fallback.
				syncDeviceSlice(ops, x)
				if qx == nil && CanUseQuantVec(layer.AttnGate) {
					qx = PrepareQuantVec(x)
				}
				ops.MatVecWithQuant(gate, layer.AttnGate, x, qx)
			}
		}

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
	ops.MatVec(m.Scratch.AttnProj, layer.Wo, attnOut[:nHead*headDim])
	return m.Scratch.AttnProj
}
