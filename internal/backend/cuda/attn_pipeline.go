//go:build cuda

package cuda

import (
	"fmt"
	"math"
	"runtime"
	"unsafe"

	instance "github.com/samcharles93/mantle/internal/backend/core"
	"github.com/samcharles93/mantle/internal/backend/cuda/native"
	"github.com/samcharles93/mantle/pkg/mcf"
)

// ropeInvFreqDevEntry caches a per-layer RoPE inverse-frequency table on the
// device. The table is converted F64 -> F32 once and reused across tokens.
type ropeInvFreqDevEntry struct {
	buf  native.DeviceBuffer
	half int
}

// InvalidateRoPECaches frees any cached RoPE inverse-frequency tables on the
// device. Must be called whenever the host-side RopeInvFreq slices are
// reallocated (e.g. UpdateRoPE / updateInstanceRoPE) because the cache is
// keyed by the host slice data pointer.
func (o *Ops) InvalidateRoPECaches() {
	o.mu.Lock()
	defer o.mu.Unlock()
	for _, entry := range o.ropeInvFreqDev {
		_ = entry.buf.Free()
	}
	o.ropeInvFreqDev = make(map[uintptr]ropeInvFreqDevEntry)
}

// ensureRoPEInvFreqDev returns a device buffer containing the RoPE
// inverse-frequency table converted to F32. The buffer is cached and reused
// for subsequent calls with the same host slice.
//
// Requires o.mu held.
func (o *Ops) ensureRoPEInvFreqDev(invFreq []float64) (native.DeviceBuffer, int, error) {
	if len(invFreq) == 0 {
		return native.DeviceBuffer{}, 0, fmt.Errorf("empty rope inv-freq table")
	}
	key := uintptr(unsafe.Pointer(&invFreq[0]))
	if entry, ok := o.ropeInvFreqDev[key]; ok && entry.half == len(invFreq) {
		return entry.buf, entry.half, nil
	}
	half := len(invFreq)
	bytes := int64(half) * int64(unsafe.Sizeof(float32(0)))
	buf, err := native.AllocDevice(bytes)
	if err != nil {
		return native.DeviceBuffer{}, 0, err
	}
	hostF32 := make([]float32, half)
	for i, v := range invFreq {
		hostF32[i] = float32(v)
	}
	if err := native.MemcpyH2D(buf, unsafe.Pointer(&hostF32[0]), bytes); err != nil {
		_ = buf.Free()
		return native.DeviceBuffer{}, 0, err
	}
	o.ropeInvFreqDev[key] = ropeInvFreqDevEntry{buf: buf, half: half}
	return buf, half, nil
}

// ensureQKVProjDev ensures the three persistent projection buffers are
// allocated and at least the requested size.
//
// Requires o.mu held.
func (o *Ops) ensureQKVProjDev(qBytes, kBytes, vBytes int) error {
	grow := func(buf *native.DeviceBuffer, have *int, want int) error {
		if want <= *have {
			return nil
		}
		if err := buf.Free(); err != nil {
			return err
		}
		nb, err := native.AllocDevice(int64(want))
		if err != nil {
			return err
		}
		*buf = nb
		*have = want
		return nil
	}
	if err := grow(&o.qProjDev, &o.qProjDevBytes, qBytes); err != nil {
		return err
	}
	if err := grow(&o.kProjDev, &o.kProjDevBytes, kBytes); err != nil {
		return err
	}
	if err := grow(&o.vProjDev, &o.vProjDevBytes, vBytes); err != nil {
		return err
	}
	return nil
}

// projectInto runs one matvec with weight w and device-resident input xDev
// of length w.C, writing w.R float32 outputs into dst. Dispatches over the
// quantized and dense weight paths without touching host memory.
//
// Requires o.mu held.
func (o *Ops) projectInto(dst native.DeviceBuffer, w *instance.Mat, xDev native.DeviceBuffer) error {
	if w == nil {
		return fmt.Errorf("projectInto: nil weight")
	}
	if dst.Ptr() == nil {
		return fmt.Errorf("projectInto: nil dst buffer")
	}

	// Quant path (either pre-cached or promotable)
	if _, ok := o.qweights[w]; ok {
		qw, err := o.ensureQuantMat(w)
		if err != nil {
			return err
		}
		return o.runQuantKernel(qw, xDev, dst, o.stream)
	}
	if useQuantKernel() && (shouldPreferQuantWeights(w, currentCUDAWeightMode()) || (w.Quant != nil && w.Quant.ValidFor(w))) {
		qw, err := o.ensureQuantMat(w)
		if err != nil {
			return err
		}
		return o.runQuantKernel(qw, xDev, dst, o.stream)
	}

	// Dense GEMM path
	devW, err := o.deviceMat(w)
	if err != nil {
		return err
	}
	xInput := xDev
	xInputType := native.BlasF32
	if devW.dtype != native.BlasF32 {
		switch devW.dtype {
		case native.BlasF16:
			xBytes := int64(w.C) * 2
			if err := o.ensureDeviceVecs(int(xBytes), 0); err != nil {
				return err
			}
			if err := native.ConvertF32ToF16(xDev, o.xDev, w.C, o.stream); err != nil {
				return err
			}
			xInput = o.xDev
			xInputType = native.BlasF16
		case native.BlasBF16:
			xBytes := int64(w.C) * 2
			if err := o.ensureDeviceVecs(int(xBytes), 0); err != nil {
				return err
			}
			if err := native.ConvertF32ToBF16(xDev, o.xDev, w.C, o.stream); err != nil {
				return err
			}
			xInput = o.xDev
			xInputType = native.BlasBF16
		default:
			return fmt.Errorf("projectInto: unsupported weight dtype %d", devW.dtype)
		}
	}
	return native.GemmEx(
		o.blas,
		native.BlasOpT,
		native.BlasOpN,
		w.R, 1, w.C,
		1.0,
		devW.buf, devW.dtype, w.C,
		xInput, xInputType, w.C,
		0.0,
		dst, native.BlasF32, w.R,
		native.BlasComputeF32,
		native.BlasGemmDefault,
	)
}

// ensureOnesDev returns a device buffer of F32 ones with at least n entries.
// Used as an identity weight vector for batched RMSNorm without a learnable
// gain (e.g. Gemma V-norm).
//
// Requires o.mu held.
func (o *Ops) ensureOnesDev(n int) (native.DeviceBuffer, error) {
	if n <= 0 {
		return native.DeviceBuffer{}, fmt.Errorf("ensureOnesDev: non-positive length %d", n)
	}
	if buf, ok := o.onesDev[n]; ok {
		return buf, nil
	}
	bytes := int64(n) * int64(unsafe.Sizeof(float32(0)))
	buf, err := native.AllocDevice(bytes)
	if err != nil {
		return native.DeviceBuffer{}, err
	}
	host := make([]float32, n)
	for i := range host {
		host[i] = 1
	}
	if err := native.MemcpyH2D(buf, unsafe.Pointer(&host[0]), bytes); err != nil {
		_ = buf.Free()
		return native.DeviceBuffer{}, err
	}
	o.onesDev[n] = buf
	return buf, nil
}

// mirrorKVRowToHost copies the just-stored K and V row back from the
// device KV cache into the layer's host AttnCache. This is used for
// shared-KV source layers whose consumer layer(s) may run on the host
// attention path and read directly from the host cache.
//
// The call blocks on the op stream at the end so that subsequent host
// reads observe the mirrored data.
//
// Requires o.mu held.
func (o *Ops) mirrorKVRowToHost(cache deviceAttnCache, layer *instance.Layer, cachePos, kvStride, blocksPerStride int) error {
	cacheRef := &layer.AttnCache
	cacheRef.EnsurePos(cachePos)
	hostOffset := cachePos * kvStride

	if cache.useQ8K {
		if len(cacheRef.KQ8) < hostOffset+kvStride {
			return fmt.Errorf("host KQ8 too small: have %d need %d", len(cacheRef.KQ8), hostOffset+kvStride)
		}
		if err := native.MemcpyD2HAsyncAt(unsafe.Pointer(&cacheRef.KQ8[hostOffset]), cache.kQ8, int64(hostOffset), int64(kvStride), o.stream); err != nil {
			return err
		}
		if cacheRef.KQ8S != nil && cachePos < len(cacheRef.KQ8S) {
			scaleOff := int64(cachePos) * int64(blocksPerStride) * int64(unsafe.Sizeof(float32(0)))
			if err := native.MemcpyD2HAsyncAt(unsafe.Pointer(&cacheRef.KQ8S[cachePos]), cache.kQ8Scales, scaleOff, int64(unsafe.Sizeof(float32(0))), o.stream); err != nil {
				return err
			}
		}
	} else if cacheRef.K16 != nil {
		if len(cacheRef.K16) < hostOffset+kvStride {
			return fmt.Errorf("host K16 too small: have %d need %d", len(cacheRef.K16), hostOffset+kvStride)
		}
		byteOff := int64(hostOffset) * 2
		if err := native.MemcpyD2HAsyncAt(unsafe.Pointer(&cacheRef.K16[hostOffset]), cache.kF16, byteOff, int64(kvStride)*2, o.stream); err != nil {
			return err
		}
	}

	if cache.useQ8V {
		if len(cacheRef.VQ8) < hostOffset+kvStride {
			return fmt.Errorf("host VQ8 too small: have %d need %d", len(cacheRef.VQ8), hostOffset+kvStride)
		}
		if err := native.MemcpyD2HAsyncAt(unsafe.Pointer(&cacheRef.VQ8[hostOffset]), cache.vQ8, int64(hostOffset), int64(kvStride), o.stream); err != nil {
			return err
		}
		if cacheRef.VQ8S != nil && cachePos < len(cacheRef.VQ8S) {
			scaleOff := int64(cachePos) * int64(blocksPerStride) * int64(unsafe.Sizeof(float32(0)))
			if err := native.MemcpyD2HAsyncAt(unsafe.Pointer(&cacheRef.VQ8S[cachePos]), cache.vQ8Scales, scaleOff, int64(unsafe.Sizeof(float32(0))), o.stream); err != nil {
				return err
			}
		}
	} else if cacheRef.V16 != nil {
		if len(cacheRef.V16) < hostOffset+kvStride {
			return fmt.Errorf("host V16 too small: have %d need %d", len(cacheRef.V16), hostOffset+kvStride)
		}
		byteOff := int64(hostOffset) * 2
		if err := native.MemcpyD2HAsyncAt(unsafe.Pointer(&cacheRef.V16[hostOffset]), cache.vF16, byteOff, int64(kvStride)*2, o.stream); err != nil {
			return err
		}
	}

	// Ensure the D2H completes before any later host-side consumer
	// layer reads the host cache directly.
	return o.stream.Synchronize()
}

// SyncAttentionCacheToHost materializes the current device-side KV cache into
// the layer's host AttnCache when a shared-KV consumer falls back to the host
// attention path.
func (o *Ops) SyncAttentionCacheToHost(layer *instance.Layer, pos int) {
	if o == nil || layer == nil || pos < 0 {
		return
	}

	o.mu.Lock()
	defer o.mu.Unlock()

	devPos, ok := o.lastDevKVPos[layer]
	if !ok || devPos != pos {
		// No device-side cache update for this token step, so the host path is
		// already authoritative (or there is nothing to sync).
		return
	}
	if hostPos, ok := o.lastHostKVPos[layer]; ok && hostPos == pos {
		return
	}

	cache, ok := o.attnCaches[layer]
	if !ok {
		return
	}

	rows := min(pos+1, cache.cacheLen)
	if rows <= 0 {
		return
	}

	cacheRef := &layer.AttnCache
	cacheRef.EnsurePos(rows - 1)

	totalElems, ok := mulInt(rows, cache.kvStride)
	if !ok {
		panic(fmt.Errorf("cuda: sync attention cache overflow (rows=%d stride=%d)", rows, cache.kvStride))
	}
	blocksPerStride := cache.kvStride / int(mcf.QuantBlockSize)
	if blocksPerStride < 1 {
		blocksPerStride = 1
	}

	copyScales := func(dst []float32, src native.DeviceBuffer, label string) {
		if len(dst) < rows {
			panic(fmt.Errorf("cuda: host %s scales too small: have %d need %d", label, len(dst), rows))
		}
		scaleCount, ok := mulInt(rows, blocksPerStride)
		if !ok {
			panic(fmt.Errorf("cuda: %s scale sync overflow (rows=%d blocks=%d)", label, rows, blocksPerStride))
		}
		scaleTmp := make([]float32, scaleCount)
		if err := native.MemcpyD2H(unsafe.Pointer(&scaleTmp[0]), src, int64(scaleCount)*4); err != nil {
			panic(fmt.Errorf("cuda: sync %s scales failed: %w", label, err))
		}
		for i := range rows {
			dst[i] = scaleTmp[i*blocksPerStride]
		}
	}

	if cache.useQ8K {
		if len(cacheRef.KQ8) < totalElems {
			panic(fmt.Errorf("cuda: host KQ8 too small: have %d need %d", len(cacheRef.KQ8), totalElems))
		}
		if err := native.MemcpyD2H(unsafe.Pointer(&cacheRef.KQ8[0]), cache.kQ8, int64(totalElems)); err != nil {
			panic(fmt.Errorf("cuda: sync KQ8 cache failed: %w", err))
		}
		if cacheRef.KQ8S != nil {
			copyScales(cacheRef.KQ8S, cache.kQ8Scales, "KQ8")
		}
	} else if cacheRef.K16 != nil {
		if len(cacheRef.K16) < totalElems {
			panic(fmt.Errorf("cuda: host K16 too small: have %d need %d", len(cacheRef.K16), totalElems))
		}
		if err := native.MemcpyD2H(unsafe.Pointer(&cacheRef.K16[0]), cache.kF16, int64(totalElems)*2); err != nil {
			panic(fmt.Errorf("cuda: sync K16 cache failed: %w", err))
		}
	}

	if cache.useQ8V {
		if len(cacheRef.VQ8) < totalElems {
			panic(fmt.Errorf("cuda: host VQ8 too small: have %d need %d", len(cacheRef.VQ8), totalElems))
		}
		if err := native.MemcpyD2H(unsafe.Pointer(&cacheRef.VQ8[0]), cache.vQ8, int64(totalElems)); err != nil {
			panic(fmt.Errorf("cuda: sync VQ8 cache failed: %w", err))
		}
		if cacheRef.VQ8S != nil {
			copyScales(cacheRef.VQ8S, cache.vQ8Scales, "VQ8")
		}
	} else if cacheRef.V16 != nil {
		if len(cacheRef.V16) < totalElems {
			panic(fmt.Errorf("cuda: host V16 too small: have %d need %d", len(cacheRef.V16), totalElems))
		}
		if err := native.MemcpyD2H(unsafe.Pointer(&cacheRef.V16[0]), cache.vF16, int64(totalElems)*2); err != nil {
			panic(fmt.Errorf("cuda: sync V16 cache failed: %w", err))
		}
	}

	o.lastHostKVPos[layer] = pos
}

// QKVAttentionProjection is the fused end-to-end attention fast path.
//
// On success it performs the entire attention block on device:
//
//	Wq/Wk/Wv matvec → bias → RMSNorm(Q,K) [+ V-norm] → RoPE(Q,K) →
//	KV-store → attention kernel → Wo matvec → lazy D2H of projOut.
//
// Only the Wo projection output is mapped back to the host via the
// existing setLastResult mechanism. When the function returns false,
// no device state has been mutated that would invalidate a fallback
// to the host attention path.
//
// If mirrorHostKV is true (source layers only), the just-stored K/V
// row is also copied back to the layer's host AttnCache so that any
// shared-KV consumer that falls back to the host attention path reads
// valid data.
//
// Shared-KV consumer layers (kvLayer != layer) are supported when the
// source's device-side KV cache was already updated at this pos by a
// prior fused call within the same token step.
func (o *Ops) QKVAttentionProjection(
	projOut []float32,
	layer, kvLayer *instance.Layer,
	x []float32,
	pos, start, nHead, headDim, kvHeads, kvStride int,
	scale, epsilon, softcap float32,
	invFreq []float64,
	ropeAttnScale float32,
	applyRope bool,
	mirrorHostKV bool,
) bool {
	if !useAttentionInnerFastPath() {
		return false
	}
	if o == nil || layer == nil {
		return false
	}
	if kvLayer == nil {
		kvLayer = layer
	}
	isConsumer := kvLayer != layer

	if layer.Wq == nil || layer.Wo == nil {
		return false
	}
	if !isConsumer {
		if layer.Wk == nil {
			return false
		}
		// Wv may be nil iff ValueFromKey (V derived from K projection).
		if layer.Wv == nil && !layer.ValueFromKey {
			return false
		}
	}
	if nHead <= 0 || headDim <= 0 || kvHeads <= 0 || kvStride <= 0 {
		return false
	}
	if pos < 0 || start < 0 || start > pos {
		return false
	}
	if applyRope && headDim%2 != 0 {
		return false
	}
	if kvStride%int(mcf.QuantBlockSize) != 0 {
		return false
	}
	qDim := nHead * headDim
	if layer.Wq.R != qDim {
		return false
	}
	if !isConsumer {
		if layer.Wk.R != kvStride {
			return false
		}
		if layer.Wv != nil && layer.Wv.R != kvStride {
			return false
		}
		if layer.Wq.C == 0 || layer.Wq.C != layer.Wk.C {
			return false
		}
		if layer.Wv != nil && layer.Wv.C != layer.Wq.C {
			return false
		}
	} else if layer.Wq.C == 0 {
		return false
	}
	projDim := layer.Wo.R
	if layer.Wo.C != qDim || projDim <= 0 {
		return false
	}
	if len(projOut) < projDim || len(x) < layer.Wq.C {
		return false
	}

	// Consumer reads KV from kvLayer's cache; source uses its own.
	cacheOwner := kvLayer
	if cacheOwner.AttnCache.CacheLen <= 0 {
		return false
	}

	// Bias sanity: allowed to be empty or exactly match projection size.
	if n := len(layer.WqBias); n != 0 && n != qDim {
		return false
	}
	if !isConsumer {
		if n := len(layer.WkBias); n != 0 && n != kvStride {
			return false
		}
		if n := len(layer.WvBias); n != 0 && n != kvStride {
			return false
		}
	}
	if n := len(layer.AttnQNorm); n != 0 && n != headDim {
		return false
	}
	if !isConsumer {
		if n := len(layer.AttnKNorm); n != 0 && n != headDim {
			return false
		}
	}

	// Reject weights intentionally kept on the host; the fused path
	// relies on all matmuls running on device.
	if _, off := o.offloadedMats[layer.Wq]; off {
		return false
	}
	if !isConsumer {
		if _, off := o.offloadedMats[layer.Wk]; off {
			return false
		}
		if layer.Wv != nil {
			if _, off := o.offloadedMats[layer.Wv]; off {
				return false
			}
		}
	}
	if _, off := o.offloadedMats[layer.Wo]; off {
		return false
	}

	o.mu.Lock()
	defer o.mu.Unlock()

	// Consumer requires the source's device KV cache to have been
	// written at this pos by the source's own fused invocation
	// earlier in this forward step.
	if isConsumer {
		if p, ok := o.lastDevKVPos[kvLayer]; !ok || p != pos {
			return false
		}
	}

	// Drain any pending deferred D2H from a previous block before we
	// reuse shared scratch (yDev, zDev, etc.). If this fails it is a
	// hard error; keep it on fastPathErr and bail.
	if err := o.flushLastResult(); err != nil {
		o.recordFastPathErrorLocked(err)
		return false
	}

	// Locate x on the device. If it is not already resident, fall back
	// to the host path - uploading here would negate the savings.
	xRef, usedDev, err := o.deviceInputForVector(x[:layer.Wq.C], native.BlasF32, layer.Wq.C)
	if err != nil {
		o.recordFastPathErrorLocked(err)
		return false
	}
	if !usedDev || xRef.dtype != native.BlasF32 {
		return false
	}
	xDev := xRef.buf

	cacheLen := cacheOwner.AttnCache.CacheLen
	actualCacheLen := cacheLen
	if o.effectiveContextLen > 0 && actualCacheLen > o.effectiveContextLen {
		actualCacheLen = o.effectiveContextLen
	}
	if pos >= actualCacheLen {
		return false
	}

	cache, err := o.ensureAttnCache(cacheOwner, kvStride, actualCacheLen)
	if err != nil {
		o.recordFastPathErrorLocked(err)
		return false
	}
	if cache.kvStride != kvStride || cache.cacheLen != actualCacheLen {
		return false
	}

	// Consumer only needs q-proj scratch; still share persistent
	// buffers with the source path for simplicity.
	if err := o.ensureQKVProjDev(qDim*4, kvStride*4, kvStride*4); err != nil {
		o.recordFastPathErrorLocked(err)
		return false
	}
	// yDev is reused as attention-inner output (size qDim*4).
	// zDev is reused as Wo projection output (size projDim*4).
	if err := o.ensureDeviceVecs(0, qDim*4); err != nil {
		o.recordFastPathErrorLocked(err)
		return false
	}
	if err := o.ensureNormTmp(projDim * 4); err != nil {
		o.recordFastPathErrorLocked(err)
		return false
	}

	// Wq projection (always). Wk/Wv/K-store only on source layers.
	if err := o.projectInto(o.qProjDev, layer.Wq, xDev); err != nil {
		o.recordFastPathErrorLocked(err)
		return false
	}
	if !isConsumer {
		if err := o.projectInto(o.kProjDev, layer.Wk, xDev); err != nil {
			o.recordFastPathErrorLocked(err)
			return false
		}
		if layer.Wv != nil {
			if err := o.projectInto(o.vProjDev, layer.Wv, xDev); err != nil {
				o.recordFastPathErrorLocked(err)
				return false
			}
		} else {
			// ValueFromKey: V starts as a copy of post-projection K,
			// before any bias/norm has been applied.
			vBytes := int64(kvStride) * int64(unsafe.Sizeof(float32(0)))
			if err := native.MemcpyD2DAsync(o.vProjDev, o.kProjDev, vBytes, o.stream); err != nil {
				o.recordFastPathErrorLocked(err)
				return false
			}
		}
	}

	// Biases (optional).
	if len(layer.WqBias) == qDim {
		bDev, err := o.deviceNormWeight(layer.WqBias)
		if err != nil {
			o.recordFastPathErrorLocked(err)
			return false
		}
		if err := native.AddVectorsF32(o.qProjDev, bDev, qDim, o.stream); err != nil {
			o.recordFastPathErrorLocked(err)
			return false
		}
	}
	if !isConsumer {
		if len(layer.WkBias) == kvStride {
			bDev, err := o.deviceNormWeight(layer.WkBias)
			if err != nil {
				o.recordFastPathErrorLocked(err)
				return false
			}
			if err := native.AddVectorsF32(o.kProjDev, bDev, kvStride, o.stream); err != nil {
				o.recordFastPathErrorLocked(err)
				return false
			}
		}
		if len(layer.WvBias) == kvStride {
			bDev, err := o.deviceNormWeight(layer.WvBias)
			if err != nil {
				o.recordFastPathErrorLocked(err)
				return false
			}
			if err := native.AddVectorsF32(o.vProjDev, bDev, kvStride, o.stream); err != nil {
				o.recordFastPathErrorLocked(err)
				return false
			}
		}
	}

	// RMSNorm on Q (per-head) always; K/V only on source layers.
	if len(layer.AttnQNorm) == headDim {
		wDev, err := o.deviceNormWeight(layer.AttnQNorm)
		if err != nil {
			o.recordFastPathErrorLocked(err)
			return false
		}
		if err := native.RMSNormBatchedF32(o.qProjDev, o.qProjDev, wDev, epsilon, headDim, nHead, o.stream); err != nil {
			o.recordFastPathErrorLocked(err)
			return false
		}
	}
	if !isConsumer {
		if len(layer.AttnKNorm) == headDim {
			wDev, err := o.deviceNormWeight(layer.AttnKNorm)
			if err != nil {
				o.recordFastPathErrorLocked(err)
				return false
			}
			if err := native.RMSNormBatchedF32(o.kProjDev, o.kProjDev, wDev, epsilon, headDim, kvHeads, o.stream); err != nil {
				o.recordFastPathErrorLocked(err)
				return false
			}
		}
		if layer.ApplyVNorm {
			wDev, err := o.ensureOnesDev(headDim)
			if err != nil {
				o.recordFastPathErrorLocked(err)
				return false
			}
			if err := native.RMSNormBatchedF32(o.vProjDev, o.vProjDev, wDev, epsilon, headDim, kvHeads, o.stream); err != nil {
				o.recordFastPathErrorLocked(err)
				return false
			}
		}
	}

	// RoPE in-place on Q (always when applyRope); on K only for source.
	if applyRope {
		rDev, half, err := o.ensureRoPEInvFreqDev(invFreq)
		if err != nil {
			o.recordFastPathErrorLocked(err)
			return false
		}
		if half*2 > headDim {
			return false
		}
		if err := native.ApplyRoPEInplaceF32(o.qProjDev, rDev, pos, ropeAttnScale, headDim, half, nHead, o.stream); err != nil {
			o.recordFastPathErrorLocked(err)
			return false
		}
		if !isConsumer {
			if err := native.ApplyRoPEInplaceF32(o.kProjDev, rDev, pos, ropeAttnScale, headDim, half, kvHeads, o.stream); err != nil {
				o.recordFastPathErrorLocked(err)
				return false
			}
		}
	}

	// Store K/V into the ring cache (source only).
	cachePos := pos
	if cacheLen > 0 {
		cachePos = pos % cacheLen
	}
	if cachePos < 0 || cachePos >= actualCacheLen {
		return false
	}
	blocksPerStride := kvStride / int(mcf.QuantBlockSize)
	if blocksPerStride < 1 {
		blocksPerStride = 1
	}

	if !isConsumer {
		if cache.useQ8K {
			if err := native.StoreKVQ8RowBroadcast(cache.kQ8, cache.kQ8Scales, o.kProjDev, cachePos, kvStride, blocksPerStride, o.stream); err != nil {
				o.recordFastPathErrorLocked(err)
				return false
			}
		} else {
			if err := native.StoreKVF16Row(cache.kF16, o.kProjDev, cachePos, kvStride, o.stream); err != nil {
				o.recordFastPathErrorLocked(err)
				return false
			}
		}
		if cache.useQ8V {
			if err := native.StoreKVQ8RowBroadcast(cache.vQ8, cache.vQ8Scales, o.vProjDev, cachePos, kvStride, blocksPerStride, o.stream); err != nil {
				o.recordFastPathErrorLocked(err)
				return false
			}
		} else {
			if err := native.StoreKVF16Row(cache.vF16, o.vProjDev, cachePos, kvStride, o.stream); err != nil {
				o.recordFastPathErrorLocked(err)
				return false
			}
		}
		// Record that this source layer's device KV cache now holds
		// valid data at this pos for any consumer dispatched later in
		// the forward step.
		o.lastDevKVPos[layer] = pos

		// Optional: mirror the just-stored K/V row back to the host
		// AttnCache so that a shared-KV consumer that falls back to
		// the host attention path reads valid data.
		if mirrorHostKV {
			if err := o.mirrorKVRowToHost(cache, layer, cachePos, kvStride, blocksPerStride); err != nil {
				o.recordFastPathErrorLocked(err)
				return false
			}
			o.lastHostKVPos[layer] = pos
		}
	}

	// Attention inner kernel.
	if scale == 0 {
		scale = float32(1.0 / math.Sqrt(float64(headDim)))
	}
	if cache.useQ8K || cache.useQ8V {
		if err := native.AttentionInnerMixedCacheF32(
			o.qProjDev,
			cache.kF16, cache.vF16,
			cache.kQ8, cache.vQ8,
			cache.kQ8Scales, cache.vQ8Scales,
			o.yDev,
			cache.useQ8K, cache.useQ8V,
			pos, start,
			kvStride, headDim, nHead, kvHeads,
			actualCacheLen,
			scale, softcap,
			o.stream,
		); err != nil {
			o.recordFastPathErrorLocked(err)
			return false
		}
	} else {
		if err := native.AttentionInnerF16CacheF32(
			o.qProjDev, cache.kF16, cache.vF16,
			o.yDev,
			pos, start,
			kvStride, headDim, nHead, kvHeads,
			actualCacheLen,
			scale, softcap,
			o.stream,
		); err != nil {
			o.recordFastPathErrorLocked(err)
			return false
		}
	}

	// Wo projection: qDim -> projDim, into zDev.
	if err := o.projectInto(o.zDev, layer.Wo, o.yDev); err != nil {
		o.recordFastPathErrorLocked(err)
		return false
	}

	// Map the projection output to the host slice via the existing
	// lazy D2H mechanism; the actual copy happens on the next
	// host-visible access.
	o.setLastResult(o.zDev, projOut, projDim)
	runtime.KeepAlive(projOut)
	runtime.KeepAlive(x)
	return true
}
