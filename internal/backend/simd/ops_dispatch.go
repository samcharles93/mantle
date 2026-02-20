package simd

import (
	"github.com/samcharles93/mantle/internal/backend/core"
	"github.com/samcharles93/mantle/internal/hostcaps"
)

type rmsNormKernel func(dst, src, weight []float32, eps float32)

type applyRoPEKernel func(x []float32, nHead, headDim, pos int, invFreq []float64, attentionFactor float32)

type boundCPUOps struct {
	DefaultOps

	headDim int

	rmsNormFn   rmsNormKernel
	applyRoPEFn applyRoPEKernel
}

func newBoundCPUOps(caps *hostcaps.Snapshot, headDim int) Ops {
	hasAVX2, hasAVX512 := dispatchCPUFeatures(caps)

	ops := &boundCPUOps{
		headDim: headDim,
	}

	switch {
	case hasAVX512:
		ops.rmsNormFn = rmsNormAVX512
	case hasAVX2:
		ops.rmsNormFn = rmsNormSIMD
	default:
		ops.rmsNormFn = rmsNormScalar
	}

	half := headDim / 2
	switch {
	case hasAVX512 && half >= 16:
		ops.applyRoPEFn = func(x []float32, nHead, headDim, pos int, invFreq []float64, attentionFactor float32) {
			applyRoPEAVX512(x, nHead, headDim, pos, invFreq, attentionFactor, half)
		}
	case hasAVX2 && half >= 8:
		ops.applyRoPEFn = func(x []float32, nHead, headDim, pos int, invFreq []float64, attentionFactor float32) {
			applyRoPESIMD(x, nHead, headDim, pos, invFreq, attentionFactor, half)
		}
	default:
		ops.applyRoPEFn = func(x []float32, nHead, headDim, pos int, invFreq []float64, attentionFactor float32) {
			applyRoPEScalar(x, nHead, headDim, pos, invFreq, attentionFactor, half)
		}
	}

	return ops
}

func dispatchCPUFeatures(caps *hostcaps.Snapshot) (hasAVX2, hasAVX512 bool) {
	hasAVX2 = cpu.HasAVX2
	hasAVX512 = cpu.HasAVX512
	if caps != nil {
		hasAVX2 = caps.CPU.HasAVX2
		hasAVX512 = caps.CPU.HasAVX512
	}
	return
}

func (o *boundCPUOps) RMSNorm(dst, src, weight []float32, eps float32) {
	if o != nil && o.rmsNormFn != nil {
		o.rmsNormFn(dst, src, weight, eps)
		return
	}
	rmsNormScalar(dst, src, weight, eps)
}

func (o *boundCPUOps) ApplyRoPE(x []float32, nHead, headDim, pos int, invFreq []float64, attentionFactor float32) {
	if o == nil || o.applyRoPEFn == nil || headDim != o.headDim {
		ApplyRoPE(x, nHead, headDim, pos, invFreq, attentionFactor)
		return
	}
	if headDim%2 != 0 {
		panic("headDim must be even for RoPE")
	}
	if attentionFactor == 0 {
		attentionFactor = 1
	}
	o.applyRoPEFn(x, nHead, headDim, pos, invFreq, attentionFactor)
}

func (m *Instance) bindDefaultOps() {
	if m == nil {
		return
	}
	cm := m.asCore()
	if cm == nil {
		return
	}
	if !isDefaultLikeOps(cm.Ops()) {
		return
	}
	cm.SetOps(newBoundCPUOps(nil, m.HeadDim))
}

func isDefaultLikeOps(ops Ops) bool {
	switch ops.(type) {
	case nil:
		return true
	case *DefaultOps:
		return true
	case core.DefaultOps:
		return true
	case *boundCPUOps:
		return true
	}
	return false
}
