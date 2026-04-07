package simd

import (
	"fmt"

	instance "github.com/samcharles93/mantle/internal/backend/core"
)

// ForwardTokenGreedy runs one autoregressive step and returns the next token id
// using greedy argmax selection. It prefers fully device-resident output head
// + argmax when supported.
func (m *Instance) ForwardTokenGreedy(tok int) (int, error) {
	rt, cleanup, err := prepareTokenRuntimeState(m, tok)
	if err != nil {
		return 0, err
	}
	defer cleanup()

	x := rt.x
	ops := rt.ops
	ds := rt.ds

	if err := runDecoderLayers(m, rt); err != nil {
		return 0, err
	}

	type greedyHeadOps interface {
		DeviceMatVecNoCopy(w *instance.Mat, x []float32) bool
		DeviceArgMaxLastResult() (idx int, ok bool)
	}

	if ds != nil && ds.DeviceRMSNorm(m.Scratch.Tmp, x, m.OutputNorm, m.RMSEpsilon) {
		if gh, ok := ops.(greedyHeadOps); ok && gh.DeviceMatVecNoCopy(m.Output, m.Scratch.Tmp) {
			if next, ok := gh.DeviceArgMaxLastResult(); ok {
				m.Pos++
				return next, nil
			}
		}
	}
	if err := consumeFastPathError(ops); err != nil {
		return 0, fmt.Errorf("output head fast path failed: %w", err)
	}

	if ds != nil {
		ds.SyncHostState(x)
		if err := consumeFastPathError(ops); err != nil {
			return 0, fmt.Errorf("output head sync failed: %w", err)
		}
	}
	FusedRMSNormMatVec(ops, m.Scratch.Logits, m.Output, x, m.OutputNorm, m.RMSEpsilon, m.Scratch.Tmp)
	if scale := m.Config.Config.LMHeadMultiplier; scale != 0 && scale != 1 {
		s := float32(scale)
		for i := range m.Scratch.Logits {
			m.Scratch.Logits[i] *= s
		}
	}

	m.Pos++
	return argmaxHost(m.Scratch.Logits), nil
}

func argmaxHost(x []float32) int {
	if len(x) == 0 {
		panic("argmax: empty slice")
	}
	bestI := 0
	bestV := x[0]
	for i := 1; i < len(x); i++ {
		if x[i] > bestV {
			bestV = x[i]
			bestI = i
		}
	}
	return bestI
}
