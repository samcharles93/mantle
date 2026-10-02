package simd

import (
	"fmt"
	"math"
	"os"
)

// ForwardTokens advances the model through each token and returns independent
// logits for every position. The single-token path owns attention and KV updates.
func (m *Instance) ForwardTokens(tokens []int) ([][]float32, error) {
	if len(tokens) == 0 {
		return nil, fmt.Errorf("no tokens to process")
	}
	if len(tokens) > m.MaxContext-m.Pos {
		return nil, fmt.Errorf("context length exceeded: %d + %d > %d", m.Pos, len(tokens), m.MaxContext)
	}

	outputs := make([][]float32, 0, len(tokens))
	for _, tok := range tokens {
		logits, err := m.ForwardToken(tok)
		if err != nil {
			return nil, err
		}
		outputs = append(outputs, append([]float32(nil), logits...))
	}
	return outputs, nil
}

// ForwardToken runs one autoregressive step for the provided token id.
// It returns a logits slice owned by the model (overwritten on next call).
// Implements model.Model interface.
func (m *Instance) ForwardToken(tok int) ([]float32, error) {
	rt, cleanup, err := prepareTokenRuntimeState(m, tok)
	if err != nil {
		return nil, err
	}
	defer cleanup()

	x := rt.x
	ops := rt.ops
	ds := rt.ds

	if err := runDecoderLayers(m, rt); err != nil {
		return nil, err
	}

	// Output norm + projection:
	// 1) prefer device-only RMSNorm + MatVec when available
	// 2) otherwise fallback to fused/host path
	usedDeviceHead := false
	if ds != nil && ds.DeviceRMSNorm(m.Scratch.Tmp, x, m.OutputNorm, m.RMSEpsilon) {
		usedDeviceHead = ds.DeviceMatVec(m.Scratch.Logits, m.Output, m.Scratch.Tmp)
		if usedDeviceHead {
			scale := m.Config.Config.LMHeadMultiplier
			if (scale != 0 && scale != 1) || m.Config.Config.FinalLogitSoftcap > 0 {
				// The device MatVec result reaches m.Scratch.Logits only when
				// EndToken flushes it. The postprocessing below reads and
				// rewrites that slice, so force the copy now: otherwise the
				// host math would run on the previous token's values and the
				// deferred flush would then overwrite the postprocessed
				// logits with the raw device result, silently ignoring
				// LMHeadMultiplier and FinalLogitSoftcap.
				//
				// Only forced when postprocessing actually applies, so the
				// common no-multiplier/no-softcap case keeps the deferred copy.
				syncDeviceSlice(ops, m.Scratch.Logits)
			}
		}
	}
	if err := consumeFastPathError(ops); err != nil {
		return nil, fmt.Errorf("output head fast path failed: %w", err)
	}
	if !usedDeviceHead {
		if ds != nil {
			ds.SyncHostState(x)
			if err := consumeFastPathError(ops); err != nil {
				return nil, fmt.Errorf("output head sync failed: %w", err)
			}
		}
		FusedRMSNormMatVec(ops, m.Scratch.Logits, m.Output, x, m.OutputNorm, m.RMSEpsilon, m.Scratch.Tmp)
	}
	if scale := m.Config.Config.LMHeadMultiplier; scale != 0 && scale != 1 {
		s := float32(scale)
		for i := range m.Scratch.Logits {
			m.Scratch.Logits[i] *= s
		}
	}
	if softcap := m.Config.Config.FinalLogitSoftcap; softcap > 0 {
		if ds != nil && ds.DeviceLogitSoftcap(m.Scratch.Logits, softcap) {
			// Done on device
		} else {
			for i := range m.Scratch.Logits {
				m.Scratch.Logits[i] = fastTanh(m.Scratch.Logits[i]/softcap) * softcap
			}
		}
	} else if os.Getenv("MANTLE_DEBUG_GEN") != "" {
		maxV := float32(0)
		for _, v := range m.Scratch.Logits[:min(5, len(m.Scratch.Logits))] {
			if v > maxV {
				maxV = v
			}
		}
		fmt.Fprintf(os.Stderr, "  DEBUG softcap: softcap=%f (skipped, <=0) max_first_5=%f\n", softcap, maxV)
	}

	m.Pos++
	return m.Scratch.Logits, nil
}

func addResidual(ds DeviceStateOps, dst, src []float32) {
	if ds != nil && ds.DeviceAdd(dst, src) {
		return
	}
	if ds != nil {
		ds.SyncHostState(dst)
	}
	Add(dst, src)
	if ds != nil {
		ds.HostStateDirty(dst)
	}
}

// Reset clears the model's internal state (KV cache, etc.).
// Implements model.Model interface.
func (m *Instance) Reset() {
	m.Pos = 0
	for i := range m.Layers {
		layer := &m.Layers[i]
		// KV caches do not need zeroing: attention reads positions [start, pos]
		// which are always written by StoreKV before being read. After Pos = 0,
		// old data is never accessed.
		if layer.ShortConvState.Buf != nil {
			for j := range layer.ShortConvState.Buf {
				layer.ShortConvState.Buf[j] = 0
			}
		}
		if layer.DeltaNet != nil {
			for j := range layer.DeltaNet.ConvState {
				layer.DeltaNet.ConvState[j] = 0
			}
			for j := range layer.DeltaNet.RecurrentState {
				layer.DeltaNet.RecurrentState[j] = 0
			}
		}
		if layer.Mamba != nil {
			if layer.Mamba.ConvState != nil {
				for j := range layer.Mamba.ConvState {
					layer.Mamba.ConvState[j] = 0
				}
			}
			if layer.Mamba.SSMState != nil {
				for j := range layer.Mamba.SSMState {
					layer.Mamba.SSMState[j] = 0
				}
			}
		}
	}
	// Invalidate device-resident conv states so they are re-uploaded from zeroed host buffers.
	type convResetter interface {
		ResetConvStates()
	}
	if cr, ok := m.Ops().(convResetter); ok {
		cr.ResetConvStates()
	}
}

// PrecomputeRoPETables precomputes RoPE tables for the maximum context length.
func (m *Instance) PrecomputeRoPETables() {
	if len(m.RopeInvFreq) == 0 {
		return // Nothing to precompute
	}

	half := len(m.RopeInvFreq)
	totalEntries := m.MaxContext * half

	m.RopeCosTable = make([]float32, totalEntries)
	m.RopeSinTable = make([]float32, totalEntries)

	// Precompute sin/cos values for all positions and frequencies
	for pos := 0; pos < m.MaxContext; pos++ {
		for i := range half {
			angle := float64(pos) * m.RopeInvFreq[i]
			cosVal := float32(math.Cos(angle)) * m.RopeAttnScale
			sinVal := float32(math.Sin(angle)) * m.RopeAttnScale

			idx := pos*half + i
			m.RopeCosTable[idx] = cosVal
			m.RopeSinTable[idx] = sinVal
		}
	}
}

// UpdateRoPE recomputes RoPE frequency scaling.
// Implements model.Runtime interface.
func (m *Instance) UpdateRoPE() {
	// Recompute RoPE tables when RoPE parameters change
	m.PrecomputeRoPETables()
}
