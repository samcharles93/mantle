package simd

import "fmt"

// ForwardToken runs one autoregressive step for the provided token id.
// It returns a logits slice owned by the model (overwritten on next call).
// Implements model.Model interface.
func (m *Instance) ForwardToken(tok int) ([]float32, error) {
	if tok < 0 || tok >= m.Config.Config.VocabSize {
		return nil, fmt.Errorf("token id out of range: %d", tok)
	}
	if m.Pos >= m.MaxContext {
		return nil, fmt.Errorf("context length exceeded: %d >= %d", m.Pos, m.MaxContext)
	}

	x := m.Scratch.X
	m.Embeddings.RowTo(x, tok)
	if scale := m.Config.Config.EmbeddingMultiplier; scale != 0 && scale != 1 {
		s := float32(scale)
		for i := range x {
			x[i] *= s
		}
	}
	if m.Config.Config.MuPEnabled && m.MuPScale != 1 {
		for i := range x {
			x[i] *= m.MuPScale
		}
	}

	var ds DeviceStateOps
	if d, ok := m.Ops().(DeviceStateOps); ok {
		ds = d
		ds.BeginToken(x)
		defer ds.EndToken(x)
	}

	type blockFlusher interface {
		FlushBlockResult()
	}
	var bf blockFlusher
	if f, ok := m.Ops().(blockFlusher); ok {
		bf = f
	}

	for i := range m.Layers {
		layer := &m.Layers[i]

		// Attention block: pre-norm, attention, optional post-norm, residual
		if ds != nil && ds.DeviceRMSNorm(m.Scratch.Tmp, x, layer.AttnNorm, m.RMSEpsilon) {
			// device-side pre-norm succeeded
		} else {
			if ds != nil {
				ds.SyncHostState(x)
			}
			m.Ops().RMSNorm(m.Scratch.Tmp, x, layer.AttnNorm, m.RMSEpsilon)
		}

		var opOut []float32
		if layer.Mamba != nil {
			var attnOut []float32
			attnIn := m.Scratch.Tmp
			if scale := m.Config.Config.AttentionInMultiplier; scale != 0 && scale != 1 {
				buf := m.Scratch.Tmp2
				s := float32(scale)
				for i := range attnIn {
					buf[i] = attnIn[i] * s
				}
				attnIn = buf
			}
			if layer.IsRecurrent {
				attnOut = ShortConv(m, layer, attnIn)
			} else {
				attnOut = Attention(m, layer, attnIn, m.Pos)
			}
			if scale := m.Config.Config.AttentionOutMultiplier; scale != 0 && scale != 1 {
				s := float32(scale)
				for i := range attnOut {
					attnOut[i] *= s
				}
			}
			mambaOut := Mamba(m, layer, m.Scratch.Tmp)
			if attnOut == nil {
				opOut = mambaOut
			} else if mambaOut == nil {
				opOut = attnOut
			} else {
				opOut = m.Scratch.Tmp2
				copy(opOut, attnOut)
				Add(opOut, mambaOut)
			}
		} else if layer.IsRecurrent {
			opOut = ShortConv(m, layer, m.Scratch.Tmp)
		} else {
			opOut = Attention(m, layer, m.Scratch.Tmp, m.Pos)
		}
		if len(layer.PostAttnNorm) > 0 {
			if bf != nil {
				bf.FlushBlockResult()
			}
			m.Ops().RMSNorm(m.Scratch.Tmp2, opOut, layer.PostAttnNorm, m.RMSEpsilon)
			opOut = m.Scratch.Tmp2
		}
		addResidual(ds, x, opOut)

		// FFN block: pre-norm, dense/MoE, optional post-norm, residual
		if ds != nil && ds.DeviceRMSNorm(m.Scratch.Tmp, x, layer.FfnNorm, m.RMSEpsilon) {
			// device-side pre-norm succeeded
		} else {
			if ds != nil {
				ds.SyncHostState(x)
			}
			m.Ops().RMSNorm(m.Scratch.Tmp, x, layer.FfnNorm, m.RMSEpsilon)
		}
		var ffnOut []float32
		if layer.MoE != nil {
			syncDeviceSlice(m.Ops(), m.Scratch.Tmp)
			ffnOut = MoE(m, layer, m.Scratch.Tmp)
		} else {
			ffnOut = FFN(m, layer, m.Scratch.Tmp)
		}
		if len(layer.PostFfnNorm) > 0 {
			if bf != nil {
				bf.FlushBlockResult()
			}
			m.Ops().RMSNorm(m.Scratch.Tmp, ffnOut, layer.PostFfnNorm, m.RMSEpsilon)
			ffnOut = m.Scratch.Tmp
		}
		addResidual(ds, x, ffnOut)
	}

	// Output norm + projection:
	// 1) prefer device-only RMSNorm + MatVec when available
	// 2) otherwise fallback to fused/host path
	usedDeviceHead := false
	if ds != nil && ds.DeviceRMSNorm(m.Scratch.Tmp, x, m.OutputNorm, m.RMSEpsilon) {
		usedDeviceHead = ds.DeviceMatVec(m.Scratch.Logits, m.Output, m.Scratch.Tmp)
	}
	if !usedDeviceHead {
		if ds != nil {
			ds.SyncHostState(x)
		}
		FusedRMSNormMatVec(m.Ops(), m.Scratch.Logits, m.Output, x, m.OutputNorm, m.RMSEpsilon, m.Scratch.Tmp)
	}
	if scale := m.Config.Config.LMHeadMultiplier; scale != 0 && scale != 1 {
		s := float32(scale)
		for i := range m.Scratch.Logits {
			m.Scratch.Logits[i] *= s
		}
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

// UpdateRoPE recomputes RoPE frequency scaling.
// Implements model.Runtime interface.
func (m *Instance) UpdateRoPE() {
	// Implementation will be moved from model/loader.go
	// For now, this is a placeholder
}
