package simd

import "fmt"

// ForwardTokenGreedy runs one autoregressive step and returns the next token id
// using greedy argmax selection. It prefers fully device-resident output head
// + argmax when supported.
func (m *Instance) ForwardTokenGreedy(tok int) (int, error) {
	if tok < 0 || tok >= m.Config.Config.VocabSize {
		return 0, fmt.Errorf("token id out of range: %d", tok)
	}
	if m.Pos >= m.MaxContext {
		return 0, fmt.Errorf("context length exceeded: %d >= %d", m.Pos, m.MaxContext)
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

	type greedyHeadOps interface {
		DeviceMatVecNoCopy(w *Mat, x []float32) bool
		DeviceArgMaxLastResult() (idx int, ok bool)
	}

	if ds != nil && ds.DeviceRMSNorm(m.Scratch.Tmp, x, m.OutputNorm, m.RMSEpsilon) {
		if gh, ok := m.Ops().(greedyHeadOps); ok && gh.DeviceMatVecNoCopy(m.Output, m.Scratch.Tmp) {
			if next, ok := gh.DeviceArgMaxLastResult(); ok {
				m.Pos++
				return next, nil
			}
		}
	}

	if ds != nil {
		ds.SyncHostState(x)
	}
	FusedRMSNormMatVec(m.Ops(), m.Scratch.Logits, m.Output, x, m.OutputNorm, m.RMSEpsilon, m.Scratch.Tmp)
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
