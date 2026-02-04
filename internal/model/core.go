package model

import (
	"fmt"

	"github.com/samcharles93/mantle/internal/tensor"
)

// ForwardToken runs one autoregressive step for the provided token id.
// It returns a logits slice owned by the model (overwritten on next call).
func (m *Instance) ForwardToken(tok int) ([]float32, error) {
	if tok < 0 || tok >= m.Config.Config.VocabSize {
		return nil, fmt.Errorf("token id out of range: %d", tok)
	}
	if m.Pos >= m.MaxContext {
		return nil, fmt.Errorf("context length exceeded: %d >= %d", m.Pos, m.MaxContext)
	}

	x := m.scratch.x
	m.Embeddings.RowTo(x, tok)
	if m.Config.Config.MuPEnabled && m.muPScale != 1 {
		for i := range x {
			x[i] *= m.muPScale
		}
	}

	for i := range m.Layers {
		layer := &m.Layers[i]

		// Attention block: pre-norm, attention, optional post-norm, residual.
		tensor.RMSNorm(m.scratch.tmp, x, layer.AttnNorm, m.RMSEpsilon)

		var opOut []float32
		if layer.IsRecurrent {
			opOut = m.shortconv(layer, m.scratch.tmp)
		} else {
			opOut = m.attention(layer, m.scratch.tmp, m.Pos)
		}
		if len(layer.PostAttnNorm) > 0 {
			tensor.RMSNorm(m.scratch.tmp2, opOut, layer.PostAttnNorm, m.RMSEpsilon)
			opOut = m.scratch.tmp2
		}
		tensor.Add(x, opOut)

		// FFN block: pre-norm, dense/MoE, optional post-norm, residual.
		tensor.RMSNorm(m.scratch.tmp, x, layer.FfnNorm, m.RMSEpsilon)
		var ffnOut []float32
		if layer.MoE != nil {
			ffnOut = m.moe(layer, m.scratch.tmp)
		} else {
			ffnOut = m.ffn(layer, m.scratch.tmp)
		}
		if len(layer.PostFfnNorm) > 0 {
			tensor.RMSNorm(m.scratch.tmp, ffnOut, layer.PostFfnNorm, m.RMSEpsilon)
			ffnOut = m.scratch.tmp
		}
		tensor.Add(x, ffnOut)
	}

	// output norm
	tensor.RMSNorm(m.scratch.tmp, x, m.OutputNorm, m.RMSEpsilon)
	ensureOps(m.ops).MatVec(m.scratch.logits, m.Output, m.scratch.tmp)

	m.Pos++
	return m.scratch.logits, nil
}

func (m *Instance) Reset() {
	m.Pos = 0
	for i := range m.Layers {
		layer := &m.Layers[i]
		if layer.AttnCache.k != nil {
			for j := range layer.AttnCache.k {
				layer.AttnCache.k[j] = 0
			}
			for j := range layer.AttnCache.v {
				layer.AttnCache.v[j] = 0
			}
		}
		if layer.ShortConvState.buf != nil {
			for j := range layer.ShortConvState.buf {
				layer.ShortConvState.buf[j] = 0
			}
		}
	}
}
