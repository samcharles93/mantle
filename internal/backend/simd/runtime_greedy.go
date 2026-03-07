package simd

import (
	"fmt"

	instance "github.com/samcharles93/mantle/internal/backend/core"
)

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

	ops := m.Ops()

	var ds DeviceStateOps
	if d, ok := ops.(DeviceStateOps); ok {
		ds = d
		ds.BeginToken(x)
		defer ds.EndToken(x)
	}

	type blockFlusher interface {
		FlushBlockResult() error
	}
	var bf blockFlusher
	if f, ok := ops.(blockFlusher); ok {
		bf = f
	}

	for i := range m.Layers {
		layer := &m.Layers[i]

		attnNormFast := ds != nil && ds.DeviceRMSNorm(m.Scratch.Tmp, x, layer.AttnNorm, m.RMSEpsilon)
		if err := consumeFastPathError(ops); err != nil {
			return 0, fmt.Errorf("attention pre-norm fast path failed: %w", err)
		}
		if attnNormFast {
			// device-side pre-norm succeeded
		} else {
			if ds != nil {
				ds.SyncHostState(x)
				if err := consumeFastPathError(ops); err != nil {
					return 0, fmt.Errorf("attention pre-norm sync failed: %w", err)
				}
			}
			ops.RMSNorm(m.Scratch.Tmp, x, layer.AttnNorm, m.RMSEpsilon)
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
			if layer.DeltaNet != nil {
				attnOut = DeltaNet(m, layer, attnIn)
			} else if layer.IsRecurrent {
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
		} else if layer.DeltaNet != nil {
			opOut = DeltaNet(m, layer, m.Scratch.Tmp)
		} else if layer.IsRecurrent {
			opOut = ShortConv(m, layer, m.Scratch.Tmp)
		} else {
			opOut = Attention(m, layer, m.Scratch.Tmp, m.Pos)
		}
		if err := consumeFastPathError(ops); err != nil {
			return 0, fmt.Errorf("attention fast path failed: %w", err)
		}
		if len(layer.PostAttnNorm) > 0 {
			if bf != nil {
				if err := bf.FlushBlockResult(); err != nil {
					return 0, fmt.Errorf("post-attention flush failed: %w", err)
				}
			}
			ops.RMSNorm(m.Scratch.Tmp2, opOut, layer.PostAttnNorm, m.RMSEpsilon)
			opOut = m.Scratch.Tmp2
		}
		addResidual(ds, x, opOut)
		if err := consumeFastPathError(ops); err != nil {
			return 0, fmt.Errorf("attention residual fast path failed: %w", err)
		}

		ffnNormFast := ds != nil && ds.DeviceRMSNorm(m.Scratch.Tmp, x, layer.FfnNorm, m.RMSEpsilon)
		if err := consumeFastPathError(ops); err != nil {
			return 0, fmt.Errorf("ffn pre-norm fast path failed: %w", err)
		}
		if ffnNormFast {
			// device-side pre-norm succeeded
		} else {
			if ds != nil {
				ds.SyncHostState(x)
				if err := consumeFastPathError(ops); err != nil {
					return 0, fmt.Errorf("ffn pre-norm sync failed: %w", err)
				}
			}
			ops.RMSNorm(m.Scratch.Tmp, x, layer.FfnNorm, m.RMSEpsilon)
		}
		var ffnOut []float32
		if layer.MoE != nil {
			syncDeviceSlice(ops, m.Scratch.Tmp)
			if err := consumeFastPathError(ops); err != nil {
				return 0, fmt.Errorf("moe sync fast path failed: %w", err)
			}
			ffnOut = MoE(m, layer, m.Scratch.Tmp)
		} else {
			ffnOut = FFN(m, layer, m.Scratch.Tmp)
		}
		if err := consumeFastPathError(ops); err != nil {
			return 0, fmt.Errorf("ffn fast path failed: %w", err)
		}
		if len(layer.PostFfnNorm) > 0 {
			if bf != nil {
				if err := bf.FlushBlockResult(); err != nil {
					return 0, fmt.Errorf("post-ffn flush failed: %w", err)
				}
			}
			ops.RMSNorm(m.Scratch.Tmp, ffnOut, layer.PostFfnNorm, m.RMSEpsilon)
			ffnOut = m.Scratch.Tmp
		}
		addResidual(ds, x, ffnOut)
		if err := consumeFastPathError(ops); err != nil {
			return 0, fmt.Errorf("ffn residual fast path failed: %w", err)
		}
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
