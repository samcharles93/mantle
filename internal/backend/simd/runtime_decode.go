package simd

import (
	"fmt"
	"os"
)

type blockFlusher interface {
	FlushBlockResult() error
}

type tokenRuntimeState struct {
	x              []float32
	perLayerInputs []float32
	ops            Ops
	ds             DeviceStateOps
	bf             blockFlusher
	debug          layerStateDebugger
}

type layerStateDebugger struct {
	enabled bool
}

func newLayerStateDebugger() layerStateDebugger {
	return layerStateDebugger{enabled: os.Getenv("MANTLE_DEBUG_LAYER_STATE") != ""}
}

func initializeTokenInput(m *Instance, tok int) ([]float32, error) {
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
	return x, nil
}

func prepareTokenRuntimeState(m *Instance, tok int) (*tokenRuntimeState, func(), error) {
	x, err := initializeTokenInput(m, tok)
	if err != nil {
		return nil, nil, err
	}

	state := &tokenRuntimeState{
		x:     x,
		ops:   m.Ops(),
		debug: newLayerStateDebugger(),
	}
	if d, ok := state.ops.(DeviceStateOps); ok {
		state.ds = d
		state.ds.BeginToken(x)
	}
	state.perLayerInputs = prepareGemma4PerLayerInputs(m, tok, x, state.ops, state.ds)
	if f, ok := state.ops.(blockFlusher); ok {
		state.bf = f
	}

	cleanup := func() {
		if state.ds != nil {
			state.ds.EndToken(x)
		}
	}
	return state, cleanup, nil
}

func runDecoderLayers(m *Instance, rt *tokenRuntimeState) error {
	x := rt.x
	ops := rt.ops
	ds := rt.ds
	bf := rt.bf

	for i := range m.Layers {
		layer := &m.Layers[i]

		prefetchLayer(ops, i+1)

		if err := rt.debug.logPreNormX(rt, i); err != nil {
			return err
		}

		attnNormFast := ds != nil && ds.DeviceRMSNorm(m.Scratch.Tmp, x, layer.AttnNorm, m.RMSEpsilon)
		if err := consumeFastPathError(ops); err != nil {
			return fmt.Errorf("attention pre-norm fast path failed: %w", err)
		}
		if !attnNormFast {
			if ds != nil {
				ds.SyncHostState(x)
				if err := consumeFastPathError(ops); err != nil {
					return fmt.Errorf("attention pre-norm sync failed: %w", err)
				}
			}
			ops.RMSNorm(m.Scratch.Tmp, x, layer.AttnNorm, m.RMSEpsilon)
		}

		if err := rt.debug.logPostNormTmp(rt, i, m.Scratch.Tmp); err != nil {
			return err
		}

		opOut := computeAttentionBranch(m, layer, m.Scratch.Tmp)
		if err := consumeFastPathError(ops); err != nil {
			return fmt.Errorf("attention fast path failed: %w", err)
		}
		if err := rt.debug.logAttentionOut(rt, i, opOut); err != nil {
			return err
		}
		if len(layer.PostAttnNorm) > 0 {
			if bf != nil {
				if err := bf.FlushBlockResult(); err != nil {
					return fmt.Errorf("post-attention flush failed: %w", err)
				}
			}
			ops.RMSNorm(m.Scratch.Tmp2, opOut, layer.PostAttnNorm, m.RMSEpsilon)
			opOut = m.Scratch.Tmp2
		}
		addResidual(ds, x, opOut)
		if usesGemma4BF16Rounding(layer) {
			if !deviceRoundBF16InPlace(ops, x) {
				syncDeviceSlice(ops, x)
				if err := consumeFastPathError(ops); err != nil {
					return fmt.Errorf("attention residual bf16 sync failed: %w", err)
				}
				roundBF16SliceInPlace(x)
				markHostStateDirty(ds, x)
			}
		}
		if err := consumeFastPathError(ops); err != nil {
			return fmt.Errorf("attention residual fast path failed: %w", err)
		}
		if err := rt.debug.logPostAttentionResidual(rt, i); err != nil {
			return err
		}

		ffnNormFast := ds != nil && ds.DeviceRMSNorm(m.Scratch.Tmp, x, layer.FfnNorm, m.RMSEpsilon)
		if err := consumeFastPathError(ops); err != nil {
			return fmt.Errorf("ffn pre-norm fast path failed: %w", err)
		}
		if !ffnNormFast {
			if ds != nil {
				ds.SyncHostState(x)
				if err := consumeFastPathError(ops); err != nil {
					return fmt.Errorf("ffn pre-norm sync failed: %w", err)
				}
			}
			ops.RMSNorm(m.Scratch.Tmp, x, layer.FfnNorm, m.RMSEpsilon)
		}
		if usesGemma4BF16Rounding(layer) {
			syncDeviceSlice(ops, m.Scratch.Tmp)
			if err := consumeFastPathError(ops); err != nil {
				return fmt.Errorf("ffn pre-norm sync failed: %w", err)
			}
			roundBF16SliceInPlace(m.Scratch.Tmp[:len(x)])
		}

		perLayerInput := rt.perLayerInput(m, i)
		if layer.Gemma4MoE != nil || layer.Gemma4PLE != nil || layer.LayerScale != 1 {
			if bf != nil {
				if err := bf.FlushBlockResult(); err != nil {
					return fmt.Errorf("gemma4 ffn flush failed: %w", err)
				}
			}
			if err := runGemma4FFNBlock(m, layer, x, m.Scratch.Tmp, perLayerInput, ds, nil); err != nil {
				return err
			}
			if err := consumeFastPathError(ops); err != nil {
				return fmt.Errorf("gemma4 ffn fast path failed: %w", err)
			}
			if err := rt.debug.logLayerHidden(rt, i); err != nil {
				return err
			}
			continue
		}

		ffnOut, err := computeStandardFFNOutput(m, ops, layer, m.Scratch.Tmp, bf, true)
		if err != nil {
			return err
		}
		addResidual(ds, x, ffnOut)
		if err := consumeFastPathError(ops); err != nil {
			return fmt.Errorf("ffn residual fast path failed: %w", err)
		}
		if err := rt.debug.logLayerHidden(rt, i); err != nil {
			return err
		}
	}

	return nil
}

func computeAttentionBranch(m *Instance, layer *Layer, normed []float32) []float32 {
	if layer.Mamba != nil {
		var attnOut []float32
		attnIn := normed
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
		mambaOut := Mamba(m, layer, normed)
		if attnOut == nil {
			return mambaOut
		}
		if mambaOut == nil {
			return attnOut
		}
		opOut := m.Scratch.Tmp2
		copy(opOut, attnOut)
		Add(opOut, mambaOut)
		return opOut
	}
	if layer.DeltaNet != nil {
		return DeltaNet(m, layer, normed)
	}
	if layer.IsRecurrent {
		return ShortConv(m, layer, normed)
	}
	return Attention(m, layer, normed, m.Pos)
}

func computeStandardFFNOutput(m *Instance, ops Ops, layer *Layer, normed []float32, bf blockFlusher, syncMoE bool) ([]float32, error) {
	var ffnOut []float32
	if layer.MoE != nil {
		if syncMoE {
			syncDeviceSlice(ops, normed)
			if err := consumeFastPathError(ops); err != nil {
				return nil, fmt.Errorf("moe sync fast path failed: %w", err)
			}
		}
		ffnOut = MoE(m, layer, normed)
	} else {
		ffnOut = FFN(m, layer, normed)
	}
	if err := consumeFastPathError(ops); err != nil {
		return nil, fmt.Errorf("ffn fast path failed: %w", err)
	}
	if len(layer.PostFfnNorm) > 0 {
		if bf != nil {
			if err := bf.FlushBlockResult(); err != nil {
				return nil, fmt.Errorf("post-ffn flush failed: %w", err)
			}
		}
		ops.RMSNorm(m.Scratch.Tmp, ffnOut, layer.PostFfnNorm, m.RMSEpsilon)
		if err := consumeFastPathError(ops); err != nil {
			return nil, fmt.Errorf("post-ffn norm failed: %w", err)
		}
		ffnOut = m.Scratch.Tmp
	}
	return ffnOut, nil
}

func (rt *tokenRuntimeState) perLayerInput(m *Instance, layerIdx int) []float32 {
	if rt.perLayerInputs == nil || m.Gemma4PerLayer == nil {
		return nil
	}
	start := layerIdx * m.Gemma4PerLayer.HiddenSize
	end := start + m.Gemma4PerLayer.HiddenSize
	return rt.perLayerInputs[start:end]
}

func (d layerStateDebugger) logPreNormX(rt *tokenRuntimeState, layerIdx int) error {
	if !d.enabled || layerIdx != 0 {
		return nil
	}
	if rt.ds != nil {
		rt.ds.SyncHostState(rt.x)
		if err := consumeFastPathError(rt.ops); err != nil {
			return fmt.Errorf("debug pre-norm sync failed: %w", err)
		}
	}
	n := min(5, len(rt.x))
	fmt.Printf("  L0 pre-norm x[0:%d] = %v\n", n, rt.x[:n])
	return nil
}

func (d layerStateDebugger) logPostNormTmp(rt *tokenRuntimeState, layerIdx int, tmp []float32) error {
	if !d.enabled || layerIdx != 0 {
		return nil
	}
	syncDeviceSlice(rt.ops, tmp)
	if err := consumeFastPathError(rt.ops); err != nil {
		return fmt.Errorf("debug post-norm sync failed: %w", err)
	}
	n := min(5, len(tmp))
	fmt.Printf("  L0 post-norm tmp[0:%d] = %v\n", n, tmp[:n])
	return nil
}

func (d layerStateDebugger) logAttentionOut(rt *tokenRuntimeState, layerIdx int, opOut []float32) error {
	if !d.enabled || layerIdx != 0 {
		return nil
	}
	if rt.bf != nil {
		if err := rt.bf.FlushBlockResult(); err != nil {
			return fmt.Errorf("debug flush failed: %w", err)
		}
	}
	syncDeviceSlice(rt.ops, opOut)
	if err := consumeFastPathError(rt.ops); err != nil {
		return fmt.Errorf("debug attention output sync failed: %w", err)
	}
	n := min(5, len(opOut))
	fmt.Printf("  L0 attn_out[0:%d] = %v\n", n, opOut[:n])
	return nil
}

func (d layerStateDebugger) logPostAttentionResidual(rt *tokenRuntimeState, layerIdx int) error {
	if !d.enabled || layerIdx != 0 {
		return nil
	}
	if rt.ds != nil {
		rt.ds.SyncHostState(rt.x)
		if err := consumeFastPathError(rt.ops); err != nil {
			return fmt.Errorf("debug post-attention residual sync failed: %w", err)
		}
	}
	n := min(5, len(rt.x))
	fmt.Printf("  L0 post-attn-residual x[0:%d] = %v\n", n, rt.x[:n])
	return nil
}

func (d layerStateDebugger) logLayerHidden(rt *tokenRuntimeState, layerIdx int) error {
	if !d.enabled {
		return nil
	}
	if rt.ds != nil {
		rt.ds.SyncHostState(rt.x)
		if err := consumeFastPathError(rt.ops); err != nil {
			return fmt.Errorf("debug layer-state sync failed: %w", err)
		}
	}
	n := min(5, len(rt.x))
	fmt.Printf("  LAYER %d x[0:%d] = %v\n", layerIdx, n, rt.x[:n])
	return nil
}
