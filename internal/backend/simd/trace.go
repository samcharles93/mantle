package simd

import "fmt"

// PromptTrace captures intermediate activations for a prompt prefill run.
// Shapes are chosen to line up with the HF comparison harness:
// - InputsEmbeds: [seq][hidden]
// - HiddenStates: [states][seq][hidden]
// - PerLayerInputs*: [seq][layers][hidden_per_layer]
// - AttentionOutputs: [layers][seq][hidden]
type PromptTrace struct {
	Prompt                  string        `json:"prompt,omitempty"`
	RenderedPrompt          string        `json:"rendered_prompt,omitempty"`
	TokenIDs                []int         `json:"token_ids"`
	Tokens                  []string      `json:"tokens,omitempty"`
	InputsEmbeds            [][]float32   `json:"inputs_embeds"`
	HiddenStates            [][][]float32 `json:"hidden_states"`
	AttentionOutputs        [][][]float32 `json:"attention_outputs,omitempty"`
	FfnOutputs              [][][]float32 `json:"ffn_outputs,omitempty"`
	PostFfnHiddenStates     [][][]float32 `json:"post_ffn_hidden_states,omitempty"`
	PerLayerResidualOutputs [][][]float32 `json:"per_layer_residual_outputs,omitempty"`
	PerLayerInputsRaw       [][][]float32 `json:"per_layer_inputs_raw,omitempty"`
	PerLayerInputsProjected [][][]float32 `json:"per_layer_inputs_projected,omitempty"`
	LastTokenLogits         []float32     `json:"last_token_logits"`
	NextTokenID             int           `json:"next_token_id"`
	NextToken               string        `json:"next_token,omitempty"`
}

type tokenTrace struct {
	InputEmbed         []float32
	HiddenStates       [][]float32
	AttentionOutputs   [][]float32
	FfnOutputs         [][]float32
	PostFfnHidden      [][]float32
	PerLayerResidual   [][]float32
	PerLayerInputsRaw  [][]float32
	PerLayerInputsProj [][]float32
}

type gemma4FFNTrace struct {
	FfnOut           []float32
	PostFfnHidden    []float32
	PerLayerResidual []float32
}

// TraceTokens runs a CPU-side traced prefill and records the prompt activations.
// The instance state is reset before tracing so the output is deterministic.
func (m *Instance) TraceTokens(tokens []int) (*PromptTrace, error) {
	if m == nil {
		return nil, fmt.Errorf("nil instance")
	}
	if len(tokens) == 0 {
		return nil, fmt.Errorf("no tokens to trace")
	}
	m.Reset()

	seqLen := len(tokens)
	stateCount := len(m.Layers) + 1
	trace := &PromptTrace{
		TokenIDs:            append([]int(nil), tokens...),
		InputsEmbeds:        make([][]float32, seqLen),
		HiddenStates:        make([][][]float32, stateCount),
		AttentionOutputs:    make([][][]float32, len(m.Layers)),
		FfnOutputs:          make([][][]float32, len(m.Layers)),
		PostFfnHiddenStates: make([][][]float32, len(m.Layers)),
	}
	for i := range trace.HiddenStates {
		trace.HiddenStates[i] = make([][]float32, seqLen)
	}
	for i := range trace.AttentionOutputs {
		trace.AttentionOutputs[i] = make([][]float32, seqLen)
	}
	for i := range trace.FfnOutputs {
		trace.FfnOutputs[i] = make([][]float32, seqLen)
	}
	for i := range trace.PostFfnHiddenStates {
		trace.PostFfnHiddenStates[i] = make([][]float32, seqLen)
	}
	if m.Gemma4PerLayer != nil {
		trace.PerLayerResidualOutputs = make([][][]float32, len(m.Layers))
		for i := range trace.PerLayerResidualOutputs {
			trace.PerLayerResidualOutputs[i] = make([][]float32, seqLen)
		}
		trace.PerLayerInputsRaw = make([][][]float32, seqLen)
		trace.PerLayerInputsProjected = make([][][]float32, seqLen)
	}

	for tokIdx, tok := range tokens {
		logits, tt, err := m.traceForwardToken(tok)
		if err != nil {
			return nil, fmt.Errorf("trace token %d: %w", tokIdx, err)
		}
		trace.InputsEmbeds[tokIdx] = tt.InputEmbed
		for stateIdx := range tt.HiddenStates {
			trace.HiddenStates[stateIdx][tokIdx] = tt.HiddenStates[stateIdx]
		}
		for layerIdx := range tt.AttentionOutputs {
			trace.AttentionOutputs[layerIdx][tokIdx] = tt.AttentionOutputs[layerIdx]
		}
		for layerIdx := range tt.FfnOutputs {
			trace.FfnOutputs[layerIdx][tokIdx] = tt.FfnOutputs[layerIdx]
		}
		for layerIdx := range tt.PostFfnHidden {
			trace.PostFfnHiddenStates[layerIdx][tokIdx] = tt.PostFfnHidden[layerIdx]
		}
		for layerIdx := range tt.PerLayerResidual {
			if len(trace.PerLayerResidualOutputs) > 0 {
				trace.PerLayerResidualOutputs[layerIdx][tokIdx] = tt.PerLayerResidual[layerIdx]
			}
		}
		if tt.PerLayerInputsRaw != nil {
			trace.PerLayerInputsRaw[tokIdx] = tt.PerLayerInputsRaw
		}
		if tt.PerLayerInputsProj != nil {
			trace.PerLayerInputsProjected[tokIdx] = tt.PerLayerInputsProj
		}
		if tokIdx == seqLen-1 {
			trace.LastTokenLogits = cloneVec(logits)
			trace.NextTokenID = argmaxHost(logits)
		}
	}

	return trace, nil
}

func (m *Instance) traceForwardToken(tok int) ([]float32, *tokenTrace, error) {
	x, err := initializeTokenInput(m, tok)
	if err != nil {
		return nil, nil, err
	}

	tt := &tokenTrace{
		InputEmbed:       cloneVec(x),
		HiddenStates:     make([][]float32, len(m.Layers)+1),
		AttentionOutputs: make([][]float32, len(m.Layers)),
		FfnOutputs:       make([][]float32, len(m.Layers)),
		PostFfnHidden:    make([][]float32, len(m.Layers)),
		PerLayerResidual: make([][]float32, len(m.Layers)),
	}
	tt.HiddenStates[0] = cloneVec(x)

	rawPerLayer, projectedPerLayer := computeGemma4PerLayerInputs(m, tok, x, m.Ops(), nil)
	if rawPerLayer != nil && projectedPerLayer != nil && m.Gemma4PerLayer != nil {
		tt.PerLayerInputsRaw = splitRows(rawPerLayer, m.Gemma4PerLayer.LayerCount, m.Gemma4PerLayer.HiddenSize)
		tt.PerLayerInputsProj = splitRows(projectedPerLayer, m.Gemma4PerLayer.LayerCount, m.Gemma4PerLayer.HiddenSize)
	}

	ops := m.Ops()

	for i := range m.Layers {
		layer := &m.Layers[i]

		ops.RMSNorm(m.Scratch.Tmp, x, layer.AttnNorm, m.RMSEpsilon)
		if err := consumeFastPathError(ops); err != nil {
			return nil, nil, fmt.Errorf("attention pre-norm failed: %w", err)
		}

		opOut := computeAttentionBranch(m, layer, m.Scratch.Tmp)
		if err := consumeFastPathError(ops); err != nil {
			return nil, nil, fmt.Errorf("attention failed: %w", err)
		}
		if len(layer.PostAttnNorm) > 0 {
			ops.RMSNorm(m.Scratch.Tmp2, opOut, layer.PostAttnNorm, m.RMSEpsilon)
			if err := consumeFastPathError(ops); err != nil {
				return nil, nil, fmt.Errorf("post-attention norm failed: %w", err)
			}
			opOut = m.Scratch.Tmp2
		}
		tt.AttentionOutputs[i] = cloneVec(opOut[:len(x)])
		Add(x, opOut)
		if usesGemma4BF16Rounding(layer) {
			roundBF16SliceInPlace(x)
		}

		ops.RMSNorm(m.Scratch.Tmp, x, layer.FfnNorm, m.RMSEpsilon)
		if err := consumeFastPathError(ops); err != nil {
			return nil, nil, fmt.Errorf("ffn pre-norm failed: %w", err)
		}
		if usesGemma4BF16Rounding(layer) {
			roundBF16SliceInPlace(m.Scratch.Tmp[:len(x)])
		}

		var perLayerInput []float32
		if projectedPerLayer != nil && m.Gemma4PerLayer != nil {
			start := i * m.Gemma4PerLayer.HiddenSize
			end := start + m.Gemma4PerLayer.HiddenSize
			perLayerInput = projectedPerLayer[start:end]
		}
		if layer.Gemma4MoE != nil || layer.Gemma4PLE != nil || layer.LayerScale != 1 {
			ffnTrace := &gemma4FFNTrace{}
			if err := runGemma4FFNBlock(m, layer, x, m.Scratch.Tmp, perLayerInput, nil, ffnTrace); err != nil {
				return nil, nil, err
			}
			if err := consumeFastPathError(ops); err != nil {
				return nil, nil, fmt.Errorf("gemma4 ffn failed: %w", err)
			}
			if ffnTrace.FfnOut != nil {
				tt.FfnOutputs[i] = ffnTrace.FfnOut
			}
			if ffnTrace.PostFfnHidden != nil {
				tt.PostFfnHidden[i] = ffnTrace.PostFfnHidden
			}
			if ffnTrace.PerLayerResidual != nil {
				tt.PerLayerResidual[i] = ffnTrace.PerLayerResidual
			}
			tt.HiddenStates[i+1] = cloneVec(x)
			continue
		}

		ffnOut, err := computeStandardFFNOutput(m, ops, layer, m.Scratch.Tmp, nil, false)
		if err != nil {
			return nil, nil, err
		}
		tt.FfnOutputs[i] = cloneVec(ffnOut)
		Add(x, ffnOut)
		tt.PostFfnHidden[i] = cloneVec(x)
		tt.HiddenStates[i+1] = cloneVec(x)
	}

	FusedRMSNormMatVec(ops, m.Scratch.Logits, m.Output, x, m.OutputNorm, m.RMSEpsilon, m.Scratch.Tmp)
	if scale := m.Config.Config.LMHeadMultiplier; scale != 0 && scale != 1 {
		s := float32(scale)
		for i := range m.Scratch.Logits {
			m.Scratch.Logits[i] *= s
		}
	}
	if softcap := m.Config.Config.FinalLogitSoftcap; softcap > 0 {
		for i := range m.Scratch.Logits {
			m.Scratch.Logits[i] = fastTanh(m.Scratch.Logits[i]/softcap) * softcap
		}
	}

	logits := cloneVec(m.Scratch.Logits)
	m.Pos++
	return logits, tt, nil
}

func cloneVec(src []float32) []float32 {
	dst := make([]float32, len(src))
	copy(dst, src)
	return dst
}

func splitRows(src []float32, rows, cols int) [][]float32 {
	if rows <= 0 || cols <= 0 {
		return nil
	}
	out := make([][]float32, rows)
	for row := range rows {
		start := row * cols
		end := start + cols
		out[row] = cloneVec(src[start:end])
	}
	return out
}
