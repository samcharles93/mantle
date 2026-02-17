package simd

import (
	"fmt"
	"math"
)

// ForwardTokens processes multiple tokens in batch using GEMM for the projections.
// This is more efficient than token-by-token for prompt prefill.
// Returns the output logits for each token position.
func (m *Instance) ForwardTokens(tokens []int) ([][]float32, error) {
	if len(tokens) == 0 {
		return nil, fmt.Errorf("no tokens to process")
	}
	if m.Pos+len(tokens) > m.MaxContext {
		return nil, fmt.Errorf("context length exceeded: %d + %d > %d", m.Pos, len(tokens), m.MaxContext)
	}

	// GEMM path requires f32 weight data; bail for BF16/quantized models
	// where Mat.Data is nil (data lives in Mat.Raw or Mat.Quant).
	if len(m.Layers) > 0 {
		l := &m.Layers[0]
		if l.Wq != nil && l.Wq.Data == nil {
			return nil, fmt.Errorf("ForwardTokens requires f32 weights, got %v", l.Wq.DType)
		}
	}
	// Bail if model has Mamba/MoE/recurrent layers (unsupported in batch path)
	for i := range m.Layers {
		if m.Layers[i].Mamba != nil || m.Layers[i].MoE != nil || m.Layers[i].IsRecurrent {
			return nil, fmt.Errorf("ForwardTokens does not support Mamba/MoE/recurrent layers")
		}
	}

	// Get tiling configuration
	tiling := m.TilingConfig
	if tiling.TileM == 0 {
		tiling = DefaultTilingConfig()
	}

	seqLen := len(tokens)
	embd := m.Config.Config.EmbeddingLength
	headDim := m.HeadDim
	nHead := m.HeadCount
	kvHeads := m.Layers[0].HeadKV

	// Build input matrix X: [seqLen x embd]
	X := NewMat(seqLen, embd)
	for i, tok := range tokens {
		if tok < 0 || tok >= m.Config.Config.VocabSize {
			return nil, fmt.Errorf("token id out of range: %d", tok)
		}
		m.Embeddings.RowTo(X.Data[i*X.Stride:], tok)
	}
	if scale := m.Config.Config.EmbeddingMultiplier; scale != 0 && scale != 1 {
		s := float32(scale)
		for i := range X.Data {
			X.Data[i] *= s
		}
	}
	if m.Config.Config.MuPEnabled && m.MuPScale != 1 {
		for i := range X.Data {
			X.Data[i] *= m.MuPScale
		}
	}

	// Temporary matrices for GEMM
	Q := NewMat(seqLen, nHead*headDim)
	K := NewMat(seqLen, kvHeads*headDim)
	V := NewMat(seqLen, kvHeads*headDim)
	attnOut := NewMat(seqLen, embd)
	ffnUp := NewMat(seqLen, m.Config.Config.FFNLength)
	ffnGate := NewMat(seqLen, m.Config.Config.FFNLength)
	ffnAct := NewMat(seqLen, m.Config.Config.FFNLength)

	// Process through layers
	for layerIdx := range m.Layers {
		layer := &m.Layers[layerIdx]

		// RMSNorm each row
		for i := range seqLen {
			rowOff := i * X.Stride
			RMSNorm(m.Scratch.Tmp, X.Data[rowOff:rowOff+embd], layer.AttnNorm, m.RMSEpsilon)
			copy(X.Data[rowOff:rowOff+embd], m.Scratch.Tmp)
		}

		// QKV projections using GEMM: X @ W^T
		cfg := SelectGemmConfigWithTiling(seqLen, embd, nHead*headDim, tiling)
		GemmPar(cfg, &Q, &X, layer.Wq, 1.0, 0.0, 0)
		GemmPar(cfg, &K, &X, layer.Wk, 1.0, 0.0, 0)
		GemmPar(cfg, &V, &X, layer.Wv, 1.0, 0.0, 0)

		// Attention for each position
		for i := range seqLen {
			pos := m.Pos + i
			qRow := Q.Data[i*Q.Stride : i*Q.Stride+nHead*headDim]
			kRow := K.Data[i*K.Stride : i*K.Stride+kvHeads*headDim]
			vRow := V.Data[i*V.Stride : i*V.Stride+kvHeads*headDim]
			outRow := attnOut.Data[i*attnOut.Stride : i*attnOut.Stride+embd]

			// Apply RoPE and attention
			copy(m.Scratch.Q, qRow)
			copy(m.Scratch.K, kRow)
			copy(m.Scratch.V, vRow)

			attnResult := Attention(m, layer, m.Scratch.Q[:embd], pos)
			copy(outRow, attnResult)
		}

		// Residual: X = X + attnOut
		for i := range seqLen {
			xRow := X.Data[i*X.Stride : i*X.Stride+embd]
			aRow := attnOut.Data[i*attnOut.Stride : i*attnOut.Stride+embd]
			Add(xRow, aRow)
		}

		// FFN block
		// RMSNorm each row
		for i := range seqLen {
			rowOff := i * X.Stride
			RMSNorm(m.Scratch.Tmp, X.Data[rowOff:rowOff+embd], layer.FfnNorm, m.RMSEpsilon)
			copy(X.Data[rowOff:rowOff+embd], m.Scratch.Tmp)
		}

		// FFN projections using GEMM
		cfg = SelectGemmConfigWithTiling(seqLen, embd, m.Config.Config.FFNLength, tiling)
		GemmPar(cfg, &ffnUp, &X, layer.FfnUp, 1.0, 0.0, 0)
		GemmPar(cfg, &ffnGate, &X, layer.FfnGate, 1.0, 0.0, 0)

		// SiLU activation: ffnAct = silu(ffnGate) * ffnUp
		for i := range seqLen {
			upRow := ffnUp.Data[i*ffnUp.Stride : i*ffnUp.Stride+m.Config.Config.FFNLength]
			gateRow := ffnGate.Data[i*ffnGate.Stride : i*ffnGate.Stride+m.Config.Config.FFNLength]
			actRow := ffnAct.Data[i*ffnAct.Stride : i*ffnAct.Stride+m.Config.Config.FFNLength]
			FusedSiluAct(actRow, gateRow, upRow)
		}

		// Down projection: ffnOut = ffnAct @ FfnDown
		ffnOut := NewMat(seqLen, embd)
		cfg = SelectGemmConfigWithTiling(seqLen, m.Config.Config.FFNLength, embd, tiling)
		GemmPar(cfg, &ffnOut, &ffnAct, layer.FfnDown, 1.0, 0.0, 0)

		// Residual: X = X + ffnOut
		for i := range seqLen {
			xRow := X.Data[i*X.Stride : i*X.Stride+embd]
			fRow := ffnOut.Data[i*ffnOut.Stride : i*ffnOut.Stride+embd]
			Add(xRow, fRow)
		}
	}

	// Output norm and projection to logits
	logits := NewMat(seqLen, m.Config.Config.VocabSize)
	for i := range seqLen {
		rowOff := i * X.Stride
		RMSNorm(m.Scratch.Tmp, X.Data[rowOff:rowOff+embd], m.OutputNorm, m.RMSEpsilon)
	}

	// Output projection using GEMM
	outCfg := SelectGemmConfigWithTiling(seqLen, embd, m.Config.Config.VocabSize, tiling)
	GemmPar(outCfg, &logits, &X, m.Output, 1.0, 0.0, 0)

	// Collect logits
	outputs := make([][]float32, seqLen)
	for i := range seqLen {
		outputs[i] = make([]float32, m.Config.Config.VocabSize)
		copy(outputs[i], logits.Data[i*logits.Stride:i*logits.Stride+m.Config.Config.VocabSize])
	}

	m.Pos += seqLen
	return outputs, nil
}

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

		// Attention block: pre-norm, attention, optional post-norm, residual
		attnNormFast := ds != nil && ds.DeviceRMSNorm(m.Scratch.Tmp, x, layer.AttnNorm, m.RMSEpsilon)
		if err := consumeFastPathError(ops); err != nil {
			return nil, fmt.Errorf("attention pre-norm fast path failed: %w", err)
		}
		if attnNormFast {
			// device-side pre-norm succeeded
		} else {
			if ds != nil {
				ds.SyncHostState(x)
				if err := consumeFastPathError(ops); err != nil {
					return nil, fmt.Errorf("attention pre-norm sync failed: %w", err)
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
		if err := consumeFastPathError(ops); err != nil {
			return nil, fmt.Errorf("attention fast path failed: %w", err)
		}
		if len(layer.PostAttnNorm) > 0 {
			if bf != nil {
				if err := bf.FlushBlockResult(); err != nil {
					return nil, fmt.Errorf("post-attention flush failed: %w", err)
				}
			}
			ops.RMSNorm(m.Scratch.Tmp2, opOut, layer.PostAttnNorm, m.RMSEpsilon)
			opOut = m.Scratch.Tmp2
		}
		addResidual(ds, x, opOut)
		if err := consumeFastPathError(ops); err != nil {
			return nil, fmt.Errorf("attention residual fast path failed: %w", err)
		}

		// FFN block: pre-norm, dense/MoE, optional post-norm, residual
		ffnNormFast := ds != nil && ds.DeviceRMSNorm(m.Scratch.Tmp, x, layer.FfnNorm, m.RMSEpsilon)
		if err := consumeFastPathError(ops); err != nil {
			return nil, fmt.Errorf("ffn pre-norm fast path failed: %w", err)
		}
		if ffnNormFast {
			// device-side pre-norm succeeded
		} else {
			if ds != nil {
				ds.SyncHostState(x)
				if err := consumeFastPathError(ops); err != nil {
					return nil, fmt.Errorf("ffn pre-norm sync failed: %w", err)
				}
			}
			ops.RMSNorm(m.Scratch.Tmp, x, layer.FfnNorm, m.RMSEpsilon)
		}
		var ffnOut []float32
		if layer.MoE != nil {
			syncDeviceSlice(ops, m.Scratch.Tmp)
			if err := consumeFastPathError(ops); err != nil {
				return nil, fmt.Errorf("moe sync fast path failed: %w", err)
			}
			ffnOut = MoE(m, layer, m.Scratch.Tmp)
		} else {
			ffnOut = FFN(m, layer, m.Scratch.Tmp)
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
			ffnOut = m.Scratch.Tmp
		}
		addResidual(ds, x, ffnOut)
		if err := consumeFastPathError(ops); err != nil {
			return nil, fmt.Errorf("ffn residual fast path failed: %w", err)
		}
	}

	// Output norm + projection:
	// 1) prefer device-only RMSNorm + MatVec when available
	// 2) otherwise fallback to fused/host path
	usedDeviceHead := false
	if ds != nil && ds.DeviceRMSNorm(m.Scratch.Tmp, x, m.OutputNorm, m.RMSEpsilon) {
		usedDeviceHead = ds.DeviceMatVec(m.Scratch.Logits, m.Output, m.Scratch.Tmp)
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

// PrecomputeRoPETables precomputes RoPE tables for the maximum context length.
func (m *Instance) PrecomputeRoPETables() {
	if len(m.RopeInvFreq) == 0 {
		return // Nothing to precompute
	}

	half := m.HeadDim / 2
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
