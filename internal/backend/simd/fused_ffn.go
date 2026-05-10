//go:build goexperiment.simd

package simd

import (
	"simd/archsimd"
)

// fusedFFN performs a fused gate+up+activation+down FFN in a single function.
//
// The exact sequence:
//  1. gateOut = gate * input  (stored in m.Scratch.FfnGate)
//  2. upOut = up * input      (stored in m.Scratch.FfnUp)
//  3. actOut[i] = activation(gateOut[i]) * upOut[i]  (stored in m.Scratch.FfnAct)
//  4. downOut = down * actOut (stored in m.Scratch.Tmp2; returned)
//
// Uses existing scratch buffers (FfnGate, FfnUp, FfnAct, Tmp2) for intermediates.
// Supports "silu", "gelu", and "gelu_erf" activations.
func fusedFFN(m *Instance, layer *Layer, input []float32, activation string) []float32 {
	m.Ops().MatVec(m.Scratch.FfnGate, layer.FfnGate, input)
	m.Ops().MatVec(m.Scratch.FfnUp, layer.FfnUp, input)

	n := len(m.Scratch.FfnAct)
	useGelu := activation == "gelu" || activation == "gelu_erf"
	i := 0

	if useGelu {
		if cpu.HasAVX2 {
			for ; i+8 <= n; i += 8 {
				vgate := archsimd.LoadFloat32x8Slice(m.Scratch.FfnGate[i:])
				vup := archsimd.LoadFloat32x8Slice(m.Scratch.FfnUp[i:])
				vact := fastGeluVec(vgate).Mul(vup)
				vact.StoreSlice(m.Scratch.FfnAct[i:])
			}
		}
		for ; i < n; i++ {
			m.Scratch.FfnAct[i] = Gelu(m.Scratch.FfnGate[i]) * m.Scratch.FfnUp[i]
		}
	} else {
		if cpu.HasAVX2 {
			for ; i+8 <= n; i += 8 {
				vgate := archsimd.LoadFloat32x8Slice(m.Scratch.FfnGate[i:])
				vup := archsimd.LoadFloat32x8Slice(m.Scratch.FfnUp[i:])
				vact := fastSiluVec(vgate).Mul(vup)
				vact.StoreSlice(m.Scratch.FfnAct[i:])
			}
		}
		for ; i < n; i++ {
			m.Scratch.FfnAct[i] = Silu(m.Scratch.FfnGate[i]) * m.Scratch.FfnUp[i]
		}
	}

	m.Ops().MatVec(m.Scratch.Tmp2, layer.FfnDown, m.Scratch.FfnAct)
	return m.Scratch.Tmp2
}
