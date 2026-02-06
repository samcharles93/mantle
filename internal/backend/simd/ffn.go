package simd

import "simd/archsimd"

type ffnFastPath interface {
	FFNBlock(layer *Layer, x []float32, out []float32) bool
}

// FFN performs feed-forward network computation with SiLU activation.
// Computes: SiLU(Gate(x)) * Up(x) -> Down
func FFN(m *Instance, layer *Layer, x []float32) []float32 {
	if fp, ok := m.Ops().(ffnFastPath); ok && fp.FFNBlock(layer, x, m.Scratch.Tmp2) {
		return m.Scratch.Tmp2
	}

	var qx *QuantVec
	if CanUseQuantVec(layer.FfnUp) || CanUseQuantVec(layer.FfnGate) {
		qx = PrepareQuantVec(x)
		defer ReleaseQuantVec(qx)
	}
	m.Ops().MatVecWithQuant(m.Scratch.FfnUp, layer.FfnUp, x, qx)
	m.Ops().MatVecWithQuant(m.Scratch.FfnGate, layer.FfnGate, x, qx)

	// Vectorized SiLU activation
	n := len(m.Scratch.FfnAct)
	i := 0
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

	m.Ops().MatVec(m.Scratch.Tmp2, layer.FfnDown, m.Scratch.FfnAct)
	return m.Scratch.Tmp2
}
