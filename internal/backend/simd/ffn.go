package simd

// FFN performs feed-forward network computation with SiLU activation.
// Computes: SiLU(Gate(x)) * Up(x) -> Down
func FFN(m *Instance, layer *Layer, x []float32) []float32 {
	var qx *QuantVec
	if CanUseQuantVec(layer.FfnUp) || CanUseQuantVec(layer.FfnGate) {
		qx = PrepareQuantVec(x)
		defer ReleaseQuantVec(qx)
	}
	m.Ops().MatVecWithQuant(m.Scratch.FfnUp, layer.FfnUp, x, qx)
	m.Ops().MatVecWithQuant(m.Scratch.FfnGate, layer.FfnGate, x, qx)
	for i := range m.Scratch.FfnAct {
		m.Scratch.FfnAct[i] = Silu(m.Scratch.FfnGate[i]) * m.Scratch.FfnUp[i]
	}
	m.Ops().MatVec(m.Scratch.Tmp2, layer.FfnDown, m.Scratch.FfnAct)
	return m.Scratch.Tmp2
}
