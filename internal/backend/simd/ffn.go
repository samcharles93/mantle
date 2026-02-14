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

	var dm deviceMatVecFastPath
	if d, ok := m.Ops().(deviceMatVecFastPath); ok {
		dm = d
	}

	var qx *QuantVec
	defer func() {
		if qx != nil {
			ReleaseQuantVec(qx)
		}
	}()
	needSync := true
	syncOnce := func() {
		if needSync {
			syncDeviceSlice(m.Ops(), x)
			needSync = false
		}
	}
	ensureQX := func(w *Mat) {
		syncOnce()
		if qx == nil && CanUseQuantVec(w) {
			qx = PrepareQuantVec(x)
		}
	}
	runInputMatVec := func(dst []float32, w *Mat) {
		if dm != nil && dm.DeviceMatVec(dst, w, x) {
			return
		}
		syncOnce()
		ensureQX(w)
		m.Ops().MatVecWithQuant(dst, w, x, qx)
	}

	runInputMatVec(m.Scratch.FfnUp, layer.FfnUp)
	runInputMatVec(m.Scratch.FfnGate, layer.FfnGate)

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
