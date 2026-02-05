package simd

import "math"

func FFNProject(m *Instance, up, gate, down *Mat, x []float32) []float32 {
	intermediate := up.R
	if intermediate <= 0 || intermediate > len(m.Scratch.FfnUp) {
		panic("ffn intermediate size exceeds scratch buffers")
	}
	upBuf := m.Scratch.FfnUp[:intermediate]
	gateBuf := m.Scratch.FfnGate[:intermediate]
	actBuf := m.Scratch.FfnAct[:intermediate]

	m.Ops().MatVec(upBuf, up, x)
	m.Ops().MatVec(gateBuf, gate, x)
	for i := range actBuf {
		actBuf[i] = Silu(gateBuf[i]) * upBuf[i]
	}
	m.Ops().MatVec(m.Scratch.Tmp2, down, actBuf)
	return m.Scratch.Tmp2
}

func MoE(m *Instance, layer *Layer, x []float32) []float32 {
	moe := layer.MoE
	if moe == nil {
		panic("moe layer missing moe config")
	}
	accum := m.Scratch.MoeAccum
	for i := range accum {
		accum[i] = 0
	}

	// Shared experts are always active.
	sharedOut := FFNProject(m, moe.Shared.Up, moe.Shared.Gate, moe.Shared.Down, x)
	for i := range accum {
		accum[i] += sharedOut[i]
	}

	raw := m.Scratch.RouterRaw
	sel := m.Scratch.RouterSel
	if len(raw) != len(sel) || len(raw) != len(moe.Experts) {
		panic("router scratch buffers do not match expert count")
	}
	m.Ops().MatVec(raw, moe.Router, x)
	for i := range raw {
		raw[i] = Sigmoid(raw[i])
		bias := float32(0)
		if i < len(moe.ExpertBias) {
			bias = moe.ExpertBias[i]
		}
		sel[i] = raw[i] + bias
	}

	idx := m.Scratch.RouterIdx
	weights := m.Scratch.RouterW
	SelectTopK(sel, raw, moe.TopK, moe.RouteScale, idx, weights)

	for j := 0; j < moe.TopK; j++ {
		expertID := idx[j]
		if expertID < 0 || expertID >= len(moe.Experts) {
			continue
		}
		w := weights[j]
		if w == 0 {
			continue
		}
		expert := moe.Experts[expertID]
		out := FFNProject(m, expert.Up, expert.Gate, expert.Down, x)
		for i := range accum {
			accum[i] += w * out[i]
		}
	}

	return accum
}

func SelectTopK(selScores, rawScores []float32, k int, routeScale float32, idxOut []int, wOut []float32) {
	if k > len(selScores) {
		k = len(selScores)
	}

	if k <= 0 {
		return
	}
	if k > len(idxOut) || k > len(wOut) {
		panic("topk scratch buffers too small")
	}

	if k <= 8 {
		selectTopKSmall(selScores, rawScores, k, routeScale, idxOut, wOut)
		return
	}
}

// selectTopKSmall uses partial selection for small k values (k <= 8).
func selectTopKSmall(selScores, rawScores []float32, k int, routeScale float32, idxOut []int, wOut []float32) {
	var bestIdx [8]int
	var bestScore [8]float32

	for i := range k {
		bestIdx[i] = -1
		bestScore[i] = float32(-math.MaxFloat32)
	}

	// Find top k using partial selection
	for i, score := range selScores {
		pos := k
		for j := range k {
			if score > bestScore[j] {
				pos = j
				break
			}
		}
		if pos == k {
			continue
		}
		for j := k - 1; j > pos; j-- {
			bestIdx[j] = bestIdx[j-1]
			bestScore[j] = bestScore[j-1]
		}
		bestIdx[pos] = i
		bestScore[pos] = score
	}

	for i := range k {
		idxOut[i] = bestIdx[i]
	}

	computeTopKWeights(rawScores, k, routeScale, idxOut, wOut)
}

// computeTopKWeights computes normalized weights for selected experts.
func computeTopKWeights(rawScores []float32, k int, routeScale float32, idxOut []int, wOut []float32) {
	var denom float32
	for j := range k {
		id := idxOut[j]
		if id < 0 || id >= len(rawScores) {
			continue
		}
		denom += rawScores[id]
	}
	if denom == 0 {
		denom = 1
	}
	for j := range k {
		id := idxOut[j]
		if id < 0 || id >= len(rawScores) {
			wOut[j] = 0
			continue
		}
		wOut[j] = (rawScores[id] / denom) * routeScale
	}
}
