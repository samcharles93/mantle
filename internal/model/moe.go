package model

import (
	"math"

	"github.com/samcharles93/mantle/internal/tensor"
)

func (m *Instance) ffnProject(up, gate, down *tensor.Mat, x []float32) []float32 {
	intermediate := up.R
	if intermediate <= 0 || intermediate > len(m.scratch.ffnUp) {
		panic("ffn intermediate size exceeds scratch buffers")
	}
	upBuf := m.scratch.ffnUp[:intermediate]
	gateBuf := m.scratch.ffnGate[:intermediate]
	actBuf := m.scratch.ffnAct[:intermediate]

	tensor.MatVec(upBuf, up, x)
	tensor.MatVec(gateBuf, gate, x)
	for i := range actBuf {
		actBuf[i] = tensor.Silu(gateBuf[i]) * upBuf[i]
	}
	tensor.MatVec(m.scratch.tmp2, down, actBuf)
	return m.scratch.tmp2
}

func (m *Instance) moe(layer *Layer, x []float32) []float32 {
	moe := layer.MoE
	if moe == nil {
		panic("moe layer missing moe config")
	}
	accum := m.scratch.moeAccum
	for i := range accum {
		accum[i] = 0
	}

	// Shared experts are always active.
	sharedOut := m.ffnProject(moe.Shared.Up, moe.Shared.Gate, moe.Shared.Down, x)
	for i := range accum {
		accum[i] += sharedOut[i]
	}

	raw := m.scratch.routerRaw
	sel := m.scratch.routerSel
	if len(raw) != len(sel) || len(raw) != len(moe.Experts) {
		panic("router scratch buffers do not match expert count")
	}
	tensor.MatVec(raw, moe.Router, x)
	for i := range raw {
		raw[i] = tensor.Sigmoid(raw[i])
		bias := float32(0)
		if i < len(moe.ExpertBias) {
			bias = moe.ExpertBias[i]
		}
		sel[i] = raw[i] + bias
	}

	idx := m.scratch.routerIdx
	weights := m.scratch.routerW
	selectTopK(sel, raw, moe.TopK, moe.RouteScale, idx, weights)

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
		out := m.ffnProject(expert.Up, expert.Gate, expert.Down, x)
		for i := range accum {
			accum[i] += w * out[i]
		}
	}

	return accum
}

func selectTopK(selScores, rawScores []float32, k int, routeScale float32, idxOut []int, wOut []float32) {
	if k <= 0 {
		return
	}
	if k > len(idxOut) || k > len(wOut) {
		panic("topk scratch buffers too small")
	}
	bestScores := make([]float32, k)
	for i := 0; i < k; i++ {
		idxOut[i] = -1
		bestScores[i] = float32(math.Inf(-1))
	}

	for i, score := range selScores {
		insert := -1
		for j := 0; j < k; j++ {
			if score > bestScores[j] || (score == bestScores[j] && (idxOut[j] == -1 || i < idxOut[j])) {
				insert = j
				break
			}
		}
		if insert == -1 {
			continue
		}
		for j := k - 1; j > insert; j-- {
			bestScores[j] = bestScores[j-1]
			idxOut[j] = idxOut[j-1]
		}
		bestScores[insert] = score
		idxOut[insert] = i
	}

	var denom float32
	for j := 0; j < k; j++ {
		id := idxOut[j]
		if id < 0 || id >= len(rawScores) {
			continue
		}
		denom += rawScores[id]
	}
	if denom == 0 {
		denom = 1
	}
	for j := 0; j < k; j++ {
		id := idxOut[j]
		if id < 0 || id >= len(rawScores) {
			wOut[j] = 0
			continue
		}
		wOut[j] = (rawScores[id] / denom) * routeScale
	}
}
