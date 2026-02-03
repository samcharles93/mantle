package model

import (
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

	// Use partial selection sort with stack allocation
	// Covers all practical MoE cases (k <= 8 typical)
	// This is O(k*n) with very low constant factor and zero allocations
	if k <= 8 {
		selectTopKSmall(selScores, rawScores, k, routeScale, idxOut, wOut)
		return
	}

	// For larger k, use heap-based selection (rare in practice)
	selectTopKHeap(selScores, rawScores, k, routeScale, idxOut, wOut)
}

// selectTopKSmall uses partial selection for small k values (k <= 8).
// Uses stack-allocated array for zero allocations.
func selectTopKSmall(selScores, rawScores []float32, k int, routeScale float32, idxOut []int, wOut []float32) {
	// Stack-allocated array for scores to avoid heap allocation
	// Support up to k=8 which covers all typical MoE configurations
	var bestIdx [8]int
	for i := 0; i < k; i++ {
		bestIdx[i] = -1
	}

	// Find top k using partial selection
	// Maintain sorted order: bestIdx[0] has highest score
	for i, score := range selScores {
		// Find position to insert (scores sorted descending)
		pos := k
		for j := 0; j < k; j++ {
			if bestIdx[j] == -1 || score > selScores[bestIdx[j]] {
				pos = j
				break
			}
		}
		if pos >= k {
			continue // Not in top k
		}
		// Shift elements to make room
		for j := k - 1; j > pos; j-- {
			bestIdx[j] = bestIdx[j-1]
		}
		bestIdx[pos] = i
	}

	// Copy to output
	for i := range k {
		idxOut[i] = bestIdx[i]
	}

	computeTopKWeights(selScores, rawScores, k, routeScale, idxOut, wOut)
}

// pair represents a score-index pair for heap operations.
type pair struct {
	score float32
	idx   int
}

// selectTopKHeap uses a min-heap for selection when k is larger.
// Complexity: O(n log k) vs O(k*n) for selection sort.
func selectTopKHeap(selScores, rawScores []float32, k int, routeScale float32, idxOut []int, wOut []float32) {
	// Simple binary min-heap storing (score, index) pairs
	// Root is the k-th largest element seen so far
	heap := make([]pair, 0, k)

	// Initialize heap with first k elements
	for i := 0; i < k && i < len(selScores); i++ {
		heap = append(heap, pair{selScores[i], i})
	}
	// Heapify (build min-heap)
	for i := len(heap)/2 - 1; i >= 0; i-- {
		heapifyDown(heap, i)
	}

	// Process remaining elements
	for i := k; i < len(selScores); i++ {
		score := selScores[i]
		if score > heap[0].score {
			// Replace min and heapify down
			heap[0] = pair{score, i}
			heapifyDown(heap, 0)
		}
	}

	// Extract from heap in descending order
	for i := k - 1; i >= 0; i-- {
		if len(heap) > 0 {
			idxOut[i] = heap[0].idx
			// Pop min and heapify
			heap[0] = heap[len(heap)-1]
			heap = heap[:len(heap)-1]
			if len(heap) > 0 {
				heapifyDown(heap, 0)
			}
		} else {
			idxOut[i] = -1
		}
	}

	computeTopKWeights(selScores, rawScores, k, routeScale, idxOut, wOut)
}

// heapifyDown maintains min-heap property by moving element at i down.
func heapifyDown(heap []pair, i int) {
	n := len(heap)
	for {
		minIdx := i
		left := 2*i + 1
		right := 2*i + 2
		if left < n && heap[left].score < heap[minIdx].score {
			minIdx = left
		}
		if right < n && heap[right].score < heap[minIdx].score {
			minIdx = right
		}
		if minIdx == i {
			break
		}
		heap[i], heap[minIdx] = heap[minIdx], heap[i]
		i = minIdx
	}
}

// computeTopKWeights computes normalized weights for selected experts.
func computeTopKWeights(selScores, rawScores []float32, k int, routeScale float32, idxOut []int, wOut []float32) {
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
