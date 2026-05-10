package graph

import "fmt"

// ScratchSlot identifies a pre-allocated scratch buffer slot available for
// tensor memory reuse during graph execution.
type ScratchSlot int

const (
	SlotX       ScratchSlot = iota // primary residual/input buffer
	SlotTmp                        // temporary buffer 1
	SlotTmp2                       // temporary buffer 2
	SlotQ                          // query projection
	SlotK                          // key projection
	SlotV                          // value projection
	SlotAttnOut                    // attention output
	SlotFfnGate                    // FFN gate projection
	SlotFfnUp                      // FFN up projection
	SlotFfnAct                     // FFN activation output
)

func (s ScratchSlot) String() string {
	switch s {
	case SlotX:
		return "SlotX"
	case SlotTmp:
		return "SlotTmp"
	case SlotTmp2:
		return "SlotTmp2"
	case SlotQ:
		return "SlotQ"
	case SlotK:
		return "SlotK"
	case SlotV:
		return "SlotV"
	case SlotAttnOut:
		return "SlotAttnOut"
	case SlotFfnGate:
		return "SlotFfnGate"
	case SlotFfnUp:
		return "SlotFfnUp"
	case SlotFfnAct:
		return "SlotFfnAct"
	default:
		return fmt.Sprintf("ScratchSlot(%d)", int(s))
	}
}

// TensorLiveness tracks which tensors are simultaneously live at each node
// index. A tensor is "live" from the node that produces it until the last
// node that consumes it. LiveAt[i] is the set of tensors live BEFORE node i
// executes — this informs memory planning about what scratch space is occupied
// when scheduling each node.
type TensorLiveness struct {
	LiveAt map[int]map[TensorID]bool
}

// MemoryPlan is the result of memory planning: a mapping from tensor IDs to
// scratch buffer slots, with reuse tracking.
type MemoryPlan struct {
	Allocations  map[TensorID]ScratchSlot
	ReuseCount   int
	InPlaceReuse bool
}

// GetSlot returns the scratch slot assigned to tid, or SlotX if unassigned.
func (mp *MemoryPlan) GetSlot(tid TensorID) ScratchSlot {
	if mp.Allocations == nil {
		return SlotX
	}
	return mp.Allocations[tid]
}

// AnalyzeLiveness performs tensor lifetime analysis on a computation graph.
// TensorID 0 is a sentinel (weights/params) and is excluded from liveness.
func AnalyzeLiveness(g *Graph) *TensorLiveness {
	if len(g.Nodes) == 0 {
		return &TensorLiveness{LiveAt: make(map[int]map[TensorID]bool)}
	}

	useCount := make(map[TensorID]int)
	for _, n := range g.Nodes {
		for _, tid := range n.Input {
			if tid != 0 {
				useCount[tid]++
			}
		}
	}

	liveAt := make(map[int]map[TensorID]bool, len(g.Nodes))
	live := make(map[TensorID]bool)

	for i, n := range g.Nodes {
		snapshot := make(map[TensorID]bool, len(live))
		for tid := range live {
			snapshot[tid] = true
		}
		liveAt[i] = snapshot

		for _, tid := range n.Input {
			if tid == 0 {
				continue
			}
			useCount[tid]--
			if useCount[tid] == 0 {
				delete(live, tid)
			}
		}

		live[n.Output] = true
	}

	return &TensorLiveness{LiveAt: liveAt}
}

// MaxLiveTensors returns the maximum number of simultaneously live tensors
// at any point in the graph.
func (tl *TensorLiveness) MaxLiveTensors() int {
	maxN := 0
	for _, s := range tl.LiveAt {
		if n := len(s); n > maxN {
			maxN = n
		}
	}
	return maxN
}

// IsLive reports whether tensor tid is live immediately before nodeIdx
// executes. Returns false for out-of-range node indices.
func (tl *TensorLiveness) IsLive(nodeIdx int, tid TensorID) bool {
	s, ok := tl.LiveAt[nodeIdx]
	if !ok {
		return false
	}
	return s[tid]
}

// PlanMemory assigns each tensor in the graph to a scratch buffer slot from
// the provided pool. It uses a two-phase algorithm:
//
// Phase 1 (reserve): compute use counts for all tensors to determine
// when each tensor becomes live and when it dies.
//
// Phase 2 (alloc): walk nodes in execution order. When a node produces
// an output tensor, assign it a free slot. When a node consumes the last
// use of an input tensor, the input dies and its slot returns to the pool.
//
// If the pool is exhausted, remaining tensors are left unassigned.
func (tl *TensorLiveness) PlanMemory(g *Graph, slots []ScratchSlot) *MemoryPlan {
	if len(g.Nodes) == 0 {
		return &MemoryPlan{
			Allocations: make(map[TensorID]ScratchSlot),
		}
	}

	useCount := make(map[TensorID]int)
	for _, n := range g.Nodes {
		for _, tid := range n.Input {
			if tid != 0 {
				useCount[tid]++
			}
		}
	}

	singleUse := make(map[TensorID]bool)
	for i := range g.Nodes {
		outID := g.Nodes[i].Output
		if outID != 0 && useCount[outID] == 1 {
			singleUse[outID] = true
		}
	}

	allocations := make(map[TensorID]ScratchSlot)
	slotFree := make([]bool, len(slots))
	for i := range slotFree {
		slotFree[i] = true
	}
	slotEverUsed := make([]bool, len(slots))
	freeCount := len(slots)

	slotIndex := make(map[ScratchSlot]int, len(slots))
	for i, s := range slots {
		slotIndex[s] = i
	}

	reused := 0
	inPlace := false

	for i := range g.Nodes {
		n := g.Nodes[i]

		var nodeInPlace bool
		for _, tid := range n.Input {
			if tid == 0 {
				continue
			}
			sl, ok := allocations[tid]
			if !ok {
				continue
			}
			si, ok := slotIndex[sl]
			if !ok {
				continue
			}
			if singleUse[tid] && !slotFree[si] {
				allocations[n.Output] = sl
				nodeInPlace = true
				inPlace = true
				if slotEverUsed[si] {
					reused++
				} else {
					slotEverUsed[si] = true
				}
				break
			}
		}

		for _, tid := range n.Input {
			if tid == 0 {
				continue
			}
			useCount[tid]--
			if useCount[tid] != 0 {
				continue
			}
			sl, ok := allocations[tid]
			if !ok {
				continue
			}
			si, ok := slotIndex[sl]
			if !ok {
				continue
			}
			if nodeInPlace && n.Output != 0 && allocations[n.Output] == sl {
				continue
			}
			slotFree[si] = true
			freeCount++
		}

		if _, already := allocations[n.Output]; already {
			continue
		}
		if freeCount == 0 {
			continue
		}

		for si := len(slotFree) - 1; si >= 0; si-- {
			if !slotFree[si] {
				continue
			}
			slotFree[si] = false
			freeCount--
			allocations[n.Output] = slots[si]
			if slotEverUsed[si] {
				reused++
			}
			slotEverUsed[si] = true
			break
		}
	}

	return &MemoryPlan{
		Allocations:  allocations,
		ReuseCount:   reused,
		InPlaceReuse: inPlace,
	}
}
