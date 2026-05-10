package graph

import "testing"

func TestPlanMemoryLinear(t *testing.T) {
	ResetTensorID()

	embedOut := NewTensorID()
	ffnOut := NewTensorID()
	outputOut := NewTensorID()

	g := &Graph{
		Nodes: []Node{
			{
				Op: OpEmbed, Branch: BranchEmbed, Name: "embed",
				Input: []TensorID{0}, Output: embedOut,
				Params: EmbedParams{VocabSize: 100, EmbDim: 16},
			},
			{
				Op: OpFFNBlock, Branch: BranchFFN, Name: "ffn",
				Input: []TensorID{embedOut}, Output: ffnOut,
				Params: FFNParams{Activation: "silu"},
			},
			{
				Op: OpOutput, Branch: BranchOutput, Name: "output",
				Input: []TensorID{ffnOut}, Output: outputOut,
				Params: OutputParams{},
			},
		},
	}

	lv := AnalyzeLiveness(g)
	slots := []ScratchSlot{SlotX}
	plan := lv.PlanMemory(g, slots)

	// All tensors should be assigned a slot in a linear chain with 1 slot.
	for _, tid := range []TensorID{embedOut, ffnOut, outputOut} {
		if plan.Allocations[tid] != SlotX {
			t.Errorf("tensor %d should be assigned a slot", tid)
		}
	}

	// In a linear chain with single-use intermediates, in-place reuse should occur.
	if !plan.InPlaceReuse {
		t.Error("expected in-place reuse for linear chain")
	}
	if plan.ReuseCount == 0 {
		t.Error("expected reuse count > 0 for linear chain with 1 slot")
	}
}

func TestPlanMemoryDAG(t *testing.T) {
	ResetTensorID()

	shared := NewTensorID()
	gateOut := NewTensorID()
	upOut := NewTensorID()
	merged := NewTensorID()

	g := &Graph{
		Nodes: []Node{
			{
				Op: OpAttentionBlock, Branch: BranchAttention, Name: "norm",
				Input: []TensorID{0}, Output: shared,
				Params: AttentionParams{LayerIndex: 0, HeadDim: 8, NHeadKV: 2},
			},
			{
				Op: OpFFNBlock, Branch: BranchFFN, Name: "gate",
				Input: []TensorID{shared}, Output: gateOut,
				Params: FFNParams{Activation: "silu"},
			},
			{
				Op: OpFFNBlock, Branch: BranchFFN, Name: "up",
				Input: []TensorID{shared}, Output: upOut,
				Params: FFNParams{Activation: "silu"},
			},
			{
				Op: OpFFNBlock, Branch: BranchFFN, Name: "merge",
				Input: []TensorID{gateOut, upOut}, Output: merged,
				Params: FFNParams{Activation: "silu"},
			},
		},
	}

	lv := AnalyzeLiveness(g)
	slots := []ScratchSlot{SlotX, SlotTmp}
	plan := lv.PlanMemory(g, slots)

	for _, tid := range []TensorID{shared, gateOut, upOut, merged} {
		if _, ok := plan.Allocations[tid]; !ok {
			t.Errorf("tensor %d should be assigned a slot", tid)
		}
	}

	assigned := make(map[ScratchSlot]bool)
	for _, s := range plan.Allocations {
		assigned[s] = true
	}
	if len(assigned) < 2 {
		t.Errorf("expected at least 2 distinct slots in use, got %d: %v", len(assigned), plan.Allocations)
	}

	if plan.GetSlot(shared) != plan.Allocations[shared] {
		t.Error("GetSlot returned wrong assignment")
	}
}

func TestPlanMemoryReuseCount(t *testing.T) {
	ResetTensorID()

	embedOut := NewTensorID()
	ffnOut := NewTensorID()
	outputOut := NewTensorID()

	g := &Graph{
		Nodes: []Node{
			{
				Op: OpEmbed, Branch: BranchEmbed, Name: "embed",
				Input: []TensorID{0}, Output: embedOut,
				Params: EmbedParams{VocabSize: 100, EmbDim: 16},
			},
			{
				Op: OpFFNBlock, Branch: BranchFFN, Name: "ffn",
				Input: []TensorID{embedOut}, Output: ffnOut,
				Params: FFNParams{Activation: "silu"},
			},
			{
				Op: OpOutput, Branch: BranchOutput, Name: "output",
				Input: []TensorID{ffnOut}, Output: outputOut,
				Params: OutputParams{},
			},
		},
	}

	lv := AnalyzeLiveness(g)
	plan := lv.PlanMemory(g, []ScratchSlot{SlotX})

	if plan.ReuseCount == 0 {
		t.Error("reuse count should be > 0 for linear chain with 1 slot")
	}
	t.Logf("reuse count: %d", plan.ReuseCount)
}

func TestInPlaceReuse(t *testing.T) {
	ResetTensorID()

	embedOut := NewTensorID()
	ffnOut := NewTensorID()
	outputOut := NewTensorID()

	g := &Graph{
		Nodes: []Node{
			{
				Op: OpEmbed, Branch: BranchEmbed, Name: "embed",
				Input: []TensorID{0}, Output: embedOut,
				Params: EmbedParams{VocabSize: 100, EmbDim: 16},
			},
			{
				Op: OpFFNBlock, Branch: BranchFFN, Name: "ffn",
				Input: []TensorID{embedOut}, Output: ffnOut,
				Params: FFNParams{Activation: "silu"},
			},
			{
				Op: OpOutput, Branch: BranchOutput, Name: "output",
				Input: []TensorID{ffnOut}, Output: outputOut,
				Params: OutputParams{},
			},
		},
	}

	lv := AnalyzeLiveness(g)
	plan := lv.PlanMemory(g, []ScratchSlot{SlotX, SlotTmp})

	if !plan.InPlaceReuse {
		t.Error("expected InPlaceReuse to be true")
	}

	ffnSlot := plan.GetSlot(ffnOut)
	embedSlot := plan.GetSlot(embedOut)
	if ffnSlot != embedSlot {
		t.Errorf("ffn output slot %v != embed output slot %v (in-place reuse expected)", ffnSlot, embedSlot)
	}

	outputSlot := plan.GetSlot(outputOut)
	if outputSlot != ffnSlot {
		t.Errorf("output slot %v != ffn slot %v (in-place reuse expected)", outputSlot, ffnSlot)
	}
}

func TestAnalyzeLivenessLinear(t *testing.T) {
	ResetTensorID()

	// Build linear chain: embed → ffn → output
	// Node 0 (embed): Input [0], Output 1 — consumed by node 1
	// Node 1 (ffn):   Input [1], Output 2 — consumed by node 2
	// Node 2 (output): Input [2], Output 3 — final output, no consumers
	embedOut := NewTensorID()  // 1
	ffnOut := NewTensorID()    // 2
	outputOut := NewTensorID() // 3

	g := &Graph{
		Nodes: []Node{
			{
				Op: OpEmbed, Branch: BranchEmbed, Name: "embed",
				Input: []TensorID{0}, Output: embedOut,
				Params: EmbedParams{VocabSize: 100, EmbDim: 16},
			},
			{
				Op: OpFFNBlock, Branch: BranchFFN, Name: "ffn",
				Input: []TensorID{embedOut}, Output: ffnOut,
				Params: FFNParams{Activation: "silu"},
			},
			{
				Op: OpOutput, Branch: BranchOutput, Name: "output",
				Input: []TensorID{ffnOut}, Output: outputOut,
				Params: OutputParams{},
			},
		},
	}

	lv := AnalyzeLiveness(g)

	// liveAt[0] = {} (nothing live before embed)
	if n := len(lv.LiveAt[0]); n != 0 {
		t.Errorf("liveAt[0] should be empty, got %d entries", n)
	}

	// liveAt[1] = {1} (tensor 1 live after embed produces it, before ffn executes)
	if n := len(lv.LiveAt[1]); n != 1 {
		t.Errorf("liveAt[1] should have 1 entry, got %d: %v", n, lv.LiveAt[1])
	} else if !lv.LiveAt[1][embedOut] {
		t.Errorf("liveAt[1] should contain embed output tensor %d", embedOut)
	}

	// liveAt[2] = {2} (tensor 1 consumed by ffn and died; tensor 2 now live)
	if n := len(lv.LiveAt[2]); n != 1 {
		t.Errorf("liveAt[2] should have 1 entry, got %d: %v", n, lv.LiveAt[2])
	} else if !lv.LiveAt[2][ffnOut] {
		t.Errorf("liveAt[2] should contain ffn output tensor %d", ffnOut)
	}

	// IsLive checks
	if !lv.IsLive(1, embedOut) {
		t.Error("IsLive(1, embedOut) should be true")
	}
	if lv.IsLive(0, embedOut) {
		t.Error("IsLive(0, embedOut) should be false (not produced yet)")
	}
	if lv.IsLive(2, embedOut) {
		t.Error("IsLive(2, embedOut) should be false (already consumed)")
	}
	if !lv.IsLive(2, ffnOut) {
		t.Error("IsLive(2, ffnOut) should be true")
	}

	// MaxLiveTensors = 1 (all live sets have at most 1 tensor)
	if lv.MaxLiveTensors() != 1 {
		t.Errorf("MaxLiveTensors should be 1, got %d", lv.MaxLiveTensors())
	}

	// TensorID 0 (sentinel) should never appear in liveness
	for idx, s := range lv.LiveAt {
		if s[0] {
			t.Errorf("sentinel TensorID 0 found in liveAt[%d]", idx)
		}
	}
}

func TestAnalyzeLivenessDAG(t *testing.T) {
	ResetTensorID()

	// DAG pattern: shared input fanning out to gate and up (FFN DAG)
	// Node 0 (norm): produces shared tensor (1), consumed by nodes 1 and 2
	// Node 1 (gate): Input [1], Output gate_out (2), consumed by node 3
	// Node 2 (up):   Input [1], Output up_out (3), consumed by node 3
	// Node 3 (merge): Input [2, 3], Output merged (4), final output
	shared := NewTensorID()  // 1
	gateOut := NewTensorID() // 2
	upOut := NewTensorID()   // 3
	merged := NewTensorID()  // 4

	g := &Graph{
		Nodes: []Node{
			{
				Op: OpAttentionBlock, Branch: BranchAttention, Name: "norm",
				Input: []TensorID{0}, Output: shared,
				Params: AttentionParams{LayerIndex: 0, HeadDim: 8, NHeadKV: 2},
			},
			{
				Op: OpFFNBlock, Branch: BranchFFN, Name: "gate",
				Input: []TensorID{shared}, Output: gateOut,
				Params: FFNParams{Activation: "silu"},
			},
			{
				Op: OpFFNBlock, Branch: BranchFFN, Name: "up",
				Input: []TensorID{shared}, Output: upOut,
				Params: FFNParams{Activation: "silu"},
			},
			{
				Op: OpFFNBlock, Branch: BranchFFN, Name: "merge",
				Input: []TensorID{gateOut, upOut}, Output: merged,
				Params: FFNParams{Activation: "silu"},
			},
		},
	}

	lv := AnalyzeLiveness(g)

	// liveAt[1] = {1} — after norm, shared is live (before gate executes)
	if !lv.IsLive(1, shared) || len(lv.LiveAt[1]) != 1 {
		t.Errorf("liveAt[1] should be {1}, got %v", lv.LiveAt[1])
	}

	// liveAt[2] = {1, 2} — after gate, shared still has one more use and gate_out is live
	if !lv.IsLive(2, shared) || !lv.IsLive(2, gateOut) || len(lv.LiveAt[2]) != 2 {
		t.Errorf("liveAt[2] should be {1, 2}, got %v", lv.LiveAt[2])
	}

	// liveAt[3] = {2, 3} — after up, shared consumed last time (died), gate_out and up_out now live
	if lv.IsLive(3, shared) {
		t.Error("shared should be dead after up consumes its last use")
	}
	if !lv.IsLive(3, gateOut) || !lv.IsLive(3, upOut) || len(lv.LiveAt[3]) != 2 {
		t.Errorf("liveAt[3] should be {2, 3}, got %v", lv.LiveAt[3])
	}

	// MaxLiveTensors should be 2 (peak at {1, 2} and {2, 3})
	if lv.MaxLiveTensors() != 2 {
		t.Errorf("MaxLiveTensors should be 2, got %d", lv.MaxLiveTensors())
	}
}

func TestMaxLiveTensors(t *testing.T) {
	ResetTensorID()

	// Graph where max live is 3: a fan produces 3 concurrent live tensors
	// Node 0 (source): produces tensor 1, consumed by nodes 1, 2, 3
	// Node 1 (a): produces tensor 2
	// Node 2 (b): produces tensor 3
	// Node 3 (c): consumes 1, 2, 3; produces tensor 4
	source := NewTensorID() // 1
	a := NewTensorID()      // 2
	b := NewTensorID()      // 3
	c := NewTensorID()      // 4

	g := &Graph{
		Nodes: []Node{
			{
				Op: OpEmbed, Branch: BranchEmbed, Name: "source",
				Input: []TensorID{0}, Output: source,
				Params: EmbedParams{VocabSize: 100, EmbDim: 16},
			},
			{
				Op: OpFFNBlock, Branch: BranchFFN, Name: "a",
				Input: []TensorID{source}, Output: a,
				Params: FFNParams{Activation: "silu"},
			},
			{
				Op: OpFFNBlock, Branch: BranchFFN, Name: "b",
				Input: []TensorID{source}, Output: b,
				Params: FFNParams{Activation: "silu"},
			},
			{
				Op: OpFFNBlock, Branch: BranchFFN, Name: "c",
				Input: []TensorID{source, a, b}, Output: c,
				Params: FFNParams{Activation: "silu"},
			},
		},
	}

	lv := AnalyzeLiveness(g)

	// After node 2 (before node 3): live = {1, 2, 3}
	if n := lv.MaxLiveTensors(); n != 3 {
		t.Errorf("MaxLiveTensors should be 3, got %d. LiveAt: %v", n, lv.LiveAt)
	}
}

func TestAnalyzeLivenessEmptyGraph(t *testing.T) {
	g := &Graph{Nodes: []Node{}}
	lv := AnalyzeLiveness(g)
	if len(lv.LiveAt) != 0 {
		t.Errorf("LiveAt should be empty for empty graph, got %v", lv.LiveAt)
	}
	if lv.MaxLiveTensors() != 0 {
		t.Errorf("MaxLiveTensors should be 0 for empty graph, got %d", lv.MaxLiveTensors())
	}
}

func TestIsLiveMissingNode(t *testing.T) {
	ResetTensorID()
	g := &Graph{
		Nodes: []Node{
			{
				Op: OpEmbed, Branch: BranchEmbed, Name: "embed",
				Input: []TensorID{0}, Output: NewTensorID(),
				Params: EmbedParams{VocabSize: 100, EmbDim: 16},
			},
		},
	}
	lv := AnalyzeLiveness(g)
	if lv.IsLive(999, TensorID(1)) {
		t.Error("IsLive with out-of-range node index should return false")
	}
}
