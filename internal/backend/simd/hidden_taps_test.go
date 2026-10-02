package simd

import (
	"fmt"
	"os"
	"testing"

	"github.com/samcharles93/mantle/internal/mcfstore"
	"github.com/samcharles93/mantle/pkg/mcf"
)

// These tests pin hidden-tap capture: the disabled path must be untouched, the
// captured values must be the real post-layer residual stream, both the batched
// and the single-token (fallback) loops must capture, and HiddenTaps.Rows must
// stay authoritative so a consumer can never mistake an incomplete capture for
// a complete one.

// enableHiddenTaps configures capture with room for one batched chunk of rows.
func enableHiddenTaps(tb testing.TB, m *Instance, layers []int) {
	tb.Helper()
	if err := m.asCore().SetHiddenTapLayers(layers, m.MaxBatch); err != nil {
		tb.Fatalf("SetHiddenTapLayers(%v): %v", layers, err)
	}
}

// tapRow returns the Hidden-wide captured row for a tap slot.
func tapRow(m *Instance, slot, row int) []float32 {
	t := &m.HiddenTaps
	start := (slot*t.Capacity + row) * t.Hidden
	return t.Host[start : start+t.Hidden]
}

func compareTapRows(tb testing.TB, label string, got, want []float32) {
	tb.Helper()
	if !floatsEqual(got, want, 1e-6) {
		tb.Fatalf("%s: tap differs: got %v, want %v", label, got, want)
	}
}

// TestHiddenTapDisabledIsNoOp proves the default path allocates nothing and
// leaves the logits bit-identical, while enabling capture changes no numeric
// result.
func TestHiddenTapDisabledIsNoOp(t *testing.T) {
	tokens := []int{1, 3, 2, 6}
	plain := newBatchParityInstance(t, batchParityContext, mcf.DTypeF32)
	want, err := plain.ForwardTokens(tokens)
	if err != nil {
		t.Fatalf("ForwardTokens: %v", err)
	}
	if plain.HiddenTaps.Host != nil || plain.HiddenTaps.Enabled() || plain.HiddenTaps.Rows != 0 ||
		plain.HiddenTaps.Capacity != 0 || plain.HiddenTaps.Layers != nil {
		t.Fatalf("disabled capture allocated state: %+v", plain.HiddenTaps)
	}

	tapped := newBatchParityInstance(t, batchParityContext, mcf.DTypeF32)
	enableHiddenTaps(t, tapped, []int{0, 1})
	got, err := (*Instance)(tapped).ForwardTokens(tokens)
	if err != nil {
		t.Fatalf("ForwardTokens (tapped): %v", err)
	}
	if !plain.batchEligible(tokens) || !tapped.batchEligible(tokens) {
		t.Fatal("both instances should take the batched path")
	}
	for r := range want {
		if !floatsEqual(got[r], want[r], 0) {
			t.Fatalf("capture changed logits at row %d: got %v, want %v", r, got[r], want[r])
		}
	}
}

// TestHiddenTapDoesNotAliasScratch proves the tap buffer is its own allocation
// sized from the instance width, not one of the reused Scratch buffers.
func TestHiddenTapDoesNotAliasScratch(t *testing.T) {
	m := newBatchParityInstance(t, batchParityContext, mcf.DTypeF32)
	enableHiddenTaps(t, m, []int{0, 1})
	hidden := m.Config.Config.EmbeddingLength
	if len(m.HiddenTaps.Host) != 2*m.MaxBatch*hidden {
		t.Fatalf("tap buffer = %d floats, want %d", len(m.HiddenTaps.Host), 2*m.MaxBatch*hidden)
	}
	if m.HiddenTaps.Hidden != hidden {
		t.Fatalf("tap row width = %d, want %d", m.HiddenTaps.Hidden, hidden)
	}
	for name, scratch := range map[string][]float32{"X": m.Scratch.X, "Tmp": m.Scratch.Tmp, "Tmp2": m.Scratch.Tmp2} {
		if len(scratch) == 0 {
			continue
		}
		if &m.HiddenTaps.Host[0] == &scratch[0] {
			t.Fatalf("tap buffer aliases Scratch.%s", name)
		}
	}
}

// TestHiddenTapBatchedMatchesSequential compares every captured row of a batched
// prompt against the same prompt run one token at a time, where each token's
// single-row capture is its own forward.
func TestHiddenTapBatchedMatchesSequential(t *testing.T) {
	tokens := []int{1, 3, 2, 6}
	layers := []int{0, 1}

	batched := newBatchParityInstance(t, batchParityContext, mcf.DTypeF32)
	enableHiddenTaps(t, batched, layers)
	if !batched.batchEligible(tokens) {
		t.Fatalf("prompt should be eligible: %s", batched.batchIneligibleReason(tokens))
	}
	if _, err := batched.ForwardTokens(tokens); err != nil {
		t.Fatalf("batched ForwardTokens: %v", err)
	}
	if !batchPathRan(batched) {
		t.Fatal("expected the batched path to run")
	}
	if batched.HiddenTaps.Rows != len(tokens) {
		t.Fatalf("Rows = %d, want %d", batched.HiddenTaps.Rows, len(tokens))
	}

	sequential := newBatchParityInstance(t, batchParityContext, mcf.DTypeF32)
	enableHiddenTaps(t, sequential, layers)
	for r, tok := range tokens {
		if _, err := sequential.ForwardToken(tok); err != nil {
			t.Fatalf("ForwardToken(%d): %v", tok, err)
		}
		if sequential.HiddenTaps.Rows != 1 {
			t.Fatalf("single-token capture Rows = %d, want 1", sequential.HiddenTaps.Rows)
		}
		for slot := range layers {
			compareTapRows(t, "batched vs sequential", tapRow(batched, slot, r), tapRow(sequential, slot, 0))
		}
	}
}

// TestHiddenTapFallbackStillCaptures takes the path the bug reports worry about:
// the batched loop refuses the prompt and the sequential loop has to capture.
// The kill switch rejects the batch without changing the math, so the last
// token's sequential residuals must equal the batched path's last row.
func TestHiddenTapFallbackStillCaptures(t *testing.T) {
	tokens := []int{1, 3, 2, 6}
	layers := []int{0, 1}

	batched := newBatchParityInstance(t, batchParityContext, mcf.DTypeF32)
	enableHiddenTaps(t, batched, layers)
	if _, err := batched.ForwardTokens(tokens); err != nil {
		t.Fatalf("batched ForwardTokens: %v", err)
	}
	if !batchPathRan(batched) {
		t.Fatal("expected the batched path to run")
	}

	t.Setenv(cpuBatchedPrefillEnv, "0")
	fallback := newBatchParityInstance(t, batchParityContext, mcf.DTypeF32)
	enableHiddenTaps(t, fallback, layers)
	if _, err := fallback.ForwardTokens(tokens); err != nil {
		t.Fatalf("fallback ForwardTokens: %v", err)
	}
	if batchPathRan(fallback) {
		t.Fatal("kill switch must keep the fallback off the batched path")
	}
	if fallback.HiddenTaps.Rows == 0 {
		t.Fatal("fallback captured no rows: the silent-skip bug")
	}
	if fallback.HiddenTaps.Rows != 1 {
		t.Fatalf("sequential fallback captures one row per step, got Rows = %d", fallback.HiddenTaps.Rows)
	}
	for slot := range layers {
		compareTapRows(t, "fallback last row", tapRow(fallback, slot, 0), tapRow(batched, slot, len(tokens)-1))
	}
}

// TestHiddenTapGateRejectionStillCaptures covers a model the batched gate
// refuses on its own merits (not the env kill switch) and still requires a
// non-empty capture.
func TestHiddenTapGateRejectionStillCaptures(t *testing.T) {
	tokens := []int{1, 3, 2}
	m := newBatchParityInstance(t, batchParityContext, mcf.DTypeF32)
	m.Layers[1].AttnWindow = 2 // sliding window: rejected by the gate
	enableHiddenTaps(t, m, []int{0, 1})
	if m.batchEligible(tokens) {
		t.Fatal("sliding-window model must not be eligible for a batch")
	}
	if _, err := m.ForwardTokens(tokens); err != nil {
		t.Fatalf("fallback ForwardTokens: %v", err)
	}
	if m.HiddenTaps.Rows == 0 {
		t.Fatal("gate-rejected model captured no rows")
	}
}

// TestHiddenTapSlotOrdering pins the concat order: slot i must hold Layers[i]'s
// output. A strictly-increasing list is required, so a permuted valid list does
// not exist; instead each slot is checked against a single-layer capture, which
// is exactly what a slot-index or sort-order bug would break.
func TestHiddenTapSlotOrdering(t *testing.T) {
	const tok = 5
	hidden := 8

	multi := newBatchParityInstance(t, batchParityContext, mcf.DTypeF32)
	enableHiddenTaps(t, multi, []int{0, 1})

	single := make([]*Instance, 2)
	for slot, layer := range []int{0, 1} {
		single[slot] = newBatchParityInstance(t, batchParityContext, mcf.DTypeF32)
		enableHiddenTaps(t, single[slot], []int{layer})
	}
	for _, m := range append([]*Instance{multi}, single...) {
		if _, err := m.ForwardToken(tok); err != nil {
			t.Fatalf("ForwardToken: %v", err)
		}
	}
	if multi.HiddenTaps.Hidden != hidden {
		t.Fatalf("tap width = %d, want %d", multi.HiddenTaps.Hidden, hidden)
	}
	for slot := range 2 {
		if multi.HiddenTaps.SlotFor(slot) != slot {
			t.Fatalf("SlotFor(%d) = %d, want %d", slot, multi.HiddenTaps.SlotFor(slot), slot)
		}
		compareTapRows(t, "slot vs single", tapRow(multi, slot, 0), tapRow(single[slot], 0, 0))
	}
	if floatsEqual(tapRow(multi, 0, 0), tapRow(multi, 1, 0), 1e-9) {
		t.Fatal("slot 0 and slot 1 captured the same layer; concat order is not pinned to the layer list")
	}
}

// TestSetHiddenTapLayersValidation pins the rejected configurations: an
// out-of-range or non-increasing list must be refused, and an empty list must
// disable capture again.
func TestSetHiddenTapLayersValidation(t *testing.T) {
	bad := []struct {
		name   string
		layers []int
	}{
		{"duplicate", []int{1, 1}},
		{"decreasing", []int{1, 0}},
		{"negative below embedding", []int{-2}},
		{"embedding not first", []int{0, -1}},
		{"past last layer", []int{0, 2}},
	}
	for _, tc := range bad {
		t.Run(tc.name, func(t *testing.T) {
			m := newBatchParityInstance(t, batchParityContext, mcf.DTypeF32)
			if err := m.asCore().SetHiddenTapLayers(tc.layers, 4); err == nil {
				t.Fatalf("SetHiddenTapLayers(%v) accepted a bad list", tc.layers)
			}
			if m.HiddenTaps.Enabled() {
				t.Fatal("failed configuration must leave capture disabled")
			}
		})
	}

	m := newBatchParityInstance(t, batchParityContext, mcf.DTypeF32)
	if err := m.asCore().SetHiddenTapLayers([]int{0, 1}, 0); err == nil {
		t.Fatal("zero capacity must be refused")
	}
	enableHiddenTaps(t, m, []int{0, 1})
	if err := m.asCore().SetHiddenTapLayers(nil, 4); err != nil {
		t.Fatalf("clearing: %v", err)
	}
	if m.HiddenTaps.Enabled() || m.HiddenTaps.Host != nil {
		t.Fatal("empty list must disable capture and release the buffer")
	}
}

// TestHiddenTapEmbeddingOutput checks the layer_id -1 tap, which is part of the
// reference API even though MiniCPM5-DSpark does not use it.
func TestHiddenTapEmbeddingOutput(t *testing.T) {
	const tok = 7
	m := newBatchParityInstance(t, batchParityContext, mcf.DTypeF32)
	if err := m.asCore().SetHiddenTapLayers([]int{-1, 0}, m.MaxBatch); err != nil {
		t.Fatalf("SetHiddenTapLayers: %v", err)
	}
	if _, err := m.ForwardToken(tok); err != nil {
		t.Fatalf("ForwardToken: %v", err)
	}
	if m.HiddenTaps.Rows != 1 {
		t.Fatalf("Rows = %d, want 1", m.HiddenTaps.Rows)
	}
	want := make([]float32, m.HiddenTaps.Hidden)
	m.Embeddings.RowTo(want, tok)
	compareTapRows(t, "embedding output", tapRow(m, 0, 0), want)
	if m.HiddenTaps.SlotFor(-1) != 0 {
		t.Fatalf("SlotFor(-1) = %d, want 0", m.HiddenTaps.SlotFor(-1))
	}
	if m.HiddenTaps.SlotFor(1) != -1 {
		t.Fatalf("SlotFor(1) = %d, want -1", m.HiddenTaps.SlotFor(1))
	}
}

// TestHiddenTapRowsAuthoritativeOnOverflow checks a forward wider than the tap
// buffer: it stores what fits and leaves Rows short, which is exactly how a
// consumer detects an incomplete capture instead of reading stale rows.
func TestHiddenTapRowsAuthoritativeOnOverflow(t *testing.T) {
	tokens := []int{1, 3, 2, 6}
	m := withLongContext(newBatchParityInstance(t, batchParityContext, mcf.DTypeF32), len(tokens)+1)
	if err := m.asCore().SetHiddenTapLayers([]int{0, 1}, 2); err != nil {
		t.Fatalf("SetHiddenTapLayers: %v", err)
	}
	if _, err := m.ForwardTokens(tokens); err != nil {
		t.Fatalf("ForwardTokens: %v", err)
	}
	if m.HiddenTaps.Rows != 2 {
		t.Fatalf("Rows = %d, want the buffer capacity 2", m.HiddenTaps.Rows)
	}
	if m.HiddenTaps.Rows >= len(tokens) {
		t.Fatal("Rows must stay short of the prompt length so the truncation is detectable")
	}
}

// TestHiddenTapMatchesIndependentLayerOutput is the correctness anchor: the tap
// for layer k must equal the residual stream after that layer, verified by
// running the same token through a model truncated to k+1 layers, whose final
// hidden state (Scratch.X) is exactly layer k's output.
func TestHiddenTapMatchesIndependentLayerOutput(t *testing.T) {
	const tok = 4
	full := newBatchParityInstance(t, batchParityContext, mcf.DTypeF32)
	enableHiddenTaps(t, full, []int{0, 1})
	if _, err := full.ForwardToken(tok); err != nil {
		t.Fatalf("full ForwardToken: %v", err)
	}

	for k := range 2 {
		truncated := newBatchParityInstance(t, batchParityContext, mcf.DTypeF32)
		truncated.Layers = truncated.Layers[:k+1]
		if _, err := truncated.ForwardToken(tok); err != nil {
			t.Fatalf("truncated ForwardToken(%d layers): %v", k+1, err)
		}
		compareTapRows(t, "layer output", tapRow(full, k, 0), truncated.Scratch.X)
	}
}

// TestHiddenTapMCFTruncationDeepLayers is the real-model correctness anchor for
// deep taps: the tap for layer k must equal the residual stream after layer k,
// verified by running the same token through a model truncated to k+1 layers,
// whose final hidden state is exactly layer k's output. Truncation cannot change
// any earlier layer, so the two runs must agree to floating-point
// reassociation. Set MANTLE_BATCH_MCF to a container this host can load.
func TestHiddenTapMCFTruncationDeepLayers(t *testing.T) {
	path := os.Getenv("MANTLE_BATCH_MCF")
	if path == "" {
		t.Skip("set MANTLE_BATCH_MCF=/path/to/model.mcf to run the real-model tap test")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("model not available: %v", err)
	}
	layers := []int{0, 10, 20, 39}
	load := func() *Instance {
		t.Helper()
		file, err := mcfstore.Open(path)
		if err != nil {
			t.Fatalf("open %s: %v", path, err)
		}
		t.Cleanup(func() { _ = file.Close() })
		m, err := LoadModelMCF(file, file.SectionData(mcf.SectionHFConfigJSON), 4, LoadModelOptions{HiddenTapLayers: layers})
		if err != nil {
			t.Fatalf("LoadModelMCF: %v", err)
		}
		return m
	}

	full := load()
	if len(full.Layers) <= layers[len(layers)-1] {
		t.Skipf("model has %d layers, need at least %d", len(full.Layers), layers[len(layers)-1]+1)
	}
	if _, err := full.ForwardToken(1); err != nil {
		t.Fatalf("full ForwardToken: %v", err)
	}
	for slot, k := range layers {
		truncated := load()
		truncated.Layers = truncated.Layers[:k+1]
		if _, err := truncated.ForwardToken(1); err != nil {
			t.Fatalf("truncated ForwardToken(%d layers): %v", k+1, err)
		}
		compareTapRows(t, fmt.Sprintf("layer %d truncation", k), tapRow(full, slot, 0), truncated.Scratch.X)
	}
}

// TestHiddenTapMCFParity is the opt-in real-model check: load the same MCF with
// and without taps and compare the batched capture against the per-token one.
// Set MANTLE_BATCH_MCF to a container this host can load.
func TestHiddenTapMCFParity(t *testing.T) {
	path := os.Getenv("MANTLE_BATCH_MCF")
	if path == "" {
		t.Skip("set MANTLE_BATCH_MCF=/path/to/model.mcf to run the real-model tap test")
	}
	if _, err := os.Stat(path); err != nil {
		t.Skipf("model not available: %v", err)
	}
	load := func(layers []int) *Instance {
		t.Helper()
		file, err := mcfstore.Open(path)
		if err != nil {
			t.Fatalf("open %s: %v", path, err)
		}
		t.Cleanup(func() { _ = file.Close() })
		m, err := LoadModelMCF(file, file.SectionData(mcf.SectionHFConfigJSON), 8, LoadModelOptions{HiddenTapLayers: layers})
		if err != nil {
			t.Fatalf("LoadModelMCF: %v", err)
		}
		return m
	}
	taps := []int{0, 1}
	tokens := []int{1, 2, 3, 4}

	batched := load(taps)
	if !batched.batchEligible(tokens) {
		t.Skipf("model is not eligible for a batched prefill: %s", batched.batchIneligibleReason(tokens))
	}
	if _, err := batched.ForwardTokens(tokens); err != nil {
		t.Fatalf("ForwardTokens: %v", err)
	}
	if batched.HiddenTaps.Rows != len(tokens) {
		t.Fatalf("Rows = %d, want %d", batched.HiddenTaps.Rows, len(tokens))
	}

	sequential := load(taps)
	for r, tok := range tokens {
		if _, err := sequential.ForwardToken(tok); err != nil {
			t.Fatalf("ForwardToken(%d): %v", tok, err)
		}
		for slot := range taps {
			if !floatsEqual(tapRow(batched, slot, r), tapRow(sequential, slot, 0), 5e-3) {
				t.Fatalf("tap slot %d row %d differs from the sequential capture", slot, r)
			}
		}
	}
}
