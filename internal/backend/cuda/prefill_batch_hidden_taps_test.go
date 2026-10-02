//go:build cuda

package cuda

import (
	"testing"

	"github.com/samcharles93/mantle/internal/backend/core"
)

// Hidden-tap capture on the CUDA batched prefill path: the per-layer residual
// is copied device-to-device into a tap buffer and read back once, and the
// per-token fallback path must capture too.

// enableCUDATaps configures capture with room for one full batch of rows.
func enableCUDATaps(t *testing.T, m *core.Instance, layers []int) {
	t.Helper()
	if err := m.SetHiddenTapLayers(layers, m.MaxContext); err != nil {
		t.Fatalf("SetHiddenTapLayers(%v): %v", layers, err)
	}
}

func cudaTapRow(m *core.Instance, slot, row int) []float32 {
	t := &m.HiddenTaps
	start := (slot*t.Capacity + row) * t.Hidden
	return t.Host[start : start+t.Hidden]
}

// TestBatchedPrefillHiddenTaps checks the batched tap capture against the
// per-token CUDA decode path, which is the untouched oracle.
func TestBatchedPrefillHiddenTaps(t *testing.T) {
	if !cudaAvailable(t) {
		return
	}
	d := defaultBatchTestDims()
	tokens := []int{1, 3, 6}
	layers := []int{0, 1}

	batchedInst := batchTestModel(d)
	enableCUDATaps(t, batchedInst, layers)
	grBatched, cleanup := newBatchTestRuntime(t, batchedInst)
	defer cleanup()
	if !grBatched.batchedPrefillEligible(tokens) {
		t.Fatal("expected the prompt to be eligible for batched prefill")
	}
	if _, err := grBatched.PrefillTokens(tokens); err != nil {
		t.Fatalf("batched PrefillTokens: %v", err)
	}
	if batchedInst.HiddenTaps.Rows != len(tokens) {
		t.Fatalf("Rows = %d, want %d", batchedInst.HiddenTaps.Rows, len(tokens))
	}

	seqInst := batchTestModel(d)
	enableCUDATaps(t, seqInst, layers)
	grSeq, cleanupSeq := newBatchTestRuntime(t, seqInst)
	defer cleanupSeq()
	for r, tok := range tokens {
		if _, err := grSeq.ForwardToken(tok); err != nil {
			t.Fatalf("sequential ForwardToken(%d): %v", tok, err)
		}
		if seqInst.HiddenTaps.Rows != 1 {
			t.Fatalf("sequential capture Rows = %d, want 1", seqInst.HiddenTaps.Rows)
		}
		for slot := range layers {
			got := cudaTapRow(batchedInst, slot, r)
			want := cudaTapRow(seqInst, slot, 0)
			if !floatsClose(got, want, 1e-4) {
				t.Fatalf("slot %d row %d mismatch: got %v, want %v", slot, r, got, want)
			}
		}
	}
}

// TestBatchedPrefillHiddenTapsFallback checks capture survives the batched
// path's refusal. The env kill switch rejects the batch without changing the
// math, so the per-token fallback's single captured row must equal the batched
// path's last row; a single token (rejected by the gate itself) must also
// record a row rather than silently capturing nothing.
func TestBatchedPrefillHiddenTapsFallback(t *testing.T) {
	if !cudaAvailable(t) {
		return
	}
	d := defaultBatchTestDims()
	tokens := []int{1, 3, 6}
	layers := []int{0, 1}

	batchedInst := batchTestModel(d)
	enableCUDATaps(t, batchedInst, layers)
	grBatched, cleanup := newBatchTestRuntime(t, batchedInst)
	defer cleanup()
	if _, err := grBatched.PrefillTokens(tokens); err != nil {
		t.Fatalf("batched PrefillTokens: %v", err)
	}
	if batchedInst.HiddenTaps.Rows != len(tokens) {
		t.Fatalf("batched Rows = %d, want %d", batchedInst.HiddenTaps.Rows, len(tokens))
	}

	t.Setenv(batchedPrefillEnv, "0")
	fallbackInst := batchTestModel(d)
	enableCUDATaps(t, fallbackInst, layers)
	grFallback, cleanupFallback := newBatchTestRuntime(t, fallbackInst)
	defer cleanupFallback()
	if batchedPrefillEnabled() {
		t.Fatal("kill switch should disable batched prefill")
	}
	if _, err := grFallback.PrefillTokens(tokens); err != nil {
		t.Fatalf("fallback PrefillTokens: %v", err)
	}
	if fallbackInst.HiddenTaps.Rows == 0 {
		t.Fatal("fallback captured no rows: the silent-skip bug")
	}
	if fallbackInst.HiddenTaps.Rows != 1 {
		t.Fatalf("sequential fallback captures one row per step, got Rows = %d", fallbackInst.HiddenTaps.Rows)
	}
	for slot := range layers {
		got := cudaTapRow(fallbackInst, slot, 0)
		want := cudaTapRow(batchedInst, slot, len(tokens)-1)
		if !floatsClose(got, want, 1e-4) {
			t.Fatalf("slot %d fallback row mismatch: got %v, want %v", slot, got, want)
		}
	}

	// A single token is rejected by the gate itself (not the env switch), which
	// is the silent-fallback path the capture must not skip.
	t.Setenv(batchedPrefillEnv, "1")
	singleInst := batchTestModel(d)
	enableCUDATaps(t, singleInst, layers)
	grSingle, cleanupSingle := newBatchTestRuntime(t, singleInst)
	defer cleanupSingle()
	single := tokens[:1]
	if grSingle.batchedPrefillEligible(single) {
		t.Fatal("single-token prompt must not be eligible for batched prefill")
	}
	if _, err := grSingle.PrefillTokens(single); err != nil {
		t.Fatalf("single-token PrefillTokens: %v", err)
	}
	if singleInst.HiddenTaps.Rows == 0 {
		t.Fatal("gate-rejected single token captured no rows")
	}
}

func floatsClose(a, b []float32, tol float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		diff := a[i] - b[i]
		if diff < 0 {
			diff = -diff
		}
		if diff > tol {
			return false
		}
	}
	return true
}
