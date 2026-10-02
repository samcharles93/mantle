//go:build cuda

package cuda

import (
	"math"
	"testing"
)

// The CUDA output head stages its device MatVec result and copies it back into
// m.Scratch.Logits when EndToken flushes it. Logit postprocessing runs on the
// host before that flush, so unless the copy is forced first the host math
// operates on the previous token's values and is then overwritten by the raw
// device result. Both LMHeadMultiplier and FinalLogitSoftcap were silently
// ignored on the CUDA path because of this.
//
// Greedy decode could not have caught it: both transforms are strictly
// monotonic, so argmax - and therefore generated tokens - is unchanged. Only
// the returned logits (used for temperature/top-k/top-p sampling) are wrong.
func TestForwardTokenAppliesLogitPostprocessCUDA(t *testing.T) {
	if !cudaAvailable(t) {
		return
	}
	const tok = 3

	base := defaultBatchTestDims()
	grBase, cleanupBase := newBatchTestRuntime(t, batchTestModel(base))
	defer cleanupBase()
	raw, err := grBase.ForwardToken(tok)
	if err != nil {
		t.Fatalf("baseline ForwardToken: %v", err)
	}
	raw = append([]float32(nil), raw...)

	t.Run("lm_head_multiplier", func(t *testing.T) {
		d := defaultBatchTestDims()
		d.lmHeadMult = 1.5
		gr, cleanup := newBatchTestRuntime(t, batchTestModel(d))
		defer cleanup()

		got, err := gr.ForwardToken(tok)
		if err != nil {
			t.Fatalf("ForwardToken: %v", err)
		}
		var identical = true
		for i := range raw {
			if got[i] != raw[i] {
				identical = false
				break
			}
		}
		if identical {
			t.Fatal("lm_head_multiplier had no effect on the returned logits")
		}
		for i := range raw {
			want := raw[i] * 1.5
			if diff := math.Abs(float64(got[i] - want)); diff > 1e-4 {
				t.Fatalf("logit[%d]: got %v want %v (diff %g)", i, got[i], want, diff)
			}
		}
	})

	t.Run("final_logit_softcap", func(t *testing.T) {
		const cap = 0.5
		d := defaultBatchTestDims()
		d.finalSoftcap = cap
		gr, cleanup := newBatchTestRuntime(t, batchTestModel(d))
		defer cleanup()

		got, err := gr.ForwardToken(tok)
		if err != nil {
			t.Fatalf("ForwardToken: %v", err)
		}
		for i := range raw {
			want := float32(math.Tanh(float64(raw[i]/cap)) * cap)
			if diff := math.Abs(float64(got[i] - want)); diff > 1e-4 {
				t.Fatalf("logit[%d]: got %v want %v (diff %g)", i, got[i], want, diff)
			}
		}
	})
}
