package simd

import "testing"

func TestTraceTokensCapturesTinyRuntimeState(t *testing.T) {
	m := tinyRuntimeInstance(&DefaultOps{})

	trace, err := m.TraceTokens([]int{0, 1})
	if err != nil {
		t.Fatalf("TraceTokens: %v", err)
	}
	if got, want := trace.TokenIDs, []int{0, 1}; !equalInts(got, want) {
		t.Fatalf("token_ids=%v want %v", got, want)
	}
	if got, want := len(trace.InputsEmbeds), 2; got != want {
		t.Fatalf("inputs_embeds len=%d want %d", got, want)
	}
	if got, want := len(trace.HiddenStates), 1; got != want {
		t.Fatalf("hidden_states len=%d want %d", got, want)
	}
	if got, want := len(trace.HiddenStates[0]), 2; got != want {
		t.Fatalf("hidden_states[0] len=%d want %d", got, want)
	}
	if got := len(trace.LastTokenLogits); got != 3 {
		t.Fatalf("last_token_logits len=%d want 3", got)
	}
	if trace.NextTokenID != 1 {
		t.Fatalf("next_token_id=%d want 1", trace.NextTokenID)
	}
	if got := m.Pos; got != 2 {
		t.Fatalf("instance pos=%d want 2", got)
	}
}

func equalInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
