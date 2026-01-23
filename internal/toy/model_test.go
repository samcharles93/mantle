package toy

import (
    "math"
    "testing"

    //"infer/internal/tensor"
)

// TestForwardMatchesNaive compares the ToyLM Forward method against a
// hand‑computed reference for a single token.  Random seeds ensure
// reproducible weights.
func TestForwardMatchesNaive(t *testing.T) {
    vocab, hidden := 8, 6
    seed := int64(5)
    model := NewToyLM(vocab, hidden, seed)
    tok := 3
    // Compute logits via Forward
    logits := model.Forward(tok)
    // Compute logits manually: embed vector then mat‑vec multiply
    // h = Emb[tok]
    h := make([]float32, hidden)
    copy(h, model.Emb.Row(tok))
    ref := make([]float32, vocab)
    for j := 0; j < vocab; j++ {
        var sum float32
        for i := 0; i < hidden; i++ {
            sum += h[i] * model.W.Row(i)[j]
        }
        ref[j] = sum + model.Bias[j]
    }
    // Compare
    for i := range logits {
        if math.Abs(float64(logits[i]-ref[i])) > 1e-4 {
            t.Fatalf("logit mismatch at %d: got %f, want %f", i, logits[i], ref[i])
        }
    }
}

// TestForwardNoAllocs verifies that Forward allocates only its output slice and
// no additional heap space during execution.  We expect one allocation for the
// return slice of size vocab.
func TestForwardNoAllocs(t *testing.T) {
    model := NewToyLM(5, 3, 2)
    allocs := testing.AllocsPerRun(100, func() {
        _ = model.Forward(1)
    })
    // One allocation is expected: the logits slice returned from Forward.
    if allocs != 1 {
        t.Fatalf("expected 1 allocation, got %v", allocs)
    }
}
