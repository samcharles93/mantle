package logits

import "testing"

// TestSamplerDeterminism ensures that two samplers configured identically
// produce identical results when sampling the same logits vector.
func TestSamplerDeterminism(t *testing.T) {
    logs := []float32{0, 1, 2, 3, 4, 5}
    s1 := NewSampler(SamplerConfig{Seed: 42, Temperature: 0.9, TopK: 4, TopP: 0.95})
    s2 := NewSampler(SamplerConfig{Seed: 42, Temperature: 0.9, TopK: 4, TopP: 0.95})
    a := s1.Sample(logs, nil)
    b := s2.Sample(logs, nil)
    if a != b {
        t.Fatalf("expected deterministic sample, got %d vs %d", a, b)
    }
}

// TestSamplerGreedy tests that greedy sampling (TopK=1, Temperature=1, TopP>=1)
// returns the index of the maximum logit.
func TestSamplerGreedy(t *testing.T) {
    logs := []float32{-1, 5, 3, 7, 2}
    s := NewSampler(SamplerConfig{Seed: 99, Temperature: 1.0, TopK: 1, TopP: 1.0})
    idx := s.Sample(logs, nil)
    if idx != 3 {
        t.Fatalf("expected greedy index 3, got %d", idx)
    }
}

// TestSamplerTopP ensures that setting TopP less than 1 restricts sampling to a
// prefix of candidates.  In this contrived example, the cumulative
// probability after the first element is >TopP, so only the first index
// should ever be returned.
func TestSamplerTopP(t *testing.T) {
    // Construct logits such that the highest value dominates after softmax.
    logs := []float32{10, 0, 0, 0, 0}
    s := NewSampler(SamplerConfig{Seed: 7, Temperature: 1.0, TopK: 5, TopP: 0.5})
    // Repeatedly sample and ensure we always get index 0.
    for i := 0; i < 10; i++ {
        idx := s.Sample(logs, nil)
        if idx != 0 {
            t.Fatalf("top‑p sampling returned unexpected index %d", idx)
        }
    }
}