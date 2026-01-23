package logits

import (
    "math"
    "math/rand"
)

// SamplerConfig configures the behaviour of a Sampler.  Temperature values <=0
// are treated as 1 (no scaling).  TopK<=0 means a default value (40) is used.
// TopP outside (0,1] is clamped to 1.
type SamplerConfig struct {
    Seed        int64
    Temperature float32
    TopK        int
    TopP        float32
}

// Sampler encapsulates a random number generator and sampling parameters.  It
// supports top‑k and nucleus (top‑p) truncation as well as temperature scaling
// when selecting an index from a logits vector.
type Sampler struct {
    rng *rand.Rand
    cfg SamplerConfig
}

// NewSampler returns a new sampler with the provided configuration.  Invalid
// configuration values are normalised to sensible defaults.  Multiple
// samplers created with the same config will sample deterministically if the
// same RNG seed is used.
func NewSampler(cfg SamplerConfig) *Sampler {
    if cfg.Temperature <= 0 {
        cfg.Temperature = 1
    }
    if cfg.TopK <= 0 {
        cfg.TopK = 40
    }
    if cfg.TopP <= 0 || cfg.TopP > 1 {
        cfg.TopP = 1
    }
    return &Sampler{
        rng: rand.New(rand.NewSource(cfg.Seed)),
        cfg: cfg,
    }
}

// Sample draws a single index from the provided logits vector.  The sample
// process involves the following steps:
//
//  1. If TopK==1, TopP>=1 and Temperature==1 then argmax is returned (greedy).
//  2. Otherwise the logits are scaled by the inverse temperature and the
//     indices of the top k values are selected.
//  3. A softmax over the shortlisted values is computed using an arbitrary
//     constant subtraction for numerical stability.
//  4. If TopP<1, the shortlist is truncated when the cumulative probability
//     reaches TopP.
//  5. A random value is drawn from [0,1) and used to select an index from the
//     truncated distribution.
func (s *Sampler) Sample(logits []float32, recent []int) int {
    // Greedy path: simply return the index of the maximum logit.
    if s.cfg.TopK == 1 && s.cfg.TopP >= 1 && s.cfg.Temperature == 1 {
        return argmax(logits)
    }

    temp := s.cfg.Temperature
    invTemp := float32(1.0) / temp

    // Bound k by the length of the logits vector.
    k := s.cfg.TopK
    if k > len(logits) {
        k = len(logits)
    }

    // Find the indices and values of the top k logits scaled by 1/temp.
    topIdx, topVal := topK(logits, k, invTemp)

    // Compute exponentials in a numerically stable way: subtract the max value.
    maxv := topVal[0]
    for i := 1; i < len(topVal); i++ {
        if topVal[i] > maxv {
            maxv = topVal[i]
        }
    }

    prob := make([]float64, len(topVal))
    var sum float64
    for i := range topVal {
        // Exponentiate difference from maxv.
        x := float64(topVal[i] - maxv)
        e := math.Exp(x)
        prob[i] = e
        sum += e
    }
    if sum == 0 {
        // Should not happen unless logits are very negative; fallback to greedy.
        return topIdx[0]
    }
    invSum := 1.0 / sum
    for i := range prob {
        prob[i] *= invSum
    }

    // Nucleus/top‑p truncation: cut off shortlist when cumulative prob >= TopP.
    cut := len(prob)
    if s.cfg.TopP < 1 {
        var c float64
        for i := 0; i < len(prob); i++ {
            c += prob[i]
            if float32(c) >= s.cfg.TopP {
                cut = i + 1
                break
            }
        }
    }

    // Sample from the truncated distribution.
    r := s.rng.Float64()
    var c float64
    for i := 0; i < cut; i++ {
        c += prob[i]
        if r <= c {
            return topIdx[i]
        }
    }
    // Should not reach here; return last candidate.
    return topIdx[cut-1]
}

// argmax returns the index of the maximum value in the slice.  If the slice is
// empty it panics.
func argmax(x []float32) int {
    if len(x) == 0 {
        panic("argmax: empty slice")
    }
    bestI := 0
    bestV := x[0]
    for i := 1; i < len(x); i++ {
        if x[i] > bestV {
            bestV = x[i]
            bestI = i
        }
    }
    return bestI
}

// topK returns the indices and values of the k largest elements in logits,
// scaled by invTemp.  The returned slices are ordered from largest to
// smallest by value.  This is an O(V*K) algorithm suitable for small K.
func topK(logits []float32, k int, invTemp float32) ([]int, []float32) {
    idx := make([]int, 0, k)
    val := make([]float32, 0, k)
    for i, l := range logits {
        v := l * invTemp
        // Find insertion point into current sorted list
        pos := len(val)
        for pos > 0 && val[pos-1] < v {
            pos--
        }
        if pos >= k {
            continue
        }
        // Extend slices
        idx = append(idx, 0)
        val = append(val, 0)
        // Shift values right to make room
        copy(idx[pos+1:], idx[pos:])
        copy(val[pos+1:], val[pos:])
        idx[pos] = i
        val[pos] = v
        // Trim to length k
        if len(val) > k {
            idx = idx[:k]
            val = val[:k]
        }
    }
    if len(idx) == 0 {
        // Provide a default candidate if logits were empty (should not happen)
        return []int{0}, []float32{0}
    }
    return idx, val
}