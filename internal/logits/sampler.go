package logits

import (
	"math"
	"math/rand"
)

// SamplerConfig configures the behaviour of a Sampler.
type SamplerConfig struct {
	Seed          int64
	Temperature   float32
	TopK          int
	TopP          float32
	RepeatPenalty float32
	RepeatLastN   int
}

type Sampler struct {
	rng *rand.Rand
	cfg SamplerConfig
}

// NewSampler returns a new sampler with the provided configuration.
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
	if cfg.RepeatPenalty <= 0 {
		cfg.RepeatPenalty = 1.0
	}
	if cfg.RepeatLastN <= 0 {
		cfg.RepeatLastN = 64
	}
	return &Sampler{
		rng: rand.New(rand.NewSource(cfg.Seed)),
		cfg: cfg,
	}
}

// Sample draws a single index from the provided logits vector.  The sample
// process involves the following steps:
//
//  1. Apply repetition penalty if configured.
//  2. If TopK==1, TopP>=1 and Temperature==1 then argmax is returned (greedy).
//  3. Otherwise the logits are scaled by the inverse temperature and the
//     indices of the top k values are selected.
//  4. A softmax over the shortlisted values is computed using an arbitrary
//     constant subtraction for numerical stability.
//  5. If TopP<1, the shortlist is truncated when the cumulative probability
//     reaches TopP.
//  6. A random value is drawn from [0,1) and used to select an index from the
//     truncated distribution.
func (s *Sampler) Sample(logits []float32, recent []int, excludePenalty []int) int {
	if s.cfg.RepeatPenalty > 1.0 && len(recent) > 0 {
		start := len(recent) - s.cfg.RepeatLastN
		if start < 0 {
			start = 0
		}
		window := recent[start:]

		seen := make(map[int]struct{}, len(window))
		for _, id := range window {
			if id >= 0 && id < len(logits) {
				seen[id] = struct{}{}
			}
		}
		
		for _, id := range excludePenalty {
			delete(seen, id)
		}

		for id := range seen {
			if logits[id] > 0 {
				logits[id] /= s.cfg.RepeatPenalty
			} else {
				logits[id] *= s.cfg.RepeatPenalty
			}
		}
	}

	if s.cfg.TopK == 1 && s.cfg.TopP >= 1 && s.cfg.Temperature == 1 {
		return argmax(logits)
	}

	temp := s.cfg.Temperature
	invTemp := float32(1.0) / temp

	k := min(s.cfg.TopK, len(logits))

	topIdx, topVal := topK(logits, k, invTemp)

	maxv := topVal[0]
	for i := 1; i < len(topVal); i++ {
		if topVal[i] > maxv {
			maxv = topVal[i]
		}
	}

	prob := make([]float64, len(topVal))
	var sum float64
	for i := range topVal {
		x := float64(topVal[i] - maxv)
		e := math.Exp(x)
		prob[i] = e
		sum += e
	}
	if sum == 0 {
		return topIdx[0]
	}
	invSum := 1.0 / sum
	for i := range prob {
		prob[i] *= invSum
	}

	cut := len(prob)
	if s.cfg.TopP < 1 {
		var c float64
		for i := range prob {
			c += prob[i]
			if float32(c) >= s.cfg.TopP {
				cut = i + 1
				break
			}
		}
	}

	r := s.rng.Float64()
	var c float64
	for i := 0; i < cut; i++ {
		c += prob[i]
		if r <= c {
			return topIdx[i]
		}
	}

	return topIdx[cut-1]
}

// argmax returns the index of the maximum value in the slice. If the slice is empty it panics.
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

// topK returns the indices and values of the k largest elements in logits, scaled by invTemp.
// The returned slices are ordered from largest to smallest by value.
// This is an O(V*K) algorithm suitable for small K.
func topK(logits []float32, k int, invTemp float32) ([]int, []float32) {
	idx := make([]int, 0, k)
	val := make([]float32, 0, k)
	for i, l := range logits {
		v := l * invTemp

		pos := len(val)
		for pos > 0 && val[pos-1] < v {
			pos--
		}
		if pos >= k {
			continue
		}

		idx = append(idx, 0)
		val = append(val, 0)

		copy(idx[pos+1:], idx[pos:])
		copy(val[pos+1:], val[pos:])
		idx[pos] = i
		val[pos] = v

		if len(val) > k {
			idx = idx[:k]
			val = val[:k]
		}
	}
	if len(idx) == 0 {
		return []int{0}, []float32{0}
	}
	return idx, val
}
