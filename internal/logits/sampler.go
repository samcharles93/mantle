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
	MinP          float32
	RepeatPenalty float32
	RepeatLastN   int
}

type Sampler struct {
	rng       *rand.Rand
	cfg       SamplerConfig
	greedy    bool
	topIdx    []int
	topVal    []float32
	prob      []float64
	seenMark  []uint32
	seenEpoch uint32
	seenList  []int
}

func (s *Sampler) CanUseDeviceGreedy() bool {
	return s.greedy && s.cfg.RepeatPenalty <= 1.0
}

// NewSampler returns a new sampler with the provided configuration.
func NewSampler(cfg SamplerConfig) *Sampler {
	greedy := cfg.Temperature <= 0
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
		rng:    rand.New(rand.NewSource(cfg.Seed)),
		cfg:    cfg,
		greedy: greedy,
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
		start := max(len(recent)-s.cfg.RepeatLastN, 0)
		window := recent[start:]

		if len(s.seenMark) < len(logits) {
			s.seenMark = make([]uint32, len(logits))
		}
		s.seenEpoch++
		if s.seenEpoch == 0 {
			for i := range s.seenMark {
				s.seenMark[i] = 0
			}
			s.seenEpoch = 1
		}
		s.seenList = s.seenList[:0]

		for _, id := range window {
			if id >= 0 && id < len(logits) && s.seenMark[id] != s.seenEpoch {
				s.seenMark[id] = s.seenEpoch
				s.seenList = append(s.seenList, id)
			}
		}

		for _, id := range excludePenalty {
			if id >= 0 && id < len(logits) {
				s.seenMark[id] = 0
			}
		}

		for _, id := range s.seenList {
			if id < 0 || id >= len(logits) || s.seenMark[id] != s.seenEpoch {
				continue
			}
			if logits[id] > 0 {
				logits[id] /= s.cfg.RepeatPenalty
			} else {
				logits[id] *= s.cfg.RepeatPenalty
			}
		}
	}

	if s.greedy || (s.cfg.TopK == 1 && s.cfg.TopP >= 1 && s.cfg.Temperature == 1) {
		return argmax(logits)
	}

	temp := s.cfg.Temperature
	invTemp := float32(1.0) / temp

	k := min(s.cfg.TopK, len(logits))

	topIdx, topVal := s.topK(logits, k, invTemp)
	if len(topVal) == 0 {
		return 0
	}

	maxv := topVal[0]
	for i := 1; i < len(topVal); i++ {
		if topVal[i] > maxv {
			maxv = topVal[i]
		}
	}

	if cap(s.prob) < len(topVal) {
		s.prob = make([]float64, len(topVal))
	}
	prob := s.prob[:len(topVal)]
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

	// Min-P Sampling
	if s.cfg.MinP > 0 {
		maxProb := prob[0]
		threshold := maxProb * float64(s.cfg.MinP)

		newLen := 0
		var newSum float64
		for i := 0; i < len(prob); i++ {
			if prob[i] >= threshold {
				prob[newLen] = prob[i]
				topIdx[newLen] = topIdx[i]
				newSum += prob[i]
				newLen++
			}
		}

		// If we filtered anything, resize and modify sum for re-normalization implicit in TopP?
		// Actually TopP logic iterates until cumulative sum triggers.
		// If we removed items, sum is < 1.0.
		// We should probably re-normalize or just let TopP handle it (TopP acts on cumulative).
		// But probability distribution should sum to 1.0.
		// Let's re-normalize.
		if newLen < len(prob) {
			prob = prob[:newLen]
			if newSum > 0 {
				scale := 1.0 / newSum
				for i := range prob {
					prob[i] *= scale
				}
			}
		}
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
func (s *Sampler) topK(logits []float32, k int, invTemp float32) ([]int, []float32) {
	if k <= 0 {
		return nil, nil
	}
	if cap(s.topIdx) < k+1 {
		s.topIdx = make([]int, 0, k+1)
		s.topVal = make([]float32, 0, k+1)
	}
	topIdx := s.topIdx[:0]
	topVal := s.topVal[:0]

	for i, l := range logits {
		v := l * invTemp

		pos := len(topVal)
		for pos > 0 && topVal[pos-1] < v {
			pos--
		}
		if pos >= k {
			continue
		}

		topIdx = append(topIdx, 0)
		topVal = append(topVal, 0)

		copy(topIdx[pos+1:], topIdx[pos:])
		copy(topVal[pos+1:], topVal[pos:])
		topIdx[pos] = i
		topVal[pos] = v

		if len(topVal) > k {
			topIdx = topIdx[:k]
			topVal = topVal[:k]
		}
	}
	if len(topIdx) == 0 {
		return []int{0}, []float32{0}
	}
	s.topIdx = topIdx
	s.topVal = topVal
	return topIdx, topVal
}
