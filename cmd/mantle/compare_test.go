package main

import (
	"math"
	"testing"
)

func TestLogSoftmaxInto(t *testing.T) {
	logits := []float32{1, 3, 2}
	dst := make([]float32, 3)
	if am := logSoftmaxInto(dst, logits); am != 1 {
		t.Fatalf("argmax %d, want 1", am)
	}
	z := math.Log(math.Exp(1) + math.Exp(3) + math.Exp(2))
	for i, v := range logits {
		if want := float64(v) - z; math.Abs(float64(dst[i])-want) > 1e-6 {
			t.Errorf("dst[%d] = %v, want %v", i, dst[i], want)
		}
	}
}

func TestKLDivergence(t *testing.T) {
	logOf := func(p ...float64) []float32 {
		out := make([]float32, len(p))
		for i, v := range p {
			out[i] = float32(math.Log(v))
		}
		return out
	}
	p, q := logOf(0.5, 0.25, 0.25), logOf(0.25, 0.5, 0.25)
	want := 0.5*math.Log(2) + 0.25*math.Log(0.5)
	if got := klDivergence(p, q); math.Abs(got-want) > 1e-6 {
		t.Fatalf("KL = %v, want %v", got, want)
	}
	if got := klDivergence(p, p); got != 0 {
		t.Fatalf("KL(p||p) = %v, want 0", got)
	}
}

func TestSummarizeCompare(t *testing.T) {
	kls := make([]float64, 100)
	for i := range kls {
		kls[i] = float64(100 - i) // 100..1, unsorted
	}
	s := summarizeCompare(kls, 75, 100*math.Log(4), 100*math.Log(5))
	want := compareStats{Positions: 100, MeanKL: 50.5, MedianKL: 50, P99KL: 99, MaxKL: 100, Top1Agree: 0.75, RefPPL: 4, ModelPPL: 5}
	approx := func(a, b float64) bool { return math.Abs(a-b) < 1e-9 }
	if s.Positions != want.Positions || !approx(s.MeanKL, want.MeanKL) || s.MedianKL != want.MedianKL ||
		s.P99KL != want.P99KL || s.MaxKL != want.MaxKL || s.Top1Agree != want.Top1Agree ||
		!approx(s.RefPPL, want.RefPPL) || !approx(s.ModelPPL, want.ModelPPL) {
		t.Fatalf("got %+v, want %+v", s, want)
	}
	if got := summarizeCompare(nil, 0, 0, 0); got != (compareStats{}) {
		t.Fatalf("empty input: %+v", got)
	}
}
